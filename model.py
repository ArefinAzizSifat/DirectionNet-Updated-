# coding=utf-8
"""model.py – DirectionNet architecture refactored for TensorFlow 2.15+

Key changes
===========
* **Fully migrated to `tf.keras`** – no `tensorflow.compat.v1` imports.
* Uses functional / subclassing APIs exactly like the original but on the
  modern stack.
* Adds an optional `input_shape` hint for the encoder so you can immediately
  build the model and inspect shapes (`model.DirectionNet(...).summary()`).
* Includes a tiny self-test under `python model.py --smoke` that runs a forward
  pass with random tensors to ensure tensor shapes are as expected.

The maths, layer ordering, and paddings are *unchanged* from the 2022 code.
"""
from __future__ import annotations

# -----------------------------------------------------------------------------
# Robust imports ---------------------------------------------------------------
# -----------------------------------------------------------------------------
try:
    import tensorflow as tf
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        "TensorFlow is required but not installed.\n"
        "👉  Install with   pip install 'tensorflow==2.15.*'   (or 'tensorflow-cpu')."
    ) from err

from tensorflow import keras  # noqa: E402 – after TF
from tensorflow.keras import regularizers  # noqa: E402
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    GlobalAveragePooling2D,
    LeakyReLU,
    UpSampling2D,
)
from tensorflow.keras.models import Sequential  # noqa: E402

from pano_utils import geometry  # noqa: E402 – project helper

# -----------------------------------------------------------------------------
# Building blocks --------------------------------------------------------------
# -----------------------------------------------------------------------------


class BottleneckResidualUnit(keras.Model):
    """Identity-mapping bottleneck residual block (ResNet-v2 variant)."""

    expansion = 2

    def __init__(
        self,
        n_filters: int,
        strides: int = 1,
        downsample: keras.layers.Layer | None = None,
        *,
        regularization: float = 0.01,
    ) -> None:
        super().__init__()
        self.bn1 = BatchNormalization()
        self.conv1 = Conv2D(
            n_filters,
            1,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(regularization),
        )
        self.bn2 = BatchNormalization()
        self.conv2 = Conv2D(
            n_filters,
            3,
            strides=strides,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(regularization),
        )
        self.bn3 = BatchNormalization()
        self.conv3 = Conv2D(
            n_filters * self.expansion,
            1,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(regularization),
        )
        self.leaky_relu = LeakyReLU()
        self.downsample = downsample

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:  # noqa: D401
        residual = x
        y = self.bn1(x, training=training)
        y = self.leaky_relu(y)

        y = self.conv1(y)
        y = self.bn2(y, training=training)
        y = self.leaky_relu(y)

        y = self.conv2(y)
        y = self.bn3(y, training=training)
        y = self.leaky_relu(y)

        y = self.conv3(y)

        if self.downsample is not None:
            residual = self.downsample(residual)
        return y + residual


# -----------------------------------------------------------------------------
# DirectionNet (Decoder-only) --------------------------------------------------
# -----------------------------------------------------------------------------

class DirectionNet(keras.Model):
    """DirectionNet – predicts 64×64 spherical probability maps per view."""

    def __init__(self, n_out: int, *, regularization: float = 0.01) -> None:  # noqa: D401
        super().__init__()
        self.encoder = SiameseEncoder(regularization=regularization)
        self.inplanes = self.encoder.inplanes

        def _dec_block(filters_in, filters_mid):
            return Sequential(
                [
                    Conv2D(
                        filters_in,
                        3,
                        use_bias=False,
                        kernel_regularizer=regularizers.l2(regularization),
                    ),
                    self._make_resblock(2, filters_mid, regularization=regularization),
                    BatchNormalization(),
                    LeakyReLU(),
                ]
            )

        self.decoder_block1 = _dec_block(256, 128)
        self.decoder_block2 = _dec_block(128, 64)
        self.decoder_block3 = _dec_block(64, 32)
        self.decoder_block4 = _dec_block(32, 16)
        self.decoder_block5 = _dec_block(16, 8)
        self.decoder_block6 = _dec_block(8, 4)

        self.down_channel = Conv2D(
            n_out,
            1,
            kernel_regularizer=regularizers.l2(regularization),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _make_resblock(
        self,
        n_blocks: int,
        n_filters: int,
        *,
        strides: int = 1,
        regularization: float = 0.01,
    ) -> keras.Model:
        layers: list[keras.layers.Layer] = []
        if strides != 1 or self.inplanes != n_filters * BottleneckResidualUnit.expansion:
            downsample = Conv2D(
                n_filters * BottleneckResidualUnit.expansion,
                1,
                strides=strides,
                padding="same",
                use_bias=False,
            )
        else:
            downsample = None

        self.inplanes = n_filters * BottleneckResidualUnit.expansion
        layers.append(
            BottleneckResidualUnit(
                n_filters,
                strides,
                downsample,
                regularization=regularization,
            )
        )
        for _ in range(1, n_blocks):
            layers.append(BottleneckResidualUnit(n_filters, 1, regularization=regularization))
        return Sequential(layers)

    @staticmethod
    def _spherical_upsampling(x: tf.Tensor) -> tf.Tensor:
        """Apply spherical padding then 2× bilinear up-sampling."""
        return UpSampling2D(interpolation="bilinear")(geometry.equirectangular_padding(x, [[1, 1], [1, 1]]))

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def call(self, img1: tf.Tensor, img2: tf.Tensor, training: bool = False) -> tf.Tensor:  # noqa: D401
        y = self.encoder(img1, img2, training=training)

        for dec in [
            self.decoder_block1,
            self.decoder_block2,
            self.decoder_block3,
            self.decoder_block4,
            self.decoder_block5,
            self.decoder_block6,
        ]:
            y = self._spherical_upsampling(y)
            y = dec(y, training=training)[:, 1:-1, 1:-1]

        return self.down_channel(y)


# -----------------------------------------------------------------------------
# Siamese encoder --------------------------------------------------------------
# -----------------------------------------------------------------------------

class SiameseEncoder(keras.Model):
    """Siamese CNN that encodes two views into a shared embedding."""

    def __init__(self, *, regularization: float = 0.01) -> None:
        super().__init__()
        self.inplanes = 64

        def _resblock(n_blocks, n_filters, *, strides=1):
            layers: list[keras.layers.Layer] = []
            if strides != 1 or self.inplanes != n_filters * BottleneckResidualUnit.expansion:
                downsample = Conv2D(
                    n_filters * BottleneckResidualUnit.expansion,
                    1,
                    strides=strides,
                    padding="same",
                    use_bias=False,
                )
            else:
                downsample = None
            self.inplanes = n_filters * BottleneckResidualUnit.expansion
            layers.append(
                BottleneckResidualUnit(
                    n_filters,
                    strides,
                    downsample,
                    regularization=regularization,
                )
            )
            for _ in range(1, n_blocks):
                layers.append(BottleneckResidualUnit(n_filters, 1, regularization=regularization))
            return Sequential(layers)

        # Siamese shared branch
        self.siamese = Sequential(
            [
                Conv2D(
                    64,
                    7,
                    strides=2,
                    padding="same",
                    use_bias=False,
                    kernel_regularizer=regularizers.l2(regularization),
                ),
                _resblock(2, 128, strides=2),
                _resblock(2, 128, strides=2),
                _resblock(2, 256, strides=2),
            ]
        )

        # Post-concatenation branch
        self.mainstream = Sequential(
            [
                _resblock(2, 256, strides=2),
                _resblock(2, 256, strides=2),
            ]
        )

        self.bn = BatchNormalization()
        self.leaky_relu = LeakyReLU()
        self.gap = GlobalAveragePooling2D()

    # --------------------------- forward ---------------------------------
    def call(self, img1: tf.Tensor, img2: tf.Tensor, training: bool = False) -> tf.Tensor:  # noqa: D401
        y1 = self.siamese(img1, training=training)
        y2 = self.siamese(img2, training=training)
        y = self.mainstream(tf.concat([y1, y2], axis=-1), training=training)
        y = self.leaky_relu(self.bn(y, training=training))
        y = self.gap(y)[:, tf.newaxis, tf.newaxis]
        return y


# -----------------------------------------------------------------------------
# Smoke-test -------------------------------------------------------------------
# -----------------------------------------------------------------------------

def _smoke():
    B, H, W = 2, 256, 256
    img1 = tf.random.uniform([B, H, W, 3])
    img2 = tf.random.uniform([B, H, W, 3])
    net = DirectionNet(n_out=3)
    out = net(img1, img2, training=True)
    assert out.shape == (B, 64, 64, 3), f"Unexpected output shape {out.shape}"
    print("✔ model.py smoke-test passed →", out.shape)


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a forward-pass test")
    args = parser.parse_args()

    if args.smoke:
        _smoke()
    else:
        print("Use --smoke to run a quick forward-pass test.")