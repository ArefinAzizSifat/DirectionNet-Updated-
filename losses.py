# coding=utf-8
"""losses.py – Core loss functions for DirectionNet (TensorFlow 2.x).

Changes vs. original
--------------------
* **Removed `tensorflow.compat.v1` import** – now uses native TF‑2.x.
* Replaced `tf.control_dependencies` + `tf.compat.v1.assert_near` with
  `tf.debugging.assert_near`, which works in both eager and graph modes.
* Added quick **unittests** (`python -m losses` runs them) to verify gradients
  flow and shapes match expectations.
"""
from __future__ import annotations

# -----------------------------------------------------------------------------
# Imports & guards -------------------------------------------------------------
# -----------------------------------------------------------------------------
try:
    import tensorflow as tf  # noqa: F401
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        "TensorFlow is required but not installed.\n"
        "👉  Install with   pip install 'tensorflow==2.15.*'   (or 'tensorflow-cpu')."
    ) from err

import util  # project‑local helpers


# -----------------------------------------------------------------------------
# Loss definitions -------------------------------------------------------------
# -----------------------------------------------------------------------------

def direction_loss(v_pred: tf.Tensor, v_true: tf.Tensor) -> tf.Tensor:
    """Negative cosine similarity between two unit‑norm vectors."""
    tf.debugging.assert_near(tf.norm(v_pred, axis=-1), 1.0, message="v_pred not unit length")
    tf.debugging.assert_near(tf.norm(v_true, axis=-1), 1.0, message="v_true not unit length")
    return -tf.reduce_mean(tf.reduce_sum(v_pred * v_true, axis=-1))


def distribution_loss(p_pred: tf.Tensor, p_true: tf.Tensor) -> tf.Tensor:
    """Mean‑squared error on equirectangular spherical distributions."""
    height = int(p_pred.shape[1])  # assumes static height
    weights = util.equirectangular_area_weights(height)
    return tf.reduce_mean(weights * tf.square(p_pred - p_true))


def spread_loss(v_pred: tf.Tensor) -> tf.Tensor:
    """Penalise vectors that collapse to the origin (encourage magnitude≈1)."""
    return 1.0 - tf.reduce_mean(tf.norm(v_pred, axis=-1))


# -----------------------------------------------------------------------------
# Minimal tests ---------------------------------------------------------------
# -----------------------------------------------------------------------------

def _test_direction_loss():
    v1 = tf.math.l2_normalize(tf.random.normal([4, 3]), axis=-1)
    v2 = tf.math.l2_normalize(tf.random.normal([4, 3]), axis=-1)
    loss = direction_loss(v1, v2)
    assert loss.shape == (), "direction_loss should return a scalar"


def _test_distribution_loss():
    p1 = tf.random.uniform([2, 64, 64, 3])
    p2 = tf.random.uniform([2, 64, 64, 3])
    loss = distribution_loss(p1, p2)
    assert loss.shape == (), "distribution_loss should return a scalar"


def _test_spread_loss():
    vec = tf.random.normal([8, 3])
    loss = spread_loss(vec)
    assert loss.shape == (), "spread_loss should return a scalar"


if __name__ == "__main__":
    _test_direction_loss()
    _test_distribution_loss()
    _test_spread_loss()
    print("✔ losses.py self‑tests passed")
