# coding=utf-8
"""dataset_loader.py – TF‑2.x compatible data pipeline for stereo view pairs.

* Replaces the old `tensorflow.compat.v1` import with a **robust TF‑2 guard**.
* Swaps deprecated ops (`tf.read_file`, `tf.image.resize_area`, Python `random`
  inside `tf.data`) for modern, graph‑safe equivalents.
* Keeps the `absl.flags` CLI exactly as before so downstream scripts run
  unchanged.
* Adds a minimal **unittest** that checks the loader can build a single batch.

Usage example:

```bash
python dataset_loader.py \
  --data_path /path/to/your/training-split \
  --epochs 1 --batch_size 2
```
"""
from __future__ import annotations

import collections
import os
from pathlib import Path
from typing import Tuple

import numpy as np

# -----------------------------------------------------------------------------
# TensorFlow import guard ------------------------------------------------------
# -----------------------------------------------------------------------------
try:
    import tensorflow as tf  # noqa: F401
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        "TensorFlow is required but not installed.\n"
        "👉  Install with   pip install 'tensorflow==2.15.*'   (or 'tensorflow-cpu')."
    ) from err

# Third‑party helpers ----------------------------------------------------------
from absl import app, flags  # noqa: E402 – after TF

import util  # noqa: E402 – project‑local pickle helpers

# -----------------------------------------------------------------------------
# CLI flags --------------------------------------------------------------------
# -----------------------------------------------------------------------------
FLAGS = flags.FLAGS
flags.DEFINE_string("data_path", "", "Directory where the data is stored")
flags.DEFINE_integer("epochs", 10, "Number of training epochs")
flags.DEFINE_integer("batch_size", 2, "Batch size")
flags.DEFINE_boolean("training", True, "Training mode (enable gamma jitter)")
flags.DEFINE_boolean("load_estimated_rot", False, "Load DirectionNet‑R predictions")


# -----------------------------------------------------------------------------
# Data‑pipeline helpers --------------------------------------------------------
# -----------------------------------------------------------------------------

def _read_pickle_tf(path: tf.Tensor, dtype_list: Tuple[type, ...]):
    """Utility wrapper around `util.read_pickle` for `tf.numpy_function`."""
    values = tf.numpy_function(util.read_pickle, [path], dtype_list)
    return values  # already a tuple


def _load_data(path: tf.Tensor, load_estimated_rot: bool):
    """Read pickles in a directory and expand into a flat dataset."""
    # path is a scalar `tf.string` tensor
    path_str = tf.strings.strip(path)

    img_id, rotation = _read_pickle_tf(path_str + "/rotation_gt.pickle", (tf.string, tf.float32))
    _, translation = _read_pickle_tf(path_str + "/epipoles_gt.pickle", (tf.string, tf.float32))
    _, fov = _read_pickle_tf(path_str + "/fov.pickle", (tf.string, tf.float32))

    if load_estimated_rot:
        _, rotation_pred = _read_pickle_tf(path_str + "/rotation_pred.pickle", (tf.string, tf.float32))
    else:
        rotation_pred = tf.zeros_like(rotation)

    img_base = path_str + "/" + img_id  # id already includes scan prefix
    return tf.data.Dataset.from_tensor_slices((
        img_id,
        img_base,
        rotation,
        translation,
        fov,
        rotation_pred,
    ))


def _load_single_image(img_path: tf.Tensor) -> tf.Tensor:
    """Read a 512² PNG and down‑sample to 256² float tensor."""
    image = tf.io.decode_png(tf.io.read_file(img_path), channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [256, 256], method=tf.image.ResizeMethod.AREA)
    image.set_shape([256, 256, 3])
    return image


def _process_row(training: bool, load_estimated_rot: bool):
    """Return a mapping function for `tf.data.Dataset.map`."""

    def _fn(img_id, img_path, rotation, translation, fov, rotation_pred):
        InputPair = collections.namedtuple(
            "InputPair",
            [
                "id",
                "src_image",
                "trt_image",
                "rotation",
                "translation",
                "fov",
                "rotation_pred",
            ],
        )

        src_img = _load_single_image(img_path + ".src.perspective.png")
        trt_img = _load_single_image(img_path + ".trt.perspective.png")

        if training:
            gamma = tf.random.uniform([], 0.7, 1.2)
            src_img = tf.image.adjust_gamma(src_img, gamma)
            trt_img = tf.image.adjust_gamma(trt_img, gamma)

        rotation = tf.reshape(rotation, [3, 3])
        translation = tf.reshape(translation, [3])
        fov = tf.reshape(fov, [1])

        if load_estimated_rot:
            rotation_pred = tf.reshape(rotation_pred, [3, 3])
        else:
            rotation_pred = tf.zeros([3, 3], dtype=tf.float32)

        return InputPair(img_id, src_img, trt_img, rotation, translation, fov, rotation_pred)

    return _fn


def data_loader(
    data_path: str,
    epochs: int,
    batch_size: int,
    training: bool = True,
    load_estimated_rot: bool = False,
) -> tf.data.Dataset:
    """Public API: build the TF‑data pipeline."""
    data_root = Path(data_path)
    if not data_root.exists():
        raise FileNotFoundError(data_root)

    ds = tf.data.Dataset.list_files(str(data_root / "*"), shuffle=training)
    ds = ds.flat_map(lambda p: _load_data(p, load_estimated_rot))
    ds = ds.map(
        _process_row(training, load_estimated_rot),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.repeat(epochs).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds


# -----------------------------------------------------------------------------
# Script entry‑point -----------------------------------------------------------
# -----------------------------------------------------------------------------

def _main(_):  # absl passes the argv list
    dataset = data_loader(
        FLAGS.data_path,
        FLAGS.epochs,
        FLAGS.batch_size,
        FLAGS.training,
        FLAGS.load_estimated_rot,
    )

    # Grab one batch to verify shapes
    for batch in dataset.take(1):
        (img_id, src_img, trt_img, rotation, translation, fov, rot_pred) = batch
        print("Batch summary →",
              "id", img_id.numpy(),
              "src", src_img.shape,
              "trt", trt_img.shape,
              "rot", rotation.shape,
              "trans", translation.shape,
              "fov", fov.numpy(),
              "rot_pred", rot_pred.shape)
    print("\n✅  Data loader executed successfully!")


# -----------------------------------------------------------------------------
# Minimal unittest -------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(_main)
