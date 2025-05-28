# coding=utf-8
"""dataset.py – panorama stereo dataset utilities (TensorFlow 2.15+)

This version adds a **robust import guard** so users get clear guidance when
TensorFlow is missing instead of the opaque *ModuleNotFoundError*.

Run once inside a fresh environment:

```bash
python -m pip install "tensorflow==2.15.*"   # add `-cpu` or `-gpu` flavour as needed
```

The rest of the file is identical to the previous refactor except for:
* `try / except` wrapper around the `import tensorflow as tf` line
* A tiny `unittest` that confirms TensorFlow is available (skipped otherwise)
"""
from __future__ import annotations

import collections
import math
from pathlib import Path
from typing import Tuple

import numpy as np

# -----------------------------------------------------------------------------
# TensorFlow import guard ------------------------------------------------------
# -----------------------------------------------------------------------------
try:
    import tensorflow as tf  # noqa: F401 – we need the side‑effect of import
except ModuleNotFoundError as err:  # pragma: no cover – user feedback path
    raise ModuleNotFoundError(
        "TensorFlow is required but not installed.\n"
        "👉  Install with   pip install 'tensorflow==2.15.*'   (or 'tensorflow-cpu')."
    ) from err

from pano_utils import math_utils, transformation  # noqa: E402 – after TF

# -----------------------------------------------------------------------------
# Geometry helpers -------------------------------------------------------------
# -----------------------------------------------------------------------------


def world_to_image_projection(
    p_world: tf.Tensor, intrinsics: tf.Tensor, pose_w2c: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Project 3‑D world points onto the image plane."""
    ones = tf.ones_like(p_world[..., :1])
    p_world_h = tf.concat([p_world, ones], axis=-1)
    p_camera = tf.squeeze(
        tf.matmul(pose_w2c[tf.newaxis, tf.newaxis, :], p_world_h[..., tf.newaxis]),
        axis=-1,
    )
    p_camera *= tf.constant([1.0, 1.0, -1.0], dtype=tf.float32)
    p_image_h = tf.squeeze(
        tf.matmul(intrinsics[tf.newaxis, tf.newaxis, :], p_camera[..., tf.newaxis]),
        axis=-1,
    )
    z = p_image_h[..., -1:]
    p_image = tf.math.divide_no_nan(p_image_h[..., :2], z)
    return p_image, z


def image_to_world_projection(
    depth: tf.Tensor, intrinsics: tf.Tensor, pose_c2w: tf.Tensor
) -> tf.Tensor:
    """Lift image‑plane pixels back to 3‑D world coordinates."""
    H, W = tf.unstack(tf.shape(depth)[:2])
    xx, yy = tf.meshgrid(
        tf.linspace(0.0, tf.cast(W - 1, tf.float32), W),
        tf.linspace(0.0, tf.cast(H - 1, tf.float32), H),
    )
    pix_h = tf.concat([tf.stack([xx, yy], axis=-1), tf.ones([H, W, 1])], axis=-1)
    p_image = tf.squeeze(
        tf.matmul(tf.linalg.inv(intrinsics)[tf.newaxis, tf.newaxis, :], pix_h[..., tf.newaxis]),
        axis=-1,
    )
    cos_angle = tf.reduce_sum(
        tf.math.l2_normalize(p_image, axis=-1) * tf.constant([[[0.0, 0.0, 1.0]]]),
        axis=-1,
        keepdims=True,
    )
    z = depth * cos_angle
    p_camera = z * p_image
    p_camera *= tf.constant([1.0, 1.0, -1.0], dtype=tf.float32)
    p_camera_h = tf.concat([p_camera, tf.ones_like(p_camera[..., :1])], axis=-1)
    p_world = tf.squeeze(
        tf.matmul(pose_c2w[tf.newaxis, tf.newaxis, :], p_camera_h[..., tf.newaxis]),
        axis=-1,
    )
    return p_world


# -----------------------------------------------------------------------------
# Overlap helpers --------------------------------------------------------------
# -----------------------------------------------------------------------------

def overlap_mask(
    depth1: tf.Tensor,
    pose1_c2w: tf.Tensor,
    depth2: tf.Tensor,
    pose2_c2w: tf.Tensor,
    intrinsics: tf.Tensor,
):
    """Return boolean masks of the overlapping regions between two views."""
    pad = tf.constant([[0.0, 0.0, 0.0, 1.0]])
    pose1_w2c = tf.linalg.inv(tf.concat([pose1_c2w, pad], axis=0))[:3]
    pose2_w2c = tf.linalg.inv(tf.concat([pose2_c2w, pad], axis=0))[:3]

    p_world1 = image_to_world_projection(depth1, intrinsics, pose1_c2w)
    p_img1_in_2, z1_c2 = world_to_image_projection(p_world1, intrinsics, pose2_w2c)

    p_world2 = image_to_world_projection(depth2, intrinsics, pose2_c2w)
    p_img2_in_1, z2_c1 = world_to_image_projection(p_world2, intrinsics, pose1_w2c)

    H, W = tf.unstack(tf.shape(depth1)[:2])
    H = tf.cast(H, tf.float32)
    W = tf.cast(W, tf.float32)
    eps = 1e-4

    def _inside(im_xy):
        y_ok = tf.logical_and(im_xy[..., 1] <= H + eps, im_xy[..., 1] >= -eps)
        x_ok = tf.logical_and(im_xy[..., 0] <= W + eps, im_xy[..., 0] >= -eps)
        return tf.logical_and(x_ok, y_ok)

    mask2_in_1 = tf.logical_and(_inside(p_img2_in_1), tf.squeeze(z2_c1, -1) > 0)
    mask1_in_2 = tf.logical_and(_inside(p_img1_in_2), tf.squeeze(z1_c2, -1) > 0)
    return mask1_in_2, mask2_in_1


def overlap_ratio(mask1: tf.Tensor, mask2: tf.Tensor) -> tf.Tensor:
    H, W = tf.unstack(tf.shape(mask1)[:2])
    area = tf.cast(H * W, tf.float32)
    return tf.minimum(
        tf.reduce_sum(tf.cast(mask1, tf.float32)) / area,
        tf.reduce_sum(tf.cast(mask2, tf.float32)) / area,
    )


# -----------------------------------------------------------------------------
# Dataset helpers --------------------------------------------------------------
# -----------------------------------------------------------------------------

def _string_to_matrix(s: tf.Tensor, shape: Tuple[int, ...]) -> tf.Tensor:  # noqa: D401
    """Convert a whitespace‑separated string to a numeric Tensor."""
    flat = tf.io.decode_csv(s, record_defaults=[0.0] * int(np.prod(shape)))
    return tf.reshape(tf.stack(flat), shape)


def generate_from_meta(
    meta_data_path: str,
    pano_data_dir: str,
    *,
    pano_height: int = 1024,
    pano_width: int = 2048,
    output_height: int = 512,
    output_width: int = 512,
) -> tf.data.Dataset:
    """Return a `tf.data.Dataset` of stereo view pairs using Matterport3D meta."""

    meta_root = Path(meta_data_path)
    pano_root = tf.constant(str(Path(pano_data_dir)))

    if not meta_root.exists():
        raise FileNotFoundError(meta_root)
    if not Path(pano_data_dir).exists():
        raise FileNotFoundError(pano_data_dir)

    def _load_text(fp: tf.Tensor, n_lines: int = 200):
        return tf.data.TextLineDataset(fp).batch(n_lines).unbatch()

    def _load_image(fname: tf.Tensor) -> tf.Tensor:
        img = tf.io.decode_jpeg(tf.io.read_file(fname), channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img.set_shape([pano_height, pano_width, 3])
        return img

    def _decode_line(line: tf.Tensor):
        DataPair = collections.namedtuple(
            "DataPair", ["src_img", "trt_img", "fov", "rotation", "translation"]
        )
        parts = tf.io.decode_csv(line, record_defaults=[""] * 10, field_delim=" ")
        scan, pano1, pano2, *rest = parts

        img1_path = tf.strings.join([pano_root, scan, "/", pano1, ".jpeg"])
        img2_path = tf.strings.join([pano_root, scan, "/", pano2, ".jpeg"])

        fov = _string_to_matrix(rest[0], (1,))
        r1 = _string_to_matrix(rest[1], (3, 3))
        t1 = _string_to_matrix(rest[2], (3,))
        r2 = _string_to_matrix(rest[3], (3, 3))
        t2 = _string_to_matrix(rest[4], (3,))
        s_r1 = _string_to_matrix(rest[5], (3, 3))
        s_r2 = _string_to_matrix(rest[6], (3, 3))

        r_c2_to_c1 = tf.matmul(s_r1, s_r2, transpose_a=True)
        t_c1 = tf.squeeze(
            tf.matmul(
                s_r1, tf.expand_dims(tf.linalg.l2_normalize(t2 - t1), -1), transpose_a=True
            )
        )

        sampled_rot = tf.matmul(tf.stack([s_r1, s_r2]), tf.stack([r1, r2]), transpose_a=True)
        views
