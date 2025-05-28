# coding=utf-8
"""util.py – Maths & geometry helpers modernised for TensorFlow 2.15.

Exactly the same public API as the original **but**:
* No `tensorflow.compat.v1` calls – everything uses eager-safe `tf.linalg`,
  `tf.math`, etc.
* Deprecated ops replaced (`tf.linspace`, `tf.math.divide_no_nan`,
  `tf.linalg.svd`, `tf.linalg.trace`, `tf.transpose`).
* Verbose print statements removed (keeps `read_pickle` side-effect-free).
* Added smoke-tests (`python util.py --test`) for critical functions.
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Tuple

# Third-party -----------------------------------------------------------------
try:
    import tensorflow as tf
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        "TensorFlow is required but not installed.\n"
        "👉  pip install 'tensorflow==2.15.*'   (or 'tensorflow-cpu')."
    ) from err

import tensorflow_probability as tfp  # type: ignore
from tensorflow_graphics.geometry.transformation import axis_angle, rotation_matrix_3d  # type: ignore

# Project-local ---------------------------------------------------------------
from pano_utils import geometry, math_utils, transformation  # type: ignore

# -----------------------------------------------------------------------------
# IO helper -------------------------------------------------------------------
# -----------------------------------------------------------------------------

def read_pickle(file: str):
    """Return keys & values from a pickle file (Python-native, no TF ops)."""
    with open(Path(file), "rb") as f:
        loaded = pickle.load(f, encoding="bytes")
    return list(loaded.keys()), list(loaded.values())

# -----------------------------------------------------------------------------
# Math helpers -----------------------------------------------------------------
# -----------------------------------------------------------------------------

def safe_sqrt(x: tf.Tensor) -> tf.Tensor:
    return tf.sqrt(tf.maximum(x, 1e-20))


def degrees_to_radians(degree: float | tf.Tensor) -> tf.Tensor:
    return math.pi * tf.cast(degree, tf.float32) / 180.0


def radians_to_degrees(radians: tf.Tensor) -> tf.Tensor:
    return 180.0 * radians / math.pi


def angular_distance(v1: tf.Tensor, v2: tf.Tensor) -> tf.Tensor:
    dot = tf.reduce_sum(v1 * v2, axis=-1)
    return tf.acos(tf.clip_by_value(dot, -1.0, 1.0))


# -----------------------------------------------------------------------------
# Spherical image utilities ----------------------------------------------------
# -----------------------------------------------------------------------------

def equirectangular_area_weights(height: int | tf.Tensor) -> tf.Tensor:
    """Area-weighting mask for equirectangular pixels (shape `[1,H,1,1]`)."""
    pixel_h = math.pi / tf.cast(height, tf.float32)
    colat = tf.linspace(pixel_h / 2, math.pi - pixel_h / 2, tf.cast(height, tf.int32))
    colat = colat[tf.newaxis, :, tf.newaxis, tf.newaxis]
    return tf.sin(colat)


def spherical_normalization(x: tf.Tensor, *, rectify: bool = True) -> tf.Tensor:
    """Normalize raw spherical maps so their weighted integral equals 1."""
    if rectify:
        x = tf.nn.softplus(x)
    h = x.shape[1] or tf.shape(x)[1]
    weights = equirectangular_area_weights(h)
    weighted = x * weights
    return tf.math.divide_no_nan(x, tf.reduce_sum(weighted, axis=[1, 2], keepdims=True))


def spherical_expectation(spherical_probs: tf.Tensor) -> tf.Tensor:
    """Return expectation vectors for *normalized* spherical maps."""
    b, h, w, c = spherical_probs.shape
    if h is None:
        h = tf.shape(spherical_probs)[1]
        w = tf.shape(spherical_probs)[2]
        c = tf.shape(spherical_probs)[3]

    grid = geometry.generate_equirectangular_grid([h, w])  # (H,W,2)
    dirs = geometry.spherical_to_cartesian(grid)  # (H,W,3)
    axis_conv = tf.constant([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    dirs = tf.linalg.matmul(axis_conv, dirs[..., tf.newaxis])[..., 0]  # (H,W,3)

    dirs = tf.tile(dirs[tf.newaxis, tf.newaxis], [b, c, 1, 1, 1])  # (B,C,H,W,3)
    weights = equirectangular_area_weights(h)
    weighted = spherical_probs * weights
    expectation = tf.reduce_sum(dirs * weighted[:, :, :, :, tf.newaxis], axis=[2, 3])  # (B,C,3)
    return expectation


# -----------------------------------------------------------------------------
# von Mises-Fisher helpers -----------------------------------------------------
# -----------------------------------------------------------------------------

def von_mises_fisher(mean: tf.Tensor, concentration: float | tf.Tensor, shape: Tuple[int, int]) -> tf.Tensor:  # noqa: D401
    if len(shape) != 2:
        raise ValueError("shape must be [height, width]")
    if mean.shape[-1] != 3:
        raise ValueError("mean must end with dim 3 (xyz)")

    b, n = mean.shape[0], mean.shape[1]
    h, w = shape

    grid = geometry.generate_equirectangular_grid([h, w])  # (H,W,2)
    cart = geometry.spherical_to_cartesian(grid)  # (H,W,3)
    axis_conv = tf.constant([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    cart = tf.linalg.matmul(axis_conv, cart[..., tf.newaxis])[..., 0]  # (H,W,3)

    cart = tf.tile(cart[tf.newaxis, tf.newaxis], [b, n, 1, 1, 1])
    mean_tiled = tf.tile(mean[:, :, tf.newaxis, tf.newaxis], [1, 1, h, w, 1])

    vmf = tfp.distributions.VonMisesFisher(mean_direction=mean_tiled, concentration=tf.cast(concentration, tf.float32))
    raw = vmf.prob(cart)  # (B,N,H,W)
    return tf.transpose(raw, [0, 2, 3, 1])  # (B,H,W,N)


# -----------------------------------------------------------------------------
# Rotation helpers -------------------------------------------------------------
# -----------------------------------------------------------------------------

def rotation_geodesic(r1: tf.Tensor, r2: tf.Tensor) -> tf.Tensor:
    diff = (tf.linalg.trace(tf.linalg.matmul(r1, r2, transpose_b=True)) - 1.0) / 2.0
    return tf.acos(tf.clip_by_value(diff, -1.0, 1.0))


def gram_schmidt(m: tf.Tensor) -> tf.Tensor:
    x, y = m[:, 0], m[:, 1]
    xn = tf.math.l2_normalize(x, axis=-1)
    z = tf.linalg.cross(xn, y)
    zn = tf.math.l2_normalize(z, axis=-1)
    y = tf.linalg.cross(zn, xn)
    return tf.stack([xn, y, zn], axis=1)


def svd_orthogonalize(m: tf.Tensor) -> tf.Tensor:
    m_t = tf.transpose(tf.math.l2_normalize(m, axis=-1), perm=[0, 2, 1])
    s, u, v = tf.linalg.svd(m_t)  # note: returns s,u,v already transposed
    det = tf.linalg.det(tf.linalg.matmul(v, u, transpose_b=True))
    v_adj = tf.concat([v[:, :, :-1], v[:, :, -1:] * det[:, tf.newaxis, tf.newaxis]], axis=2)
    return tf.linalg.matmul(v_adj, u, transpose_b=True)


def perturb_rotation(r: tf.Tensor, perturb_limits):
    x, y, z = tf.split(r, [1, 1, 1], axis=1)
    x = math_utils.normal_sampled_vector_within_cone(tf.squeeze(x, 1), degrees_to_radians(perturb_limits[0]), 0.5)
    y = math_utils.normal_sampled_vector_within_cone(tf.squeeze(y, 1), degrees_to_radians(perturb_limits[1]), 0.5)
    z = math_utils.normal_sampled_vector_within_cone(tf.squeeze(z, 1), degrees_to_radians(perturb_limits[2]), 0.5)
    return svd_orthogonalize(tf.stack([x, y, z], axis=1))


def half_rotation(rotation: tf.Tensor) -> tf.Tensor:
    axes, angles = axis_angle.from_rotation_matrix(rotation)
    return rotation_matrix_3d.from_axis_angle(axes, angles / 2.0)


# -----------------------------------------------------------------------------
# Distribution → direction -----------------------------------------------------
# -----------------------------------------------------------------------------

def distributions_to_directions(x: tf.Tensor):
    dist = spherical_normalization(x)
    expectation = spherical_expectation(dist)
    return tf.nn.l2_normalize(expectation, axis=-1), expectation, dist


# -----------------------------------------------------------------------------
# Image derotation -------------------------------------------------------------
# -----------------------------------------------------------------------------

def derotation(
    src_img: tf.Tensor,
    trt_img: tf.Tensor,
    rotation: tf.Tensor,
    input_fov: tf.Tensor,
    output_fov: float,
    output_shape: Tuple[int, int],
    derotate_both: bool,
):
    batch = tf.shape(src_img)[0]
    if derotate_both:
        half_rot = half_rotation(rotation)
        src = transformation.rotate_image_in_3d(src_img, tf.transpose(half_rot, perm=[0, 2, 1]), input_fov, output_fov, output_shape)
        trt = transformation.rotate_image_in_3d(trt_img, half_rot, input_fov, output_fov, output_shape)
    else:
        eye = tf.eye(3, batch_shape=[batch])
        src = transformation.rotate_image_in_3d(src_img, eye, input_fov, output_fov, output_shape)
        trt = transformation.rotate_image_in_3d(trt_img, rotation, input_fov, output_fov, output_shape)
    return src, trt


# -----------------------------------------------------------------------------
# Smoke-tests ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def _tests():
    # Basic directional maths
    v1 = tf.math.l2_normalize(tf.random.normal([8, 3]), axis=-1)
    v2 = tf.math.l2_normalize(tf.random.normal([8, 3]), axis=-1)
    ad = angular_distance(v1, v2)
    assert ad.shape == (8,), ad.shape

    # vMF shapes
    mean = tf.math.l2_normalize(tf.random.normal([2, 3, 3]), axis=-1)
    vmf = von_mises_fisher(mean, 10.0, [64, 64])
    assert vmf.shape == (2, 64, 64, 3), vmf.shape

    print("✔ util.py smoke-tests passed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test:
        _tests()
    else:
        print("Run with --test for smoke checks.")