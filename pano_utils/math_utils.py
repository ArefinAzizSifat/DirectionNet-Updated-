# coding=utf-8
"""math_utils.py – TF-2.x utilities (no tf.compat.v1).

Updates vs. 2022 code
---------------------
* **Removed every `tensorflow.compat.v1` import**; uses native TF 2.16 APIs.
* Replaced deprecated ops: `tf.random_uniform` → `tf.random.uniform`,
  `tf.cross` → `tf.linalg.cross`, `tf.arg_max` → `tf.argmax`,
  `tf.truncated_normal` → `tf.random.truncated_normal`.
* Added dtype consistency and eager-safe `tf.debugging` checks.
"""
from __future__ import annotations

import math
import tensorflow as tf
import tensorflow_probability as tfp

__all__ = [
    "degrees_to_radians",
    "radians_to_degrees",
    "safe_sqrt",
    "argmax2d",
    "lookat_matrix",
    "skew_symmetric",
    "random_vector_on_sphere",
    "uniform_sampled_vector_within_cone",
    "normal_sampled_vector_within_cone",
    "rotation_between_vectors",
]

# -----------------------------------------------------------------------------
# General helpers --------------------------------------------------------------
# -----------------------------------------------------------------------------

def degrees_to_radians(degree):
    return math.pi * degree / 180.0


def radians_to_degrees(radians):
    return 180.0 * radians / math.pi


def safe_sqrt(x):
    return tf.sqrt(tf.maximum(x, 1e-20))

# -----------------------------------------------------------------------------
# Array helpers ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def argmax2d(tensor: tf.Tensor):
    """Return (row, col) of max value for each channel in a 4-D tensor."""
    tf.debugging.assert_rank(tensor, 4)
    batch = tf.shape(tensor)[0]
    height = tf.shape(tensor)[1]
    width = tf.shape(tensor)[2]
    channels = tf.shape(tensor)[3]

    flat = tf.reshape(tensor, (batch, -1, channels))
    index = tf.cast(tf.argmax(flat, axis=1), tf.int32)
    y = index // width
    x = index % width
    return tf.stack([y, x], axis=-1)

# -----------------------------------------------------------------------------
# Geometry helpers -------------------------------------------------------------
# -----------------------------------------------------------------------------

def lookat_matrix(up: tf.Tensor, lookat_direction: tf.Tensor):
    """Return camera-to-world rotation matrix given look-at & up vectors."""
    z = tf.linalg.l2_normalize(-lookat_direction, axis=-1)
    x = tf.linalg.l2_normalize(tf.linalg.cross(up, z), axis=-1)
    y = tf.linalg.cross(z, x)
    return tf.stack([x, y, z], axis=-1)


def skew_symmetric(v: tf.Tensor):
    tf.debugging.assert_equal(tf.shape(v)[-1], 3, "v must be [...,3]")
    batch = tf.shape(v)[0]
    v1, v2, v3 = tf.split(v, [1, 1, 1], axis=-1)
    zeros = tf.zeros([batch, 1], dtype=v.dtype)
    lower = tf.concat([zeros, v1, -v2, zeros, zeros, v3], axis=-1)
    upper = tf.concat([zeros, -v3, v2, zeros, zeros, -v1], axis=-1)
    return tfp.math.fill_triangular(lower) + tfp.math.fill_triangular(upper, upper=True)

# -----------------------------------------------------------------------------
# Random sampling on S² ---------------------------------------------------------
# -----------------------------------------------------------------------------

def _random_on_cone(batch, angle, y_sampler):
    y = tf.cos(angle * y_sampler([batch, 1]))
    phi = tf.random.uniform([batch, 1], 0.0, 2 * math.pi)
    r = safe_sqrt(1 - y ** 2)
    x = r * tf.cos(phi)
    z = r * tf.sin(phi)
    return tf.concat([x, y, z], axis=-1)


def random_vector_on_sphere(batch, limits):
    min_y, max_y = limits[0]
    min_theta, max_theta = limits[1]
    y = tf.random.uniform([batch, 1], min_y, max_y)
    theta = tf.random.uniform([batch, 1], min_theta, max_theta)
    cos_phi = tf.sqrt(tf.maximum(1 - tf.square(y), 0))
    x = cos_phi * tf.cos(theta)
    z = -cos_phi * tf.sin(theta)
    return tf.concat([x, y, z], axis=-1)


def uniform_sampled_vector_within_cone(axis, angle):
    if not (0.0 < angle < math.pi / 2):
        raise ValueError("angle must be within (0, π/2)")
    batch = tf.shape(axis)[0]
    v = _random_on_cone(batch, angle, tf.random.uniform)
    y_axis = tf.tile([[0.0, 1.0, 0.0]], [batch, 1])
    rot = rotation_between_vectors(y_axis, axis)
    return tf.squeeze(tf.matmul(rot, v[..., tf.newaxis]), -1)


def normal_sampled_vector_within_cone(axis, angle, std=1.0):
    if not (0.0 < angle < math.pi / 2):
        raise ValueError("angle must be within (0, π/2)")
    batch = tf.shape(axis)[0]
    v = _random_on_cone(batch, angle, lambda shape: tf.random.truncated_normal(shape, stddev=std))
    y_axis = tf.tile([[0.0, 1.0, 0.0]], [batch, 1])
    rot = rotation_between_vectors(y_axis, axis)
    return tf.squeeze(tf.matmul(rot, v[..., tf.newaxis]), -1)

# -----------------------------------------------------------------------------
# Rotations --------------------------------------------------------------------
# -----------------------------------------------------------------------------

def rotation_between_vectors(v1: tf.Tensor, v2: tf.Tensor):
    tf.debugging.assert_equal(tf.shape(v1)[-1], 3)
    tf.debugging.assert_equal(tf.shape(v2)[-1], 3)

    v1 = tf.linalg.l2_normalize(v1, axis=-1)
    v2 = tf.linalg.l2_normalize(v2, axis=-1)
    cross = tf.linalg.cross(v1, v2)
    cos_ang = tf.reduce_sum(v1 * v2, axis=-1, keepdims=True)
    sin_ang = tf.linalg.norm(cross, axis=-1)

    batch = tf.shape(v1)[0]
    identity = tf.eye(3, batch_shape=[batch], dtype=v1.dtype)

    # handle v1 ≈ -v2  (180°) by choosing arbitrary orthogonal axis
    mask = tf.abs(cos_ang + 1.0) < 1e-6
    alt_axis = tf.linalg.cross(v1, tf.tile([[1.0, 0.0, 0.0]], [batch, 1])) + \
               tf.linalg.cross(v1, tf.tile([[0.0, 1.0, 0.0]], [batch, 1])) + \
               tf.linalg.cross(v1, tf.tile([[0.0, 0.0, 1.0]], [batch, 1]))
    axis = tf.where(mask, alt_axis, cross)
    axis = tf.linalg.l2_normalize(axis, axis=-1)

    ss = skew_symmetric(axis)
    sin_ang = sin_ang[:, tf.newaxis, tf.newaxis]
    cos_ang = cos_ang[:, tf.newaxis]
    return identity + sin_ang * ss + (1.0 - cos_ang) * tf.matmul(ss, ss)
