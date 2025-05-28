# coding=utf-8
"""geometry.py – spherical geometry helpers for DirectionNet (TF 2.16).

Changes vs. original (2022 TF-1.x version)
-----------------------------------------
* **Pure TensorFlow 2 API** – removed all `tensorflow.compat.v1` symbols.
* Replaced deprecated ops: `tf.lin_space → tf.linspace`, `tf.matrix_transpose → tf.transpose`.
* Added runtime checks via `tf.debugging.assert_*` instead of Python `raise` inside graph contexts.
* All functions run eagerly by default yet work inside `@tf.function`.
"""
from __future__ import annotations
import math
import tensorflow as tf
import tensorflow_graphics.math.math_helpers as tfg_math_helpers

__all__ = [
    "cartesian_to_equirectangular_coordinates",
    "equirectangular_coordinates_to_cartesian",
    "generate_cartesian_grid",
    "generate_equirectangular_grid",
    "spherical_to_cartesian",
    "cartesian_to_spherical",
    "equirectangular_padding",
]

# -----------------------------------------------------------------------------
# Coordinate transforms --------------------------------------------------------
# -----------------------------------------------------------------------------

def cartesian_to_equirectangular_coordinates(v: tf.Tensor, shape: list[int]) -> tf.Tensor:
    """Cartesian → pixel coords on an equirectangular image."""
    height, width = shape
    axis_convert = tf.constant([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=v.dtype)
    v = tf.squeeze(tf.matmul(axis_convert, v[..., tf.newaxis]), -1)
    colatitude, azimuth = tf.split(cartesian_to_spherical(v), [1, 1], axis=-1)
    x = width * tf.math.mod(azimuth, 2 * math.pi) / (2 * math.pi)
    y = height * (colatitude / math.pi)
    return tf.concat([x, y], axis=-1)


def equirectangular_coordinates_to_cartesian(p: tf.Tensor, shape: list[int]) -> tf.Tensor:
    """Pixel coords on equirectangular image → Cartesian direction vectors."""
    height, width = shape
    x, y = tf.split(p, [1, 1], axis=-1)
    azimuth = x * (2 * math.pi) / width
    colatitude = math.pi * (y / height)
    spherical = tf.concat([colatitude, azimuth], axis=-1)
    cartesian = spherical_to_cartesian(spherical)
    axis_convert = tf.constant([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=cartesian.dtype)
    cartesian = tf.squeeze(tf.matmul(axis_convert, cartesian[..., tf.newaxis], transpose_a=True), -1)
    return cartesian

# -----------------------------------------------------------------------------
# Grid generators --------------------------------------------------------------
# -----------------------------------------------------------------------------

def generate_cartesian_grid(resolution: list[int], fov: float) -> tf.Tensor:
    """Return Cartesian grid (x,y,z) for every pixel centre in a perspective image."""
    tf.debugging.assert_equal(tf.shape(resolution)[0], 2, message="resolution must be length-2 list")
    fov_rad = tf.constant(fov, tf.float32) * math.pi / 180.0
    width = 2.0 * tf.tan(fov_rad / 2.0)
    height = width * resolution[0] / resolution[1]
    pixel_size = width / resolution[1]
    x_range = width - pixel_size
    y_range = height - pixel_size
    xx, yy = tf.meshgrid(
        tf.linspace(-x_range / 2.0, x_range / 2.0, resolution[1]),
        tf.linspace(y_range / 2.0, -y_range / 2.0, resolution[0]),
    )
    grid = tf.stack([xx, yy, -tf.ones_like(xx)], axis=-1)
    return grid


def generate_equirectangular_grid(shape: list[int]) -> tf.Tensor:
    """Return spherical grid (colatitude, azimuth) for an equirectangular map."""
    tf.debugging.assert_equal(tf.shape(shape)[0], 2, message="shape must be length-2 list")
    height, width = shape
    pixel_w = 2 * math.pi / tf.cast(width, tf.float32)
    pixel_h = math.pi / tf.cast(height, tf.float32)
    azimuth, colatitude = tf.meshgrid(
        tf.linspace(pixel_w / 2.0, 2.0 * math.pi - pixel_w / 2.0, width),
        tf.linspace(pixel_h / 2.0, math.pi - pixel_h / 2.0, height),
    )
    return tf.stack([colatitude, azimuth], axis=-1)

# -----------------------------------------------------------------------------
# Spherical <-> Cartesian ------------------------------------------------------
# -----------------------------------------------------------------------------

def spherical_to_cartesian(spherical: tf.Tensor) -> tf.Tensor:
    """(colatitude, azimuth) → (x, y, z)"""
    colatitude, azimuth = tf.split(spherical, [1, 1], axis=-1)
    return tfg_math_helpers.spherical_to_cartesian_coordinates(
        tf.concat([tf.ones_like(colatitude), colatitude, azimuth], axis=-1)
    )


def cartesian_to_spherical(cartesian: tf.Tensor) -> tf.Tensor:
    """(x, y, z) → (colatitude, azimuth)"""
    _, sph = tf.split(
        tfg_math_helpers.cartesian_to_spherical_coordinates(cartesian), [1, 2], axis=-1
    )
    return sph

# -----------------------------------------------------------------------------
# Padding for equirectangular images ------------------------------------------
# -----------------------------------------------------------------------------

def equirectangular_padding(
    images: tf.Tensor, num_paddings: list[list[int]]
) -> tf.Tensor:  # noqa: D401 – keep signature
    """Pad an equirectangular panorama with spherical wrap-around logic."""
    tf.debugging.assert_equal(tf.rank(images), 4, message="images must be NHWC")
    top, down = num_paddings[0]
    left, right = num_paddings[1]
    H = tf.shape(images)[1]
    W = tf.shape(images)[2]
    tf.debugging.assert_less_equal(top + down, H, "padding exceeds height")
    tf.debugging.assert_less_equal(left + right, W, "padding exceeds width")

    semicircle = W // 2
    top_pad = tf.reverse(tf.roll(images[:, :top], shift=semicircle, axis=2), axis=[1])
    bottom_pad = tf.roll(tf.reverse(images, axis=[1])[:, :down], shift=semicircle, axis=2)
    padded = tf.concat([top_pad, images, bottom_pad], axis=1)

    left_pad = tf.reverse(tf.reverse(padded, axis=[2])[:, :, :left], axis=[2])
    right_pad = padded[:, :, :right]
    return tf.concat([left_pad, padded, right_pad], axis=2)
