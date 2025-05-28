# coding=utf-8
"""transformation.py – image projection & rotation utilities (TF 2.16).

Key upgrades
============
* No **tensorflow.compat.v1** – uses plain TF 2.x ops (`tf.linalg.cross`,
  `tf.random.uniform`, `tf.reverse`, `tf.transpose`, etc.).
* Replaced deprecated `tf.mod` → `tf.math.mod`, `tf.lin_space` → `tf.linspace`.
* Added eager‑safe `tf.debugging.assert_*` checks in all public functions.
* Works with `tensorflow‑addons` **nightly** wheel (needed for Python 3.12) or
  the future 0.24 stable release.
"""
from __future__ import annotations
import math
import tensorflow as tf
import tensorflow_addons as tfa  # requires tf‑addons‑nightly on Py 3.12
from pano_utils import geometry, math_utils

__all__ = [
    "equirectangular_sampler",
    "rectilinear_projection",
    "rotate_pano",
    "rotate_image_in_3d",
    "rotate_image_on_pano",
]

# -----------------------------------------------------------------------------
# Low‑level sampler ------------------------------------------------------------
# -----------------------------------------------------------------------------

def equirectangular_sampler(images: tf.Tensor, spherical_coordinates: tf.Tensor) -> tf.Tensor:
    """Sample panorama images at given (colatitude, azimuth) grids."""
    tf.debugging.assert_rank(images, 4)
    tf.debugging.assert_equal(tf.shape(spherical_coordinates)[-1], 2)

    B, H, W = tf.unstack(tf.shape(images)[:3])
    padded = geometry.equirectangular_padding(images, [[1, 1], [1, 1]])

    colat, azim = tf.split(spherical_coordinates, [1, 1], axis=-1)
    x_pano = (tf.math.mod(azim / math.pi, 2.0) * tf.cast(W, tf.float32) / 2.0 - 0.5) + 1.0
    y_pano = ((colat / math.pi) * tf.cast(H, tf.float32) - 0.5) + 1.0
    coords = tf.concat([x_pano, y_pano], axis=-1)
    return tfa.image.resampler(padded, coords)

# -----------------------------------------------------------------------------
# Rectilinear projection -------------------------------------------------------
# -----------------------------------------------------------------------------

def rectilinear_projection(images: tf.Tensor, resolution: list[int], fov: float, rotations: tf.Tensor) -> tf.Tensor:
    """Project panorama → perspective after applying rotation."""
    tf.debugging.assert_rank(images, 4)
    tf.debugging.assert_equal(tf.shape(rotations)[-2:], [3, 3])

    B = tf.shape(images)[0]
    grid = geometry.generate_cartesian_grid(resolution, fov)               # [H,W,3]
    grid = tf.tile(grid[tf.newaxis], [B, 1, 1, 1])                         # make batch

    flip_x = tf.constant([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=grid.dtype)
    rot = tf.matmul(flip_x, tf.matmul(rotations, flip_x, transpose_a=True))
    grid_rot = tf.matmul(rot[:, tf.newaxis, tf.newaxis], grid[..., tf.newaxis], transpose_a=True)
    axis_convert = tf.constant([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=grid.dtype)
    grid_rot = tf.squeeze(tf.matmul(axis_convert, grid_rot), -1)
    spherical = geometry.cartesian_to_spherical(grid_rot)
    spherical = tf.reverse(spherical, axis=[2])      # azimuth increases L→R
    return equirectangular_sampler(images, spherical)

# -----------------------------------------------------------------------------
# Rotate equirectangular panorama ---------------------------------------------
# -----------------------------------------------------------------------------

def rotate_pano(images: tf.Tensor, rotations: tf.Tensor) -> tf.Tensor:
    tf.debugging.assert_rank(images, 4)
    tf.debugging.assert_equal(tf.shape(rotations)[-2:], [3, 3])
    B, H, W = tf.unstack(tf.shape(images)[:3])
    spherical = geometry.generate_equirectangular_grid([H, W])
    spherical = tf.tile(spherical[tf.newaxis], [B, 1, 1, 1])
    cart = geometry.spherical_to_cartesian(spherical)
    axis_convert = tf.constant([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]], dtype=cart.dtype)
    cart = axis_convert @ cart[..., tf.newaxis]
    cart_rot = rotations[:, tf.newaxis, tf.newaxis] @ cart
    cart_rot = tf.squeeze(axis_convert @ cart_rot, -1)
    spherical_rot = geometry.cartesian_to_spherical(cart_rot)
    return equirectangular_sampler(images, spherical_rot)

# -----------------------------------------------------------------------------
# Perspective → Perspective rotation ------------------------------------------
# -----------------------------------------------------------------------------

def rotate_image_in_3d(images: tf.Tensor, input_rotations: tf.Tensor, input_fov: tf.Tensor, output_fov: float, output_shape: list[int]) -> tf.Tensor:
    tf.debugging.assert_rank(images, 4)
    tf.debugging.assert_equal(tf.shape(input_rotations)[-2:], [3, 3])

    B = tf.shape(images)[0]
    grid = geometry.generate_cartesian_grid(output_shape, output_fov)     # [h,w,3]
    grid = tf.tile(grid[tf.newaxis, ..., tf.newaxis], [B, 1, 1, 1, 1])   # [B,h,w,3,1]
    R = tf.tile(input_rotations[:, tf.newaxis, tf.newaxis, :, :], [1] + output_shape + [1, 1])
    cart = tf.squeeze(R @ grid, -1)
    xy = -cart[..., :2] / cart[..., 2:3]

    W_in = tf.cast(tf.shape(images)[2], tf.float32)
    H_in = tf.cast(tf.shape(images)[1], tf.float32)
    w = 2.0 * tf.tan(math_utils.degrees_to_radians(input_fov / 2.0))
    h = w * (H_in / W_in)
    w = w[:, tf.newaxis, tf.newaxis, tf.newaxis]
    h = h[:, tf.newaxis, tf.newaxis, tf.newaxis]
    nx = xy[..., 0:1] * W_in / w + W_in / 2.0 - 0.5
    ny = -xy[..., 1:2] * H_in / h + H_in / 2.0 - 0.5
    return tfa.image.resampler(images, tf.concat([nx, ny], axis=-1))

# -----------------------------------------------------------------------------
# Perspective → Panorama embed -------------------------------------------------
# -----------------------------------------------------------------------------

def rotate_image_on_pano(images: tf.Tensor, rotations: tf.Tensor, fov: float, output_shape: list[int]) -> tf.Tensor:
    tf.debugging.assert_rank(images, 4)
    tf.debugging.assert_equal(tf.shape(rotations)[-2:], [3, 3])

    B = tf.shape(images)[0]
    sph = geometry.generate_equirectangular_grid(output_shape)
    cart = geometry.spherical_to_cartesian(sph)
    cart = tf.tile(cart[tf.newaxis, ..., tf.newaxis], [B, 1, 1, 1, 1])
    axis_convert = tf.constant([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=cart.dtype)
    cart = axis_convert @ cart
    cart_rot = rotations[:, tf.newaxis, tf.newaxis] @ cart
    cart_rot = tf.squeeze(axis_convert @ cart_rot, -1)

    hemi_mask = tf.cast(cart_rot[..., 2:3] < 0.0, cart.dtype)
    xy = cart_rot[..., :2] / cart_rot[..., 2:3]

    W_in = tf.cast(tf.shape(images)[2], tf.float32)
    H_in = tf.cast(tf.shape(images)[1], tf.float32)
    nx = -xy[..., 0:1] * W_in / (2.0 * tf.tan(math_utils.degrees_to_radians(fov / 2.0))) + W_in / 2.0 - 0.5
    ny =  xy[..., 1:2] * H_in / (2.0 * tf.tan(math_utils.degrees_to_radians(fov / 2.0))) + H_in / 2.0 - 0.5
    out = hemi_mask * tfa.image.resampler(images, tf.concat([nx, ny], axis=-1))
    return out
