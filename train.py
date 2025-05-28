# coding=utf-8
"""train.py – TF‑2.x training loop for DirectionNet variants.

Highlights
----------
* **End‑to‑end eager / `tf.function`** – no more `tf.compat.v1` graph mode.
* Uses `tf.keras.optimizers.SGD` and `tf.train.Checkpoint` for fault‑tolerant
  checkpointing.
* Re‑implements the three variant‑specific loss heads (9D/6D‑R, T, Single) in
  eager style, re‑using the same maths from `losses.py`.
* Automatic `tf.summary` writing with a summary writer in `--checkpoint_dir`.
* CLI flags match the original so existing scripts run unchanged.

Quick start
-----------
```bash
python train.py \
  --data_dir   /path/to/train_split \
  --checkpoint_dir  /tmp/dnet_ckpt \
  --model 9D --batch 8 --n_epoch 5
```
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

from absl import app, flags
import tensorflow as tf

import dataset_loader
import losses
import model as dnet_model
import util  # project‑local maths helpers

# -----------------------------------------------------------------------------
# Flags -----------------------------------------------------------------------
# -----------------------------------------------------------------------------
FLAGS = flags.FLAGS
flags.DEFINE_string("master", "", "TensorFlow master (unused in eager mode)")
flags.DEFINE_integer("task", 0, "Task id (unused in eager mode)")
flags.DEFINE_string("checkpoint_dir", "", "Directory to save checkpoints & logs")
flags.DEFINE_string("data_dir", "", "Training data root directory")
flags.DEFINE_string("model", "9D", "Variant: 9D, 6D, T, Single")
flags.DEFINE_integer("batch", 20, "Mini‑batch size")
flags.DEFINE_integer("n_epoch", 5, "Number of epochs (-1 for infinite)")
flags.DEFINE_float("lr", 1e-3, "Learning rate")
flags.DEFINE_float("alpha", 8e7, "Weight for distribution loss")
flags.DEFINE_float("beta", 0.1, "Weight for spread loss")
flags.DEFINE_float("kappa", 10.0, "Concentration coefficient (vMF)")
flags.DEFINE_integer("distribution_height", 64, "Height of output distributions")
flags.DEFINE_integer("distribution_width", 64, "Width of output distributions")
flags.DEFINE_integer("transformed_height", 344, "Height after derotation")
flags.DEFINE_integer("transformed_width", 344, "Width after derotation")
flags.DEFINE_float("transformed_fov", 105.0, "FOV after derotation (deg)")
flags.DEFINE_bool("derotate_both", True, "Derotate both images for T variant")

# -----------------------------------------------------------------------------
# Variant‑specific loss functions ---------------------------------------------
# -----------------------------------------------------------------------------

LossReturn = Tuple[tf.Tensor, dict]


def _rotation_head(
    net: dnet_model.DirectionNet,
    batch,
    n_output_distributions: int,
) -> LossReturn:
    src_img, trt_img = batch.src_image, batch.trt_image
    rotation_gt = batch.rotation
    directions_gt = rotation_gt[:, :n_output_distributions]

    dist_gt = util.spherical_normalization(
        util.von_mises_fisher(
            directions_gt,
            tf.constant(FLAGS.kappa, tf.float32),
            [FLAGS.distribution_height, FLAGS.distribution_width],
        ),
        rectify=False,
    )

    pred = net(src_img, trt_img, training=True)
    directions, expectation, dist_pred = util.distributions_to_directions(pred)

    if n_output_distributions == 3:
        rot_est = util.svd_orthogonalize(directions)
    else:
        rot_est = util.gram_schmidt(directions)

    loss_dir = losses.direction_loss(directions, directions_gt)
    loss_dist = tf.constant(FLAGS.alpha, tf.float32) * losses.distribution_loss(dist_pred, dist_gt)
    loss_spread = tf.cast(FLAGS.beta, tf.float32) * losses.spread_loss(expectation)

    rot_err = tf.reduce_mean(util.rotation_geodesic(rot_est, rotation_gt))
    dir_err = tf.reduce_mean(tf.acos(tf.clip_by_value(tf.reduce_sum(directions * directions_gt, -1), -1.0, 1.0)))

    total = loss_dir + loss_dist + loss_spread
    logs = {
        "loss": total,
        "rotation_error_deg": util.radians_to_degrees(rot_err),
        "direction_error_deg": util.radians_to_degrees(dir_err),
    }
    return total, logs


def _translation_head(net: dnet_model.DirectionNet, batch) -> LossReturn:
    src_img, trt_img = batch.src_image, batch.trt_image
    rotation_gt, translation_gt = batch.rotation, batch.translation
    fov_gt = tf.squeeze(batch.fov, -1)
    rotation_pred = batch.rotation_pred

    # 50% chance: perturb gt, else use pred rot
    perturbed_rot = tf.cond(
        tf.random.uniform([]) < 0.5,
        lambda: util.perturb_rotation(rotation_gt, [10.0, 5.0, 10.0]),
        lambda: rotation_pred,
    )

    transformed_src, transformed_trt = util.derotation(
        src_img,
        trt_img,
        perturbed_rot,
        fov_gt,
        FLAGS.transformed_fov,
        [FLAGS.transformed_height, FLAGS.transformed_width],
        FLAGS.derotate_both,
    )

    half_derot = util.half_rotation(perturbed_rot)
    translation_gt_rot = tf.squeeze(
        tf.matmul(half_derot, translation_gt[..., tf.newaxis], transpose_a=True),
        -1,
    )
    translation_gt_rot = translation_gt_rot[:, tf.newaxis]

    dist_gt = util.spherical_normalization(
        util.von_mises_fisher(
            translation_gt_rot,
            tf.constant(FLAGS.kappa, tf.float32),
            [FLAGS.distribution_height, FLAGS.distribution_width],
        ),
        rectify=False,
    )

    pred = net(transformed_src, transformed_trt, training=True)
    directions, expectation, dist_pred = util.distributions_to_directions(pred)

    loss_dir = losses.direction_loss(directions, translation_gt_rot)
    loss_dist = tf.constant(FLAGS.alpha, tf.float32) * losses.distribution_loss(dist_pred, dist_gt)
    loss_spread = tf.cast(FLAGS.beta, tf.float32) * losses.spread_loss(expectation)

    dir_err = tf.reduce_mean(tf.acos(tf.clip_by_value(tf.reduce_sum(directions * translation_gt_rot, -1), -1.0, 1.0)))

    total = loss_dir + loss_dist + loss_spread
    logs = {"loss": total, "translation_error_deg": util.radians_to_degrees(dir_err)}
    return total, logs


def _single_head(net: dnet_model.DirectionNet, batch) -> LossReturn:
    src_img, trt_img = batch.src_image, batch.trt_image
    rotation_gt, translation_gt = batch.rotation, batch.translation

    directions_gt = tf.concat([rotation_gt, translation_gt], axis=1)
    dist_gt = util.spherical_normalization(
        util.von_mises_fisher(
            directions_gt,
            tf.constant(FLAGS.kappa, tf.float32),
            [FLAGS.distribution_height, FLAGS.distribution_width],
        ),
        rectify=False,
    )

    pred = net(src_img, trt_img, training=True)
    directions, expectation, dist_pred = util.distributions_to_directions(pred)
    rot_est = util.svd_orthogonalize(directions[:, :3])

    loss_dir = losses.direction_loss(directions, directions_gt)
    loss_dist = tf.constant(FLAGS.alpha, tf.float32) * losses.distribution_loss(dist_pred, dist_gt)
    loss_spread = tf.cast(FLAGS.beta, tf.float32) * losses.spread_loss(expectation)

    rot_err = tf.reduce_mean(util.rotation_geodesic(rot_est, rotation_gt))
    trans_err = tf.reduce_mean(
        tf.acos(tf.clip_by_value(tf.reduce_sum(directions[:, -1] * directions_gt[:, -1], -1), -1.0, 1.0))
    )

    total = loss_dir + loss_dist + loss_spread
    logs = {
        "loss": total,
        "rotation_error_deg": util.radians_to_degrees(rot_err),
        "translation_error_deg": util.radians_to_degrees(trans_err),
    }
    return total, logs

# -----------------------------------------------------------------------------
# Training loop ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def main(argv):  # noqa: D401 – absl entry point
    if len(argv) > 1:
        raise app.UsageError("Too many positional args")

    # Data pipeline
    dataset = dataset_loader.data_loader(
        data_path=FLAGS.data_dir,
        epochs=FLAGS.n_epoch if FLAGS.n_epoch > 0 else 1_000_000_000,
        batch_size=FLAGS.batch,
        training=True,
        load_estimated_rot=(FLAGS.model == "T"),
    )
    data_iter = iter(dataset)

    # Model & optimizer
    if FLAGS.model in ("9D", "6D"):
        n_out = 3 if FLAGS.model == "9D" else 2
    elif FLAGS.model == "T":
        n_out = 1
    else:  # Single
        n_out = 4

    net = dnet_model.DirectionNet(n_out)
    optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.lr)

    # Summary & checkpoint setup
    ckpt_dir = Path(FLAGS.checkpoint_dir) if FLAGS.checkpoint_dir else Path.cwd() / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(str(ckpt_dir))

    global_step = tf.Variable(0, dtype=tf.int64)
    ckpt = tf.train.Checkpoint(step=global_step, optimizer=optimizer, net=net)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=5)

    # Variant‑specific loss fn
    if FLAGS.model in ("9D", "6D"):
        compute_loss = lambda b: _rotation_head(net, b, 3 if FLAGS.model == "9D" else 2)
    elif FLAGS.model == "T":
        compute_loss = lambda b: _translation_head(net, b)
    else:
        compute_loss = lambda b: _single_head(net, b)

    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            loss_val, logs = compute_loss(batch)
        grads = tape.gradient(loss_val, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        global_step.assign_add(1)
        return loss_val, logs

    # Main loop
    start_time = time.time()
    for batch in data_iter:
        loss_val, logs = train_step(batch)

        # Logging
        step = int(global_step.numpy())
        if step % 10 == 0:
            elapsed = time.time() - start_time
            tf.print(
                "step", step,
                "loss", logs.get("loss"),
                ":", {k: v for k, v in logs.items() if k != "loss"},
                "(%.2fs)" % elapsed,
            )
            start_time = time.time()
            with summary_writer.as_default():
                for k, v in logs.items():
                    tf.summary.scalar(k, v, step=step)

        if step % 2_000 == 0:
            save_path = manager.save()
            tf.print("Checkpoint saved to", save_path)

        if FLAGS.n_epoch > 0 and step >= (FLAGS.n_epoch * (dataset.cardinality().numpy() // FLAGS.batch)):
            tf.print("Training complete – reached requested epochs")
            break


if __name__ == "__main__":
    app.run(main)
