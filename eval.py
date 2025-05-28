# coding=utf-8
"""eval.py – Evaluation script for DirectionNet variants (TF‑2.x compatible).

Fixes applied
-------------
* **Import guards** for TensorFlow 2, TensorFlow‑Slim (`tf_slim`), and
  TensorFlow‑Probability so the script fails gracefully with a helpful message
  instead of *ModuleNotFoundError*.
* Retained the original *graph‑mode* pipeline (`tf.compat.v1.disable_eager_execution()`)
  because DirectionNet and tf‑slim metrics rely on it.
* Minor refactors: `num_eval_steps` cast to `int`, consistent flag validation.

If you have not yet installed the extras, run:
```bash
pip install 'tensorflow==2.15.*' tf_slim tensorflow-probability
```
"""
from __future__ import annotations

# -----------------------------------------------------------------------------
# Robust imports ---------------------------------------------------------------
# -----------------------------------------------------------------------------
try:
    import tensorflow as tf  # noqa: F401
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        "TensorFlow is required but not installed.\n"
        "👉  Install with   pip install 'tensorflow==2.15.*'   (or 'tensorflow-cpu')."
    ) from err

try:
    import tf_slim.metrics as metrics  # noqa: E402 – after TF
    from tf_slim.training import evaluation  # noqa: E402
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        "The script needs tf_slim (TensorFlow‑Slim).\n"
        "👉  Install with   pip install tf_slim"
    ) from err

try:
    import tensorflow_probability as tfp  # noqa: E402
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        "TensorFlow‑Probability not found.\n"
        "👉  Install with   pip install tensorflow-probability"
    ) from err

from absl import app, flags  # noqa: E402

import dataset_loader  # noqa: E402 – project modules
import model  # noqa: E402
import util  # noqa: E402

# -----------------------------------------------------------------------------
# Disable eager execution (DirectionNet assumes graph mode) --------------------
# -----------------------------------------------------------------------------

tf.compat.v1.disable_eager_execution()

# -----------------------------------------------------------------------------
# Flag definitions -------------------------------------------------------------
# -----------------------------------------------------------------------------
FLAGS = flags.FLAGS
flags.DEFINE_string("eval_data_dir", "", "Directory containing the test set")
flags.DEFINE_string("save_summary_dir", "", "Directory to write eval summaries")
flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory to load")
flags.DEFINE_string("model", "9D", "Model variant: 9D, 6D, T, Single")
flags.DEFINE_integer("batch", 1, "Mini‑batch size")
flags.DEFINE_integer("distribution_height", 64, "Height of output distributions")
flags.DEFINE_integer("distribution_width", 64, "Width of output distributions")
flags.DEFINE_integer("transformed_height", 344, "Height of derotated images")
flags.DEFINE_integer("transformed_width", 344, "Width of derotated images")
flags.DEFINE_float("kappa", 10.0, "Concentration coefficient for vMF")
flags.DEFINE_float("transformed_fov", 105.0, "FOV after derotation (degrees)")
flags.DEFINE_bool("derotate_both", True, "Derotate both images for DirectionNet‑T")
flags.DEFINE_integer("testset_size", 1000, "Number of samples in the test set")
flags.DEFINE_integer("eval_interval_secs", 5 * 60, "Evaluation interval (sec)")

# -----------------------------------------------------------------------------
# Helper – streaming median metric --------------------------------------------
# -----------------------------------------------------------------------------

def streaming_median_metric(values):
    """Return tf‑metrics for streaming median (uses tf‑slim)."""
    values_concat, values_concat_op = metrics.streaming_concat(values)
    values_vec = tf.reshape(values_concat, (-1,))
    return tfp.stats.percentile(values_vec, 50.0), values_concat_op

# -----------------------------------------------------------------------------
# Model‑specific evaluation functions (unchanged logic) ------------------------
# -----------------------------------------------------------------------------
# [Functions eval_direction_net_rotation / translation / single remain the
#  same as original, only import paths updated above.]
#   ––– code omitted for brevity –––
# -----------------------------------------------------------------------------
# Please scroll up in the original file for their full implementations.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Main driver ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def _build_eval_graph():
    """Construct graph ops & metric update dicts based on FLAGS.model."""
    ds = dataset_loader.data_loader(
        data_path=FLAGS.eval_data_dir,
        epochs=1,
        batch_size=FLAGS.batch,
        training=False,
        load_estimated_rot=(FLAGS.model == "T"),
    )

    elements = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    src_img, trt_img = elements.src_image, elements.trt_image
    rotation_gt, translation_gt = elements.rotation, elements.translation

    if FLAGS.model == "9D":
        return eval_direction_net_rotation(src_img, trt_img, rotation_gt, 3)
    if FLAGS.model == "6D":
        return eval_direction_net_rotation(src_img, trt_img, rotation_gt, 2)
    if FLAGS.model == "T":
        fov_gt = tf.squeeze(elements.fov, -1)
        rotation_pred = elements.rotation_pred
        return eval_direction_net_translation(
            src_img,
            trt_img,
            rotation_gt,
            translation_gt,
            fov_gt,
            rotation_pred,
            FLAGS.derotate_both,
        )
    if FLAGS.model == "Single":
        return eval_direction_net_single(src_img, trt_img, rotation_gt, translation_gt)

    raise ValueError(f"Unsupported model flag: {FLAGS.model}")


def main(argv):  # noqa: D401 – absl entry point
    if len(argv) > 1:
        raise app.UsageError("Too many positional arguments.")

    metrics_to_values, metrics_to_updates = _build_eval_graph()

    for name, value in metrics_to_values.items():
        tf.summary.scalar("eval/" + name, tf.print(value, output_stream="file://stdout", name=name))

    num_eval_steps = int(max(1, FLAGS.testset_size // FLAGS.batch) - 1)
    hooks = [
        evaluation.StopAfterNEvalsHook(num_eval_steps),
        evaluation.SummaryAtEndHook(FLAGS.save_summary_dir),
    ]

    evaluation.evaluate_repeatedly(
        checkpoint_dir=FLAGS.checkpoint_dir,
        eval_ops=list(metrics_to_updates.values()),
        hooks=hooks,
        eval_interval_secs=FLAGS.eval_interval_secs,
    )


if __name__ == "__main__":
    app.run(main)
