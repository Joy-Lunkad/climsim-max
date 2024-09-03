"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# pylint: disable=g-bad-todo, abstract-method, consider-using-with, ungrouped-imports
"""Training loop and Decoding of the model."""

# Calling jax.device_count here prevents a "TPU platform already registered" error.
# See github.com/google/maxtext/issues/20 for more
import kaggle
import datetime
import os
import sys
import functools

from typing import Sequence
from absl import app
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
import rich
import grain.python as grain
import jax
import jax.tree_util as jtu
import numpy as np
import optax
import orbax.checkpoint
import tensorflow as tf
from etils import etree

import checkpointing
import max_utils
import maxtext_utils
import max_logging
import optimizers
import pyconfig
# pylint: disable-next=unused-import
import register_jax_proxy_backend
from vertex_tensorboard import VertexTensorboardManager

from input_pipeline.input_pipeline_interface import create_data_iterator_with_tokenizer
from layers import models

import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh
from jax.experimental import checkify

from cloud_tpu_diagnostics import diagnostic
from cloud_tpu_diagnostics.configuration import debug_configuration
from cloud_tpu_diagnostics.configuration import diagnostic_configuration
from cloud_tpu_diagnostics.configuration import stack_trace_configuration

from layers import quantizations

from ml_goodput_measurement import goodput
import wandb

Transformer = models.Transformer
EPS = 1e-8


def validate_train_config(config):
  """Validates the configuration is set correctly for train.py"""

  assert config.run_name, "Erroring out, need a real run_name"
  if not config.dataset_path.startswith("gs://"):
    max_logging.log("WARNING: 'dataset_path' might be pointing your local file system")
  if not config.base_output_directory.startswith("gs://"):
    max_logging.log("WARNING: 'base_output_directory' might be pointing your local file system")
  assert config.steps > 0, "You must set steps or learning_rate_schedule_steps to a positive integer."


def get_first_step(state):
  with jax.spmd_mode("allow_all"):
    return int(state.step)


def load_next_batch(train_iter, example_batch, config):
  """Loads the next batch. Can keep reusing the same batch for performance reasons"""
  if config.reuse_example_batch and example_batch is not None:
    return example_batch
  else:
    return next(train_iter)


def record_scalar_metrics(metrics, step_time_delta, per_device_tflops, lr):
  """Records scalar metrics to be written to tensorboard"""
  metrics["scalar"].update({"perf/step_time_seconds": step_time_delta.total_seconds()})
  metrics["scalar"].update({"perf/per_device_tflops": per_device_tflops})
  metrics["scalar"].update({"perf/per_device_tflops_per_sec": per_device_tflops / step_time_delta.total_seconds()})
  metrics["scalar"].update({"learning/current_learning_rate": lr})


_buffered_step = None
_buffered_metrics = None


def write_metrics(writer, local_metrics_file, running_gcs_metrics, metrics, step, config):
  """Entry point for all metrics writing in Train's Main.
  TODO: would be better as a Class in the future (that initialized all state!)

  To avoid introducing an unnecessary dependency, we "double buffer" -- we hold
  onto the last metrics and step and only publish when we receive a new metrics and step.
  The logic is that this ensures that Jax is able to queues train_steps and we
  don't block when turning "lazy" Jax arrays into real Python numbers.
  """
  global _buffered_step, _buffered_metrics

  if _buffered_metrics is not None:
    if _buffered_step is None:
      raise ValueError(f"When writing metrics, {_buffered_step=} was none")
    write_metrics_to_tensorboard(writer, _buffered_metrics, _buffered_step, config)

    if config.metrics_file:
      max_utils.write_metrics_locally(_buffered_metrics, _buffered_step, config, local_metrics_file)

    if config.gcs_metrics and jax.process_index() == 0:
      running_gcs_metrics = max_utils.write_metrics_for_gcs(_buffered_metrics, _buffered_step, config, running_gcs_metrics)

  _buffered_step = step
  _buffered_metrics = metrics

y_cols = np.load("saved_data/output_col_names.npy")
def write_metrics_to_tensorboard(writer, metrics, step, config):
  """Writes metrics to tensorboard"""
  with jax.spmd_mode("allow_all"):
    if jax.process_index() == 0:
      for metric_name in metrics.get("scalar", []):
        writer.add_scalar(metric_name, np.array(metrics["scalar"][metric_name]), step)
      # for metric_name in metrics.get("scalars", []):
      #   writer.add_scalars(metric_name, metrics["scalars"][metric_name], step)

    full_log = step % config.log_period == 0

    log_str = f"completed step: {step}, seconds: {metrics['scalar']['perf/step_time_seconds']:.3f}, "
    log_str += f"TFLOP/s/device: {metrics['scalar']['perf/per_device_tflops_per_sec']:.3f}, "
    log_str += f"loss: {metrics['scalar']['learning/loss']:.3f}, "
    log_str += f"r2_score: {metrics['scalar']['learning/r2_score']:.3f}, "
    
    max_logging.log(log_str)

    full_r2 = metrics["scalars"]["learning/full_r2"]
    r2_dict = {f"learning/{str(k)}": float(v) for k, v in zip(y_cols, full_r2)}
    metrics["scalar"].update(r2_dict)

    if full_log and jax.process_index() == 0:
      wandb.log(metrics["scalar"], step)
      max_logging.log(f"To see full metrics 'tensorboard --logdir={config.tensorboard_dir}'")
      writer.flush()


def save_checkpoint(checkpoint_manager, step, state, dataset_type="c4", data_iterator=None):
  """Wrapper for saving checkpoint"""
  if dataset_type == "c4-array_record":
    return checkpoint_manager.save(
        step,
        args=orbax.checkpoint.args.Composite(
            items=orbax.checkpoint.args.PyTreeSave(item=state),
            iter=grain.PyGrainCheckpointSave(data_iterator.local_iterator),
        ),
    )
  else:
    return checkpoint_manager.save(
        step, args=orbax.checkpoint.args.Composite(items=orbax.checkpoint.args.PyTreeSave(item=state))
    )


# -----------------------------------------------------------------------------
# Top-level Functions
# -----------------------------------------------------------------------------


def record_activation_metrics(output_metrics, intermediate_outputs, config):
  """Adds the activation metrics to the metrics dict"""

  if config.scan_layers:
    metrics_dict = intermediate_outputs["intermediates"]["decoder"]["decoder"]

    for layer_num in range(config.num_decoder_layers):
      output_metrics["scalar"][f"activ_fraction_zero/layer_{layer_num:03d}"] = metrics_dict["activation_fraction_zero"][0][
          layer_num
      ]
      output_metrics["scalar"][f"activ_mean/layer_{layer_num:03d}"] = metrics_dict["activation_mean"][0][layer_num]
      output_metrics["scalar"][f"activ_stdev/layer_{layer_num:03d}"] = metrics_dict["activation_stdev"][0][layer_num]
  else:
    for layer_num in range(config.num_decoder_layers):
      layer = intermediate_outputs["intermediates"]["decoder"][f"layers_{layer_num}"]
      output_metrics["scalar"][f"activ_fraction_zero/layer_{layer_num:03d}"] = layer["activation_fraction_zero"][0]
      output_metrics["scalar"][f"activ_mean/layer_{layer_num:03d}"] = layer["activation_mean"][0]
      output_metrics["scalar"][f"activ_stdev/layer_{layer_num:03d}"] = layer["activation_stdev"][0]

mean_y = np.load("saved_data/keras_mean_y.npy")[0]
r2_mask = np.load("saved_data/scale_output_weights_new.npy").astype('bool')

def calc_idv_r2_score(y_true, y_pred):
    numerator = jnp.sum((y_true - y_pred) ** 2, axis=0)
    denominator = jnp.sum((y_true - mean_y) ** 2, axis=0)
    denominator = jnp.where(denominator == 0, 1.0, denominator)
    r2 = 1 - (numerator / denominator)
    return r2

@jax.vmap
def calc_r2_score(y_true, y_pred):
    numerator = jnp.sum((y_true - y_pred) ** 2, axis=0)
    denominator = jnp.sum((y_true - jnp.mean(y_true, axis=0)) ** 2, axis=0)
    r2 = 1 - (numerator / denominator)
    r2 = jnp.where(jnp.isinf(r2), 0.0, r2)
    return r2


def loss_fn(model, config, data, dropout_rng, params, is_train=True):
  """loss_fn for both train and eval.

  Args:
    model: A nn.Module
    config: Config of parameters
    data: Batch of data to apply to the model
    dropout_rng: A key to use to generate rng for dropout
    params: Model params
    is_train: True for train_step and False for eval_step

  Returns:
    loss: average loss
    aux: a dictionary including intermediate_outputs, total_loss, and total_weights
  """
  # inputs, targets, segments, positions = apply_args
  rng1, aqt_rng = jax.random.split(dropout_rng)

  # decimate proportion of data when per_device_batch_size<1
  if is_train:
    for k, v in data.items():
      if k == "inputs" or k == "targets":
        data[k] = v[: config.global_batch_size_to_train_on, :]

  y_pred, intermediate_outputs = model.apply(
      params,
      data["inputs"],
      grid_positions=data.get("grid_pos", None),
      decoder_segment_ids=None,
      enable_dropout=config.enable_dropout if is_train else False,
      rngs={"dropout": rng1, "params": aqt_rng},
      mutable="intermediates",
  )
  
  y_true = data["targets"].reshape(-1, *data["targets"].shape[2:])
  
  y_true = max_utils.scale_output(y_true)
  y_pred = max_utils.scale_output(y_pred)
  
  loss = jnp.abs(y_true - y_pred)
  loss = nn.with_logical_constraint(loss, ("activation_batch", "activation_length"))
  loss = jnp.sum(loss[:, r2_mask], axis=-1)
  if "sample_weight" in data:
    loss = loss * data["sample_weight"]
  total_loss = jnp.mean(loss)

  r2_score = calc_r2_score(y_true[:, r2_mask], y_pred[:, r2_mask])
  r2_score = nn.with_logical_constraint(r2_score, ("activation_batch",))
  total_r2_score = jnp.mean(r2_score)

  full_r2 = calc_idv_r2_score(y_true, y_pred)
  
  total_weights = jnp.sum(data["targets"] != 0.0)

  loss = total_loss
  aux = {
      "intermediate_outputs": intermediate_outputs,
      "total_loss": total_loss,
      "total_r2_score": total_r2_score,
      "total_weights": total_weights,
      "y_pred": y_pred,
      "full_r2": full_r2,
  }
  return loss, aux


def train_step(model, config, state, data, dropout_rng):
  """

  Args:
    model: A nn.Module
    state: A pytree of the current state of the model
    data: Batch of data to apply to the model
    dropout_rng: A key to use to generate rng for dropout

  Returns:
    new_state: Same format as state.
    metrics: Dictionary of model metrics such as loss, training rate, etc.
    rng2: A new rng key that can be used in future calls.

  """
  train_loss_fn = functools.partial(loss_fn, model, config, data, dropout_rng, is_train=True)
  grad_fn = jax.value_and_grad(train_loss_fn, has_aux=True)
  (loss, aux), raw_grads = grad_fn(state.params)
  intermediate_outputs = aux["intermediate_outputs"]

  if config.gradient_clipping_threshold > 0:
    grads, _ = optax.clip_by_global_norm(config.gradient_clipping_threshold).update(raw_grads, state, None)
  else:
    grads = raw_grads
  new_state = state.apply_gradients(grads=grads)
  metrics = {
      "scalar": {
          "learning/loss": loss,
          "learning/r2_score": aux["total_r2_score"],
          "learning/grad_norm": max_utils.l2norm_pytree(grads),
          "learning/raw_grad_norm": max_utils.l2norm_pytree(raw_grads),
          "learning/param_norm": max_utils.l2norm_pytree(new_state.params),
      },
      "scalars": {"learning/full_r2": aux["full_r2"]},
  }

  if config.record_internal_nn_metrics:
    record_activation_metrics(metrics, intermediate_outputs, config)

  return new_state, metrics


def eval_step(model, config, state, data, dropout_rng):
  """eval_step no backprop and new state compared with train_step."""
  eval_loss_fn = functools.partial(loss_fn, model, config, data, dropout_rng, is_train=False)
  loss, aux = eval_loss_fn(state.params)
  total_loss = aux["total_loss"]
  total_weights = aux["total_weights"]
  metrics = {
    "scalar": {
      "evaluation/loss": loss,
      "evaluation/total_loss": total_loss,
      "evaluation/total_weights": total_weights,
      "evaluation/r2_score": aux["total_r2_score"],
    },
    "y_pred": aux["y_pred"],  # for evaluation
    "full_r2": aux["full_r2"],
  }

  return metrics


def forward(model, config, state, data, dropout_rng):
  """test_step forward pass only."""
  rng1, aqt_rng = jax.random.split(dropout_rng)
  
  for k, v in data.items():
    if k == "inputs" or k == "targets":
      data[k] = v[: config.global_batch_size_to_train_on, :]
  
  y_pred, intermediate_outputs = model.apply(
      state.params,
      data["inputs"],
      None,
      decoder_segment_ids=None,
      enable_dropout=False,
      rngs={"dropout": rng1, "params": aqt_rng},
      mutable="intermediates",
  )
  
  y_pred = nn.with_logical_constraint(y_pred, ("activation_batch", "activation_length"))
  return y_pred


def create_submission_csv(state, p_forward_fn, test_data_iterator, config, mesh, nextrng):
  
  import pandas as pd
  from tqdm import tqdm
  
  batch_size = config.per_device_batch_size * mesh.size
  test_samples = 625_000
  test_steps = test_samples // batch_size
  print(f"test_steps = {test_steps}")
  
  sample_submission = os.path.join("gs://us2-climsim", "sample_submission.csv")
  sub_df = pd.read_csv(sample_submission, nrows=1)
  python_state_dir = os.path.join("submissions/maxtext", config.run_name, "checkpoints")
  os.makedirs(python_state_dir, exist_ok=True)
  python_state_path = os.path.join(python_state_dir, "submission.csv")
  
  if not os.path.exists(python_state_path):

    results = []
    for t_step, test_batch in enumerate(test_data_iterator):
      if t_step == 0:
        max_logging.log(f"{test_batch['inputs'][0]=}")
      if t_step == test_steps + 1:
        break
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        if t_step == 1:
          max_logging.log(maxtext_utils.tree_shape(test_batch))
          rich.print(test_batch)
        
        sample_ids = test_batch.pop("sample_id")
        y_pred = p_forward_fn(state, test_batch, nextrng)
        y_pred = jax.experimental.multihost_utils.process_allgather(y_pred)
        
        if t_step == 0:
          max_logging.log(f"Output: {y_pred[0]=}")
          
        sample_ids = sample_ids.reshape(-1)
        sample_ids = jax.experimental.multihost_utils.process_allgather(sample_ids)
        sample_ids = [f"test_{id}" for id in sample_ids]
        
        if config.normalize:
          y_pred = max_utils.which_denorm_output(y_pred)
        else:
          y_pred = max_utils.denorm_output(y_pred)
        
        if config.use_submission_trick:
          x_test = jax.experimental.multihost_utils.process_allgather(test_batch["inputs"])
          if config.normalize:
            x_test = max_utils.which_denorm_input(x_test)
          else:
            x_test = max_utils.denorm_input(x_test)
          # y_pred = max_utils.change_y_std_0_to_mean(y_pred)
          y_pred[:, 132:150] = -x_test[:, 132:150] / 1200
          
        y_pred = max_utils.scale_output(y_pred)
        
        if t_step == 0:
          max_logging.log(f"{y_pred[0]=}")
        
        data = {"sample_id": sample_ids}
        for col_idx, col_name in enumerate(sub_df.columns.tolist()):
            if col_name == "sample_id":
                continue
            data[col_name] = y_pred[:, col_idx - 1]
        results.append(pd.DataFrame(data))
        
        if t_step % 10 == 0:
          max_logging.log(f"Processed {t_step} batches")
    
    final_df = pd.concat(results)
    final_df = final_df.drop_duplicates(subset="sample_id", keep="first")
    final_df.to_csv(python_state_path, index=False)
    print(final_df.describe())
    
    maxtext_utils.upload_blob(
          bucket_name="us2-climsim",
          source_file_name=python_state_path,
          destination_blob_name=python_state_path,
      )
  
  if config.submit_to_kaggle and jax.process_index() == 0:
    max_logging.log("Submitting to Kaggle...")
    maxtext_utils.kaggle_submit_csv(
        python_state_path,
        msg=f"{config.run_name} submission",
    )

def create_goodput_recorder(config):
  if config.enable_goodput_recording:
    logger_name = f"goodput_{config.run_name}"
    recorder = goodput.GoodputRecorder(config.run_name, logger_name, jax.process_index() == 0)
    return recorder
  return None


def record_goodput(recorder, config, step=None, job_start=False, job_end=False):
  if recorder and config.enable_goodput_recording:
    if job_start and step is None:
      recorder.record_job_start_time()
    if job_end and step is None:
      recorder.record_job_end_time()
    if step is not None:
      recorder.record_step_start_time(step)

def check_example_batch(config, example_batch):
  if config.max_checkify:
    jittable_f = checkify.checkify(
        lambda x: checkify.check(jnp.any(x > -1), "Batch contains bad synthetic data!")
    )
    # Check if inputs in batch contains bad synthetic data.
    err, _ = jax.jit(jittable_f)(example_batch['inputs'][: config.global_batch_size_to_train_on, :])
    err.throw()

def setup_mesh_and_model(config):
  """Set up the mesh and the model for training

  Args:
    config

  Returns:
    init_rng: RNG key
    writer: Summary writer for tensorboard
    checkpoint_manager: Orbax checkpointer
    state_mesh_annotations: the mesh annotations for the train state
    model:
    mesh:
    learning_rate_schedule:
    tx:
  """

  init_rng = random.PRNGKey(config.init_weights_seed)
  writer = max_utils.initialize_summary_writer(config)
  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
      config.checkpoint_dir,
      config.enable_checkpointing,
      config.async_checkpointing,
      config.checkpoint_period,
      config.dataset_type,
      [config.steps-1],
  )
  # Mesh definition
  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  # Model and Optimizer definition
  quant = quantizations.configure_quantization(config)
  model_cls = Transformer
  # if config.only_mlp:
  #   model_cls = models.MLP
  model = model_cls(config, mesh, quant=quant)
  learning_rate_schedule = max_utils.create_learning_rate_schedule(config)
  tx = optimizers.get_optimizer(config, learning_rate_schedule)
  return init_rng, writer, checkpoint_manager, mesh, model, learning_rate_schedule, tx


def setup_train_loop(config):
  """Set up prerequisites for the training loop -
      checkpoint_manager, PRNG keys, Mesh, Model and optimizer.
      Set up data iterator and tokenizer, initialize the model.

  Args:
    config

  Returns:
    init_rng:
    writer: Summary writer for tensorboard
    checkpoint_manager: Orbax checkpointer
    state_mesh_annotations: the mesh annotations for the train state
    model:
    mesh:
    learning_rate_schedule:
    data_iterator:
    state: the initialized train state
  """
  init_rng, writer, checkpoint_manager, mesh, model, learning_rate_schedule, tx = (
    setup_mesh_and_model(config)
    )
  
  print("inside setup train loop")
  
  data_iterator, eval_data_iterator, test_data_iterator = (
    create_data_iterator_with_tokenizer(config, mesh)
    )

  state, state_mesh_annotations, data_iterator = max_utils.setup_training_state(
      model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager
  )
  
  maxtext_utils.assert_params_sufficiently_sharded(state.params, mesh)

  return (
      init_rng,
      writer,
      checkpoint_manager,
      state_mesh_annotations,
      model,
      mesh,
      learning_rate_schedule,
      data_iterator,
      eval_data_iterator,
      test_data_iterator,
      state,
  )


def train_loop(config, state=None):
  """Main Training loop.
  Args:
    config:
    state:
    ckpt_path:
  Returns:
  """
  # Create a GoodputRecorder to log information
  recorder = create_goodput_recorder(config)
  record_goodput(recorder, config, job_start=True)

  wandb.login(key="your-wandb-key")
  if jax.process_index() == 0:
    
    raw_keys = config.get_keys()
    keys = [k for k in raw_keys]
    keys.sort()
    config_dict = {k: raw_keys[k] for k in keys}
    
    wandb.init(
        project="ClimSim",
        entity="joylunkad",
        name=config.run_name,
        config=config_dict,
        id=config.run_name,
        resume="allow",
    )

  (
      init_rng,
      writer,
      checkpoint_manager,
      state_mesh_annotations,
      model,
      mesh,
      learning_rate_schedule,
      data_iterator,
      eval_data_iterator,
      test_data_iterator,
      state,
  ) = setup_train_loop(config)
  # pylint: disable=line-too-long
  
  print("Input data -> ")
  for data in data_iterator:
    print(maxtext_utils.tree_shape(data))
    break
  
  (
      functional_train,
      in_shard_train,
      out_shard_train,
      static_argnums_train,
      donate_argnums_train,
  ) = maxtext_utils.get_functional_train_with_signature(
    train_step, mesh, state_mesh_annotations, model, config
    )

  if eval_data_iterator:
    
    print("Eval data -> ")
    for data in eval_data_iterator:
      print(maxtext_utils.tree_shape(data))
      break
    
    
    # pylint: disable=line-too-long
    (
        functional_eval,
        in_shard_eval,
        out_shard_eval,
        static_argnums_eval,
        donate_argnums_eval,
    ) = maxtext_utils.get_functional_eval_with_signature(
      eval_step, mesh, state_mesh_annotations, model, config
      )
  
  
  if test_data_iterator:

    # pylint: disable=line-too-long
    (
        functional_forward,
        in_shard_forward,
        out_shard_forward,
        static_argnums_forward,
        donate_argnums_forward,
    ) = maxtext_utils.get_functional_forward_with_signature(
      forward, mesh, state_mesh_annotations, model, config
      )

  rich.print(maxtext_utils.tree_shape(state.params))
  
  num_model_parameters = max_utils.calculate_num_params_from_pytree(state.params)
  max_logging.log(f"number parameters: {num_model_parameters/1e9:.3f} billion")
  per_device_tflops, _, _ = maxtext_utils.calculate_tflops_training_per_device(config)

  # Write train config params, num model params, and XLA flags to tensorboard
  max_utils.add_text_to_summary_writer("num_model_parameters", str(num_model_parameters), writer)
  max_utils.add_text_to_summary_writer("libtpu_init_args", os.environ["LIBTPU_INIT_ARGS"], writer)
  max_utils.add_config_to_summary_writer(config, writer)

  # Define the compilation of functional_train, either by loading the compiled version or wrapping a new one in a jit
  if config.compiled_trainstep_file != "":
    print("Loading the compiled function...", flush=True)
    # Need to pass train signature and state to determine i/o shapes of train_state for now.
    p_train_step = maxtext_utils.load_compiled(config, functional_train, state)
    print("Loaded compiled function!", flush=True)
  else:
    p_train_step = jax.jit(
        functional_train,
        in_shardings=in_shard_train,
        out_shardings=out_shard_train,
        static_argnums=static_argnums_train,
        donate_argnums=donate_argnums_train,
    )

    if eval_data_iterator:
      p_eval_step = jax.jit(
          functional_eval,
          in_shardings=in_shard_eval,
          out_shardings=out_shard_eval,
          static_argnums=static_argnums_eval,
          donate_argnums=donate_argnums_eval,
      )
    
    if test_data_iterator:
      p_forward = jax.jit(
          functional_forward,
          in_shardings=in_shard_forward,
          out_shardings=out_shard_forward[0],
          static_argnums=static_argnums_forward,
          donate_argnums=donate_argnums_forward,
      )

  local_metrics_file = open(config.metrics_file, "a", encoding="utf8") if config.metrics_file else None
  running_gcs_metrics = [] if config.gcs_metrics else None

  start_step = get_first_step(state)  # this is the start_step for training
  # start_step = 0 # this is the start_step for training
  first_profiling_step = start_step + config.skip_first_n_steps_for_profiler
  if config.enable_profiler and first_profiling_step >= config.steps:
    raise ValueError("Profiling requested but initial profiling step set past training final step")
  last_profiling_step = np.clip(first_profiling_step + config.profiler_steps - 1, first_profiling_step, config.steps - 1)

  example_batch = None
  last_step_completion = datetime.datetime.now()
  shuffle_rng = jax.random.PRNGKey(config.data_shuffle_seed)
  
  for step in np.arange(start_step, config.steps):
    if step == first_profiling_step:
      max_utils.activate_profiler(config)

    with jax.profiler.StepTraceAnnotation("train", step_num=step):
      example_batch = load_next_batch(data_iterator, example_batch, config)
      if step == 0:
        max_logging.log(maxtext_utils.tree_shape(example_batch))
        max_logging.log(example_batch)
        
      check_example_batch(config, example_batch=example_batch)
      nextrng = jax.jit(jax.random.fold_in)(init_rng, step)
      record_goodput(recorder, config, step=step)
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        state, metrics = p_train_step(state, example_batch, nextrng)

    if step == 1:
        max_logging.log(maxtext_utils.tree_shape(metrics))
        rich.print(metrics)
    
    new_time = datetime.datetime.now()
    record_scalar_metrics(
        metrics,
        new_time - last_step_completion,
        per_device_tflops,
        learning_rate_schedule(step),
      )
    last_step_completion = new_time

    if checkpoint_manager is not None:
      if save_checkpoint(checkpoint_manager, step, state, config.dataset_type, data_iterator):
        max_logging.log(f"saved a checkpoint at step {step}")

      # Upon preemption, exit when and only when all ongoing saves are complete.
      if checkpoint_manager.reached_preemption(step):
        checkpoint_manager.wait_until_finished()
        sys.exit()

    write_metrics(writer, local_metrics_file, running_gcs_metrics, metrics, step, config)

    if config.eval_interval > 0 and step > start_step and (step % config.eval_interval == 0 or step == config.steps - 1):
      assert eval_data_iterator
      cumulative_eval_metrics = {"total_loss": 0.0, "total_weights": 0.0, "total_r2_score": 0.0}
      for estep, eval_batch in enumerate(eval_data_iterator):
        if estep >= config.eval_steps:
          break
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
          eval_metrics = p_eval_step(state, eval_batch, nextrng)
        cumulative_eval_metrics["total_loss"] += float(eval_metrics["scalar"]["evaluation/total_loss"])
        cumulative_eval_metrics["total_weights"] += float(eval_metrics["scalar"]["evaluation/total_weights"])
        cumulative_eval_metrics["total_r2_score"] += float(eval_metrics["scalar"]["evaluation/r2_score"])
        
      eval_loss = cumulative_eval_metrics["total_loss"] / (estep + 1)
      r2_score = cumulative_eval_metrics["total_r2_score"] / (estep + 1)
      max_logging.log(
        f"average loss after {step=}: {eval_loss=}, {r2_score=}"
      )
      if jax.process_index() == 0:
        eval_scalars = {
          "eval/loss": eval_loss,
          "eval/r2_score": cumulative_eval_metrics["total_r2_score"] / (estep + 1)
        }
        wandb.log(eval_scalars, step=step)
      
      if eval_loss <= config.target_eval_loss:
        max_logging.log(f"Early stop and exit loop after reaching {config.target_eval_loss=}")
        max_utils.deactivate_profiler(config)
        break

    if step == last_profiling_step:
      max_utils.deactivate_profiler(config)

  if checkpoint_manager is not None:
    checkpoint_manager.wait_until_finished()
  
  max_utils.close_summary_writer(writer)
  record_goodput(recorder, config, job_end=True)

  if test_data_iterator:
    nextrng = jax.jit(jax.random.fold_in)(init_rng, config.steps)
    create_submission_csv(state, p_forward, test_data_iterator, config, mesh, nextrng)

  return state


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
  pyconfig.initialize(argv)
  config = pyconfig.config
  validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  vertex_tensorboard_manager = VertexTensorboardManager()
  if config.use_vertex_tensorboard or os.environ.get("UPLOAD_DATA_TO_TENSORBOARD"):
    vertex_tensorboard_manager.configure_vertex_tensorboard(config)

  debug_config = debug_configuration.DebugConfig(
      stack_trace_config=stack_trace_configuration.StackTraceConfig(
          collect_stack_trace=config.collect_stack_trace,
          stack_trace_to_cloud=config.stack_trace_to_cloud,
          stack_trace_interval_seconds=config.stack_trace_interval_seconds,
      )
  )
  diagnostic_config = diagnostic_configuration.DiagnosticConfig(debug_config)
  with diagnostic.diagnose(diagnostic_config):
    train_loop(config)


if __name__ == "__main__":
  app.run(main)
