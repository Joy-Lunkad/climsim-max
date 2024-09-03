import functools
import os
import warnings
import time

import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jaxline.utils as jl_utils
import dill

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import optax
import wandb

warnings.simplefilter(action="ignore", category=FutureWarning)

from dataclasses import dataclass
from typing import Any, Callable, Hashable, NamedTuple

import jmp
import rich
from absl import app
from etils import eapp

import model, exp_configs, utils
from climsim_low_res_train.climsim_low_res_train_dataset_builder import (
    ABLATED_COL_NAMES,
    USELESS_FEATURES,
)

Array = jax.Array


@dataclass
class Args:
    exp_version: str
    batch_size: int | None = None
    vm_name: str = ""
    test: bool = False
    only_submit_csv: bool = False


# from train import *
# args = Args(exp_version="2.0")


def turn_wandb_offline():
    import os

    os.environ["WANDB_API_KEY"] = "60042b6f73f3a032fb1097389f3c7bf42fe046f2"
    os.environ["WANDB_MODE"] = "offline"


# wandb.login(key="60042b6f73f3a032fb1097389f3c7bf42fe046f2")


#  _             _
# | |_ _ __ __ _(_)_ __
# | __| '__/ _` | | '_ \
# | |_| | | (_| | | | | |
#  \__|_|  \__,_|_|_| |_|
#


class EXPERIMENT_STATE(NamedTuple):
    params: hk.Params
    state: hk.State
    opt_state: optax.OptState
    ema_params: hk.Params | None
    ema_state: hk.State | None


def _build_input(config: exp_configs.ExpConfig):
    """Partial Params -> config"""

    num_devices = jax.local_device_count()

    bs_per_device, ragged = divmod(config.batch_size, num_devices)

    if ragged:
        raise ValueError(
            f"Global batch size {config.batch_size} must"
            f"be divisible by num devices {num_devices}"
        )

    if config.local:
        print("Using Dummy Data")
        N_IN = (490,)
        N_OUT = (308,)

        def dummy_data_gen(b_split: bytes):
            split = b_split.decode("utf-8")
            counter = 0
            while True:
                inputs = {
                    "inputs": tf.random.normal(shape=N_IN),
                    "sample_id": f"{split}_{counter}",
                }
                if split == "train" or split == "val":
                    inputs["targets"] = tf.random.normal(shape=N_OUT)
                counter += 1
                yield inputs

        train_ds = tf.data.Dataset.from_generator(
            dummy_data_gen,
            output_signature={
                "inputs": tf.TensorSpec(shape=N_IN, dtype=tf.float32),
                "targets": tf.TensorSpec(shape=N_OUT, dtype=tf.float32),
                "sample_id": tf.TensorSpec(shape=(), dtype=tf.string),
            },
            args=("train",),
        )

        val_ds = tf.data.Dataset.from_generator(
            dummy_data_gen,
            output_signature={
                "inputs": tf.TensorSpec(shape=N_IN, dtype=tf.float32),
                "targets": tf.TensorSpec(shape=N_OUT, dtype=tf.float32),
                "sample_id": tf.TensorSpec(shape=(), dtype=tf.string),
            },
            args=("val",),
        )

        test_ds = tf.data.Dataset.from_generator(
            dummy_data_gen,
            output_signature={
                "inputs": tf.TensorSpec(shape=N_IN, dtype=tf.float32),
                "sample_id": tf.TensorSpec(shape=(), dtype=tf.string),
            },
            args=("test",),
        )

    else:
        print(f"Loading {config.train_dataset_name} from {config.data_dir}")
        train_ds: tf.data.Dataset = tfds.load(
            config.train_dataset_name,
            data_dir=config.data_dir,
            download=False,
            shuffle_files=True,
            split="train[:95%]",
        )  # type: ignore

        val_ds: tf.data.Dataset = tfds.load(
            config.train_dataset_name,
            data_dir=config.data_dir,
            download=False,
            shuffle_files=True,
            split="train[95%:]",  # 500_000 samples
        )  # type: ignore

        test_ds: tf.data.Dataset = tfds.load(
            config.test_dataset_name,
            data_dir=config.data_dir,
            split="test",
        )  # type: ignore

    # Remove "sample_id" from train_ds and val_ds
    def remove_sample_id(elem):
        elem.pop("sample_id")
        return elem

    train_ds = train_ds.map(remove_sample_id)
    val_ds = val_ds.map(remove_sample_id)

    train_ds = train_ds.repeat().batch(bs_per_device).batch(num_devices)
    val_ds = val_ds.repeat().batch(bs_per_device).batch(num_devices)
    test_ds = test_ds.repeat().batch(bs_per_device).batch(num_devices)

    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

    train_data = tfds.as_numpy(train_ds)
    val_data = tfds.as_numpy(val_ds)
    test_data = tfds.as_numpy(test_ds)

    train_data: Any = iter(train_data)
    val_data: Any = iter(val_data)
    test_data: Any = iter(test_data)

    return train_data, val_data, test_data


def _forward_fn(
    inputs: dict[str, Any],
    is_training: bool,
    config: exp_configs.ExpConfig,
):
    """Partial Params -> config"""

    clim_simulator = model.ClimSimulator(config.model_args, verbose=config.verbose)
    x = inputs["inputs"]
    if config.bfloat16 and is_training:
        x = utils.to_bf16(x)

    return clim_simulator(x, is_training=is_training)


def _build_schedule_fn(config: exp_configs.ExpConfig):
    """Partial Params -> config"""
    opt_config = config.optimizer_config

    which_schedule = getattr(optax, opt_config["lr_schedule"]["name"])
    schedule = which_schedule(**opt_config["lr_schedule"]["kwargs"])

    def _get_learning_rate(step) -> float:
        return schedule(step)

    return _get_learning_rate


def _optimizer(learning_rate: jnp.ndarray, config: exp_configs.ExpConfig):
    """Partial Params -> config"""

    opt_config = config.optimizer_config

    optimizer_fn = getattr(optax, opt_config["name"])

    if "adamW" == opt_config["name"]:
        mask = functools.partial(
            utils.params_label_fn,
            layers_to_change=["rn"],
            labels=[True, False],
        )
        opt_config["opt_kwargs"]["mask"] = mask

    GC = opt_config["opt_addons"]["clip_grads"]
    AGC = opt_config["opt_addons"]["adaptive_grad_clip"]

    if AGC:
        print(f"Adaptive Clip Grads: {AGC}")
        clip_fn = optax.adaptive_grad_clip(AGC)
    elif GC:
        print(f"Clip Grads: {GC}")
        clip_fn = optax.clip(max_delta=GC)
    else:
        clip_fn = optax.identity()

    print(f"Optimizer: {opt_config['name']}")
    print(f"Optimizer kwargs: {opt_config['opt_kwargs']}")
    print(f"Optimizer addons: {opt_config['opt_addons']}")

    optimizer = optax.chain(
        clip_fn,
        optimizer_fn(learning_rate=learning_rate, **opt_config["opt_kwargs"]),
    )

    multi_steps: int = opt_config["opt_addons"]["grad_acc_steps"]
    if multi_steps:
        print(f"Using Multi-Step Grad Accumulation: {multi_steps}")
        opt = optax.MultiSteps(optimizer, every_k_schedule=multi_steps)
        optimizer = optax.GradientTransformation(
            init=opt.init,
            update=opt.update,  # type: ignore
        )

    return optimizer


def init_model(net, init_rng, inputs):
    init_net = jax.pmap(
        lambda *a: net.init(*a, is_training=True),
        axis_name="i",
    )
    init_key = jl_utils.bcast_local_devices(init_rng)
    _params, _state = init_net(init_key, inputs)
    return _params, _state, init_net, init_key


def get_opt(_params, optimizer):
    opt_init, _ = optimizer(
        jl_utils.bcast_local_devices(jnp.zeros([], jnp.int32)),
    )
    _opt_state = jax.pmap(opt_init)(_params)
    return _opt_state


def get_input(build_input):
    train_data, val_data, test_data = build_input()
    inputs = next(train_data)

    # wb_utils.upload_sample_to_wandb(
    #     image=inputs["images"][0][0], label=int(inputs["labels"][0][0])
    # )

    return train_data, val_data, test_data, inputs


def print_param_metrics(_params: hk.Params):
    num_params = hk.data_structures.tree_size(_params)
    num_params = num_params / jax.local_device_count()
    rich.print(f"num_params = {num_params / 1e6:.3f} M")


def _initialize_train(
    init_rng: Array,
    build_train_state: bool,
    config: exp_configs.ExpConfig,
    build_input: Callable,
    net: hk.TransformedWithState,
    optimizer: Callable,
):
    """
    Partial Params -> config, build_input, net, optimizer
    """
    train_data, val_data, test_data, inputs = get_input(build_input)
    if not build_train_state:
        return None, train_data, val_data, test_data

    _params, _state, init_net, init_key = init_model(net, init_rng, inputs)
    print_param_metrics(_params)
    _opt_state = get_opt(_params, optimizer)

    if config.use_ema:
        _ema_params, _ema_state = init_net(init_key, inputs)
    else:
        _ema_params, _ema_state = None, None

    train_state = EXPERIMENT_STATE(
        _params,
        _state,
        _opt_state,
        _ema_params,
        _ema_state,
    )
    return train_state, train_data, val_data, test_data


@jax.vmap
def calc_r2_score(y_true, y_pred):
    numerator = jnp.sum((y_true - y_pred) ** 2)
    denominator = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
    r2 = 1 - (numerator / (denominator + 1e-7))
    return r2


def collect_loss_and_metrics(
    y_pred: Array,
    inputs: dict[str, Any],
    config: exp_configs.ExpConfig,
    is_training: bool,
) -> tuple[Array, dict[str, Array]]:

    if is_training and config.bfloat16:
        y_pred = y_pred.astype(jnp.float32)

    mae_loss = jnp.mean(jnp.abs(inputs["targets"] - y_pred))
    y_pred = config.post_process_y(y_pred)
    y_true = config.post_process_y(inputs["targets"])

    r2_score = calc_r2_score(y_true, y_pred)
    r2_score = jnp.mean(r2_score)

    prefix = "train" if is_training else "eval"
    metrics = {
        f"{prefix}_mae": mae_loss,
        f"{prefix}_r2_score": r2_score,
    }

    return mae_loss, metrics


def _loss_fn(
    params: hk.Params,
    state: hk.State,
    inputs: dict[str, Any],
    rng: Array,
    is_training: bool,
    config: exp_configs.ExpConfig,
    net: hk.TransformedWithState,
):
    """Partial Params -> config, net"""

    y_pred, state = net.apply(
        params,
        state,
        rng,
        inputs,
        is_training=is_training,
    )

    loss, metrics = collect_loss_and_metrics(
        y_pred=y_pred,
        inputs=inputs,
        config=config,
        is_training=is_training,
    )

    scaled_loss = loss / jax.device_count()  # Grads get psummed so do divide
    return scaled_loss, (metrics, state)


def process_ema(
    params,
    states,
    global_step,
    ema_params,
    ema_states,
    config: exp_configs.ExpConfig,
):

    def tf1_ema(ema_value, current_value, decay, step):
        """Implements EMA with TF1-style decay warmup."""
        decay = jnp.minimum(decay, (1.0 + step) / (10.0 + step))
        return ema_value * decay + current_value * (1 - decay)

    def ema(x, y):
        return tf1_ema(x, y, config.ema_decay, global_step)

    ema_params = jtu.tree_map(ema, ema_params, params)
    ema_states = jtu.tree_map(ema, ema_states, states)
    return ema_params, ema_states


def _train_fn(
    params,
    states,
    opt_states,
    inputs,
    rng,
    global_step: int,
    ema_params,
    ema_states,
    config: exp_configs.ExpConfig,
    get_learning_rate,
    loss_fn,
    optimizer,
):
    """Partial Params -> config, get_learning_rate, loss_fn, optimizer"""

    learning_rate = get_learning_rate(global_step)
    inputs["learning_rate"] = learning_rate
    inputs["global_step"] = global_step

    in_params = params
    if config.bfloat16:
        in_params, states = jtu.tree_map(utils.to_bf16, (params, states))

    grad_fn = jax.grad(loss_fn, argnums=0, has_aux=True)
    grads, outputs = grad_fn(in_params, states, inputs, rng, True)
    metrics, states = outputs

    if config.bfloat16:
        states, metrics, grads = jtu.tree_map(
            utils.from_bf16,
            (states, metrics, grads),
        )

    if not config.local:
        grads = jax.lax.psum(grads, "i")
        metrics = jax.lax.pmean(metrics, "i")

    metrics["learning_rate"] = learning_rate

    _, opt_apply = optimizer(learning_rate)
    updates, opt_states = opt_apply(grads, opt_states, params)
    params = optax.apply_updates(params, updates)

    if ema_params is not None:
        ema_params, ema_states = process_ema(
            params,
            states,
            global_step,
            ema_params,
            ema_states,
            config,
        )

    outputs = {
        "params": params,
        "states": states,
        "opt_states": opt_states,
        "ema_params": ema_params,
        "ema_states": ema_states,
        "metrics": metrics,
    }

    return outputs


def _train_step(
    inputs: dict[str, Any],
    train_state,
    global_step,
    rng,
    train_fn,
):
    """Partial Params -> train_fn"""

    out = train_fn(
        params=train_state.params,
        states=train_state.state,
        opt_states=train_state.opt_state,
        inputs=inputs,
        rng=rng,
        global_step=global_step,
        ema_params=train_state.ema_params,
        ema_states=train_state.ema_state,
    )

    _params, _state = out["params"], out["states"]
    _opt_state = out["opt_states"]
    _ema_params, _ema_state = out["ema_params"], out["ema_states"]

    scalars = jl_utils.get_first(out["metrics"])
    train_state = EXPERIMENT_STATE(_params, _state, _opt_state, _ema_params, _ema_state)

    return scalars, train_state


def _eval_fn(params, state, inputs, loss_fn):
    """Partial Params -> _loss_fn"""

    _, (metrics, _) = loss_fn(
        params=params,
        state=state,
        inputs=inputs,
        rng=None,
        is_training=False,
    )

    return jax.lax.pmean(metrics, "i")


def _eval_epoch(
    params,
    state,
    test_data,
    eval_fn,
    config: exp_configs.ExpConfig,
):
    """partial params -> eval_fn, config"""

    summed_metrics = None
    for _ in range(config.val_steps_per_epoch):
        data = next(test_data)
        metrics = eval_fn(params, state, data)
        metrics = jtu.tree_map(lambda x: x[0], metrics)
        if summed_metrics is None:
            summed_metrics = metrics
        else:
            summed_metrics = jtu.tree_map(jnp.add, summed_metrics, metrics)

    metrics = jtu.tree_map(lambda x: x / config.val_steps_per_epoch, summed_metrics)

    return metrics


def _evaluate(
    _params,
    _state,
    _ema_params,
    _ema_state,
    test_data,
    config: exp_configs.ExpConfig,
    eval_epoch,
):
    """Partial Params -> config, eval_epoch"""
    metrics = eval_epoch(_params, _state, test_data)
    if config.use_ema:
        ema_metrics = eval_epoch(_ema_params, _ema_state, test_data)
        metrics.update({f"ema_{key}": val for key, val in ema_metrics.items()})

    return metrics


def build_checkpointer(config: exp_configs.ExpConfig):

    save_path = config.save_path
    bucket = config.data_dir[5:]

    def save_checkpoint(step, train_state):

        state_dict = {
            "step": int(step),
            "params": train_state.params,
            "state": train_state.state,
            "ema_params": train_state.ema_params,
            "ema_state": train_state.ema_state,
            "opt_state": train_state.opt_state,
        }

        save_dir = os.path.join(save_path, f"step_{step}")
        os.makedirs(save_dir, exist_ok=True)
        python_state_path = os.path.join(save_dir, "checkpoint.dill")
        with open(python_state_path, "wb") as f:
            dill.dump(state_dict, f)
        print(
            f"Saved step_{step} checkpoint to {os.path.join(bucket, python_state_path)}"
        )
        utils.upload_blob(
            bucket_name=bucket,
            source_file_name=python_state_path,
            destination_blob_name=python_state_path,
        )
        os.remove(python_state_path)

        if config.keep_latest_checkpoint_only:
            utils.keep_only_latest_checkpoint(bucket, save_path)

    return save_checkpoint


def get_checkpoints(config: exp_configs.ExpConfig):

    # First find latest checkpoint in the current run directory.
    current_run_dir = config.save_path
    bucket = config.data_dir[5:]

    run_checkpoints = utils.get_checkpoint_names(current_run_dir, bucket)
    if len(run_checkpoints):
        return run_checkpoints

    # if there are no checkpoints in the current run directory, then look for
    # checkpoints in the restore path, in the restore path bucket.
    print("No checkpoints found in current run directory")

    if config.restore_path:
        # Assert restore_path in the same bucket as the data_dir
        restore_path_checkpoints = utils.get_checkpoint_names(
            config.restore_path, bucket
        )
        if len(restore_path_checkpoints):
            return restore_path_checkpoints

    return None


def extract_train_state_from_pretrained_state(pretrained_state):

    step = jl_utils.get_first(pretrained_state["step"])
    _params = pretrained_state["params"]
    _state = pretrained_state["state"]
    _ema_params = pretrained_state["ema_params"]
    _ema_state = pretrained_state["ema_state"]

    fields = getattr(pretrained_state["opt_state"], "_fields", None)
    if fields and "mini_step" in fields:
        rich.print("Converting opt_state floats to ints")
        rich.print("*" * 80)
        _opt_state = utils.opt_state_float_to_int(
            opt_state_float=pretrained_state["opt_state"]
        )
    elif (
        isinstance(pretrained_state["opt_state"], optax.MultiTransformState)
        and hasattr(pretrained_state["opt_state"], "inner_states")
        and "type_1" in pretrained_state["opt_state"].inner_states
        and "type_2" in pretrained_state["opt_state"].inner_states
    ):
        rich.print("Converting opt_state floats to ints")
        rich.print("*" * 80)
        _opt_state = utils.MultiTransformState_float_to_int(
            opt_state_float=pretrained_state["opt_state"]
        )
    else:
        _opt_state = pretrained_state["opt_state"]
    train_state = EXPERIMENT_STATE(_params, _state, _opt_state, _ema_params, _ema_state)
    return step, train_state


def _restore_state_from_checkpoint(
    init_rng,
    config: exp_configs.ExpConfig,
    initialize_train,
):
    """Partial Params -> config, initialize_train"""

    checkpoints = None
    pretrained_state = None
    bucket = config.data_dir[5:]

    if config.local and config.restore_path:
        pretrained_state = utils.load_checkpoint(config.restore_path)

    if not config.local:
        checkpoints = get_checkpoints(config)
        if checkpoints:
            pretrained_state = utils.get_latest_checkpoint(checkpoints, bucket)

    if pretrained_state is None:
        print("Starting from scratch.")
        train_state, train_data, val_data, test_data = initialize_train(
            init_rng, build_train_state=True
        )
        return dict(
            step=0,
            train_state=train_state,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
        )

    pretrained_state = jl_utils.bcast_local_devices(pretrained_state)

    _, train_data, val_data, test_data = initialize_train(
        init_rng, build_train_state=False
    )
    step, train_state = extract_train_state_from_pretrained_state(pretrained_state)
    return dict(
        step=step,
        train_state=train_state,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
    )


def build_partial_functions(config: exp_configs.ExpConfig):
    build_input = functools.partial(_build_input, config=config)
    forward_fn = functools.partial(_forward_fn, config=config)
    net = hk.transform_with_state(forward_fn)
    get_learning_rate = _build_schedule_fn(config=config)
    optimizer = functools.partial(_optimizer, config=config)
    initialize_train = functools.partial(
        _initialize_train,
        config=config,
        build_input=build_input,
        net=net,
        optimizer=optimizer,
    )

    loss_fn = functools.partial(_loss_fn, config=config, net=net)

    train_fn = functools.partial(
        _train_fn,
        config=config,
        get_learning_rate=get_learning_rate,
        loss_fn=loss_fn,
        optimizer=optimizer,
    )

    donate_argnums = (0, 1, 2, 6, 7) if config.use_ema else (0, 1, 2)
    train_fn = jax.pmap(train_fn, axis_name="i", donate_argnums=donate_argnums)

    train_step = functools.partial(
        _train_step,
        train_fn=train_fn,
    )

    eval_fn = functools.partial(_eval_fn, loss_fn=loss_fn)
    eval_fn = jax.pmap(eval_fn, axis_name="i")

    eval_epoch = functools.partial(_eval_epoch, eval_fn=eval_fn, config=config)
    evaluate = functools.partial(_evaluate, config=config, eval_epoch=eval_epoch)

    restore_state_from_checkpoint = functools.partial(
        _restore_state_from_checkpoint,
        config=config,
        initialize_train=initialize_train,
    )

    return dict(
        build_input=build_input,
        forward_fn=forward_fn,
        net=net,
        get_learning_rate=get_learning_rate,
        optimizer=optimizer,
        initialize_train=initialize_train,
        loss_fn=loss_fn,
        train_fn=train_fn,
        train_step=train_step,
        eval_fn=eval_fn,
        eval_epoch=eval_epoch,
        evaluate=evaluate,
        restore_state_from_checkpoint=restore_state_from_checkpoint,
    )


def create_submission_csv(
    train_state: EXPERIMENT_STATE,
    net: hk.TransformedWithState,
    test_data,
    config: exp_configs.ExpConfig,
):

    import pandas as pd
    from tqdm import tqdm

    net_apply = jax.pmap(
        lambda *a: net.apply(*a, is_training=False),
        axis_name="i",
    )

    sample_submission = os.path.join(config.data_dir, "sample_submission.csv")
    sub_df = pd.read_csv(sample_submission, nrows=1)

    python_state_path = os.path.join(config.save_path, "submission.csv")

    results = []
    for test_step in tqdm(range(config.test_steps + 1)):
        inputs = next(test_data)
        sample_ids = inputs.pop("sample_id")
        y_pred, _ = net_apply(
            train_state.ema_params,
            train_state.ema_state,
            None,
            inputs,
        )

        sample_ids = sample_ids.reshape(-1)
        sample_ids = [id.decode("utf-8") for id in sample_ids]

        y_pred = y_pred.reshape(-1, y_pred.shape[2])
        y_pred = config.post_process_y(y_pred)

        data = {"sample_id": sample_ids}
        for col_idx, col_name in enumerate(sub_df.columns.tolist()):
            if col_name == "sample_id":
                continue
            data[col_name] = y_pred[:, col_idx - 1]
        results.append(pd.DataFrame(data))

    final_df = pd.concat(results)
    final_df = final_df.drop_duplicates(subset="sample_id", keep="first")
    final_df.to_csv(python_state_path, index=False)
    print(final_df.describe())

    utils.upload_blob(
        bucket_name=config.data_dir[5:],
        source_file_name=python_state_path,
        destination_blob_name=python_state_path,
    )

    if config.submit_to_kaggle:
        utils.kaggle_submit_csv(
            python_state_path,
            msg=f"{config.name} submission",
        )


def run_train_loop(
    cur_state,
    fns,
    config: exp_configs.ExpConfig,
):

    rng_key: Array = jax.random.PRNGKey(config.random_seed)
    rng_key_seq = hk.PRNGSequence(rng_key)

    init_step = cur_state["step"]
    train_state: EXPERIMENT_STATE = cur_state["train_state"]
    train_data = cur_state["train_data"]
    val_data = cur_state["val_data"]
    previous_best_model_eval_metric = 1e6
    save_model_fn = build_checkpointer(config=config)

    num_steps = 0.0
    summed_scalars = None
    start = time.perf_counter()
    for step in range(init_step, config.training_steps):
        inputs = next(train_data)
        # inputs = jtu.tree_map(policy.cast_to_compute, inputs)
        rng = next(rng_key_seq)
        num_steps += 1

        scalars, train_state = fns["train_step"](
            inputs,
            train_state,
            global_step=jl_utils.bcast_local_devices(step),
            rng=jl_utils.bcast_local_devices(rng),
        )

        if summed_scalars is None:
            summed_scalars = scalars
        else:
            summed_scalars = jtu.tree_map(jnp.add, summed_scalars, scalars)

        if (
            (step % config.log_every == 0)
            or step == config.training_steps - 1
            or step == 10
            or step == 100
        ):
            scalars = jtu.tree_map(lambda x: x / num_steps, summed_scalars)
            scalars = jax.device_get(scalars)
            time_per_step = (time.perf_counter() - start) / (num_steps)
            scalars["step"] = step
            scalars["time_per_step"] = time_per_step
            wandb.log(scalars, step)
            rich.print(f"[Step {step}] Train scalars: {scalars}")

            num_steps = 0.0
            summed_scalars = None
            start = time.perf_counter()

        if (
            (step % config.eval_every == 0 and step)
            or step == config.training_steps - 1
            or step == 100
        ):
            eval_scalars = fns["evaluate"](
                train_state.params,
                train_state.state,
                train_state.ema_params,
                train_state.ema_state,
                val_data,
            )

            wandb.log(eval_scalars, step)
            rich.print(f"[Step {step}] Eval scalars: {eval_scalars}")

            start = time.perf_counter()

        if (
            step % config.checkpoint_every == 0
            or step == config.training_steps - 1
            or step == 100
        ):
            if step == init_step:
                continue

            if config.save_best_checkpoint:
                best_model_eval_metric = eval_scalars[config.best_model_eval_metric]
                print(
                    f"Previous_Best/Current {config.best_model_eval_metric}: ",
                    previous_best_model_eval_metric,
                    best_model_eval_metric,
                )
                if previous_best_model_eval_metric >= best_model_eval_metric:
                    previous_best_model_eval_metric = best_model_eval_metric
                    print("Saving best model...")
                    save_model_fn(step, jl_utils.get_first(train_state))

            else:
                save_model_fn(step, jl_utils.get_first(train_state))

            start = time.perf_counter()

    return train_state


def main(args: Args):

    config = exp_configs.get_experiment_config(
        args.exp_version,
        local=args.test,
        vm_name=args.vm_name,
        batch_size=args.batch_size,
    )

    if config.bfloat16:
        policy = jmp.get_policy("p=f32,c=bf16,o=bf16")
        norm_policy = jmp.get_policy("p=f32,c=f32,o=bf16")
        hk.mixed_precision.set_policy(hk.RMSNorm, norm_policy)
        hk.mixed_precision.set_policy(model.ClimSimulator, policy)

    fns = build_partial_functions(config)

    wandb.login(key="60042b6f73f3a032fb1097389f3c7bf42fe046f2")
    wandb.init(
        project="ClimSim",
        entity="joylunkad",
        name=config.name,
        config=config.__dict__,
        id=config.name,
        resume="allow",
    )

    init_rng = jax.random.PRNGKey(config.random_seed)
    cur_state = fns["restore_state_from_checkpoint"](init_rng)

    if args.only_submit_csv:
        train_state = cur_state["train_state"]

    else:
        train_state = run_train_loop(
            config=config,
            fns=fns,
            cur_state=cur_state,
        )

    if config.create_submission_csv:
        create_submission_csv(
            train_state=train_state,
            net=fns["net"],
            config=config,
            test_data=cur_state["test_data"],
        )

    return config


if __name__ == "__main__":
    app.run(main, flags_parser=eapp.make_flags_parser(Args))  # type: ignore
