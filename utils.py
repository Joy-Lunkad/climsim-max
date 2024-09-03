import os
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import functools
from etils.etree import tree as etree
import numpy as np
import tree
from google.cloud import storage
import natsort
import dill
from kaggle.api.kaggle_api_extended import KaggleApi


def tree_shape(tree):
    return jax.tree_util.tree_map(jnp.shape, tree)


def tree_dtype(tree):
    return jax.tree_util.tree_map(jnp.dtype, tree)


def tree_size(tree):
    return jax.tree_util.tree_map(lambda x: x.size, tree)


@functools.lru_cache(maxsize=1)
def _is_backend_tpu() -> bool:
    """Returns true iff running on TPUs."""
    return jax.default_backend() == "tpu" or jax.devices()[0].platform == "tpu"


def inspect(func: str = "", **kwargs):
    """
    Print the type, shape, the mean and variance of all variables in kwargs.
    Also print the function applied to the var if func is not None.
    """

    def pad_repr(R, L):
        l = len(R)
        pad = L - l
        R += " " * pad
        return R

    for k, v in kwargs.items():
        re_pr = f"{func}"
        re_pr += f"({k})" if k != "_" else ""
        re_pr = pad_repr(re_pr, 30)

        re_pr += f" = {etree.spec_like(v)}"
        re_pr = pad_repr(re_pr, 55)

        # Check if backend is TPU or if v is not a tracer
        if not _is_backend_tpu() and not isinstance(v, jax.core.Tracer):  # type: ignore
            axis = range(v.ndim - 1)

            mean = jnp.mean(jnp.mean(v, axis=axis) ** 2)
            var = jnp.mean(
                jnp.var(v, axis=axis),
            )

            mean = float(mean)
            var = float(var)

            re_pr += f"| mean={mean:.3f}, var={var:.3f}"
        print(re_pr)


def to_bf16(x):
    return x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x


def from_bf16(x):
    return x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x


def params_label_fn(params, layers_to_change, exceptions=[], labels=["type1", "type2"]):
    def mask_fn(p):
        return jax.tree_util.tree_map(lambda x: labels[0], p)

    mask = mask_fn(params)
    for (mod_name, var_name), _ in tree.flatten_with_path(params):  # type: ignore
        for layer in layers_to_change:
            if layer in var_name or layer in mod_name:
                mask[mod_name][var_name] = labels[1]
        for layer in exceptions:
            if layer in var_name or layer in mod_name:
                mask[mod_name][var_name] = labels[0]
    return mask


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)


def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


def list_blobs(bucket_name, prefix: str | None = None):
    """Lists all the blobs in the bucket. https://cloud.google.com/storage/docs/"""
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
    all_files = []
    for blob in blobs:
        all_files.append(blob.name)
    return all_files


def delete_checkpoint(checkpoint, bucket_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(checkpoint)
    if blob.exists():
        print("Deleting Step -> ", checkpoint.split("/")[2])
        blob.delete()
    else:
        print("Step does not exist -> ", checkpoint.split("/")[2])
    return


def clean_run_checkpoints(run, all_checkpoints, bucket_name):
    run_checkpoints = [path for path in all_checkpoints if run in path]
    sorted_checkpoints = natsort.natsorted(run_checkpoints)
    latest_checkpoint = sorted_checkpoints[-1]
    checkpoints_to_delete = sorted_checkpoints[:-1]
    assert latest_checkpoint not in checkpoints_to_delete

    if any(p in latest_checkpoint for p in ["trial", "del", "test", "Trial"]):
        print(".......Deleting Latest Checkpoint.....")
        checkpoints_to_delete.append(latest_checkpoint)
        assert latest_checkpoint in checkpoints_to_delete
        latest_checkpoint = None

    delete_fn = lambda x: delete_checkpoint(x, bucket_name)
    etree.parallel_map(delete_fn, checkpoints_to_delete)

    return latest_checkpoint, checkpoints_to_delete


def clean_exp_bucket(bucket_name: str):
    all_checkpoints = list_blobs(bucket_name, prefix="checkpoints")
    all_runs = set()
    for checkpoint in all_checkpoints:
        run_name, *_ = checkpoint.split("/")
        all_runs.add(run_name)
    print("Total Runs -> ", len(all_runs))

    total_deleted = 0
    for run in list(all_runs):
        print("-" * 50)
        print("Run -> ", run)
        latest_checkpoint, checkpoints_to_delete = clean_run_checkpoints(
            run, all_checkpoints, bucket_name
        )
        total_deleted += len(checkpoints_to_delete)
        print("Kept Checkpoint -> ", latest_checkpoint)
        print("Deleted Checkpoints -> ", len(checkpoints_to_delete))
        print("-" * 50)

    print("Total Deleted Checkpoints -> ", total_deleted)


def clean_buckets(
    bucket_names: list[str] = [
        "us1-climsim",
        "us2-climsim",
        "eu-climsim",
    ]
):
    for bucket_name in bucket_names:
        print("Cleaning Bucket -> ", bucket_name)
        clean_exp_bucket(bucket_name)
        print("Cleaned Bucket -> ", bucket_name)
        print("-" * 50)
    print("Cleaned All Buckets")


def keep_only_latest_checkpoint(bucket, save_path):
    all_checkpoints = list_blobs(bucket)
    latest_checkpoint, checkpoints_to_delete = clean_run_checkpoints(
        save_path, all_checkpoints, bucket
    )
    print("Kept Checkpoint -> ", latest_checkpoint)
    print("Deleted Checkpoints -> ", len(checkpoints_to_delete))


def get_checkpoint_names(current_save_dir, checkpoint_bucket):
    print(f"Looking for checkpoints in {current_save_dir} in {checkpoint_bucket}")
    all_dirs = list_blobs(checkpoint_bucket)
    current_checkpoints = [
        path for path in all_dirs if current_save_dir in path and "step" in path
    ]
    return current_checkpoints


def load_checkpoint(python_state_path):
    with open(python_state_path, "rb") as f:
        pretrained_state = dill.load(f)
    print(f"Restored checkpoint from {python_state_path}")
    return pretrained_state


def get_latest_checkpoint(checkpoints, bucket):
    python_state_path = natsort.natsorted(checkpoints)[-1]
    idx = python_state_path.rfind("/")
    os.makedirs(python_state_path[:idx], exist_ok=True)
    print("Downloading checkpoint from ", os.path.join(bucket, python_state_path))
    download_blob(bucket, python_state_path, python_state_path)
    return load_checkpoint(python_state_path)


def opt_state_float_to_int(opt_state_float):
    to_int = lambda x: x.astype(jnp.int32)
    opt_state_float = opt_state_float._replace(
        mini_step=jtu.tree_map(to_int, opt_state_float.mini_step)
    )
    opt_state_float = opt_state_float._replace(
        gradient_step=jtu.tree_map(to_int, opt_state_float.gradient_step)
    )
    return opt_state_float


def MultiTransformState_float_to_int(opt_state_float):
    x_1 = opt_state_float_to_int(opt_state_float.inner_states["type_1"].inner_state)
    x_2 = opt_state_float_to_int(opt_state_float.inner_states["type_2"].inner_state)
    opt_state_float.inner_states["type_1"] = opt_state_float.inner_states[
        "type_1"
    ]._replace(inner_state=x_1)
    opt_state_float.inner_states["type_2"] = opt_state_float.inner_states[
        "type_2"
    ]._replace(inner_state=x_2)
    return opt_state_float


def kaggle_submit_csv(csv_path, msg=""):
    api = KaggleApi()
    api.authenticate()
    api.competition_submit(
        csv_path,
        msg,
        competition="leap-atmospheric-physics-ai-climsim",
    )


def denorm_and_scale_wrapper(denorm_arr_path: str):

    x_denorm_mean = np.load(os.path.join(denorm_arr_path, "x_denorm_mean.npy"))
    x_denorm_std = np.load(os.path.join(denorm_arr_path, "x_denorm_std.npy"))
    y_denorm_mean = np.load(os.path.join(denorm_arr_path, "y_denorm_mean.npy"))
    y_denorm_std = np.load(os.path.join(denorm_arr_path, "y_denorm_std.npy"))
    scale_output_weights = np.load(
        os.path.join(denorm_arr_path, "scale_output_weights.npy")
    )
    not_ablated_col_indices = np.load(
        os.path.join(denorm_arr_path, "not_ablated_col_indices.npy")
    )

    def expand_add_ablated_cols(y):
        B = y.shape[0]
        placeholder = jnp.zeros((B, 368))
        placeholder = placeholder.at[:, not_ablated_col_indices].set(
            y, indices_are_sorted=True, unique_indices=True
        )
        return placeholder

    def denorm_output(y):
        y = y * y_denorm_std + y_denorm_mean
        return y

    def denorm_and_scale_output(y):
        y = y * y_denorm_std + y_denorm_mean
        y = y * scale_output_weights
        return y

    def denorm_input(x):
        x = x * x_denorm_std + x_denorm_mean
        return x

    def denorm_scale_and_expand_output(y):
        y = denorm_and_scale_output(y)
        y = expand_add_ablated_cols(y)
        return y

    out_fns = {
        "denorm_output": denorm_output,
        "denorm_and_scale_output": denorm_and_scale_output,
        "denorm_input": denorm_input,
        "expand_add_ablated_cols": expand_add_ablated_cols,
        "denorm_scale_and_expand_output": denorm_scale_and_expand_output,
    }

    return out_fns


def bcast_devices(value):
    """Broadcasts an object to num devices."""
    num_devices = len(jax.devices())
    return jax.tree_util.tree_map(
        lambda v: jnp.broadcast_to(v, (num_devices, *v.shape)), value
    )
