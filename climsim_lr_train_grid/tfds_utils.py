import xarray as xr

norm_path = "../climsim/preprocessing/normalizations/"
input_mean = xr.open_dataset(norm_path + "inputs/input_mean.nc")
input_max = xr.open_dataset(norm_path + "inputs/input_max.nc")
input_min = xr.open_dataset(norm_path + "inputs/input_min.nc")
output_scale = xr.open_dataset(norm_path + "outputs/output_scale.nc")

import functools
import shutil
import numpy as np
from typing import Any
from huggingface_hub import hf_hub_download
from climsim.climsim_utils.data_utils import *
import fnmatch
from etils import etree
import time

import functools
import os
from typing import Any

from etils import etree

from google.cloud import storage


def create_gcs_folder(bucket_name, folder_path):
    # Initialize the storage client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    # Create a blob with the folder path and a trailing slash
    folder_blob = bucket.blob(f"{folder_path}/")
    folder_blob.upload_from_string("")
    print(f"Folder '{folder_path}' created in bucket '{bucket_name}'.")


def list_gcs_files(bucket_name, prefix):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    all_files = []
    for blob in blobs:
        all_files.append(blob.name)
    return all_files


def get_splits(all_file_paths, n_splits, time_stride=7):

    input_files = [f for f in all_file_paths if "mli" in f]
    output_files = set([f for f in all_file_paths if "mlo" in f])

    print(f"Total input files: {len(input_files)}")
    print(f"Total output files: {len(output_files)}")

    regexps = [
        "*/*/E3SM-MMF.mli.000[1234567]-*-*-*.nc",  # years 1 through 7
        "*/*/E3SM-MMF.mli.0008-01-*-*.nc",  # first month of year 8
    ]

    start = time.time()

    matched_files = []
    for pattern in regexps:
        matched_files.extend(fnmatch.filter(input_files, pattern))

    print(f"Total matched files: {len(matched_files)}")
    print(f"Matching files took {time.time() - start} seconds")

    start = time.time()

    filtered_files = [
        ip_f for ip_f in matched_files if ip_f.replace("mli", "mlo") in output_files
    ]
    print(f"Filtering files took {time.time() - start} seconds")

    filtered_files = filtered_files[::time_stride]
    filtered_files = np.array(filtered_files)

    print(f"Total filtered files: {len(filtered_files)}")

    total_files = len(filtered_files)
    files_per_split = total_files // n_splits
    print(f"Files per split: {files_per_split}")

    np.random.seed(42)
    np.random.shuffle(filtered_files)

    splits = np.array_split(filtered_files, n_splits)
    splits = [s.tolist() for s in splits]

    return splits


def get_dataset_from_file_names(
    train_files,
    repo_id="LEAP/ClimSim_high-res",
) -> dict[str, dict[str, Any]]:
    download_file_fn = functools.partial(
        hf_hub_download,
        repo_id=repo_id,
        repo_type="dataset",
    )

    def download_file(filename: str) -> str:
        try:
            return download_file_fn(filename=filename)
        except Exception as e:
            print(f"Failed to download {filename} with error {e}")
            return None

    download_paths = etree.parallel_map(download_file, train_files)
    download_paths = [x for x in download_paths if x is not None]
    file_paths = dict(zip(train_files, download_paths))
    data_path = "/".join(file_paths[train_files[0]].split("/")[:-3])
    return file_paths, data_path


def extract_data_from_file_paths(data_path, grid_file_name, keep_grid=False):

    grid_path = os.path.join(data_path, grid_file_name)
    grid_info = xr.open_dataset(grid_path)

    data = data_utils(
        grid_info=grid_info,
        input_mean=input_mean,
        input_max=input_max,
        input_min=input_min,
        output_scale=output_scale,
    )

    data.set_to_v2_vars()

    data.normalize = False
    data.data_path = f"{data_path}/*/"

    data.set_regexps(
        data_split="train",
        regexps=[
            "E3SM-MMF.mli.000[1234567]-*-*-*.nc",  # years 1 through 7
            "E3SM-MMF.mli.0008-01-*-*.nc",  # first month of year 8
        ],
    )

    data.set_stride_sample(data_split="train", stride_sample=1)
    data.set_filelist(data_split="train")
    data_loader = data.load_ncdata_with_generator(data_split="train")
    npy_iterator = list(data_loader.as_numpy_iterator())

    filelist = np.array(
        [npy_iterator[x][2] for x in range(len(npy_iterator))]
    ).flatten()
    filelist = [x.decode() for x in filelist]
    file_ids = [f.split("/")[-1].split(".")[-2] for f in filelist]

    train_index = []
    for f_idx in range(len(file_ids)):
        sample_id = f"train_{file_ids[f_idx]}"
        if keep_grid:
            train_index.append(sample_id)
        else:
            for x in range(len(npy_iterator[f_idx][0])):
                sample_id += f"_{str(x)}"
                train_index.append(sample_id)

    which_join = np.array if keep_grid else np.concatenate

    train_index = which_join(train_index)

    npy_input = which_join([npy_iterator[x][0] for x in range(len(npy_iterator))])

    print("dropping cam_in_SNOWHICE because of strange values")
    train_col_names = np.load("../train_col_names.npy").tolist()
    drop_idx = train_col_names.index("cam_in_SNOWHICE")
    npy_input = np.delete(npy_input, drop_idx, axis=-1)

    npy_output = which_join([npy_iterator[x][1] for x in range(len(npy_iterator))])

    num_grid_cols = npy_iterator[0][1].shape[0]
    grid_ids = np.arange(0, num_grid_cols)
    grid_ids = np.broadcast_to(grid_ids, (len(npy_iterator), num_grid_cols))

    if not keep_grid:
        grid_ids = grid_ids.flatten()

    print("deleting file_paths to free up space")   
    shutil.rmtree(
        "/home/joylunkad/.cache/huggingface/hub/datasets--LEAP--ClimSim_low-res/"
    )

    return train_index, npy_input, npy_output, grid_ids
