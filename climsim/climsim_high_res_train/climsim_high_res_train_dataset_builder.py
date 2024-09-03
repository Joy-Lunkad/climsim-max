"""climsim_high_res_train dataset."""

import dataclasses
import functools
import shutil
import tensorflow_datasets as tfds
import pandas as pd
import tensorflow as tf
import pickle
import numpy as np
from typing import Any
from huggingface_hub import hf_hub_download
from climsim_utils.data_utils import *
import fnmatch
from etils import etree
import time
import gc
import rich

from google.cloud import storage


def create_folder(bucket_name, folder_path):
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


v2_inputs = [
    "state_t",
    "state_q0001",
    "state_q0002",
    "state_q0003",
    "state_u",
    "state_v",
    "state_ps",
    "pbuf_SOLIN",
    "pbuf_LHFLX",
    "pbuf_SHFLX",
    "pbuf_TAUX",
    "pbuf_TAUY",
    "pbuf_COSZRS",
    "cam_in_ALDIF",
    "cam_in_ALDIR",
    "cam_in_ASDIF",
    "cam_in_ASDIR",
    "cam_in_LWUP",
    "cam_in_ICEFRAC",
    "cam_in_LANDFRAC",
    "cam_in_OCNFRAC",
    "cam_in_SNOWHICE",
    "cam_in_SNOWHLAND",
    "pbuf_ozone",  # outside of the upper troposphere lower stratosphere (UTLS, corresponding to indices 5-21), variance in minimal for these last 3
    "pbuf_CH4",
    "pbuf_N2O",
]

v2_outputs = [
    "ptend_t",
    "ptend_q0001",
    "ptend_q0002",
    "ptend_q0003",
    "ptend_u",
    "ptend_v",
    "cam_out_NETSW",
    "cam_out_FLWDS",
    "cam_out_PRECSC",
    "cam_out_PRECC",
    "cam_out_SOLS",
    "cam_out_SOLL",
    "cam_out_SOLSD",
    "cam_out_SOLLD",
]

vertically_resolved = [
    "state_t",
    "state_q0001",
    "state_q0002",
    "state_q0003",
    "state_u",
    "state_v",
    "pbuf_ozone",
    "pbuf_CH4",
    "pbuf_N2O",
    "ptend_t",
    "ptend_q0001",
    "ptend_q0002",
    "ptend_q0003",
    "ptend_u",
    "ptend_v",
]

ablated_vars = [
    "ptend_q0001",
    "ptend_q0002",
    "ptend_q0003",
    "ptend_u",
    "ptend_v",
]

v2_vars = v2_inputs + v2_outputs

train_col_names = []
ablated_col_names = []
for var in v2_vars:
    if var in vertically_resolved:
        for i in range(60):
            train_col_names.append(var + "_" + str(i))
            if i < 12 and var in ablated_vars:
                ablated_col_names.append(var + "_" + str(i))
    else:
        train_col_names.append(var)

input_col_names = []
for var in v2_inputs:
    if var in vertically_resolved:
        for i in range(60):
            input_col_names.append(var + "_" + str(i))
    else:
        input_col_names.append(var)

output_col_names = []
for var in v2_outputs:
    if var in vertically_resolved:
        for i in range(60):
            output_col_names.append(var + "_" + str(i))
    else:
        output_col_names.append(var)


repo_id = "LEAP/ClimSim_high-res"
local_dir = "./high_res_data/"


def get_dataset_from_file_names(train_files) -> dict[str, dict[str, Any]]:
    download_file_fn = functools.partial(
        hf_hub_download,
        repo_id=repo_id,
        repo_type="dataset",
        # cache_dir=local_dir,
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


def file_paths_to_df(file_paths, data_path):

    grid_path = os.path.join(data_path, "ClimSim_high-res_grid-info.nc")
    norm_path = "../preprocessing/normalizations/"
    grid_info = xr.open_dataset(grid_path)
    input_mean = xr.open_dataset(norm_path + "inputs/input_mean.nc")
    input_max = xr.open_dataset(norm_path + "inputs/input_max.nc")
    input_min = xr.open_dataset(norm_path + "inputs/input_min.nc")
    output_scale = xr.open_dataset(norm_path + "outputs/output_scale.nc")

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
    npy_input = np.concatenate([npy_iterator[x][0] for x in range(len(npy_iterator))])
    npy_output = np.concatenate([npy_iterator[x][1] for x in range(len(npy_iterator))])
    train_npy = np.concatenate([npy_input, npy_output], axis=1)
    train_index = ["train_" + str(x) for x in range(train_npy.shape[0])]

    df = pd.DataFrame(train_npy, index=train_index, columns=train_col_names)
    df.index.name = "sample_id"
    print("dropping cam_in_SNOWHICE because of strange values")
    df.drop("cam_in_SNOWHICE", axis=1, inplace=True)

    print("deleting file_paths to free up space")
    for file_path in file_paths.values():
        if "ClimSim_high-res_grid-info.nc" not in file_path:
            os.remove(file_path)
    shutil.rmtree(
        "/home/joylunkad/.cache/huggingface/hub/datasets--LEAP--ClimSim_high-res/"
    )
    return df


def get_splits(n_splits=1000):
    with open("../all_files_in_high_res.pkl", "rb") as f:
        all_file_paths = pickle.load(f)

    input_files = [f for f in all_file_paths if "mli" in f]
    output_files = set([f for f in all_file_paths if "mlo" in f])
    regexps = [
        "*/*/E3SM-MMF.mli.000[1234567]-*-*-*.nc",  # years 1 through 7
        "*/*/E3SM-MMF.mli.0008-01-*-*.nc",  # first month of year 8
    ]

    start = time.time()

    matched_files = []
    for pattern in regexps:
        matched_files.extend(fnmatch.filter(input_files, pattern))

    print(f"Matching files took {time.time() - start} seconds")
    start = time.time()

    filtered_files = [
        ip_f for ip_f in matched_files if ip_f.replace("mli", "mlo") in output_files
    ]
    print(f"Filtering files took {time.time() - start} seconds")

    splits = np.array_split(np.array(filtered_files), n_splits)
    splits = [s.tolist() for s in splits]
    return splits


@dataclasses.dataclass
class MyDatasetConfig(tfds.core.BuilderConfig):
    which_split: int = 0

    # Only Uncomment for the first run with config_idx = last
    # def __post_init__(self):
    #     create_folder('us2-climsim', f'climsim_high_res_train/{self.which_split}')


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for climsim_high_res_train dataset."""

    VERSION = tfds.core.Version("2.0.0")

    n = 512 if VERSION == "1.0.0" else 128
    BUILDER_CONFIGS = [
        MyDatasetConfig(
            name=f"{i}", description=f"{i}th split of the data", which_split=i
        )
        for i in range(n)
    ]

    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
        "2.0.0": "Throughly Shuffled 1.0.0",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    # "sample_id": tfds.features.Text(),
                    "inputs": tfds.features.Tensor(shape=(556,), dtype=tf.float32),
                    "targets": tfds.features.Tensor(shape=(368,), dtype=tf.float32),
                }
            ),
            supervised_keys=("inputs", "targets"),
            homepage="https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/data/",
            disable_shuffling=False,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        generate_examples_fn = (
            self._generate_examples
            if self.VERSION == "1.0.0"
            else self._shuffle_generate_examples
        )

        return {
            "train": generate_examples_fn(),
        }

    def _generate_examples(self):
        """Yields examples."""
        # Each Split Downloads, process and deletes the data to save disk space
        train_stats_df = pd.read_csv("train_stats.csv", index_col=0)
        patch_idx = self.builder_config.which_split
        n_splits = 4096
        num_splits_per_version = n_splits // 512
        start_idx = patch_idx * num_splits_per_version
        end_idx = start_idx + num_splits_per_version
        print(f"Generating examples for split {start_idx} to {end_idx} of {n_splits}")
        splits = get_splits(n_splits)[start_idx:end_idx]
        # create_folder('us2-climsim', f'climsim_high_res_train/{patch_idx}')

        hash_key = 0

        for train_files in splits:
            train_files.extend([f.replace("mli", "mlo") for f in train_files])
            train_files.append("ClimSim_high-res_grid-info.nc")
            file_paths, data_path = get_dataset_from_file_names(train_files)
            df = file_paths_to_df(file_paths, data_path)
            sample_ids = df.index
            x_train = df.iloc[:, :556]
            y_train = df.iloc[:, 556:]

            for col in x_train.columns.to_list():
                std = train_stats_df[col]["std"]
                if std != 0:
                    x_train[col] = (x_train[col] - train_stats_df[col]["mean"]) / std

            for col in y_train.columns.to_list():
                std = train_stats_df[col]["std"]
                if std != 0:
                    y_train[col] = (y_train[col] - train_stats_df[col]["mean"]) / std

            for sample_id, x, y in zip(sample_ids, x_train.values, y_train.values):
                hash_key += 1
                yield hash_key, {
                    # "sample_id": sample_id,
                    "inputs": x.astype("float32"),
                    "targets": y.astype("float32"),
                }

    def _shuffle_generate_examples(self):
        """Yields examples."""

        which_split = self.builder_config.which_split
        # create_folder("us2-climsim", f"climsim_high_res_train/{which_split}")

        bucket_name = "us2-climsim"
        prefix_list = [f"climsim_high_res_train/{i}/1.0.0" for i in range(512)]
        all_tfrecords = {
            prefix: list_gcs_files(bucket_name, prefix) for prefix in prefix_list
        }

        # Clean
        all_tfrecords = {
            k: [f for f in v if "train.tfrecord" in f] for k, v in all_tfrecords.items()
        }

        # Shuffle
        np.random.seed(42)
        {k: np.random.shuffle(v) for k, v in all_tfrecords.items()}
        all_tfrecords_list = [v for k, v in all_tfrecords.items()]
        all_tfrecords_list = np.concatenate(all_tfrecords_list)
        np.random.shuffle(all_tfrecords_list)
        assert len(all_tfrecords_list) == 256 * 512

        # Download, shuffle and yield
        files_per_split = len(all_tfrecords_list) // self.n
        start_idx = which_split * files_per_split
        end_idx = start_idx + files_per_split
        print(
            f"Generating examples for split {start_idx} to {end_idx}"
            f" of {self.n * files_per_split}"
        )
        files = all_tfrecords_list[start_idx:end_idx]
        files = [f"gs://us2-climsim/{f}" for f in files]
        rich.print(f"Processing Files: {files}")

        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(read_tfrecord)
        dataset = dataset.shuffle(32_000_000)

        for i, record in enumerate(dataset):
            yield i, {k: record[k].numpy() for k in record.keys()}


def read_tfrecord(serialized_example):

    feature_description = {
        "inputs": tf.io.FixedLenFeature([556], tf.float32),
        "targets": tf.io.FixedLenFeature([368], tf.float32),
    }
    features = tf.io.parse_single_example(
        serialized_example,
        feature_description,
    )
    return features
