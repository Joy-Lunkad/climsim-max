import collections
import xarray as xr

norm_path = "climsim/preprocessing/normalizations/"
input_mean = xr.open_dataset(norm_path + "inputs/input_mean.nc")
input_max = xr.open_dataset(norm_path + "inputs/input_max.nc")
input_min = xr.open_dataset(norm_path + "inputs/input_min.nc")
output_scale = xr.open_dataset(norm_path + "outputs/output_scale.nc")


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
from climsim.climsim_utils.data_utils import *
import fnmatch
from etils import etree
import time
import gc
import rich

from collections.abc import Iterable, Sequence
import dataclasses
import functools
import itertools
import json
import os
from typing import Any

from absl import app
from etils import epy
from etils import eapp
from etils import etree

with epy.lazy_imports():
    # pylint: disable=g-import-not-at-top
    from absl import logging
    from etils import epath
    from tensorflow_datasets.core import example_parser
    from tensorflow_datasets.core import naming
    from tensorflow_datasets.core import shuffle
    from tensorflow_datasets.core import utils
    from tensorflow_datasets.core.utils import shard_utils

from google.cloud import storage
import pandas as pd
from tqdm import tqdm
import subprocess
import rich


@dataclasses.dataclass
class Args:
    local_idx: int = 0
    write_initial_tfrecords: bool = False
    shuffle_tfrecords: bool = False
    finalize_shuffled_tfrecords: bool = False
    delete_tmp_files: bool = False
    delete_split_files: str = ""
    shuffle_split_key: str = ""
    rename_split_to_config_structure: str = ""
    restructure_tfrecords: bool = False


train_col_names = np.load("saved_data/train_col_names.npy").tolist()
repo_id = "LEAP/ClimSim_high-res"
local_dir = "./high_res_data/"

s_mean = np.load("saved_data/s_mean.npy")
s_std = np.load("saved_data/s_std.npy")

ip_means = np.load("saved_data/input_means.npy")
ip_stds = np.load("saved_data/input_stds.npy")

train_mean_and_std_df = pd.read_csv("saved_data/train_mean_and_std.csv", index_col=0)
op_means = train_mean_and_std_df["mean"][556:].values
op_stds = train_mean_and_std_df["std"][556:].values

TFRECORDS_DIR_BASE = "gs://us2-climsim/climsim_high_res_train/"
VERSION = "1.0.0"


def z_norm_input(x):
    # For full data -> (556,) only
    x_mean = x - ip_means
    x_norm = x_mean / ip_stds
    x_norm = np.where(ip_stds < 1e-10, x_mean, x_norm)
    return x_norm


def z_norm_output(y):
    # For full data -> (368,) only
    y_mean = y - op_means
    y_norm = y_mean / op_stds
    y_norm = np.where(op_stds < 1e-10, y_mean, y_norm)
    return y_norm


def calc_s_score(x):
    return np.sqrt(np.sum(x**2, axis=1))


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


def extract_data_from_file_paths(file_paths, data_path):

    grid_path = os.path.join(data_path, "ClimSim_high-res_grid-info.nc")
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

    train_index = [
        [
            f"train_{file_ids[f_idx]}_{str(x)}"
            for x in range(len(npy_iterator[f_idx][0]))
        ]
        for f_idx in range(len(file_ids))
    ]
    train_index = np.concatenate(train_index)

    npy_input = np.concatenate([npy_iterator[x][0] for x in range(len(npy_iterator))])
    print("dropping cam_in_SNOWHICE because of strange values")
    drop_idx = train_col_names.index("cam_in_SNOWHICE")
    npy_input = np.delete(npy_input, drop_idx, axis=1)
    npy_output = np.concatenate([npy_iterator[x][1] for x in range(len(npy_iterator))])

    print("deleting file_paths to free up space")
    for file_path in file_paths.values():
        if "ClimSim_high-res_grid-info.nc" not in file_path:
            os.remove(file_path)

    shutil.rmtree(
        "/home/joylunkad/.cache/huggingface/hub/datasets--LEAP--ClimSim_high-res/"
    )

    return train_index, npy_input, npy_output


def get_splits(n_splits):
    with open("climsim/all_files_in_high_res.pkl", "rb") as f:
        all_file_paths = pickle.load(f)

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

    filtered_files = filtered_files[::7]  # Time Stride
    filtered_files = np.array(filtered_files)

    print(f"Total filtered files: {len(filtered_files)}")

    total_files = len(filtered_files)
    files_per_split = total_files // n_splits
    print(f"Files per split: {files_per_split}")

    np.random.seed(42)
    np.random.shuffle(filtered_files)

    splits = np.array_split(filtered_files, n_splits)
    splits = [s.tolist() for s in splits]

    print(len(splits[0]))
    return splits


def assign_buckets(x, y, sample_ids, buckets):
    x_norm = z_norm_input(x)
    Sx = calc_s_score(x_norm)
    Sx_norm = (Sx - s_mean) / s_std
    Sx_norm_bucket = np.digitize(Sx_norm, buckets)
    bucket_indices = [np.where(Sx_norm_bucket == i)[0] for i in range(len(buckets) + 1)]
    sample_id_buckets = [sample_ids[i] for i in bucket_indices]
    x_buckets = [x[i] for i in bucket_indices]
    y_buckets = [y[i] for i in bucket_indices]
    return (sample_id_buckets, x_buckets, y_buckets)


class ExampleWriter:
    """Writes examples to files."""

    def __init__(self):
        self.file_format = tfds.core.file_adapters.FileFormat.TFRECORD

    def write(self, path, examples):
        """Write examples from iterator."""
        adapter = tfds.core.file_adapters.ADAPTER_FOR_FORMAT[self.file_format]
        return adapter.write_examples(path, examples)


_INDEX_PATH_SUFFIX = "_index.json"


@dataclasses.dataclass(frozen=True)
class _ShardSpec:
    """Spec to write a final records shard.

    Attributes:
      shard_index: Index of the shard.
      path: The path where to write the shard.
      index_path: The path where to write index of the records in the
        corresponding shard. NOTE: Value for this attribute is always set, but
        usage depends on whether `write_examples` returned a list of record
        positions for each example.
      examples_number: Number of examples in shard.
      file_instructions: Reading instructions.
    """

    shard_index: int
    path: str
    index_path: str
    examples_number: int
    file_instructions: Sequence[shard_utils.FileInstruction]


def read_example(serialized_example):

    feature_description = {
        "sample_id": tf.io.FixedLenFeature([], tf.string),
        "inputs": tf.io.FixedLenFeature([556], tf.float32),
        "targets": tf.io.FixedLenFeature([368], tf.float32),
    }
    example = tf.io.parse_single_example(
        serialized_example,
        feature_description,
    )
    return example


def _raise_error_for_duplicated_keys(example1, example2):
    """Log information about the examples and raise an AssertionError."""
    msg = "Two examples share the same hashed key!"
    logging.error(msg)
    try:
        ex1 = read_example(example1)
        ex2 = read_example(example2)
        logging.error("1st example: %s", ex1)
        logging.error("2nd example: %s", ex2)
    except ValueError:
        logging.error(
            "Failed to parse examples! Cannot log them to see the examples behind"
            " the duplicated keys. Raw example 1: %s, raw example 2: %s",
            example1,
            example2,
        )
    raise AssertionError(msg + " See logs above to view the examples.")


def _get_index_path(path: str) -> epath.PathLike:
    """Returns path to the index file of the records stored at the given path.

    E.g: Say the path to a shard of records (of a particular split) are stored at
    `your/path/to/records/foo.riegeli-00001-of-00005`, it is transformed into
    `your/path/to/records/foo.riegeli-00001-of-00005_index.json`.

    Args:
      path: Path to the record file.

    Returns:
      Path of the index file of a shard of records stored at given path.
    """
    return path + _INDEX_PATH_SUFFIX


def _get_shard_specs(
    num_examples: int,
    total_size: int,
    bucket_lengths: Sequence[int],
    filename_template: naming.ShardedFileTemplate,
    shard_config: shard_utils.ShardConfig,
) -> Sequence[_ShardSpec]:
    """Returns list of _ShardSpec instances, corresponding to shards to write.

    Args:
      num_examples: int, number of examples in split.
      total_size: int (bytes), sum of example sizes.
      bucket_lengths: list of ints, number of examples in each bucket.
      filename_template: template to format sharded filenames.
      shard_config: the configuration for creating shards.
    """
    num_shards = shard_config.get_number_shards(total_size, num_examples)
    shard_boundaries = shard_utils.get_shard_boundaries(num_examples, num_shards)
    shard_specs = []
    bucket_indexes = [str(i) for i in range(len(bucket_lengths))]
    from_ = 0
    for shard_index, to in enumerate(shard_boundaries):
        # Read the bucket indexes
        file_instructions = shard_utils.get_file_instructions(
            from_, to, bucket_indexes, bucket_lengths
        )
        shard_path = filename_template.sharded_filepath(
            shard_index=shard_index, num_shards=num_shards
        )
        index_path = _get_index_path(os.fspath(shard_path))
        shard_specs.append(
            _ShardSpec(
                shard_index=shard_index,
                path=os.fspath(shard_path),
                index_path=index_path,
                examples_number=to - from_,
                file_instructions=file_instructions,
            )
        )
        from_ = to
    return shard_specs


def _write_index_file(sharded_index_path: epath.PathLike, record_keys: Sequence[Any]):
    """Writes index file for records of a shard at given `sharded_index_path`.

    NOTE: Each record position (i.e shard_key in shard_keys) is stored as a
    string for better readability of the index files. The reader should parse the
    position from string using `RecordPosition.from_str()` method.

    Args:
      sharded_index_path: Path to the sharded index path.
      record_keys: Sequence of keys/indices of the records in the shard.
    """
    # Store string representation of each record position.
    #
    # NOTE: This makes the index file more readable. Although the reader should
    # parse the record position from a string.
    index_info = {"index": [str(record_key) for record_key in record_keys]}
    epath.Path(sharded_index_path).write_text(json.dumps(index_info))
    logging.info("Wrote index file to %s", os.fspath(sharded_index_path))


class Writer:
    """Shuffles and writes Examples to sharded files.

    The number of shards is computed automatically.
    """

    def __init__(
        self,
        serializer,
        filename_template: naming.ShardedFileTemplate,
        hash_salt,
        disable_shuffling: bool,
        example_writer: ExampleWriter,
        shard_config: shard_utils.ShardConfig | None = None,
        ignore_duplicates: bool = False,
    ):
        """Initializes Writer.

        Args:
          serializer: class that can serialize examples.
          filename_template: template to format sharded filenames.
          hash_salt (str or bytes): salt to hash keys.
          disable_shuffling (bool): Specifies whether to shuffle the records.
          example_writer: class that writes examples to disk or elsewhere.
          shard_config: the configuration for creating shards.
          ignore_duplicates: whether to ignore duplicated examples with the same
            key. If False, a `DuplicatedKeysError` will be raised on duplicates.
        """
        self._serializer = serializer
        self._shuffler = shuffle.Shuffler(
            dirpath=filename_template.data_dir,
            hash_salt=hash_salt,
            disable_shuffling=disable_shuffling,
            ignore_duplicates=ignore_duplicates,
        )
        self._filename_template = filename_template
        self._shard_config = shard_config or shard_utils.ShardConfig()
        self._example_writer = example_writer

    def write(self, key: int | bytes, serialized_example):
        """Writes given example.

        The given example is not directly written to the shard, but to a
        temporary file (or memory). The finalize() method writes all the shards.

        Args:
          key (int|bytes): the key associated with the example. Used for shuffling.
          example: the Example to write to the shard.
        """
        # serialized_example = self._serializer.serialize_example(example)
        self._shuffler.add(key, serialized_example)

    def finalize(self) -> tuple[list[int], int]:
        """Effectively writes examples to the shards."""
        if self._shuffler.num_examples == 0:
            raise AssertionError("No examples were yielded.")

        shard_specs = _get_shard_specs(
            num_examples=self._shuffler.num_examples,
            total_size=self._shuffler.size,
            bucket_lengths=self._shuffler.bucket_lengths,
            filename_template=self._filename_template,
            shard_config=self._shard_config,
        )
        filename = self._filename_template.sharded_filepaths_pattern()
        # Here we just loop over the examples, and don't use the instructions, just
        # the final number of examples in every shard. Instructions could be used to
        # parallelize, but one would need to be careful not to sort buckets twice.
        examples_generator = iter(
            utils.tqdm(
                self._shuffler,
                desc=f"Shuffling {filename}...",
                total=self._shuffler.num_examples,
                unit=" examples",
                leave=False,
                mininterval=1.0,
            )
        )
        try:
            for shard_spec in shard_specs:
                iterator = itertools.islice(
                    examples_generator, 0, shard_spec.examples_number
                )
                record_keys = self._example_writer.write(shard_spec.path, iterator)

                # No shard keys returned (e.g: TFRecord format), index cannot be
                # created.
                if not record_keys:
                    continue

                # Number of `shard_keys` received should match the number of examples
                # written in this shard.
                if len(record_keys) != int(shard_spec.examples_number):
                    raise RuntimeError(
                        f"Length of example `keys` ({len(record_keys)}) does not match "
                        f"`shard_spec.examples_number: (`{shard_spec.examples_number})"
                    )
                _write_index_file(shard_spec.index_path, record_keys)

        except shuffle.DuplicatedKeysError as err:
            _raise_error_for_duplicated_keys(err.item1, err.item2)

        # Finalize the iterator to clear-up TQDM
        try:
            val = next(examples_generator)
        except StopIteration:
            pass
        else:
            raise ValueError(
                f"Shuffling more elements than expected. Additional element: {val}"
            )

        shard_lengths = [int(spec.examples_number) for spec in shard_specs]

        logging.info(
            "Done writing %s. Number of examples: %s (shards: %s)",
            filename,
            sum(shard_lengths),
            shard_lengths,
        )
        return shard_lengths, self._shuffler.size


FEATURES = tfds.features.FeaturesDict(
    {
        "sample_id": tfds.features.Text(),
        "inputs": tfds.features.Tensor(shape=(556,), dtype=np.float32),
        "targets": tfds.features.Tensor(shape=(368,), dtype=np.float32),
    },
)


def write_tfrecords(local_idx=0):

    tfrecords_dir = f"{TFRECORDS_DIR_BASE}/LOCAL_{local_idx}/{VERSION}/"
    create_folder(
        bucket_name="us2-climsim",
        folder_path=f"climsim_high_res_train/LOCAL_{local_idx}/{VERSION}",
    )

    bucket_size = 0.05
    buckets = np.arange(-1.25, 4 + bucket_size, bucket_size)

    DATASET = "climsim_high_res_train"
    ALL_BUCKET_SPLIT_NAMES = [f"bucket_{i:03d}" for i in range(len(buckets) + 1)]
    FILEFORMAT = "tfrecord"
    LOCAL_SAMPLES_IN_BUCKET = {bucket_name: 0 for bucket_name in ALL_BUCKET_SPLIT_NAMES}
    N_SPLITS = 512  # Divides the data into 512 splits
    # Each checkpoint writes 8 splits, each checkpoint can be run in parallel
    N_CHECKPOINTS = 64
    num_splits_per_version = N_SPLITS // N_CHECKPOINTS
    start_idx = local_idx * num_splits_per_version
    end_idx = start_idx + num_splits_per_version

    print(f"Generating examples for split {start_idx} to {end_idx} of {N_SPLITS}")
    splits = get_splits(N_SPLITS)[start_idx:end_idx]

    total_train_files = sum([len(s) for s in splits])
    print(f"Total train files: {total_train_files}")

    assumed_total_count = total_train_files * 21600
    print(f"Assumed total count: {assumed_total_count}")

    example_writer = ExampleWriter()
    SPLIT_WRITERS = {}

    for i, SPLIT_NAME in enumerate(ALL_BUCKET_SPLIT_NAMES):

        filename_template = tfds.core.ShardedFileTemplate(
            data_dir=tfrecords_dir,
            dataset_name=DATASET,
            split=SPLIT_NAME,
            filetype_suffix=FILEFORMAT,
        )

        SPLIT_WRITERS[SPLIT_NAME] = Writer(
            serializer=FEATURES,
            filename_template=filename_template,
            hash_salt=str(local_idx) + SPLIT_NAME,
            disable_shuffling=False,
            example_writer=example_writer,
        )

    for train_files in splits:

        train_files.extend([f.replace("mli", "mlo") for f in train_files])
        train_files.append("ClimSim_high-res_grid-info.nc")
        file_paths, data_path = get_dataset_from_file_names(train_files)

        sample_ids, X, Y = extract_data_from_file_paths(file_paths, data_path)

        print(f"Extracted data from {len(train_files)} files")
        print(X.shape, Y.shape)

        sample_id_buckets, x_buckets, y_buckets = assign_buckets(
            X, Y, sample_ids, buckets
        )

        for i, SPLIT_NAME in enumerate(ALL_BUCKET_SPLIT_NAMES):
            split_sid = sample_id_buckets[i]
            split_x = x_buckets[i].astype(np.float32)
            split_y = y_buckets[i].astype(np.float32)
            LOCAL_SAMPLES_IN_BUCKET[SPLIT_NAME] = len(split_sid)

            split_writer = SPLIT_WRITERS[SPLIT_NAME]

            def write_split_data():
                for sid, eg_x, eg_y in zip(split_sid, split_x, split_y):
                    data = {
                        "sample_id": sid,
                        "inputs": eg_x,
                        "targets": eg_y,
                    }
                    data_bytes = FEATURES.serialize_example(data)
                    split_writer.write(sid, data_bytes)

            write_split_data()

    for split_writer in SPLIT_WRITERS.values():
        split_writer.finalize()

    split_infos = tfds.folder_dataset.compute_split_info(
        filename_template=filename_template
    )

    tfds.folder_dataset.write_metadata(
        data_dir=tfrecords_dir,
        features=FEATURES,
        split_infos=split_infos,
        homepage="https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/data/",
        supervised_keys=("inputs", "targets"),
    )


def _extract_split_files(
    filename_template: naming.ShardedFileTemplate,
):
    """Extract the files."""
    files = sorted(filename_template.data_dir.iterdir())
    file_infos = [filename_template.parse_filename_info(f.name) for f in files]
    file_infos = [f for f in file_infos if f is not None]
    if not file_infos:
        raise ValueError(
            f"No example files detected in {filename_template.data_dir}. "
            f"Make sure to follow the pattern: {filename_template.template}"
        )
    files_without_splits = []
    split_files = collections.defaultdict(list)
    for file_info in file_infos:
        if file_info.split is not None:
            split_files[file_info.split].append(file_info)
        else:
            files_without_splits.append(file_info)
    if files_without_splits:
        raise ValueError(
            f"Some matched files did not specify the split: {files_without_splits}"
        )
    return split_files


def get_completed_splits(filename_template):
    split_files = _extract_split_files(filename_template)
    print("Auto-detected splits:")
    for split_name, file_infos in split_files.items():
        print(f" * {split_name}: {file_infos[0].num_shards} shards")
    return list(split_files.keys())


def shuffle_written_tfrecords(
    split_name,
    split_size="[:100%]",
    new_split_name=None,
):

    DATASET = "climsim_high_res_train"
    FILEFORMAT = "tfrecord"

    tfrecords_dir = f"{TFRECORDS_DIR_BASE}/{VERSION}/"
    create_folder(
        bucket_name="us2-climsim",
        folder_path=f"{DATASET}/{VERSION}",
    )

    all_tfrecords_dir = [
        f"{TFRECORDS_DIR_BASE}/LOCAL_{idx}/{VERSION}/" for idx in range(64)
    ]
    builder = tfds.builder_from_directories(builder_dirs=all_tfrecords_dir)
    splits = builder.info.splits

    num_examples = splits[split_name + split_size].num_examples

    if num_examples > 15_000_000:
        # Workaround to handle large splits
        # TFDS writes to /tmp by default, which runs OOM for large splits
        print(
            f"Split {split_name} has {num_examples} examples, splitting into two smaller splits"
        )
        shuffle_written_tfrecords(
            split_name,
            split_size="[:80%]",
            new_split_name=f"{split_name}_0",
        )
        shuffle_written_tfrecords(
            split_name,
            split_size="[80%:]",
            new_split_name=f"{split_name}_1",
        )
        return

    split_ds = builder.as_dataset(
        shuffle_files=True,
        split=split_name + split_size,
    )
    split_ds = split_ds.shuffle(buffer_size=num_examples, seed=42)

    example_writer = ExampleWriter()

    filename_template = tfds.core.ShardedFileTemplate(
        data_dir=tfrecords_dir,
        dataset_name=DATASET,
        split=new_split_name or split_name,
        filetype_suffix=FILEFORMAT,
    )

    split_writer = Writer(
        serializer=FEATURES,
        filename_template=filename_template,
        hash_salt=split_name + split_size,
        disable_shuffling=False,
        example_writer=example_writer,
    )

    for i, example in enumerate(tqdm(split_ds, total=num_examples)):

        data = {
            "sample_id": example["sample_id"].numpy().decode("utf-8"),
            "inputs": example["inputs"].numpy(),
            "targets": example["targets"].numpy(),
        }
        data_bytes = FEATURES.serialize_example(data)
        split_writer.write(str(i), data_bytes)

    split_writer.finalize()


def shuffle_runner_local(local_idx):

    all_tfrecords_dir = [
        f"{TFRECORDS_DIR_BASE}/LOCAL_{idx}/{VERSION}/" for idx in range(64)
    ]
    builder = tfds.builder_from_directories(builder_dirs=all_tfrecords_dir)
    splits = builder.info.splits
    split_keys = sorted(splits.keys())

    if local_idx == -1:
        local_split_keys = split_keys
    else:
        # interleave the splits with stride 8
        all_local_split_keys = [split_keys[i::8] for i in range(8)]
        local_split_keys = all_local_split_keys[local_idx]
        print(
            f"Running shuffle for local index {local_idx} with splits {local_split_keys}"
        )

    DATASET = "climsim_high_res_train"
    FILEFORMAT = "tfrecord"

    tfrecords_dir = f"{TFRECORDS_DIR_BASE}/{VERSION}/"

    filename_template = tfds.core.ShardedFileTemplate(
        data_dir=tfrecords_dir,
        dataset_name=DATASET,
        filetype_suffix=FILEFORMAT,
    )

    completed_splits = get_completed_splits(filename_template)

    for split_key in local_split_keys:
        if local_idx != -1:
            completed_splits = get_completed_splits(filename_template)

        if split_key in completed_splits or (
            f"{split_key}_0" in completed_splits
            and f"{split_key}_1" in completed_splits
        ):
            print(f"Skipping {split_key}")
            continue

        print("-" * 100)
        print(f"Shuffling {split_key}")
        print("\n" * 2)
        shuffle_written_tfrecords(split_key)


def finalize_tfrecords_split(split_name):

    DATASET = "climsim_high_res_train"
    FILEFORMAT = "tfrecord"

    tfrecords_dir = f"{TFRECORDS_DIR_BASE}/{split_name}/{VERSION}/"

    filename_template = tfds.core.ShardedFileTemplate(
        data_dir=tfrecords_dir,
        dataset_name=DATASET,
        filetype_suffix=FILEFORMAT,
    )

    split_infos = tfds.folder_dataset.compute_split_info(
        filename_template=filename_template
    )

    tfds.folder_dataset.write_metadata(
        data_dir=tfrecords_dir,
        features=FEATURES,
        split_infos=split_infos,
        homepage="https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/data/",
        supervised_keys=("inputs", "targets"),
    )


def test_finalized_split(split_name):
    tfrecords_dir = f"{TFRECORDS_DIR_BASE}/{split_name}/{VERSION}/"
    builder = tfds.builder_from_directory(tfrecords_dir)
    splits = builder.info.splits

    num_examples_per_bucket = [s.num_examples for s in splits.values()]
    print(num_examples_per_bucket)

    total_examples = sum(num_examples_per_bucket)
    print(total_examples)

    sample_weights_per_bucket = 1 / np.array(num_examples_per_bucket)
    sampling_probs_per_bucket = (
        sample_weights_per_bucket / (sample_weights_per_bucket).sum()
    )

    all_ds = builder.as_dataset()

    train_ds = tf.data.Dataset.sample_from_datasets(
        datasets=all_ds.values(),
        weights=sampling_probs_per_bucket,
        seed=42,
        rerandomize_each_iteration=True,
    )

    for data in train_ds.take(5):
        sample_id_str_decoded = tf.strings.as_string(data["sample_id"])
        sample_id = sample_id_str_decoded.numpy().decode("utf-8")
        print(f"Sample ID: {sample_id}")


def delete_blob(blob_name, bucket_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if blob.exists():
        print("Deleting blob -> ", blob_name)
        blob.delete()
    else:
        print("blob does not exist -> ", blob_name)
    return


def delete_tmp_files_from_bucket():
    files = list_gcs_files("us2-climsim", "climsim_high_res_train/1.0.0")
    tmp_files = [f for f in files if ".tmp" in f]
    delete_fn = lambda x: delete_blob(x, "us2-climsim")
    etree.parallel_map(delete_fn, tmp_files)


def delete_particular_split_files_from_bucket(split_name):
    files = list_gcs_files("us2-climsim", "climsim_high_res_train/1.0.0")
    split_files = [f for f in files if split_name in f]
    delete_fn = lambda x: delete_blob(x, "us2-climsim")
    etree.parallel_map(delete_fn, split_files)


def rename_blob(bucket_name, old_blob_name, new_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(old_blob_name)
    if blob.exists():
        print(f"{old_blob_name} -> {new_blob_name}")
        new_blob = bucket.rename_blob(blob, new_blob_name)
    else:
        print("blob does not exist -> ", old_blob_name)


def make_split_folders(split_name, merge=False):
    # Make switch from split to config type dataset
    # Each config contains 1 split
    bucket_name = "us2-climsim"
    prefix = f"climsim_high_res_train/1.0.0/"
    if merge:
        split_files_0 = list_gcs_files(
            bucket_name,
            prefix + f"climsim_high_res_train-{split_name}_0",
        )
        split_files_1 = list_gcs_files(
            bucket_name,
            prefix + f"climsim_high_res_train-{split_name}_1",
        )
        split_files = split_files_0 + split_files_1

    else:
        split_files = list_gcs_files(
            bucket_name,
            prefix + f"climsim_high_res_train-{split_name}",
        )

    create_folder("us2-climsim", f"climsim_high_res_train/{split_name}/1.0.0")

    def get_new_name(split_name, old_name):
        old_name_split = old_name.split("/")
        prefix, suffix = old_name_split[0], "/".join(old_name_split[1:])
        new_name = f"{prefix}/{split_name}/{suffix}"
        return new_name

    rename_fn = lambda old_name: rename_blob(
        bucket_name,
        old_name,
        get_new_name(split_name, old_name),
    )

    etree.parallel_map(rename_fn, split_files)

    # switch_names = dict(
    #     zip(split_files, [get_new_name(split_name, f) for f in split_files])
    # )
    # rich.print(switch_names)


def restructure_tfrecords():

    all_tfrecords_dir = [
        f"{TFRECORDS_DIR_BASE}/LOCAL_{idx}/{VERSION}/" for idx in range(64)
    ]
    builder = tfds.builder_from_directories(builder_dirs=all_tfrecords_dir)
    splits = builder.info.splits
    split_keys = sorted(splits.keys())

    DATASET = "climsim_high_res_train"
    FILEFORMAT = "tfrecord"

    tfrecords_dir = f"{TFRECORDS_DIR_BASE}/{VERSION}/"

    filename_template = tfds.core.ShardedFileTemplate(
        data_dir=tfrecords_dir,
        dataset_name=DATASET,
        filetype_suffix=FILEFORMAT,
    )

    completed_splits = get_completed_splits(filename_template)

    for split_name in split_keys:

        if split_name in completed_splits:
            make_split_folders(split_name)
        elif (
            f"{split_name}_0" in completed_splits
            and f"{split_name}_1" in completed_splits
        ):
            make_split_folders(split_name, merge=True)
        else:
            print(f"Skipping {split_name}")


def finalize_tfrecords(local_idx):
    all_tfrecords_dir = [
        f"{TFRECORDS_DIR_BASE}/LOCAL_{idx}/{VERSION}/" for idx in range(64)
    ]
    builder = tfds.builder_from_directories(builder_dirs=all_tfrecords_dir)
    splits = builder.info.splits
    split_keys = sorted(splits.keys())

    all_local_split_keys = [split_keys[i::8] for i in range(8)]
    local_split_keys = all_local_split_keys[local_idx]
    print(f"Running shuffle for local index {local_idx} with splits {local_split_keys}")

    for split_name in local_split_keys:
        
        print(f"Finalizing {split_name}")
        finalize_tfrecords_split(split_name)
        test_finalized_split(split_name)


def main(args: Args):

    if args.write_initial_tfrecords:
        write_tfrecords(local_idx=args.local_idx)

        tfrecords_dir = f"{TFRECORDS_DIR_BASE}/LOCAL_{args.local_idx}/{VERSION}/"
        builder = tfds.builder_from_directory(tfrecords_dir)
        splits = builder.info.splits

        num_examples_per_bucket = [s.num_examples for s in splits.values()]
        print(num_examples_per_bucket)

        total_examples = sum(num_examples_per_bucket)
        print(total_examples)

        sample_weights_per_bucket = 1 / np.array(num_examples_per_bucket)
        sampling_probs_per_bucket = (
            sample_weights_per_bucket / (sample_weights_per_bucket).sum()
        )

        all_ds = builder.as_dataset()

        train_ds = tf.data.Dataset.sample_from_datasets(
            datasets=all_ds.values(),
            weights=sampling_probs_per_bucket,
            seed=42,
            rerandomize_each_iteration=True,
        )

        for data in train_ds.take(5):
            sample_id_str_decoded = tf.strings.as_string(data["sample_id"])
            sample_id = sample_id_str_decoded.numpy().decode("utf-8")
            print(f"Sample ID: {sample_id}")

    if args.shuffle_tfrecords:
        shuffle_runner_local(local_idx=args.local_idx)

    if args.finalize_shuffled_tfrecords:
        finalize_tfrecords(args.local_idx)

    if args.delete_tmp_files:
        delete_tmp_files_from_bucket()

    if args.delete_split_files:
        delete_particular_split_files_from_bucket(args.delete_split_files)

    if args.shuffle_split_key:
        shuffle_written_tfrecords(args.shuffle_split_key)

    if args.rename_split_to_config_structure:
        make_split_folders(args.rename_split_to_config_structure)

    if args.restructure_tfrecords:
        restructure_tfrecords()


if __name__ == "__main__":
    app.run(main, flags_parser=eapp.make_flags_parser(Args))
