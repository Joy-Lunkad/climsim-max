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

"""Input pipeline for a LM1B dataset."""

import functools
import os
import random
import sys
from typing import Optional

import ml_collections
import numpy as np
import pandas as pd
import rich
import tensorflow as tf
import tensorflow_datasets as tfds
import jax

import multihost_dataloading
import tokenizer
import sequence_packing

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, parent_dir)
print(sys.path)
from etils import etree
from google.cloud import storage

from climsim import climsim_high_res_train
import max_utils
from scipy.stats import norm
import rich


AUTOTUNE = tf.data.experimental.AUTOTUNE


# Right-shifting token inputs for teacher-forced training.
# -----------------------------------------------------------------------------


def shift_right_tf(x, axis=1):
    """Shift the input to the right by padding and slicing on axis."""
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[axis] = (1, 0)
    slices = [
        slice(None),
    ] * len(x.shape)
    slices[axis] = slice(0, -1)
    padded = tf.pad(
        x,
        tf.constant(pad_widths),
        mode="constant",
        constant_values=tf.constant(0, x.dtype),
    )
    return padded[tuple(slices)]


def shift_inputs_tf(x, segment_ids=None, axis=1):
    """Shift inputs and replace EOS by 0 for packed inputs."""
    shifted = shift_right_tf(x, axis=axis)
    # For packed targets, the first shifted token of a new sequence is made
    # 0, rather than being the EOS token for the last sequence.
    if segment_ids is not None:
        shifted *= tf.cast(
            segment_ids == shift_right_tf(segment_ids, axis=axis), x.dtype
        )
    return shifted


def shift_data(x, axis=0, segmented=True):
    segment_ids = x["inputs_segmentation"] if segmented else None
    x["inputs"] = shift_inputs_tf(x["inputs"], segment_ids=segment_ids, axis=axis)
    return x


def shift_data_by_truncation(x):
    x["inputs"] = x["inputs"][:-1]
    x["targets"] = x["targets"][1:]
    return x


def normalize_features(ds):
    """Normalize text feature keys."""

    def _normalize_features(features):
        # features["inputs"] = features.pop("text")
        features["inputs"] = features.pop("inputs")
        features["targets"] = features["targets"]
        # features["targets"] = features["inputs"]
        return features

    return ds.map(_normalize_features, num_parallel_calls=AUTOTUNE)


def length_trim(ds, max_len):
    """ "Trim to Max length"""

    def _trim_fn(features):
        if tf.shape(features["inputs"])[0] > max_len:
            features["inputs"] = features["inputs"][:max_len]
        if tf.shape(features["targets"])[0] > max_len:
            features["targets"] = features["targets"][:max_len]
        return features

    return ds.map(_trim_fn, num_parallel_calls=AUTOTUNE)


# -----------------------------------------------------------------------------
# Main dataset preparation.
# -----------------------------------------------------------------------------

sample_weights = tf.convert_to_tensor(np.load("sample_weights_s_scores.npy"))
sid_to_sw = pd.read_csv("sample_id_to_sample_weights.csv", index_col=0)
sid_to_sw = sid_to_sw.to_dict()["sample_weight"]
MIN_SW = 0.024394170768953

sample_weights_lookup_table = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(list(sid_to_sw.keys()), dtype=tf.string),
        values=tf.constant(list(sid_to_sw.values()), dtype=tf.float32),
    ),
    default_value=tf.constant(-1.0, dtype=tf.float32),
)

min_skew = 0.05
max_skew = 0.25
num_samples = 10_000_000

skew_value_eq = (max_skew / min_skew) ** (1 / (num_samples - 1))


def get_skew_values(x):
    return min_skew * (skew_value_eq**x)


def tf_get_skew_values(x):
    return tf.multiply(min_skew, tf.pow(skew_value_eq, tf.cast(x, "float32")))


def preprocessing_pipeline(
    dataset,
    batch_size: int,
    global_mesh,
    shuffle: bool,
    num_epochs: Optional[int] = 1,
    pack_examples: bool = True,
    shuffle_buffer_size: int = 1024,
    max_input_length: int = 512,
    max_target_length: int = 512,
    shift: bool = True,
    drop_remainder: bool = True,
    prefetch_size=tf.data.experimental.AUTOTUNE,
    data_shuffle_seed=0,
    has_targets=True,
    normalize=True,
):
    """Shuffle and batch/pack the given dataset."""

    def norm_fn(example):
        example["inputs"] = max_utils.which_norm_input(example["inputs"])
        example["inputs"] = tf.cast(example["inputs"], tf.float32)
        if has_targets:
            example["targets"] = max_utils.which_norm_output(example["targets"])
            example["targets"] = tf.cast(example["targets"], tf.float32)
        return example

    if normalize:
        dataset = dataset.map(norm_fn)

    def truncate_to_max_allowable_length(x):
        x["inputs"] = x["inputs"][:max_input_length]
        sample_id_str_decoded = tf.strings.as_string(
            x["sample_id"]
        )  # eg: test_{f_name}_553446
        sample_id = tf.strings.split(sample_id_str_decoded, "_")[-1]  # eg: 553446
        sample_id = tf.strings.to_number(sample_id, out_type=tf.int32)
        x["sample_id"] = sample_id

        if has_targets:
            x["targets"] = x["targets"][:max_target_length]
        return x

    dataset = dataset.map(lambda x: truncate_to_max_allowable_length(x))

    # Shuffle and repeat.
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size, seed=data_shuffle_seed)

    dataset = dataset.repeat(num_epochs)

    # Shift inputs for teacher-forced training
    if shift:
        dataset = dataset.map(
            shift_data_by_truncation,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )

    # Perform greedy sequence packing
    if pack_examples:
        dataset = sequence_packing.pack_dataset(dataset, max_input_length)
    assert (
        batch_size % global_mesh.size == 0
    ), "Batch size should be divisible number of global devices."

    for example in dataset.take(1):
        assign_sample_weights = "sample_weight" in example.keys()

    # Batch examples.
    if pack_examples:
        dataset = dataset.batch(
            batch_size // jax.process_count(), drop_remainder=drop_remainder
        )
    else:
        # simple (static-shape) padded batching
        pad_shapes = {"inputs": max_input_length}
        pad_values = {"inputs": 0.0}
        pad_shapes["sample_id"] = []
        pad_values["sample_id"] = -1

        if has_targets:
            pad_shapes["targets"] = max_target_length
            pad_values["targets"] = 0.0

        if assign_sample_weights:
            pad_shapes["sample_weight"] = []
            pad_values["sample_weight"] = -1.0

        dataset = dataset.padded_batch(
            batch_size // jax.process_count(),
            padded_shapes=pad_shapes,
            padding_values=pad_values,
            drop_remainder=drop_remainder,
        )

    if prefetch_size:
        dataset = dataset.prefetch(prefetch_size)

    multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(
        dataset, global_mesh
    )

    # Return multi-host jax.Array prep iterator
    return multihost_gen


def get_datasets(
    config: ml_collections.ConfigDict,
    dataloading_host_index,
    dataloading_host_count,
    read_config=None,
):
    """Load and return dataset of batched examples for use during training."""
    # Training dataset.
    # train_ds_builder = tfds.builder(config.dataset_name)
    version = "1.0.0"
    test_version = "1.0.0"
    train_split = "train[:95%]"
    eval_split = "train[95%:]"
    extra_eval = False
    if config.use_full_low_res_data:
        version = config.full_low_res_version
        if version == "2.0.0":
            test_version = "4.0.0"
        elif version == "3.0.0":
            test_version = "5.0.0"
        elif version == "6.0.0":
            test_version = "6.0.0"
            extra_eval = False
        elif version == "7.0.0":
            test_version = "6.0.0"
            train_split = "train"
            eval_split = "val"

    train_ds_builder = tfds.builder(
        f"climsim_low_res_train:{version}", data_dir="gs://us2-climsim"
    )
    # train_data = get_raw_dataset(train_ds_builder, 'train')
    train_ds = train_ds_builder.as_dataset(
        split=train_split,
        read_config=read_config,
        shuffle_files=config.enable_data_shuffling,
    )
    # shard the dataset as soon as it is loaded
    train_ds = train_ds.shard(
        num_shards=dataloading_host_count, index=dataloading_host_index
    )

    if config.use_full_low_res_data:
        
        def assign_weight(example):
            sample_id_str_decoded = tf.strings.as_string(example["sample_id"])
            sample_weight = sample_weights_lookup_table.lookup(sample_id_str_decoded)

            # Skewing sample weights to benefit the commoner datapoints
            sample_weight += 0.05
            example["sample_weight"] = tf.cast(sample_weight, tf.float32)
            return example

        train_ds = train_ds.map(assign_weight)


    # Evaluation dataset.
    if config.eval_dataset_name:
        eval_ds_builder = tfds.builder(config.eval_dataset_name)
    else:
        eval_ds_builder = train_ds_builder
    # eval_data = get_raw_dataset(eval_ds_builder, config.eval_split)
    eval_ds = eval_ds_builder.as_dataset(
        split=eval_split, read_config=read_config, shuffle_files=False
    )

    if extra_eval:
        # THIS LEAKS SOME DATA FROM THE TRAINING SET INTO THE VAL SET
        extra_eval_ds = tfds.load(
            "climsim_low_res_train:7.0.0",
            data_dir="gs://us2-climsim",
            split="val",
            read_config=read_config,
            shuffle_files=False,
        )
        eval_ds = eval_ds.concatenate(extra_eval_ds)

    eval_ds = eval_ds.shard(num_shards=jax.process_count(), index=jax.process_index())

    test_ds_builder = tfds.builder(
        f"climsim_low_res_test:{test_version}", data_dir="gs://us2-climsim"
    )

    test_ds = test_ds_builder.as_dataset(
        split="test",
        read_config=read_config,
        shuffle_files=False,
    )

    test_ds = test_ds.shard(
        num_shards=dataloading_host_count, index=dataloading_host_index
    )

    return train_ds, eval_ds, test_ds


def get_high_res_datasets(
    config: ml_collections.ConfigDict,
    dataloading_host_index,
    dataloading_host_count,
    read_config=None,
):
    """Load and return dataset of batched examples for use during training."""

    TFRECORDS_DIR_BASE = "gs://us2-climsim/climsim_high_res_train/"
    VERSION = "1.0.0"
    ALL_BUCKET_SPLIT_NAMES = [f"bucket_{i:03d}" for i in range(107)]

    all_tfrecords_dir = [
        f"{TFRECORDS_DIR_BASE}/{split_name}/{VERSION}/"
        for split_name in ALL_BUCKET_SPLIT_NAMES
    ]

    builder = tfds.builder_from_directories(builder_dirs=all_tfrecords_dir)
    splits = builder.info.splits

    # Merge splits from the same bucket

    all_splits = []
    for split_name in ALL_BUCKET_SPLIT_NAMES:
        if split_name not in splits:
            all_splits.append(f"{split_name}_0 + {split_name}_1")
        else:
            all_splits.append(split_name)

    all_splits = sorted(all_splits)
    splits = {split_name: splits[split_name] for split_name in all_splits}

    num_examples_per_bucket = np.array([s.num_examples for s in splits.values()])
    # Skew the number of examples per bucket to benefit the commoner datapoints
    num_examples_per_bucket = np.clip(num_examples_per_bucket, 1, 5_000_000)
    
    total_examples = np.sum(num_examples_per_bucket)
    print(total_examples)

    sample_weights_per_bucket = 1 / num_examples_per_bucket
    sampling_probs_per_bucket = (
        sample_weights_per_bucket / (sample_weights_per_bucket).sum()
    )
    print(sampling_probs_per_bucket)

    all_ds = builder.as_dataset(
        split=all_splits,
        read_config=read_config,
        shuffle_files=True,
    )

    train_ds = tf.data.Dataset.sample_from_datasets(
        datasets=all_ds,
        weights=sampling_probs_per_bucket,
        seed=42,
        rerandomize_each_iteration=True,
        stop_on_empty_dataset=True,
    )

    # shard the dataset as soon as it is loaded
    train_ds = train_ds.shard(
        num_shards=dataloading_host_count, index=dataloading_host_index
    )

    eval_ds_builder = tfds.builder(
        f"climsim_low_res_train:3.0.0", data_dir="gs://us2-climsim"
    )
    eval_ds = eval_ds_builder.as_dataset(
        split="train[75%:]", read_config=read_config, shuffle_files=True
    )
    eval_ds = eval_ds.shard(num_shards=jax.process_count(), index=jax.process_index())

    test_ds_builder = tfds.builder(
        "climsim_low_res_test:5.0.0", data_dir="gs://us2-climsim"
    )

    test_ds = test_ds_builder.as_dataset(
        split="test",
        read_config=read_config,
        shuffle_files=False,
    )

    test_ds = test_ds.shard(
        num_shards=dataloading_host_count, index=dataloading_host_index
    )

    return train_ds, eval_ds, test_ds


def get_mixed_datasets(
    config: ml_collections.ConfigDict,
    dataloading_host_index,
    dataloading_host_count,
    read_config=None,
):

    lr_train_ds, lr_eval_ds, lr_test_ds = get_datasets(
        config, dataloading_host_index, dataloading_host_count, read_config
    )
    
    hr_train_ds, _, _ = get_high_res_datasets(
        config, dataloading_host_index, dataloading_host_count, read_config
    )
        
    def assign_min_sample_weight(example):
        example["sample_weight"] = tf.cast(MIN_SW, tf.float32)
        return example
    
    hr_train_ds = hr_train_ds.map(assign_min_sample_weight)

    hr_weight = config.mix_high_res_ratio
    train_ds = tf.data.Dataset.sample_from_datasets(
        datasets=[lr_train_ds, hr_train_ds],
        weights=[1-hr_weight, hr_weight],
        seed=42,
        rerandomize_each_iteration=True,
        stop_on_empty_dataset=True,
    )
    
    return train_ds, lr_eval_ds, lr_test_ds


def preprocess_dataset(
    config: ml_collections.ConfigDict,
    global_mesh,
    train_ds,
    eval_ds,
    test_ds,
    sp_tokenizer,
    data_shuffle_seed=0,
):
    """Pre-process the dataset and return iterators"""
    # Tokenize data.

    # TokenizeOp = tokenizer.TokenizeOp(sp_tokenizer)
    # TokenizeOp = functools.partial(pretend_tokenize_op, vocab_size=config.vocab_size)

    # Tokenize no op
    TokenizeOp = lambda x: x

    train_ds = train_ds.map(TokenizeOp, num_parallel_calls=AUTOTUNE)
    if eval_ds is not None:
        eval_ds = eval_ds.map(TokenizeOp, num_parallel_calls=AUTOTUNE)

    # Set global batch size.
    global_batch_size_to_load = config.global_batch_size_to_load

    if config.eval_per_device_batch_size > 0:
        eval_batch_size = config.eval_per_device_batch_size * global_mesh.size
    else:
        eval_batch_size = global_batch_size_to_load


    shuffle_buffer_size = 1024
    if config.use_high_res_data or config.mix_high_res_ratio > 0.0:
        shuffle_buffer_size = 8

    train_iter = preprocessing_pipeline(
        train_ds,
        global_batch_size_to_load,
        global_mesh,
        shuffle=config.enable_data_shuffling,
        shuffle_buffer_size=shuffle_buffer_size,
        num_epochs=None,
        pack_examples=False,
        max_input_length=config.N_IN,
        max_target_length=config.N_OUT,
        shift=False,
        data_shuffle_seed=data_shuffle_seed,
        normalize=config.normalize,
    )

    eval_iter = preprocessing_pipeline(
        eval_ds,
        eval_batch_size,
        global_mesh,
        shuffle=False,
        pack_examples=False,
        max_input_length=config.N_IN,
        max_target_length=config.N_OUT,
        shift=False,
        drop_remainder=True,
        data_shuffle_seed=data_shuffle_seed,
        normalize=config.normalize,
    )

    predict_iter = preprocessing_pipeline(
        test_ds,
        eval_batch_size,
        global_mesh,
        shuffle=False,
        num_epochs=None,
        pack_examples=False,
        max_input_length=config.N_IN,
        max_target_length=config.N_OUT,
        shift=False,
        drop_remainder=False,
        data_shuffle_seed=data_shuffle_seed,
        has_targets=False,
        normalize=config.normalize,
    )

    return train_iter, eval_iter, predict_iter
