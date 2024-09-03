import dataclasses
from gc import enable
import math
import os
from typing import Any, Union

from plum import Val
from sympy import comp
import app
from etils import eapp
from MaxText import max_utils, max_logging, accelerator_to_spec_map
import jax
import socket
import time
from jax.experimental.compilation_cache import compilation_cache


@dataclasses.dataclass
class Args:
    config_no: str = "0.0"


@dataclasses.dataclass
class Config:

    # If there is already a checkpoint under this run, that checkpoint will auto-resume.
    run_name: str = ""

    # override config settings to match a specific model.
    # other than the override, nothing should use this!
    model_name: str = "default"

    normalization_layer_epsilon: float = 1.0e-05

    ################################## CHECKPOINTING ##################################
    # Checkpointing makes the following choices in the following order, starting with (1):
    #   (1) If there is already a checkpoint for this run_name,
    #       we load the latest entire checkpoint.
    #     This ensures if we're resuming a run after preemption or hardware failure we lose minimum state.
    #   (2) Same priority and mutually exclusive -- you can't set both!
    #      * If load_parameters_path is set, we load a parameter only checkpoint from that path.
    #      * If load_full_state_path is set, we load a full state checkpoint from that path.
    #   (3) We don't load a checkpoint and initialize state instead!

    # Loads a just parameters from a specific directory
    # e.g. gs://my-base-output-directory/my-previous-run-name/checkpoints/items/NUMBER or NUMBER/items
    load_parameters_path: str = ""
    # Loads a full checkpoint including optimizer state and step count from a specific directory
    # e.g. gs://my-base-output-directory/my-previous-run-name/checkpoints/items/NUMBER or NUMBER/items
    load_full_state_path: str = ""

    # If enable_checkpointing is true, an asynchronous checkpointer will be used if
    # async_checkpointing is true, else a synchronous one is used. If you have
    # problems with the checkpointer we recommend trying the synchronous one.
    enable_checkpointing: bool = True
    async_checkpointing: bool = True
    n_checkpoints_every_epoch: int | None = 2  # If set, overrides checkpoint_interval
    checkpoint_period: int = 4000
    # enables one replica to read the ckpt then broadcast to the rest
    enable_single_replica_ckpt_restoring: bool = False

    # during generate_param_only_checkpoint should we unroll the loop?
    force_unroll: bool = False

    ############################### END CHECKPOINTING ##################################

    # for testing TPU performance, this options repeated uses the same batch.
    reuse_example_batch: int = 0

    metrics_file: str = (
        ""  # for testing, local file that stores scalar metrics. If empty, no metrics are written.
    )
    # If true save metrics such as loss and TFLOPS to GCS in {base_output_directory}/{run_name}/metrics/
    gcs_metrics: bool = False

    # If true save config to GCS in {base_output_directory}/{run_name}/
    save_config_to_gcs: bool = False

    # Activation dtypes.
    dtype: jax.typing.DTypeLike = "float32"
    # defaults to no quantization, i.e. bf16. possible alternative setting is 'int8'
    # or use fp8 to run with 8-bit floating-point GeMMs on NVIDIA GPUs.
    quantization: str = ""
    quantize_kvcache: bool = False

    # Shard the range finding operation for quantization. By default this is set to number of slices.
    quantization_local_shard_count: int = -1

    decoder_block: str = "llama2"  # which style of DecoderBlock to use.
    # Global parameter scale needs to be a power of 2.
    # If you want finer grained control of the model sizes then you should
    # explicitly set base_embed_dim, base_num_query_heads, base_num_kv_heads,
    # base_mlp_dim, base_num_decoder_layers and/or head_dim.
    weight_dtype: str = "float32"
    global_parameter_scale: int = 1
    base_emb_dim: int = 128
    base_num_query_heads: int = 1
    base_num_kv_heads: int = 1
    base_mlp_dim: int = 512
    base_num_decoder_layers: int = 8
    head_dim: int = 128
    use_value_masks: bool = True
    num_experts: int = 1
    num_experts_per_tok: int = 1
    mlp_activations: list[str] = ["silu", "linear"]
    dropout_rate: float = 0.0
    logits_via_embedding: bool = False
    # whether to normlize pre-softmax logits if logits_via_embedding is true
    normalize_embedding_logits: bool = True
    # whether to use fp32 in logits_dense or shared_embedding dot product for stability
    logits_dot_in_fp32: bool = True

    only_mlp: bool = False
    stack_inputs: bool = False
    use_feature_specific_mlp: bool = False
    stack_input_seq_length: int = 1

    # Choose 'remat_policy' between 'minimal', 'save_dot_except_mlpwi', 'save_dot_except_mlp', 'save_qkv_proj', 'qkv_proj_offloaded', 'minimal_offloaded' and 'full'.
    # These options offer a trade-off between speed (fastest to slowest) and HBM usage (highest to lowest)
    remat_policy: str = "minimal"
    scan_layers: bool = True
    param_scan_axis: int = 1
    # Supported attention: autoselected, dot_product, flash, cudnn_flash_te
    attention: str = "autoselected"
    # Combine matmuls for QKV and MLP
    fused_qkv: bool = False
    fused_mlp: bool = False

    record_internal_nn_metrics: int = 0

    # Output directory
    # Create a GCS bucket, e.g. my-maxtext-outputs and set this to "gs://my-maxtext-outputs/"
    base_output_directory: str = "gs://us2-climsim/maxtext"

    # Jax cache directory
    jax_cache_dir: str = "~/jax_cache"

    # Supported hardware types are 'tpu', 'gpu', 'gpu_multiprocess' and 'cpu'
    hardware: str = "tpu"

    # One axis for each parallelism type may hold a placeholder (-1)
    # value to auto-shard based on available slices and devices.
    # By default, product of the DCN axes should equal number of slices
    # and product of the ICI axes should equal number of devices per slice.
    dcn_data_parallelism: int = -1  # recommended DCN axis to be auto-sharded
    dcn_fsdp_parallelism: int = 1
    dcn_fsdp_transpose_parallelism: int = 1
    dcn_sequence_parallelism: int = 1  # never recommended
    dcn_tensor_parallelism: int = 1  # never recommended
    dcn_autoregressive_parallelism: int = 1  # never recommended
    ici_data_parallelism: int = -1
    ici_fsdp_parallelism: int = 1  # recommended ICI axis to be auto-sharded
    ici_fsdp_transpose_parallelism: int = 1
    ici_sequence_parallelism: int = 1
    ici_tensor_parallelism: int = 1
    ici_autoregressive_parallelism: int = 1

    # The number of TPU slices is automatically determined, you should not set
    # this explicitly. For ahead of time compilation, you should set
    # compile_toplogy_num_slices, which will in turn set this value.
    # For non-TPU environments this is set to 1.
    num_slices: int = -1

    # Dataset
    # Replace with your path given as argument in download_dataset.sh, e.g. "gs://my-maxtext-dataset/"
    dataset_path: str = "gs://us2-climsim/"
    vocab_size: int = 1024  # powers of 2 for sharding
    tokenizer_path: str = "assets/tokenizer.llama2"
    dataset_name: str = "c4/en:3.0.1"
    eval_dataset_name: str = ""
    eval_split: str = ""
    per_device_batch_size: float = 512.0
    # if -1 then all hosts will load real data,
    # else total_hosts//expansion_factor_real_data will pull data from GCS.
    expansion_factor_real_data: int = -1
    eval_per_device_batch_size: int = 0
    max_corpus_chars: int = 10_000_000
    dataset_type: str = "c4"  # must be c4 or synthetic
    use_full_low_res_data: bool = False
    full_low_res_version: str = "3.0.0"
    normalize: bool = False
    using_mixed_y_norm: bool = False
    use_high_res_data: bool = False
    mix_high_res_ratio: float = 0.0
    n_ds_per_split: int = 128
    n_repeat_ds_per_epoch: int = 16
    shuffle_buffer_multiplier: int = 2

    # Setting for grain
    grain_worker_count: int = 4

    # Training loop
    num_epochs: float | None = (
        1.0  # Number of epochs to train for, if set, overrides steps
    )
    # If set to -1 then will inherit value from learning_rate_schedule_steps
    steps: int | None = None
    log_period: int = 100  # Flushes Tensorboard

    # We take inspiration from Llama2's learning rate (LR) schedule,
    # see https://arxiv.org/pdf/2307.09288.pdf section 2.2
    # Learning rate schedule has either two or three parts:
    # 1) Linear warmup from 0 to [learning_rate] over steps 0
    #     to [learning_rate_schedule_steps * warmup_steps_fraction]
    # 2) Cosine decay from [learning_rate] to
    #     [learning_rate * cosine_learning_rate_final_fraction]
    #     from warmup to learning_rate_schedule_steps
    # 3) Constant learning rate of 0 from learning_rate_schedule_steps to steps.
    #     The zero learning rate section can be used to more accurately measure
    #     the fully trained model's performance.

    learning_rate: float = 5.0e-4
    cosine_learning_rate_final_fraction: float = 0.001
    warmup_steps_fraction: float = 0.167
    # By default the length of the schedule is set to the number of steps.
    learning_rate_schedule_steps: int = -1
    # However you may choose a longer schedule (learning_rate_schedule_steps > steps),
    # in which case the training will end before
    # dropping fully down. Or you may choose a shorter schedule,
    # where the unspecified steps will have a learning rate of 0.

    max_target_length: int = 512  # Maximum sequence length

    # Maximum length for the prefill when doing autoregression
    max_prefill_predict_length: int = 64

    prompt: str = "I love to"  # Prompt for language model sampling.

    # If true, decode.py doesn't "prefill" but just reads from directory
    load_from_prefill_dir: bool = False

    # If set and load_from_prefill_dir, decode.py reads from directory.
    # If set, decode.py writes to directory
    prefill_cache_dir: str = ""
    autoregressive_decode_assert: str = ""

    enable_profiler: bool = False
    # If set to true, upload all profiler xplane results from all hosts.
    # Otherwise, only upload the xplane result from the first host.
    upload_all_profiler_results: bool = False
    # Skip first n steps for profiling, to omit things like compilation and to give
    # the iteration time a chance to stabilize.
    skip_first_n_steps_for_profiler: int = 1
    # Profile for a small number of steps to avoid a large profile file size.
    profiler_steps: int = 5

    # When dropout is false the model is a deterministic function of the
    # data_shuffle_seed and init_weights_seed (i.e. reproducible losses)
    enable_dropout: bool = False
    enable_data_shuffling: bool = True
    data_shuffle_seed: int = 0
    init_weights_seed: int = 0

    # You may disable clipping by setting gradient_clipping_threshold to zero.
    gradient_clipping_threshold: float = 0.1

    # AdamW optimizer parameters
    # We use AdamW following Llama2's training details,
    # see https://arxiv.org/pdf/2307.09288.pdf section 2.2
    opt_type: str = "adamw"  # one of "adam_pax" or "adamw"
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    adam_eps: float = 1.0e-8
    adam_eps_root: float = 0.0
    adam_weight_decay: float = 0.0

    # Stack trace parameters
    collect_stack_trace: bool = False
    # Uploads to cloud logging if True, else to the console if False.
    stack_trace_to_cloud: bool = False
    # Stack trace collection frequency in seconds.
    stack_trace_interval_seconds: int = 600

    # Use iota operator in Embed
    use_iota_embed: bool = False
    # use positional embedding
    use_untrainable_positional_embedding: bool = False
    # enable gpt3 position embedding with a positive trainable_position_size
    trainable_position_size: int = -1

    # Ahead of time Compilation (aka AOT)
    # Only set these arguments if you are running train_compile or loading a compiled train step.
    # Name of saved serialized compiled train_step, e.g. compiled_train_v5e-256.pickle
    compiled_trainstep_file: str = ""
    compile_topology: str = ""  # Target hardware version, e.g. 'v5e-256'
    # Number of target slices, set to a positive integer.
    compile_topology_num_slices: int = -1

    # decode_sampling_strategy should be one of greedy, weighted, nucleus, or topk
    decode_sampling_strategy: str = "greedy"
    decode_sampling_nucleus_p: int = -1  # set if you're doing nucleus / top-p
    decode_sampling_top_k: int = 0  # set if you're doing top-k
    decode_sampling_temperature: float = 1.0

    n_evals_every_epoch: int | None = 2  # If set, overrides eval_interval
    # the specific number of train step between eval_step -> eval every 0.1 epochs
    eval_interval: int = 4000
    target_eval_loss: float = 0.0  # early stop once reaching target eval_loss

    # Goodput parameters
    enable_goodput_recording: bool = False

    # Vertex AI Tensorboard Configurations -
    # https://github.com/google/maxtext/tree/main/getting_started/Use_Vertex_AI_Tensorboard.md
    # Set to True for GCE, False if running via XPK
    use_vertex_tensorboard: bool = False
    # Project to create Vertex AI Tensorboard in for GCE, blank if project
    # is set using 'gcloud config set project'
    # Set this to blank if running via XPK
    vertex_tensorboard_project: str = ""
    # Region to create Vertex AI Tensorboard in for GCE, blank if running via XPK
    # Vertex AI supported regions:
    # https://cloud.google.com/vertex-ai/docs/general/locations#available-regions
    vertex_tensorboard_region: str = ""

    # If set to True, MaxText will perform extra checks using jax.checkify.
    # Note that this will effect performance.
    max_checkify: bool = False

    # Inference
    inference_microbenchmark_prefill_lengths: str = "64,128,256,512,1024"
    inference_microbenchmark_stages: str = "prefill,generate"
    inference_microbenchmark_loop_iters: int = 10
    inference_microbenchmark_log_file_path: str = ""

    submit_to_kaggle: bool = False
    use_submission_trick: bool = False

    tensorboard_dir: str = dataclasses.field(init=False)
    checkpoint_dir: str = dataclasses.field(init=False)
    metrics_dir: str = dataclasses.field(init=False)
    global_batch_size_to_load: int = dataclasses.field(init=False)
    global_batch_size_to_train_on: int = dataclasses.field(init=False)
    N_IN: int = dataclasses.field(init=False)
    N_OUT: int = dataclasses.field(init=False)
    train_steps_per_epoch: int = dataclasses.field(init=False)
    eval_steps: int = dataclasses.field(init=False)
    mesh_axes: list[str] = dataclasses.field(init=False)
    logical_axis_rules: tuple[str | tuple[str | tuple[str]]] = dataclasses.field(
        init=False
    )
    data_sharding: tuple[tuple[str]] = dataclasses.field(init=False)

    def __post_init__(self):
        max_utils.maybe_initialize_jax_distributed_system(
            enable_checkpointing=self.enable_checkpointing,
            async_checkpointing=self.async_checkpointing,
            compile_topology_num_slices=self.compile_topology_num_slices,
            hardware=self.hardware,
        )

        if self.jax_cache_dir:
            compilation_cache.set_cache_dir(os.path.expanduser(self.jax_cache_dir))

        if self.run_name == "":
            raise ValueError("Set run_name")

        self.tensorboard_dir = os.path.join(
            self.base_output_directory, self.run_name, "tensorboard", ""
        )
        self.checkpoint_dir = os.path.join(
            self.base_output_directory, self.run_name, "checkpoints", ""
        )
        self.metrics_dir = os.path.join(
            self.base_output_directory, self.run_name, "metrics", ""
        )

        self.global_batch_size_to_load, self.global_batch_size_to_train_on = (
            calculate_global_batch_sizes(
                self.per_device_batch_size,
                self.expansion_factor_real_data,
                self.compile_topology,
                self.compile_topology_num_slices,
            )
        )

        USE_FULL_DATA = self.use_full_low_res_data or self.use_high_res_data
        self.N_IN = 556 if USE_FULL_DATA else 490
        self.N_OUT = 368 if USE_FULL_DATA else 308

        if USE_FULL_DATA:
            self.max_target_length = 556

        train_samples_per_epoch = 10_000_000
        if self.use_high_res_data:
            train_samples_per_epoch = 563_176_800

        val_samples_per_epoch = 500_000

        self.train_steps_per_epoch = (
            train_samples_per_epoch // self.global_batch_size_to_train_on
        )
        self.eval_steps = val_samples_per_epoch // self.global_batch_size_to_train_on

        if self.num_epochs is None and self.steps is None:
            raise ValueError("Either steps or num_epochs should be provided")

        if self.num_epochs is not None:
            print("Inheriting steps from num_epochs")
            self.steps = int(self.num_epochs * self.train_steps_per_epoch)
            if self.learning_rate_schedule_steps == -1:
                self.learning_rate_schedule_steps = self.steps
            if self.steps == -1:
                self.steps = self.learning_rate_schedule_steps
        print("Total Training Steps -> ", self.steps)

        if self.n_evals_every_epoch is not None:
            if self.use_high_res_data:
                self.n_evals_every_epoch *= 50
            self.eval_interval = int(
                self.train_steps_per_epoch * (1 / self.n_evals_every_epoch)
            )
        print("eval_interval -> ", self.eval_interval)

        if self.n_checkpoints_every_epoch is not None:
            if self.use_high_res_data:
                self.n_checkpoints_every_epoch *= 50
            self.checkpoint_period = int(
                self.train_steps_per_epoch * (1 / self.n_checkpoints_every_epoch)
            )
        print("checkpoint_period -> ", self.checkpoint_period)

        emb_scale, num_head_scale, mlp_dim_scale, layer_scale = get_individual_scales(
            self.global_parameter_scale
        )
        self.emb_dim = 2**emb_scale * self.base_emb_dim
        self.num_query_heads = 2**num_head_scale * self.base_num_query_heads
        self.num_kv_heads = 2**num_head_scale * self.base_num_kv_heads
        self.mlp_dim = 2**mlp_dim_scale * self.base_mlp_dim
        self.num_decoder_layers = 2**layer_scale * self.base_num_decoder_layers

        self.normalize = False
        if (
            self.use_full_low_res_data
            and self.full_low_res_version
            in [
                "3.0.0",
                "6.0.0",
                "7.0.0",
            ]
        ) or self.use_high_res_data:
            self.normalize = True

        self.num_slices = get_num_slices(self.compile_topology_num_slices)
        self.quantization_local_shard_count = get_quantization_local_shard_count(
            self.quantization_local_shard_count, self.num_slices
        )

        print_system_information()

        self.dtype = jax.numpy.dtype(self.dtype)

        # Parallelism
        self.mesh_axes = [
            "data",
            "fsdp",
            "fsdp_transpose",
            "sequence",
            "tensor",
            "autoregressive",
        ]

        logical_axis_rules: list[str | list[str | list[str]]] = [
            [
                "activation_batch",
                [
                    "data",
                    "fsdp",
                    "fsdp_transpose",
                ],
            ],
            ["activation_heads", ["tensor", "sequence"]],
            ["activation_length", "sequence"],
            ["activation_embed", "tensor"],
            ["activation_mlp", "tensor"],
            ["activation_kv", "tensor"],
            ["activation_vocab", ["tensor", "sequence"]],
            ["activation_vocab", "tensor"],
            ["activation_vocab", "sequence"],
            ["mlp", ["fsdp_transpose", "tensor", "autoregressive"]],
            ["vocab", ["tensor", "autoregressive"]],
            ["embed", ["fsdp", "fsdp_transpose", "sequence"]],
            ["embed", ["fsdp", "sequence"]],
            ["norm", "tensor"],
            ["heads", ["tensor", "autoregressive"]],
            ["kv", []],
            ["cache_batch", []],
            ["cache_heads", ["autoregressive", "tensor"]],
            ["cache_kv", []],
            ["cache_sequence", []],
        ]

        data_sharding: list[list[str]] = [
            ["data", "fsdp", "fsdp_transpose", "sequence", "tensor", "autoregressive"]
        ]

        self.logical_axis_rules = _lists_to_tuples(logical_axis_rules)
        self.data_sharding = _lists_to_tuples(data_sharding)

        validate_keys(self)


def _lists_to_tuples(l: list[Any]):
    return tuple(_lists_to_tuples(x) for x in l) if isinstance(l, list) else l


def get_num_slices(compile_topology_num_slices):
    if int(compile_topology_num_slices) > 0:
        return compile_topology_num_slices
    else:
        devices = jax.devices()
        try:
            return 1 + max([d.slice_index for d in devices])
        except:
            return 1


def get_quantization_local_shard_count(quantization_local_shard_count, num_slices):
    if quantization_local_shard_count == -1:
        return num_slices
    else:
        return quantization_local_shard_count


def get_individual_scales(scale):
    """Choose appropriate scales for individual dimensions based on global scale
    We choose to rotate between doubling:
      num_head and mlp_dim
      embed_dim
      num_layers
    Any one of these steps is not a perfect doubling, although going through a cycle
    of three is a near perfect 8x scaling except for the linear -> softmax -> output step
    """

    log_2_scale = math.floor((math.log2(scale)))
    if 2**log_2_scale != scale:
        raise ValueError(
            "Global parameter scale should be a power of 2. If you want finer grained control of the model sizes "
            "then you can explicitly set base_embed_dim, base_num_heads, base_mlp_dim, base_num_decoder_layers and/or head_dim."
        )
    base_scale, rem = divmod(log_2_scale, 3)
    num_head_scale = base_scale + int(rem > 0)
    mlp_dim_scale = num_head_scale
    emb_scale = base_scale + int(rem > 1)
    layer_scale = base_scale
    return emb_scale, num_head_scale, mlp_dim_scale, layer_scale


def get_num_target_devices(compile_topology, compile_topology_num_slices):
    compile_topology = accelerator_to_spec_map.get_system_characteristics(
        compile_topology
    )
    if compile_topology is not None:
        devices_per_slice = compile_topology.devices_per_slice
        return int(devices_per_slice * compile_topology_num_slices)
    else:
        return len(jax.devices())


def calculate_global_batch_sizes(
    per_device_batch_size,
    expansion_factor_real_data,
    compile_topology,
    compile_topology_num_slices,
):
    """Calculates target global batch size from target devices and per_device_batch"""
    num_devices = get_num_target_devices(compile_topology, compile_topology_num_slices)
    if per_device_batch_size < 1.0:
        # For per_device_batch_size<1, we load the data as if per_device_batch_size=1
        if expansion_factor_real_data != -1:
            global_batch_size_to_load = num_devices * expansion_factor_real_data
        else:
            global_batch_size_to_load = num_devices
    else:
        if expansion_factor_real_data != -1:
            global_batch_size_to_load = int(
                num_devices * per_device_batch_size * expansion_factor_real_data
            )
        else:
            global_batch_size_to_load = int(num_devices * per_device_batch_size)

    global_batch_size_to_train_on = int(num_devices * per_device_batch_size)
    return global_batch_size_to_load, global_batch_size_to_train_on


def validate_attention_type(s: str) -> None:
    valid_attention_types = ("autoselected", "dot_product", "flash", "cudnn_flash_te")
    if s not in valid_attention_types:  # currently supported attention
        raise ValueError(
            "Invalid attention type was passed. Valid options ", valid_attention_types
        )


def validate_keys(config: Config):
    validate_attention_type(config.attention)
    assert (
        config.load_parameters_path == "" and config.load_full_state_path == ""
    ) or config.enable_checkpointing, (
        "You must set enable_checkpointing to load a checkpoint"
    )
    assert (
        config.load_parameters_path == "" or config.load_full_state_path == ""
    ), "At most one of `load_parameters_path` or `load_full_state_path` should be set"


def validate_model_name(s: str):
    """Validate provided model name."""
    # currently supported models
    valid_model_names = (
        "default",
        "llama2-7b",
        "llama2-13b",
        "llama2-70b",
        "mistral-7b",
        "mixtral-8x7b",
        "gemma-7b",
        "gemma-2b",
        "gpt3-175b",
        "gpt3-22b",
        "gpt3-6b",
        "gpt3-52k",
    )
    if s not in valid_model_names:
        raise ValueError(
            "Invalid model name was passed. Valid options ", valid_model_names
        )


def validate_no_keys_overwritten_twice(keys1: list[str], keys2: list[str]):
    overwritten_keys = [k for k in keys1 if k in keys2]
    if overwritten_keys:
        raise ValueError(
            f"Keys {overwritten_keys} are overwritten from both the model"
            " and the environment/command line. This isn't allowed."
        )


def print_system_information():
    max_logging.log(f"System Information: Jax Version: {jax.__version__}")
    max_logging.log(f"System Information: Jaxlib Version: {jax.lib.__version__}")  # type: ignore
    max_logging.log(
        f"System Information: Jax Backend: {jax.lib.xla_bridge.get_backend().platform_version}"  # type: ignore
    )


def get_config(args: Args):
    pass


if __name__ == "__main__":
    app.run(get_config, flags_parser=eapp.make_flags_parser(Args))
