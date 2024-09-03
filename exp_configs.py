from dataclasses import dataclass
import os
from typing import Any
import jax
from etils import eapp
from absl import app
import haiku as hk
import tpu_utils
import utils


@dataclass
class Args:
    exp_version: str = "0.0"
    csv: bool = False


@dataclass
class ModelArgs:
    width: int = 128
    num_layers: int = 8
    num_heads: int = 1
    n_out: int = 308
    drop_rate: float = 0.0
    activation: Any = jax.nn.silu
    which_n_out_act: Any = None
    w_init: Any = hk.initializers.VarianceScaling()
    name: str = "ClimSimulator"


@dataclass
class ExpConfig:
    exp_version: str
    model_version: str
    run_name: str = ""

    train_dataset_name: str = "climsim_low_res_train"
    test_dataset_name: str = "climsim_low_res_test"

    num_epochs: int = 90
    warmup_epochs: float = 0.2
    batch_size: int = 1024
    lr: float = 0.1
    bfloat16: bool = True
    use_ema: bool = True
    ema_decay: float = 0.9999
    verbose: int = 1
    scale_lr_with_bs: bool = True
    random_seed: int = 42

    which_optimizer: str = "adamw"
    opt_kwargs_weight_decay: float = 0.05
    clip_grads: float | None = None
    grad_acc_steps: int | None = None
    adaptive_grad_clip: float | None = None

    local: bool = False
    vm_name: str = ""
    restore_path: str = ""
    log_every: int = 100  # steps
    eval_every: int = 2000  # steps ~ 5 evals per epoch
    checkpoint_every: int = 2000  # steps
    keep_latest_checkpoint_only: bool = True
    save_best_checkpoint: bool = False
    best_model_eval_metric: str = "eval_mae"
    create_submission_csv: bool = True
    submit_to_kaggle: bool = True
    denorm_arr_path: str = "./"

    def __post_init__(self):
        self.model_args = get_model_args(self.model_version)
        if self.local:
            self.model_args.num_layers = 2
            self.model_args.width = 128
            self.model_args.num_heads=2

        if not self.local:
            self.zone = tpu_utils.get_tpu_vm_zone()
            if self.zone == "europe-west4-a":
                self.data_dir = "eu-climsim"
            elif self.zone == "us-central1-f":
                self.data_dir = "us1-climsim"
            elif self.zone == "us-central2-b":
                self.data_dir = "us2-climsim"
            else:
                raise ValueError(f"Zone {self.zone} not supported")

            self.data_dir = f"gs://{self.data_dir}"

            print("Zone", self.zone)
            print("Bucket", self.data_dir)

        else:
            self.zone = None
            self.data_dir = (
                "/mnt/c/Users/Admin/Documents/GitHub/LEAP-Weather-Prediction/"
            )

        self.train_samples_per_epoch = 10_000_000
        self.val_samples_per_epoch = 500_000
        self.test_samples = 625_000
        self.train_steps_per_epoch = self.train_samples_per_epoch // self.batch_size
        self.val_steps_per_epoch = self.val_samples_per_epoch // self.batch_size
        self.test_steps = self.test_samples // self.batch_size

        self.training_steps = self.num_epochs * self.train_steps_per_epoch
        self.warmup_steps = int(self.warmup_epochs * self.train_steps_per_epoch)

        self.base_bs = 256
        peak_lr = (
            self.lr * self.batch_size / self.base_bs
            if self.scale_lr_with_bs
            else self.lr
        )
        self.lr = peak_lr

        self.opt_kwargs = {
            "weight_decay": self.opt_kwargs_weight_decay,
        }


        self.optimizer_config = {
            "name": self.which_optimizer,
            "opt_kwargs": self.opt_kwargs,
            "opt_addons": {
                "clip_grads": self.clip_grads,
                "grad_acc_steps": self.grad_acc_steps,
                "adaptive_grad_clip": self.adaptive_grad_clip,
            },
            "lr_schedule": {
                "name": "warmup_cosine_decay_schedule",
                "kwargs": {
                    "decay_steps": self.training_steps,
                    "peak_value": self.lr,
                    "init_value": 0,
                    "end_value": 0,
                    "warmup_steps": self.warmup_steps,
                },
            },
        }

        self.name = f"ClimSimulator_{self.exp_version}"
        self.name += f"-{self.run_name}" if len(self.run_name) else ""

        self.save_path = os.path.join("checkpoints", f"checkpoint_{self.name}")

        dnorm_fns = utils.denorm_and_scale_wrapper(self.denorm_arr_path)
        self.post_process_y = dnorm_fns["denorm_scale_and_expand_output"]


def model_args_0(version: int):
    return ModelArgs(
        width=128,
        num_layers=2,
        num_heads=2,
        n_out=308,
        drop_rate=0,
        activation=jax.nn.silu,
        which_n_out_act=None,
    )


def exp_0(version: int, kwargs: dict):
    """test - experiment"""
    batch_size = kwargs.pop("batch_size", 1024)
    return ExpConfig(
        exp_version=f"0.{version}",
        model_version=f"0.{version}",
        bfloat16=bool(version),
        verbose=1,
        batch_size=batch_size,
        num_epochs=1,
        warmup_epochs=0,
        log_every=1,
        eval_every=10,
        checkpoint_every=20,
        **kwargs,
    )


def model_args_2(version: int):
    return ModelArgs(
        width=512,
        num_layers=12,
        num_heads=4,
        n_out=308,
        drop_rate=0,
        activation=jax.nn.silu,
        which_n_out_act=None,
        w_init=hk.initializers.Constant(0.0),
    )


def exp_2(version: int, kwargs: dict):
    """1st experiment"""

    batch_size = kwargs.pop("batch_size", None)
    if batch_size is None:
        batch_size = 512

    return ExpConfig(
        exp_version=f"2.{version}",
        model_version=f"2.{version}",
        bfloat16=True,
        verbose=1,
        batch_size=batch_size,
        num_epochs=1,
        warmup_epochs=0.2,
        log_every=100,
        eval_every=4000,  # 5 evals per epoch
        checkpoint_every=4000,  # 5 checkpoints per epoch
        clip_grads=1.0,
        lr=1e-3,
        **kwargs,
    )

def model_args_3(version: int):
    return ModelArgs(
        width=512,
        num_layers=12,
        num_heads=8,
        n_out=308,
        drop_rate=0,
        activation=jax.nn.silu,
        which_n_out_act=None,
        w_init=hk.initializers.Constant(0.0),
    )


def exp_3(version: int, kwargs: dict):
    """1st experiment"""

    batch_size = kwargs.pop("batch_size", None)
    if batch_size is None:
        batch_size = 512

    return ExpConfig(
        exp_version=f"3.{version}",
        model_version=f"3.{version}",
        bfloat16=True,
        verbose=1,
        batch_size=batch_size,
        num_epochs=3,
        warmup_epochs=0.5,
        log_every=100,
        eval_every=4000,  # 5 evals per epoch
        checkpoint_every=4000,  # 5 checkpoints per epoch
        clip_grads=1.0,
        lr=5e-4,
        **kwargs,
    )

def model_args_4(version: int):
    return ModelArgs(
        width=512,
        num_layers=12,
        num_heads=8,
        n_out=308,
        drop_rate=0,
        activation=jax.nn.silu,
        which_n_out_act=None,
        w_init=hk.initializers.Constant(0.0),
    )


def exp_4(version: int, kwargs: dict):
    """1st experiment"""

    batch_size = kwargs.pop("batch_size", None)
    if batch_size is None:
        batch_size = 2048

    return ExpConfig(
        exp_version=f"4.{version}",
        model_version=f"4.{version}",
        bfloat16=True,
        verbose=1,
        batch_size=batch_size,
        num_epochs=3,
        warmup_epochs=0.5,
        log_every=25,
        eval_every=1000,  # 5 evals per epoch
        checkpoint_every=1000,  # 5 checkpoints per epoch
        clip_grads=1.0,
        lr=5e-4,
        **kwargs,
    )


def get_model_args(model_version) -> ModelArgs:
    major, minor = model_version.split(".")
    print(major, minor)
    return eval(f"model_args_{major}({minor})")


def get_experiment_config(exp_version: str, **kwargs) -> ExpConfig:
    exp_no, version = exp_version.split(".")
    print(exp_no, version)
    return eval(f"exp_{exp_no}({version}, {kwargs})")


def get_all_version_kwargs(exp_version):
    raise NotImplementedError


def main(args: Args):
    import rich

    if args.csv:
        table = get_all_version_kwargs(args.exp_version)
        rich.inspect(table, value=True, title="table")
    else:
        exp_config = get_experiment_config(args.exp_version)
        rich.inspect(
            exp_config,
            value=False,
            title=f"exp_config_{args.exp_version}",
        )


if __name__ == "__main__":
    app.run(main, flags_parser=eapp.make_flags_parser(Args))  # type: ignore
