from dataclasses import dataclass
from typing import Any

import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

import rich
from absl import app
from etils import eapp
from etils.etree import tree as etree

import utils
from utils import inspect as spec
import exp_configs


@dataclass
class Args:
    exp_version: str = '0.0'

class IdentityLayer(hk.Module):

    def __init__(self, name: str = "Identity"):
        super().__init__(name=name)

    def __call__(self, x: jax.Array):
        return x


class AddPositionEmbs(hk.Module):
    def __init__(
        self,
        name: str = "posemb",
    ):
        super().__init__(name=name)
        self.posemb_init = hk.initializers.RandomNormal(stddev=0.02)

    def __call__(self, x: jax.Array):
        pos_emb_shape = (1, x.shape[1], x.shape[2])
        pe = hk.get_parameter(
            "pos_embedding",
            shape=pos_emb_shape,
            init=self.posemb_init,
        )

        return x + pe


class MLPBlock(hk.Module):
    """Transformer MLP / feed-forward block."""

    def __init__(
        self,
        width: int,
        verbose: int,
        activation: Any,
        drop_rate: float = 0.1,
        name: str = "MLPBlock",
    ):
        super().__init__(name=name)
        self.width = width
        self.drop_rate = drop_rate
        self.verbose = verbose
        self.act = activation

    def __call__(self, x: jax.Array, is_training: bool) -> jax.Array:
        if self.verbose:
            print(f"Inside {self.name}")
            print(" .... " * 4)

        x = hk.Linear(self.width * 4)(x)
        x = self.act(x)

        if self.verbose:
            spec("gelu-dense", x=x)

        if self.drop_rate > 0.0 and is_training:
            x = hk.dropout(hk.next_rng_key(), self.drop_rate, x)

            if self.verbose:
                spec(f"dropout({self.drop_rate})", x=x)

        out = hk.Linear(self.width)(x)

        if self.verbose:
            spec("dense", x=out)

        if self.drop_rate > 0.0 and is_training:
            out = hk.dropout(hk.next_rng_key(), self.drop_rate, out)

            if self.verbose:
                spec(f"dropout({self.drop_rate})", out=out)

        return out


class SelfAttention(hk.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        drop_rate: float,
        verbose: int,
        name: str = "SelfAttention",
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_size = d_model // num_heads
        self.w_init = hk.initializers.VarianceScaling()
        self.drop_rate = drop_rate
        self.verbose = verbose

        self.in_proj_weight = hk.get_parameter(
            "in_proj_weight",
            shape=[self.d_model * 3, self.d_model],
            init=self.w_init,
        )
        self.in_proj_bias = hk.get_parameter(
            "in_proj_bias", shape=[self.d_model * 3], init=self.w_init
        )
        self.out_proj = hk.Linear(self.d_model, name="out_proj")
        self.verbose = verbose

    def __call__(self, x: jax.Array, is_training: bool = False):
        """Compute (optionally masked) MHA with queries, keys & values."""
        if self.verbose:
            print(f"Inside {self.name}")
            print(" .... " * 4)
            spec("inputs", x=x)

        all_out = jnp.dot(x, self.in_proj_weight.transpose())
        all_out += self.in_proj_bias
        if self.verbose:
            spec("linear", x=all_out)

        q, k, v = jnp.array_split(all_out, 3, axis=-1)
        query_heads = self._split(q)
        key_heads = self._split(k)
        value_heads = self._split(v)

        if self.verbose:
            spec("split", q=query_heads, k=key_heads, v=value_heads)

        attention_logits = jnp.einsum("bthd,bThd->bhtT", query_heads, key_heads)
        sqrt_key_size = np.sqrt(self.head_size).astype(k.dtype)
        attention_logits = attention_logits / sqrt_key_size
        attention_weights = jax.nn.softmax(attention_logits)

        if self.verbose:
            spec("", attn_logits=attention_logits)
            spec("", attn_weights=attention_weights)

        if self.drop_rate > 0.0 and is_training:
            attention_weights = hk.dropout(
                hk.next_rng_key(), self.drop_rate, attention_weights
            )
            if self.verbose:
                spec(f"dropout({self.drop_rate})", attn_weights=attention_weights)

        attention = jnp.einsum("bhtT,bThd->bthd", attention_weights, value_heads)
        if self.verbose:
            spec("", attn=attention)

        # Concatenate attention matrix of all heads into a single vector.
        attention_vec = jnp.reshape(attention, (*q.shape[:2], -1))
        if self.verbose:
            spec("merge", attn_vec=attention_vec)

        out = self.out_proj(attention_vec)
        if self.verbose:
            spec("out_proj", out=out)

        return out

    def _split(self, x: jnp.ndarray):
        return x.reshape((*x.shape[:2], self.num_heads, self.d_model // self.num_heads))


def test_SelfAttention():

    B, L, D = 2, 197, 768

    x = jax.random.normal(jax.random.PRNGKey(0), shape=(B, L, D))

    def forward(x, is_training: bool):
        x = SelfAttention(num_heads=12, d_model=D, drop_rate=0.1, verbose=True)(
            x, is_training=is_training
        )
        return x

    f = hk.transform_with_state(forward)

    p, s = f.init(jax.random.PRNGKey(0), x, is_training=True)

    compiled = (
        jax.jit(f.apply, static_argnames=("is_training",))
        .lower(p, s, jax.random.PRNGKey(1), x, is_training=True)
        .compile()
    )

    rich.print(utils.tree_shape(p))
    rich.print(utils.tree_size(p))
    print(f"{hk.data_structures.tree_size(p):,}")

    gigaflops = compiled.cost_analysis()[0]["flops"] / (10**9)  # type: ignore

    print("GFLOPs = ", gigaflops)
    print(f"{gigaflops/B = }")


class TransformerBlock(hk.Module):
    def __init__(
        self,
        width: int,
        num_heads: int,
        activation: Any,
        verbose: int,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        name: str = "transformer_block",
    ):
        super().__init__(name=name)

        self.width = width
        self.num_heads = num_heads
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.verbose = verbose

        self.rn_1 = hk.RMSNorm(
            axis=-1,
            eps=1e-6,
            create_scale=True,
            name="rn_1",
        )
        self.rn_2 = hk.RMSNorm(
            axis=-1,
            eps=1e-6,
            create_scale=True,
            name="rn_2",
        )

        self.mha = SelfAttention(
            num_heads=self.num_heads,
            d_model=self.width,
            drop_rate=self.attn_drop_rate,
            verbose=self.verbose - 1,
        )

        self.mlp = MLPBlock(
            width=self.width,
            drop_rate=self.drop_rate,
            verbose=self.verbose - 1,
            activation=activation,
        )

    def __call__(self, inputs: jax.Array, is_training: bool) -> jax.Array:
        """Transformer block."""
        if self.verbose > 0:
            print("\n" * 2)
            print(f"Inside {self.name}")
            print("-" * 25)

        x = self.rn_1(inputs)
        x = self.mha(x, is_training=is_training)

        if self.verbose > 0:
            spec("mha-ln_1", inputs=x)
        if self.attn_drop_rate > 0.0 and is_training:
            x = hk.dropout(hk.next_rng_key(), self.attn_drop_rate, x)
            if self.verbose:
                spec(f"dropout({self.attn_drop_rate})", x=x)

        x = x + inputs
        if self.verbose > 0:
            spec("add", inputs_and_x=x)
        if self.verbose - 1 > 0:
            print("_" * 55)

        y = self.rn_2(x)
        y = self.mlp(y, is_training=is_training)
        out = x + y

        if self.verbose > 0:
            spec("add-mlp-ln_2", out=out)

        return out


class ClimSimulator(hk.Module):
    def __init__(
        self,
        model_args: exp_configs.ModelArgs,
        verbose: int = 0,
    ):
        super().__init__(name=model_args.name)

        self.num_layers = model_args.num_layers
        self.width = model_args.width
        self.num_heads = model_args.num_heads
        self.drop_rate = model_args.drop_rate
        self.activation = model_args.activation
        self.n_out = model_args.n_out
        self.which_n_out_act = model_args.which_n_out_act
        self.w_init = model_args.w_init
        self.verbose = verbose

    def __call__(self, x: jax.Array, is_training: bool) -> jax.Array:
        """ClimSimulator."""
        if self.verbose > 0:
            print()
            print(f"{self.name}")
            print("-" * 25)
            spec("inputs", x=x)

        B, N_IN = x.shape

        x = jnp.expand_dims(x, axis=-1)  # [bs, n_in, 1]
        if self.verbose > 0:
            spec("expand_dims", x=x)

        x_learn_width = hk.get_parameter(
            "x_width",
            shape=[x.shape[1], self.width - 1],
            init=self.w_init,
        )  # [n_in, width-1]
        x_learn_width = jnp.broadcast_to(x_learn_width, (B, N_IN, self.width - 1))
        x = jnp.concatenate([x, x_learn_width], axis=-1)  # [bs, n_in, width]
        if self.verbose > 0:
            spec("concat_x_learn", x=x)

        if self.drop_rate > 0.0 and is_training:
            x = hk.dropout(hk.next_rng_key(), self.drop_rate, x)
            if self.verbose > 0:
                spec(f"dropout({self.drop_rate})", x=x)

        y_learn_width = hk.get_parameter(
            "y_width",
            shape=[self.n_out, self.width],
            init=self.w_init,
        )
        y_learn_width = jnp.broadcast_to(y_learn_width, (B, self.n_out, self.width))
        x = jnp.concatenate([x, y_learn_width], axis=1)  # [bs, n_in + n_out, width]
        if self.verbose > 0:
            spec("concat_y_learn", x=x)

        x = AddPositionEmbs(name="posembed_input")(x)
        if self.verbose > 0:
            spec("pos_emb", x=x)

        for i in range(self.num_layers):
            x = TransformerBlock(
                width=self.width,
                num_heads=self.num_heads,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.drop_rate,
                activation=self.activation,
                verbose=self.verbose - 1,
                name=f"{self.name}_block_{i}",
            )(x, is_training=is_training)

            if self.verbose:
                print("-" * 80)

        x = hk.RMSNorm(axis=-1, create_scale=True, name="rn_f")(x)
        if self.verbose > 0:
            print()
            spec("rn_f", x=x)

        x = x[:, -self.n_out:, :]
        if self.verbose > 0:
            spec("n_out_slice", x=x)
        
        y_pred = x[..., 0]
        if self.verbose > 0:
            spec("", y_pred=y_pred)
        
        if self.which_n_out_act is not None:
            y_pred = self.which_n_out_act(y_pred)
            if self.verbose > 0:
                spec(f"{self.which_n_out_act}", y_pred=y_pred)
        
        return y_pred


def test_ClimSimulator(args: Args):
    import exp_configs

    exp_version = args.exp_version

    model_args = exp_configs.get_model_args(exp_version)

    verbose = 4

    B, N_IN = 8, 490

    x = jax.random.normal(jax.random.PRNGKey(0), shape=(B, N_IN))

    def forward(x, is_training: bool):
        x = ClimSimulator(model_args, verbose)(x, is_training=is_training)
        return x

    f = hk.transform_with_state(forward)

    p, s = f.init(jax.random.PRNGKey(0), x, is_training=True)

    compiled = (
        jax.jit(f.apply, static_argnames=("is_training",))
        .lower(p, s, jax.random.PRNGKey(1), x, is_training=True)
        .compile()
    )

    rich.print(utils.tree_shape(p))
    rich.print(utils.tree_size(p))
    print(f"{hk.data_structures.tree_size(p):,}")

    gigaflops = compiled.cost_analysis()[0]["flops"] / (10**9)  # type: ignore

    print("GFLOPs = ", gigaflops)
    print(f"{gigaflops/B = }")


# def main(exp_version: str,):
#     # M = [0, 128, 512, 512, 512]
#     # print(calc_stagewise_extra_out_ch(M))

#     import jmp
#     from exp_configs import get_experiment_config
#     import rich
#     import tree

#     exp_config = get_experiment_config(exp_version)

#     if exp_config.bfloat16:
#         policy = jmp.get_policy("p=f32,c=bf16,o=bf16")
#         norm_policy = jmp.get_policy("p=f32,c=f32,o=bf16")
#     else:
#         policy = jmp.get_policy("p=f32,c=f32,o=f32")
#         norm_policy = jmp.get_policy("p=f32,c=f32,o=f32")

#     hk.mixed_precision.set_policy(hk.BatchNorm, norm_policy)
#     hk.mixed_precision.set_policy(hk.LayerNorm, norm_policy)
#     hk.mixed_precision.set_policy(hk.RMSNorm, norm_policy)
#     hk.mixed_precision.set_policy(ClimSimulator, policy)

#     rng_key_seq = hk.PRNGSequence(jax.random.PRNGKey(0))  # type: ignore
#     rng = next(rng_key_seq)

#     rich.inspect(exp_config, value=False, title=f"exp_config_{exp_version}")

#     def forward_fn(
#         x,
#         is_training=True,
#     ):
#         net = ClimSimulator(exp_config.model_args, exp_config.verbose)
#         return net(x, is_training=is_training)

#     f = hk.transform_with_state(forward_fn)

#     B = exp_config.batch_size
#     x = jax.random.normal(rng, (B, 224, 224, 3))

#     x = policy.cast_to_compute(x)
#     _params, _state = f.init(next(rng_key_seq), x)

#     if exp_config.verbose:
#         print()
#         print()

#     num_params = hk.data_structures.tree_size(_params)
#     print(f"num parameters: {num_params:,}")
#     params_count = jtu.tree_map(lambda x: x.size, _params)

#     rich.print(params_count)
#     num_state_var = hk.data_structures.tree_size(_state)
#     print(f"num state variables: {num_state_var:,}")

#     if compile_model:
#         f_apply = jax.jit(f.apply, static_argnums=(4, 5))

#         compiled = f_apply.lower(_params, _state, rng, x, False).compile()

#         gigaflops = compiled.cost_analysis()[0]["flops"] / (10**9)  # type: ignore

#         if exp_config.pmap:
#             print("GFLOPs/device = ", gigaflops / jax.device_count())
#         else:
#             print("GFLOPs = ", gigaflops)

#         # Calculate Runtime of outputs, _ = f.apply(p, s, rng, x)
#     if exp_config.outputs:
#         import time

#         import rich
#         from ml_collections.config_dict.config_dict import ConfigDict

#         from functions import utils

#         start = time.time()

#         apply = f_apply if exp_config.compile else f.apply  # type: ignore
#         outputs, _ = apply(p, s, rng, x, False, False)

#         rich.print(utils.tree_shape(outputs))


if __name__ == "__main__":
    # test_SelfAttention()
    # test_ViT()
    app.run(test_ClimSimulator, flags_parser=eapp.make_flags_parser(Args))  # type: ignore
