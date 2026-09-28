"""
history_gemma_xf.py

Forks mme_vla_suite.models.integration.history_gemma's HistoryBlock/Module
so the action expert reads XF's symbolic memory through its OWN cross-
attention path ("symbolic route"), in parallel with the released perceptual
memory modulator, instead of both streams competing in one softmax.

Why: measured 2026-09-23 (RESEARCH_LOG.md, measure_coord_usage.py) -- with
[F'; E] in the single released MemoryAttention, every caption-side
intervention (moving/deleting the target coords, swapping captions, hiding
E, bypassing fusion) changed predicted actions by ~0.2%, while swapping the
frames changed them 34-45%. 14 event tokens among 526 in one softmax, read
by a modulator warm-started for 80k steps on frames only, never got used.
Arm D's single shared cross-attention collapsed the same way.

What changes, per action-expert layer, at the memory-modulation site ONLY:
    released:  m = mem_attn(x, mem);  x = norm(x)*(1+s(m)) + b(m)
    XF route:  m_p = mem_attn(x, mem[:, :perc_len])        # frames F', warm-started
               m_s = sym_attn(x, mem[:, perc_len:])        # symbolic [E; C], new
               x = norm(x)*(1 + s_p(m_p) + s_s(m_s)) + b_p(m_p) + b_s(m_s)
Both readings reach the action expert in every layer and are summed; nothing
is dropped. The single (mem_seq, mem_mask) broadcast argument is unchanged --
the block splits it at the static `perc_len`, so the nn.scan plumbing is
untouched.

Warm-start compatibility: the perceptual path keeps the released parameter
paths exactly ("mem_attn/...", "mem_rms_norm_ffn/Dense_0/..."), so the
FrameSamp+Modul@79999 modulator loads unchanged. New params: "sym_mem_attn/..."
(same architecture as the released MemoryAttention) and "sym_mem_mod_dense"
(same kernel_init_out_proj init the released modulator's Dense used when it
was trained from scratch -- one small factor on the path, not two stacked
like the gate x out_proj pair that froze under AdamW's eps). With
sym_mem_mod_dense at zero the block is bit-identical to the released one.

NAMING IS LOAD-BEARING: both new names contain "mem". HistoryPi0Config.
get_freeze_filter (LoRA recipe) freezes every ".*llm.*" param whose path
contains none of "_1", "lora", "mem" -- a param named e.g. "sym_mod_dense"
would be SILENTLY FROZEN (caught 2026-09-23 before any run). smoke_test
checks these params are trainable.

robomme_policy_learning/ is never edited. Only integration_type
"modulation" is supported (the only one XF uses).

Role in the system: xf_pi0.XFModel builds its llm from XFModule (this file)
instead of history_gemma.Module when history_config.symbolic_route.enabled is true.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp

from openpi.models.gemma import PALIGEMMA_VOCAB_SIZE
from openpi.models.gemma import Attention
from openpi.models.gemma import Config
from openpi.models.gemma import Embedder
from openpi.models.gemma import RMSNorm
from openpi.models.gemma import _gated_residual
import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

from mme_vla_suite.models.integration.history_gemma import MemoryAttention
from mme_vla_suite.models.integration.history_gemma import Module as _ReleasedModule
from mme_vla_suite.models.integration.utils import _name
from mme_vla_suite.models.representation.utils import kernel_init_out_proj


class XFDualCondRMSNorm(nn.Module):
    """RMSNorm conditioned on the perceptual reading (released Dense_0) plus an additive symbolic modulation."""

    @nn.compact
    def __call__(self, x, cond_perc, sym_scale_shift):
        """
        What it does:
            Same math as history_gemma.MemoryRMSNorm's conditioned branch
            (param path "Dense_0" kept identical so the released weights
            load), with the symbolic path's (scale, shift) added on top:
            normed * (1 + scale_p + scale_s) + shift_p + shift_s.

        Returns:
            jax.Array -- same shape and dtype as `x`.

        Example input:
            XFDualCondRMSNorm(name="mem_rms_norm_ffn")(x, m_perc, sym_mod)
            # x, m_perc: [B, T, 1024]; sym_mod: [B, T, 2048]

        Example output:
            Array of shape (B, T, 1024)
        """
        dtype = x.dtype  # DTypeLike, original (possibly bf16) dtype
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)  # float32[..., 1]
        normed = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))  # [..., D]
        modulation = nn.Dense(x.shape[-1] * 2, kernel_init=kernel_init_out_proj, dtype=dtype, name="Dense_0")(cond_perc)  # [..., 2D]
        modulation = modulation + sym_scale_shift.astype(modulation.dtype)  # [..., 2D]
        scale, shift = jnp.split(modulation, 2, axis=-1)  # each [..., D]
        return (normed * (1 + scale) + shift).astype(dtype)


@at.typecheck
class XFHistoryBlock(nn.Module):
    """history_gemma.HistoryBlock with a parallel symbolic memory route at the modulation site."""

    configs: tuple[Config, ...]

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    integration_type: str | None = None
    perc_len: int = 512  # int, number of leading mem_seq tokens that are perceptual (F'); the rest are symbolic

    @nn.compact
    def __call__(
        self,
        xs,
        kv_cache,
        positions,
        attn_mask,
        adarms_cond,
        mem_seq,
        mem_mask,
        deterministic=True,
    ):  # noqa: FBT002
        """
        What it does:
            One layer of the (VLM expert, action expert) stack -- a verbatim
            copy of history_gemma.HistoryBlock's "modulation" path except at
            the memory-modulation site, where the action expert's hidden
            state reads the perceptual prefix of mem_seq through the released
            mem_attn and the symbolic suffix through the new sym_attn, and
            both readings modulate the pre-FFN RMSNorm additively.

        Returns:
            tuple[list[jax.Array | None], KVCache] -- per-expert outputs and
            the updated kv cache, same as HistoryBlock.

        Example input:
            block(xs, kv_cache, positions, attn_mask, adarms_cond, mem_seq, mem_mask)
            # mem_seq: [B, 512 + S_sym, 1024]

        Example output:
            ([None or Array (B, T_prefix, 2048), Array (B, T_suffix, 1024)], kv_cache)
        """
        assert self.integration_type == "modulation", "XFHistoryBlock only supports integration_type='modulation'"
        mem_attn = MemoryAttention(name="mem_attn")  # released params, warm-started
        sym_attn = MemoryAttention(name="sym_mem_attn")  # "mem" in name keeps it out of the LoRA freeze filter  # new, same architecture, fresh

        xs = sharding.activation_sharding_constraint(xs)
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        attn = Attention(configs=self.configs, name="attn")

        pre_attn = []  # list[jax.Array | None]
        gates = []  # list[jax.Array | None]
        for i, x in enumerate(xs):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
            pre_attn.append(x)
            gates.append(gate if x is not None else None)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        out = []  # list[jax.Array | None]
        gates = []  # list[jax.Array | None]
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                if i == len(xs) - 1:
                    mem = mem_seq[-1]  # [B, perc_len + S_sym, 1024]
                    mmask = mem_mask[-1]  # bool[B, perc_len + S_sym]
                    m_perc = mem_attn(x, mem[:, : self.perc_len], mmask[:, : self.perc_len])  # [B, T, 1024]
                    m_sym = sym_attn(x, mem[:, self.perc_len :], mmask[:, self.perc_len :])  # [B, T, 1024]
                    sym_mod = nn.Dense(
                        x.shape[-1] * 2, kernel_init=kernel_init_out_proj, dtype=x.dtype, name="sym_mem_mod_dense"
                    )(m_sym)  # [B, T, 2048], additive (scale, shift)
                    x = XFDualCondRMSNorm(name="mem_rms_norm_ffn")(x, m_perc, sym_mod)  # noqa: PLW2901

                x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
                x = lora.FeedForward(  # noqa: PLW2901
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=_name("mlp", i),
                    lora_config=config.lora_configs.get("ffn"),
                )(x)

            out.append(x)
            gates.append(gate if x is not None else None)

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        return xs, kv_cache


@at.typecheck
class XFModule(_ReleasedModule):
    """history_gemma.Module whose layer stack is built from XFHistoryBlock (symbolic route)."""

    perc_len: int = 512  # int, forwarded to every XFHistoryBlock

    def setup(self):
        """
        What it does:
            Same as history_gemma.Module.setup (embedder, nn.scan over depth
            with remat, final norms -- identical names, so every released
            param path is unchanged), except the scanned block class is
            XFHistoryBlock carrying `perc_len`. embed/__call__/init are
            inherited unchanged.

        Returns:
            None -- sets self.embedder, self.layers, self.final_norms.

        Example input:
            XFModule(configs=[paligemma_cfg, action_cfg], embed_dtype="bfloat16",
                     adarms=True, integration_type="modulation", perc_len=512)

        Example output:
            <XFModule ...> (a flax.linen Module)
        """
        assert all(config.depth == self.configs[0].depth for config in self.configs)
        assert self.integration_type == "modulation", "XFModule only supports integration_type='modulation'"
        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,  # embedder for first expert only
            name="embedder",
        )
        block_cls = nn.remat(
            XFHistoryBlock,
            prevent_cse=False,
            static_argnums=(7,),  # deterministic
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
            integration_type=self.integration_type,
            perc_len=self.perc_len,
        )
        self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]

    def init(self, use_adarms, mem_mods):
        """
        What it does:
            Same as history_gemma.Module.init (runs embed + one forward on
            dummy inputs to create every param), except the dummy memory is
            perc_len + 2 tokens long instead of 4, so both halves of the
            split -- the perceptual prefix AND the symbolic suffix -- are
            non-empty when the params are created.

        Returns:
            None -- side effect: params created under linen's init.

        Example input:
            llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True], mem_mods=[False, True])

        Example output:
            None
        """
        mem_len = self.perc_len + 2  # int
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[jnp.zeros((1, c.width)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)],
            mem_seq=[jnp.zeros((1, mem_len, c.width)) if m else None for c, m in zip(self.configs, mem_mods, strict=True)],
            mem_mask=[jnp.ones((1, mem_len), dtype=bool) if m else None for m in mem_mods],
        )
