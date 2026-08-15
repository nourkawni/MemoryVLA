"""
history_gemma_dual.py

Forks mme_vla_suite.models.integration.history_gemma's HistoryBlock/Module to
carry TWO memory streams (symbolic + perceptual) through the action expert
instead of the released code's single stream, and to replace the single
MemoryAttention + MemoryRMSNorm modulation call with
joint_gated_modulator.JointGatedModulator when integration_type ==
"dynamic_fusion".

robomme_policy_learning/ is never edited -- everything outside the memory-
modulation site (self-attention, RoPE, the per-timestep AdaLN-Zero FFN
sublayer, the nn.scan-over-depth plumbing) is copied unchanged from
history_gemma.py, since that machinery is already working and not what Arm D
is testing. The only structural difference in the scan wiring is that the
single (mem_seq, mem_mask) pair of broadcast arguments becomes two pairs
((mem_seq_sym, mem_mask_sym), (mem_seq_perc, mem_mask_perc)) -- both still
plain nn.broadcast arguments, exactly like the original's mem_seq/mem_mask,
so the parts of the scan this file does NOT touch (how xs and kv_cache flow
through nn.scan) are unaffected by this change.
"""

from collections.abc import Sequence
from typing import TypeAlias

import flax.linen as nn
import jax
import jax.numpy as jnp

from openpi.models.gemma import PALIGEMMA_VOCAB_SIZE
from openpi.models.gemma import Config
from openpi.models.gemma import Embedder
from openpi.models.gemma import RMSNorm
from openpi.models.gemma import _gated_residual
from openpi.models.gemma import Attention
import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

from mme_vla_suite.models.integration.utils import _name

from arm_d_dynamic_fusion.models.joint_gated_modulator import JointGatedModulator


@at.typecheck
class DualMemoryHistoryBlock(nn.Module):
    """
    What it does:
        One transformer layer of the multi-expert stack (VLM expert + action
        expert), identical to history_gemma.HistoryBlock except that when
        integration_type == "dynamic_fusion", the action expert's memory
        modulation is computed jointly from two memory streams via
        JointGatedModulator instead of from a single stream.

    Returns:
        n/a -- see __call__.

    Example input:
        DualMemoryHistoryBlock(configs=(vlm_config, action_config),
                                integration_type="dynamic_fusion")

    Example output:
        a callable module; see __call__ below.
    """

    configs: tuple[Config, ...]  # tuple[Config, ...], one per expert (VLM, action)

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    integration_type: str | None = None  # str | None, "dynamic_fusion" for Arm D

    @nn.compact
    def __call__(
        self,
        xs,             # Sequence[jax.Array | None], one per expert
        kv_cache,       # KVCache | None
        positions,      # jax.Array [b, t]
        attn_mask,      # jax.Array [b, 1, t, s]
        adarms_cond,    # Sequence[jax.Array | None], per-expert AdaLN-Zero conditioning
        mem_seq_sym,    # Sequence[jax.Array | None], per-expert symbolic memory tokens
        mem_mask_sym,   # Sequence[jax.Array | None], per-expert symbolic memory mask
        mem_seq_perc,   # Sequence[jax.Array | None], per-expert perceptual memory tokens
        mem_mask_perc,  # Sequence[jax.Array | None], per-expert perceptual memory mask
        deterministic=True,  # bool, disables dropout when True
    ):
        if self.integration_type == "dynamic_fusion":
            joint_modulator = JointGatedModulator(name="joint_gated_modulator")

        xs = sharding.activation_sharding_constraint(xs)
        drop = (
            nn.Dropout(self.dropout, self.dropout_bdims)
            if self.dropout
            else lambda x, _: x
        )

        attn = Attention(configs=self.configs, name="attn")

        pre_attn = []
        gates = []
        for i, x in enumerate(xs):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_attention_norm", i))(
                    x, adarms_cond[i]
                )  # noqa: PLW2901
            pre_attn.append(x)
            gates.append(gate if x is not None else None)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = [
            _gated_residual(x, y, gate)
            for x, y, gate in zip(xs, post_attn, gates, strict=True)
        ]
        xs = sharding.activation_sharding_constraint(xs)

        out = []
        gates = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                # Joint cross-modal memory modulation, action expert only
                # (last slot in xs), mirroring where history_gemma.py's
                # single-stream modulation is inserted.
                if i == len(xs) - 1 and self.integration_type == "dynamic_fusion":
                    x, gate_stats = joint_modulator(  # noqa: PLW2901
                        x,
                        mem_seq_sym[-1], mem_mask_sym[-1],
                        mem_seq_perc[-1], mem_mask_perc[-1],
                    )
                    # Collected per layer via nn.scan's "intermediates" axis
                    # (see DualMemoryModule.setup) -- the standard flax idiom
                    # for reading out per-scan-step diagnostics without
                    # changing this block's carry/output structure.
                    self.sow("intermediates", "gate_sym", gate_stats["gate_sym"])
                    self.sow("intermediates", "gate_perc", gate_stats["gate_perc"])
                    self.sow("intermediates", "balance_loss", gate_stats["balance_loss"])

                x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(
                    x, adarms_cond[i]
                )  # noqa: PLW2901
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
        xs = [
            _gated_residual(x, y, gate)
            for x, y, gate in zip(xs, out, gates, strict=True)
        ]
        xs = sharding.activation_sharding_constraint(xs)

        return xs, kv_cache


KVCache: TypeAlias = tuple[
    at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]
]


@at.typecheck
class DualMemoryModule(nn.Module):
    """
    What it does:
        Transformer model over the VLM + action expert stack, scanned over
        depth, supporting Arm D's two simultaneous memory streams. Identical
        to history_gemma.Module except __call__ takes mem_seq_sym/mem_mask_sym
        and mem_seq_perc/mem_mask_perc instead of a single mem_seq/mem_mask,
        and the scan collects each layer's gate/balance-loss diagnostics via
        a `variable_axes={"intermediates": 0}` collection.

    Returns:
        n/a -- see __call__.

    Example input:
        DualMemoryModule(configs=(vlm_config, action_config), embed_dtype="bfloat16",
                          adarms=True, integration_type="dynamic_fusion")

    Example output:
        a callable module; see __call__ below.
    """

    configs: Sequence[Config]  # Sequence[Config], one per expert
    embed_dtype: str  # str, e.g. "bfloat16"

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    adarms: bool = False

    integration_type: str | None = None  # str | None, "dynamic_fusion" for Arm D

    def setup(self):
        assert all(config.depth == self.configs[0].depth for config in self.configs)
        embed_dim = self.configs[0].width  # int, VLM expert width
        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=embed_dim,
            name="embedder",
        )
        block_cls = nn.remat(
            DualMemoryHistoryBlock,
            prevent_cse=False,
            # deterministic is the 10th positional arg to __call__ (index 9,
            # after self): xs=0, kv_cache=1, positions=2, attn_mask=3,
            # adarms_cond=4, mem_seq_sym=5, mem_mask_sym=6, mem_seq_perc=7,
            # mem_mask_perc=8, deterministic=9.
            static_argnums=(9,),
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0, "intermediates": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(
                0,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
            ),  # xs, kv_cache, positions, mask, adarms_cond, mem_seq_sym, mem_mask_sym, mem_seq_perc, mem_mask_perc
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
            integration_type=self.integration_type,
        )
        self.final_norms = [
            RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))
        ]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    @at.typecheck
    def __call__(
        self,
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None = None,
        *,
        kv_cache: KVCache | None = None,
        # NOTE: mem_seq_sym and mem_seq_perc deliberately use different
        # jaxtyping axis names (lmem_sym vs lmem_perc, not a shared "lmem")
        # -- jaxtyping's @at.typecheck binds an axis name to one concrete
        # size for the whole call, and the two streams have different
        # budgets (~64 vs 512), so a shared name would wrongly reject valid
        # inputs whenever the two lengths differ.
        mem_seq_sym: Sequence[at.Float[at.Array, "b lmem_sym _d"] | None] | None = None,
        mem_mask_sym: Sequence[at.Bool[at.Array, "b lmem_sym"] | None] | None = None,
        mem_seq_perc: Sequence[at.Float[at.Array, "b lmem_perc _d"] | None] | None = None,
        mem_mask_perc: Sequence[at.Bool[at.Array, "b lmem_perc"] | None] | None = None,
        deterministic: bool = True,
    ):
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        embedded, kv_cache = self.layers(
            embedded,
            kv_cache,
            positions,
            mask,
            adarms_cond,
            mem_seq_sym,
            mem_mask_sym,
            mem_seq_perc,
            mem_mask_perc,
            deterministic,
        )

        assert all(
            e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None
        )

        return [
            f(e, a)[0] if e is not None else e
            for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ], kv_cache

    def init(self, use_adarms: Sequence[bool], mem_mods: Sequence[bool]):
        """Convenience method for initializing all parameters, necessary due to the quirks of linen."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[
                jnp.zeros((1, c.width)) if u else None
                for u, c in zip(use_adarms, self.configs, strict=True)
            ],
            mem_seq_sym=[
                jnp.zeros((1, 4, c.width)) if m else None
                for c, m in zip(self.configs, mem_mods, strict=True)
            ],
            mem_mask_sym=[jnp.ones((1, 4), dtype=bool) if m else None for m in mem_mods],
            mem_seq_perc=[
                jnp.zeros((1, 4, c.width)) if m else None
                for c, m in zip(self.configs, mem_mods, strict=True)
            ],
            mem_mask_perc=[jnp.ones((1, 4), dtype=bool) if m else None for m in mem_mods],
        )
