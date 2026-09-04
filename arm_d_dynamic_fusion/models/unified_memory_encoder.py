"""
unified_memory_encoder.py

Fixes two problems found empirically in Arm D's two memory streams before
they reach joint_gated_modulator.JointGatedModulator (see RESEARCH_LOG.md's
2026-08-28 21:24 and 22:14 entries, from a real trained checkpoint on real
data): (1) a large and growing raw-magnitude mismatch between the streams
(perceptual tokens ended up ~15x larger in RMS norm than symbolic tokens
after training, up from ~9x at init), and (2) zero measured representation-
space alignment between a given example's two streams (matched-pair
retrieval sat exactly at the 1/batch_size chance floor, both before and
after training) -- likely contributing, alongside the still-unwired
JointGatedModulator.balance_loss (see arm_d_pi0.ArmDModel.compute_loss), to
the full gate collapse onto the perceptual stream also measured on that same
checkpoint (RESEARCH_LOG.md's 2026-08-28 22:14 entry).

UnifiedMemoryEncoder addresses (1) directly (parameter-free RMSNorm) and
provides the shared computational path for (2) -- the SAME module instance
is applied independently to M_sym and M_perc (see arm_d_pi0.ArmDModel.
embed_memory), forcing both through identical weights rather than each
keeping its own private post-projection transform forever. This module alone
does not guarantee alignment, though: only contrastive_alignment_loss below,
wired into ArmDModel.compute_loss's training objective, actually supplies a
gradient that rewards the two streams for landing close together per
example. Reuses SymbolicMemoryEncoder/PerceptualMemory's existing per-stream
projectors unchanged -- this sits strictly after them, not instead of them.
"""

import jax
import jax.numpy as jnp
import flax.nnx as nnx

import openpi.shared.array_typing as at
from mme_vla_suite.models.representation.utils import kernel_init_out_proj


class UnifiedMemoryEncoder(nnx.Module):
    """
    What it does:
        Applies a parameter-free RMSNorm (removes the raw-magnitude mismatch
        between streams) followed by a small residual 2-layer MLP, with a
        near-zero-initialized output projection so the MLP's own contribution
        starts near-zero -- the same "near-identity at init" convention this
        codebase uses elsewhere (joint_gated_modulator's router/mlp_sym/
        mlp_perc), applied here to the new MLP specifically. The RMSNorm step
        itself is a deliberate, necessary exception to that convention: fixing
        a real ~15x scale mismatch requires actually changing the numbers, so
        this module is NOT an exact identity at init, unlike every other new
        Arm D module. Called independently on M_sym and M_perc with the SAME
        instance (same weights) by arm_d_pi0.ArmDModel.embed_memory -- that
        sharing, not anything in this class's structure alone, is what gives
        both streams a chance to land in a common space; contrastive_
        alignment_loss below is what actually trains them to.

    Returns:
        n/a -- see __call__.

    Example input:
        UnifiedMemoryEncoder(rngs=nnx.Rngs(0), dtype=jnp.float32, width=1024, hidden_dim=1024)

    Example output:
        a callable module; see __call__ below.
    """

    def __init__(self, rngs: nnx.Rngs, dtype: at.DTypeLike, width: int, hidden_dim: int):
        self.dtype = dtype  # jax.numpy.dtype, compute dtype for the MLP
        self.dense_in = nnx.Linear(width, hidden_dim, rngs=rngs, dtype=dtype)  # nnx.Linear, width -> hidden_dim
        self.dense_out = nnx.Linear(
            hidden_dim, width, rngs=rngs, dtype=dtype, kernel_init=kernel_init_out_proj
        )  # nnx.Linear, hidden_dim -> width, near-zero-init (matches joint_gated_modulator's mlp_sym/mlp_perc convention)

    @at.typecheck
    def __call__(self, tokens: at.Float[at.Array, "b l d"]) -> at.Float[at.Array, "b l d"]:
        """
        What it does:
            Normalizes token magnitude (parameter-free RMSNorm over the last
            axis, computed in float32 for stability -- matching the
            convention already used in joint_gated_modulator.py and
            history_gemma.MemoryRMSNorm), then adds a small shared-weight
            residual MLP on top.

        Returns:
            jax.Array -- same shape and dtype as `tokens`, [b, l, d].

        Example input:
            encoder(jnp.ones((2, 64, 1024), dtype=jnp.bfloat16))

        Example output:
            Array of shape (2, 64, 1024), dtype bfloat16
        """
        variance = jnp.mean(jnp.square(tokens.astype(jnp.float32)), axis=-1, keepdims=True)  # jax.Array [b, l, 1], float32
        normed = (tokens.astype(jnp.float32) * jax.lax.rsqrt(variance + 1e-6)).astype(self.dtype)  # jax.Array [b, l, d]

        hidden = nnx.gelu(self.dense_in(normed))  # jax.Array [b, l, hidden_dim]
        residual = self.dense_out(hidden)  # jax.Array [b, l, d], near-zero at init
        return normed + residual


@at.typecheck
def contrastive_alignment_loss(
    mem_sym: at.Float[at.Array, "b l_sym d"],
    mem_sym_mask: at.Bool[at.Array, "b l_sym"],
    mem_perc: at.Float[at.Array, "b l_perc d"],
    mem_perc_mask: at.Bool[at.Array, "b l_perc"],
    temperature: float = 0.07,
) -> at.Float[at.Array, ""]:
    """
    What it does:
        Symmetric InfoNCE / CLIP-style contrastive loss between the two
        streams' per-example mean-pooled summaries: rewards a given batch
        example's symbolic summary for being closest (by cosine similarity)
        to its OWN perceptual summary, among every other example's
        perceptual summary in the same batch, and vice versa. This is the
        differentiable, training-time counterpart of the matched-vs-shuffled
        retrieval check in arm_d_dynamic_fusion/analysis/measure_
        representation_alignment.py, which measured this exact quantity on
        the pre-existing checkpoint (matched cosine similarity
        indistinguishable from shuffled, retrieval accuracy at chance -- see
        RESEARCH_LOG.md's 2026-08-28 21:24 entry) and found nothing was
        training it. Pairing is at the batch-example level (both streams
        come from the same training example/timestep), not at the individual
        token level -- there is no ground truth for which specific symbolic
        token corresponds to which specific perceptual token.

    Returns:
        jax.Array -- scalar float32. Not floored at 0: a batch with no
        alignment at all gives approximately log(batch_size), not 0 -- lower
        is better, not "close to zero is good".

    Example input:
        contrastive_alignment_loss(
            mem_sym=jnp.zeros((4, 64, 1024)), mem_sym_mask=jnp.ones((4, 64), dtype=bool),
            mem_perc=jnp.zeros((4, 512, 1024)), mem_perc_mask=jnp.ones((4, 512), dtype=bool),
        )

    Example output:
        Array(1.3862944, dtype=float32)
    """
    sym_mask_f = mem_sym_mask.astype(jnp.float32)[..., None]  # jax.Array [b, l_sym, 1]
    sym_pooled = jnp.sum(mem_sym.astype(jnp.float32) * sym_mask_f, axis=1) / jnp.clip(
        jnp.sum(sym_mask_f, axis=1), 1.0
    )  # jax.Array [b, d]
    perc_mask_f = mem_perc_mask.astype(jnp.float32)[..., None]  # jax.Array [b, l_perc, 1]
    perc_pooled = jnp.sum(mem_perc.astype(jnp.float32) * perc_mask_f, axis=1) / jnp.clip(
        jnp.sum(perc_mask_f, axis=1), 1.0
    )  # jax.Array [b, d]

    sym_normed = sym_pooled / (jnp.linalg.norm(sym_pooled, axis=-1, keepdims=True) + 1e-8)  # jax.Array [b, d]
    perc_normed = perc_pooled / (jnp.linalg.norm(perc_pooled, axis=-1, keepdims=True) + 1e-8)  # jax.Array [b, d]

    logits = (sym_normed @ perc_normed.T) / temperature  # jax.Array [b, b], float32
    batch_size = logits.shape[0]  # int
    diag_idx = jnp.arange(batch_size)  # jax.Array [b]

    log_probs_sym_to_perc = jax.nn.log_softmax(logits, axis=-1)  # jax.Array [b, b]
    log_probs_perc_to_sym = jax.nn.log_softmax(logits.T, axis=-1)  # jax.Array [b, b]
    loss_sym_to_perc = -jnp.mean(log_probs_sym_to_perc[diag_idx, diag_idx])  # jax.Array []
    loss_perc_to_sym = -jnp.mean(log_probs_perc_to_sym[diag_idx, diag_idx])  # jax.Array []
    return 0.5 * (loss_sym_to_perc + loss_perc_to_sym)
