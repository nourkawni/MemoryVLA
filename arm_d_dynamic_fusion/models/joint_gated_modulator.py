"""
joint_gated_modulator.py

The core new mechanism for Arm D (cross_modal_gated_fusion_proposal.md, section
3.3, "F3" fusion position). Implements a per-layer softmax router that observes
BOTH memory streams (symbolic subgoals, perceptual visual tokens) at once and
uses that joint view to produce the AdaLN scale/shift that conditions one layer
of the pi0.5 action expert.

This is the piece that differs from every other MME-VLA variant in the RoboMME
paper: the released memory-as-modulator design (history_gemma.MemoryAttention +
MemoryRMSNorm, see robomme_policy_learning) computes a gate/modulation from a
single memory stream. Here, two streams are cross-attended independently
(r_sym, r_perc), and a small router observes both results together before
deciding how much of each to use -- so the gate can express "the symbolic
subgoal is contradicted by what I see," not just "history vs. current frame."

Role in the system: instantiated once per action-expert layer inside
history_gemma_dual.DualMemoryHistoryBlock (which is scanned over depth, so
each layer gets its own independently-learned router/MLP parameters, matching
proposal Eq. "g_k = softmax(W_route . [r_sym; r_perc])" being layer-indexed by
k). Reuses history_gemma.MemoryAttention unchanged for the two per-stream
cross-attentions, since that block's RoPE/multi-head-attention math is already
part of the working released implementation and duplicating it here would only
add risk without adding anything new -- the only genuinely new part is the
router and the dual-stream combination.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from mme_vla_suite.models.integration.history_gemma import MemoryAttention
from mme_vla_suite.models.representation.utils import kernel_init_out_proj


class JointGatedModulator(nn.Module):
    """
    What it does:
        Cross-attends the action-expert features into the symbolic stream and
        into the perceptual stream independently (r_sym, r_perc), routes a
        joint 2-way softmax gate from the concatenation of both results, and
        combines each stream's AdaLN-Zero-style (scale, shift) proposal by
        that gate to modulate the (RMS-normalized) action features. Also
        returns the per-stream gate values and a load-balancing auxiliary
        loss term for the training objective.

    Returns:
        n/a -- see __call__.

    Example input:
        JointGatedModulator(name="joint_gated_modulator")

    Example output:
        a callable module; see __call__ below.
    """

    @nn.compact
    def __call__(
        self,
        x,           # jax.Array [b, t, d] -- action-expert features before the FFN sublayer
        mem_sym,     # jax.Array [b, s_sym, d] -- symbolic memory tokens (M_sym)
        mem_sym_mask,   # jax.Array [b, s_sym] -- True where mem_sym is a real (non-padding) token
        mem_perc,    # jax.Array [b, s_perc, d] -- perceptual memory tokens (M_perc)
        mem_perc_mask,  # jax.Array [b, s_perc] -- True where mem_perc is a real (non-padding) token
    ):
        """
        What it does:
            Runs the F3 fusion mechanism (proposal section 3.3) for one
            action-expert layer: r_sym/r_perc via cross-attention, a joint
            softmax router g in R^2, combined AdaLN-Zero scale/shift, then
            RMS-normalizes x and applies the combined modulation.

        Returns:
            tuple[jax.Array, dict] -- (modulated_x, stats). modulated_x has
            the same shape as x, [b, t, d]. stats is a dict with keys
            "gate_sym" [b, t, 1], "gate_perc" [b, t, 1], and "balance_loss"
            (scalar float32) -- the load-balancing auxiliary term, floored at
            0.0 when the two streams' average gate mass is exactly 0.5/0.5.

        Example input:
            modulator(x=jnp.zeros((2, 20, 1024)),
                      mem_sym=jnp.zeros((2, 64, 1024)), mem_sym_mask=jnp.ones((2, 64), dtype=bool),
                      mem_perc=jnp.zeros((2, 512, 1024)), mem_perc_mask=jnp.ones((2, 512), dtype=bool))

        Example output:
            (Array of shape (2, 20, 1024), {"gate_sym": ..., "gate_perc": ..., "balance_loss": Array(0.0)})
        """
        dtype = x.dtype  # jax numpy dtype, e.g. bfloat16 during training
        width = x.shape[-1]  # int, action expert width (1024 for gemma_300m)

        # Cross-attend action features into each stream independently. Reused
        # unchanged from the released single-stream modulator (see module
        # docstring) -- each instance gets its own params via a distinct name.
        r_sym = MemoryAttention(name="mem_attn_sym")(x, mem_sym, mem_sym_mask)  # jax.Array [b, t, d]
        r_perc = MemoryAttention(name="mem_attn_perc")(x, mem_perc, mem_perc_mask)  # jax.Array [b, t, d]

        # Joint router: observes both streams' cross-attended results at once,
        # so it can express suppression ("symbolic contradicted by perceptual")
        # rather than each stream only ever asking "history or current frame?"
        # Zero-init (kernel AND bias) so g is exactly uniform (0.5, 0.5) at the
        # start of fine-tuning, per proposal section 3.3 ("W_route is
        # initialized to yield uniform g"). Router logits/softmax computed in
        # float32 for stability, matching how attention logits are computed
        # in float32 elsewhere in this codebase (history_gemma.MemoryAttention).
        router_input = jnp.concatenate([r_sym, r_perc], axis=-1)  # jax.Array [b, t, 2*d]
        router_logits = nn.Dense(
            2,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            dtype=jnp.float32,
            name="router",
        )(router_input)  # jax.Array [b, t, 2], float32
        gate = jax.nn.softmax(router_logits, axis=-1).astype(dtype)  # jax.Array [b, t, 2]
        gate_sym = gate[..., 0:1]  # jax.Array [b, t, 1]
        gate_perc = gate[..., 1:2]  # jax.Array [b, t, 1]

        # Per-stream AdaLN-Zero-style modulation proposals. kernel_init_out_proj
        # (small non-zero noise, not exact zero) matches the existing
        # MemoryRMSNorm convention in history_gemma.py -- "add small
        # randomness instead of pure zeros" so the two streams' MLPs don't
        # start perfectly tied and can differentiate during training.
        mod_sym = nn.Dense(
            width * 2, kernel_init=kernel_init_out_proj, dtype=dtype, name="mlp_sym"
        )(r_sym)  # jax.Array [b, t, 2*d]
        mod_perc = nn.Dense(
            width * 2, kernel_init=kernel_init_out_proj, dtype=dtype, name="mlp_perc"
        )(r_perc)  # jax.Array [b, t, 2*d]
        scale_sym, shift_sym = jnp.split(mod_sym, 2, axis=-1)  # each jax.Array [b, t, d]
        scale_perc, shift_perc = jnp.split(mod_perc, 2, axis=-1)  # each jax.Array [b, t, d]

        # (gamma_k, beta_k) = g_sym . MLP_sym(r_sym) + g_perc . MLP_perc(r_perc)
        # Both MLPs are zero-init, so scale/shift are exactly zero at init
        # regardless of the (already-uniform) gate values -- the (1+scale)
        # convention below then gives an identity modulation at the start of
        # fine-tuning, matching AdaLN-Zero and this codebase's existing
        # MemoryRMSNorm.
        scale = gate_sym * scale_sym + gate_perc * scale_perc  # jax.Array [b, t, d]
        shift = gate_sym * shift_sym + gate_perc * shift_perc  # jax.Array [b, t, d]

        # s_hat = gamma_k (dot) Norm(s_tilde_k) + beta_k -- plain RMSNorm (no
        # extra learnable per-channel scale), matching the cond-branch of
        # history_gemma.MemoryRMSNorm exactly.
        variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)  # jax.Array [b, t, 1], float32
        normed_x = (x.astype(jnp.float32) * jax.lax.rsqrt(variance + 1e-6)).astype(dtype)  # jax.Array [b, t, d]
        modulated_x = normed_x * (1 + scale) + shift  # jax.Array [b, t, d]

        # Load-balancing auxiliary loss against modality collapse (proposal
        # section 3.3, citing Fedus et al. 2022). This router is dense (both
        # streams always contribute -- no top-1 sparse dispatch), so the
        # Switch-Transformer loss (which weights a dispatch fraction by a
        # router probability) doesn't directly apply. This uses the closest
        # applicable form for a 2-way dense gate -- the importance/load loss
        # from Shazeer et al. 2017 -- penalizing the squared batch-and-token
        # mean gate mass per stream, floored at 0 (its analytic minimum,
        # reached when both streams average exactly 0.5 gate mass).
        mean_gate = jnp.mean(gate.astype(jnp.float32), axis=(0, 1))  # jax.Array [2], float32
        num_streams = 2  # int, fixed: symbolic + perceptual
        balance_loss = num_streams * jnp.sum(jnp.square(mean_gate)) - 1.0  # jax.Array [], float32

        stats = {
            "gate_sym": gate_sym,
            "gate_perc": gate_perc,
            "balance_loss": balance_loss,
        }  # dict[str, jax.Array]
        return modulated_x, stats
