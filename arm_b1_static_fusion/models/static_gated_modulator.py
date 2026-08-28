"""
static_gated_modulator.py

The core mechanism for Arm B1 (cross_modal_gated_fusion_proposal.md, section 4.1
arms table: "B1 | g = [0.5, 0.5], fixed | Static fusion"). Structurally identical
to arm_d_dynamic_fusion.models.joint_gated_modulator.JointGatedModulator EXCEPT
the gate itself: instead of a learned softmax router that looks at both memory
streams and decides how much to trust each one, B1's gate is a hard-coded 50/50
split that never changes, no matter what the two streams contain.

Why this arm exists: Arm D's whole claim is that a gate which sees BOTH streams
at once and adjusts per-input beats a fixed compromise. B1 is the control that
makes that claim testable -- if D doesn't beat B1, the learned router isn't
adding anything over just averaging the two streams.

Role in the system: instantiated once per action-expert layer inside
history_gemma_static.StaticMemoryHistoryBlock (scanned over depth, same as
Arm D's DualMemoryHistoryBlock), the fixed drop-in replacement for
JointGatedModulator wherever integration_type == "static_fusion". Reuses
history_gemma.MemoryAttention unchanged for the two per-stream cross-
attentions, same as JointGatedModulator does, for the same reason (that
block's RoPE/attention math is already working and this arm isn't testing it).
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from mme_vla_suite.models.integration.history_gemma import MemoryAttention
from mme_vla_suite.models.representation.utils import kernel_init_out_proj


class StaticGatedModulator(nn.Module):
    """
    What it does:
        Cross-attends the action-expert features into the symbolic stream and
        into the perceptual stream independently (r_sym, r_perc), exactly
        like JointGatedModulator -- but then combines each stream's AdaLN-
        Zero-style (scale, shift) proposal with a FIXED 0.5/0.5 weighting
        instead of a learned, input-dependent gate. No router network exists
        in this module at all: gate_sym and gate_perc are the Python literal
        0.5, not a Dense layer's output.

    Returns:
        n/a -- see __call__.

    Example input:
        StaticGatedModulator(name="static_gated_modulator")

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
            Runs the B1 fusion mechanism for one action-expert layer: r_sym/
            r_perc via cross-attention (same as Arm D), then a FIXED 50/50
            AdaLN-Zero scale/shift combine (no router, unlike Arm D), then
            RMS-normalizes x and applies the combined modulation.

        Returns:
            tuple[jax.Array, dict] -- (modulated_x, stats). modulated_x has
            the same shape as x, [b, t, d]. stats is a dict with keys
            "gate_sym" [b, t, 1] and "gate_perc" [b, t, 1] -- both the
            constant 0.5, broadcast to shape, kept for interface parity with
            JointGatedModulator's stats dict so training/logging code that
            reads gate_sym/gate_perc works unmodified against either arm.
            No "balance_loss" key: a load-balancing auxiliary loss exists to
            fight modality collapse (one stream's gate drifting to 0), which
            cannot happen here since the gate is fixed at 0.5/0.5 by
            construction -- there is nothing for that loss to defend against.

        Example input:
            modulator(x=jnp.zeros((2, 20, 1024)),
                      mem_sym=jnp.zeros((2, 64, 1024)), mem_sym_mask=jnp.ones((2, 64), dtype=bool),
                      mem_perc=jnp.zeros((2, 512, 1024)), mem_perc_mask=jnp.ones((2, 512), dtype=bool))

        Example output:
            (Array of shape (2, 20, 1024), {"gate_sym": <all 0.5>, "gate_perc": <all 0.5>})
        """
        dtype = x.dtype  # jax numpy dtype, e.g. bfloat16 during training
        width = x.shape[-1]  # int, action expert width (1024 for gemma_300m)

        # Cross-attend action features into each stream independently.
        # Identical call to JointGatedModulator's -- this part of the
        # mechanism is not what B1 vs. D isolates.
        r_sym = MemoryAttention(name="mem_attn_sym")(x, mem_sym, mem_sym_mask)  # jax.Array [b, t, d]
        r_perc = MemoryAttention(name="mem_attn_perc")(x, mem_perc, mem_perc_mask)  # jax.Array [b, t, d]

        # Fixed gate: the proposal's "g = [0.5, 0.5], fixed" (section 4.1).
        # No nn.Dense, no learned parameters -- gate_sym/gate_perc are the
        # Python float 0.5, broadcast against r_sym's leading dims below.
        gate_sym = jnp.full(r_sym.shape[:-1] + (1,), 0.5, dtype=dtype)  # jax.Array [b, t, 1]
        gate_perc = jnp.full(r_perc.shape[:-1] + (1,), 0.5, dtype=dtype)  # jax.Array [b, t, 1]

        # Per-stream AdaLN-Zero-style modulation proposals -- identical to
        # JointGatedModulator's mlp_sym/mlp_perc, same near-zero init
        # convention (kernel_init_out_proj) so the two streams' MLPs start
        # differentiated rather than perfectly tied.
        mod_sym = nn.Dense(
            width * 2, kernel_init=kernel_init_out_proj, dtype=dtype, name="mlp_sym"
        )(r_sym)  # jax.Array [b, t, 2*d]
        mod_perc = nn.Dense(
            width * 2, kernel_init=kernel_init_out_proj, dtype=dtype, name="mlp_perc"
        )(r_perc)  # jax.Array [b, t, 2*d]
        scale_sym, shift_sym = jnp.split(mod_sym, 2, axis=-1)  # each jax.Array [b, t, d]
        scale_perc, shift_perc = jnp.split(mod_perc, 2, axis=-1)  # each jax.Array [b, t, d]

        # (gamma, beta) = 0.5 . MLP_sym(r_sym) + 0.5 . MLP_perc(r_perc) --
        # same combine formula as Arm D's proposal equation, with g replaced
        # by the constant 0.5 instead of a learned per-token value. Both
        # MLPs are near-zero-init, so scale/shift are ~zero at init
        # regardless of the (already-fixed) gate values -- the (1+scale)
        # convention below then gives a near-identity modulation at the
        # start of fine-tuning, matching AdaLN-Zero.
        scale = gate_sym * scale_sym + gate_perc * scale_perc  # jax.Array [b, t, d]
        shift = gate_sym * shift_sym + gate_perc * shift_perc  # jax.Array [b, t, d]

        # s_hat = gamma (dot) Norm(s_tilde) + beta -- plain RMSNorm (no extra
        # learnable per-channel scale), matching JointGatedModulator exactly.
        variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)  # jax.Array [b, t, 1], float32
        normed_x = (x.astype(jnp.float32) * jax.lax.rsqrt(variance + 1e-6)).astype(dtype)  # jax.Array [b, t, d]
        modulated_x = normed_x * (1 + scale) + shift  # jax.Array [b, t, d]

        stats = {"gate_sym": gate_sym, "gate_perc": gate_perc}  # dict[str, jax.Array]
        return modulated_x, stats
