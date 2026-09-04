"""
joint_gated_modulator.py

The core fusion mechanism for Arm D (cross_modal_gated_fusion_proposal.md, section
3.3), REVISED 2026-08-29 per the user's supervisor's explicit direction: fuse the two
memory streams into ONE representation BEFORE cross-attention, with a single
cross-attention reading that fused memory -- not, as this file originally
implemented, two independent per-stream cross-attentions combined by a router
AFTER attention. See RESEARCH_LOG.md's 2026-08-29 entries for the discussion
and reasoning behind this change (in particular: why the two memory streams
needing to be in a genuinely comparable representation space,
arm_d_dynamic_fusion/models/unified_memory_encoder.py, is a precondition for
this design being meaningful at all -- concatenating two streams only makes
sense once every token, regardless of stream, is measured in the same units).

Mechanism, per action-expert layer:
    M_sym_tagged  = M_sym  + tag_sym    (a learned "I am symbolic" vector, broadcast)
    M_perc_tagged = M_perc + tag_perc   (a learned "I am perceptual" vector, broadcast)
    M_fused       = concat([M_sym_tagged, M_perc_tagged], axis=tokens)   -- ONE sequence
    r_fused       = FusedMemoryAttention(x, M_fused, M_fused_mask)       -- ONE cross-attention
    (scale, shift) = MLP(r_fused)                                       -- ONE modulation MLP
    s_hat         = (1 + scale) * Norm(x) + shift

This replaces the released single-stream design's one MemoryAttention + one
modulation MLP with the SAME shape (one attention, one MLP) -- the only
difference from the single-stream released code is that the memory being
attended to is now a concatenation of two (now-comparable) streams instead of
one, plus the two additions described below that specifically address the
64-token-symbolic-vs-512-token-perceptual imbalance:

1. Modality tags (tag_sym/tag_perc, learned per-layer vectors added before
   concatenation): let attention use "which stream is this token from" as a
   content signal, not just each token's own features. Needed because after
   concatenation, a token's position in the sequence no longer implies
   anything on its own (attention has no innate sense of "the first 64
   positions are one kind of thing") -- the tag is what makes stream identity
   available to attention at all.
2. A learned per-stream score bias (bias_sym/bias_perc, two scalars per
   layer), added to every symbolic/perceptual token's raw attention score
   before the softmax that combines all 576 tokens. Necessary because a
   *plain* single softmax over 64 symbolic + 512 perceptual tokens has a
   structural bias toward whichever stream has more tokens: if the model
   can't yet distinguish relevant from irrelevant content (e.g. early in
   training), softmax tends toward assigning roughly equal weight per TOKEN,
   which means perceptual's 512 tokens would claim ~89% of the total
   attention mass (512/576) purely from being more numerous, independent of
   actual relevance. Both biases start at exactly 0 (no correction at all,
   pure content-driven attention) and are free to move if training finds the
   count imbalance is actually distorting results -- see this module's
   FusedMemoryAttention docstring for the exact mechanics.

Removed from the prior design: the two-stream router (`g_k =
softmax(W_route . [r_sym; r_perc])`) and the two separate per-stream
MemoryAttention/MLP pairs it combined. There is no longer a single scalar
"gate_sym"/"gate_perc" value to log -- "which stream matters" is now decided
per-token, per-head, per-layer, inside the one softmax, not as a separate
2-way decision afterward. FusedMemoryAttention instead sows the REALIZED
attention mass landing on symbolic vs. perceptual positions (attn_mass_sym/
attn_mass_perc) as a read-only diagnostic -- a description of what actually
happened on a given forward pass, not a trained decision variable the way
gate_sym/gate_perc was.

Also removed: JointGatedModulator's balance_loss (the load-balancing
auxiliary loss against modality collapse, Fedus et al. 2022). It penalized a
two-way gate specifically for collapsing to one side; there is no longer a
two-way gate for it to apply to. If a new form of collapse (e.g. attn_mass_sym
staying pinned near 0 despite the bias terms being free to move) shows up
empirically once this is trained, that would be the point to design a new
anti-collapse mechanism suited to THIS architecture -- not before there's
evidence it's needed.

Role in the system: EarlyFusionModulator is instantiated once per action-
expert layer inside history_gemma_dual.DualMemoryHistoryBlock (scanned over
depth, so each layer gets its own independently-learned tags/bias/attention/
MLP parameters), under the attribute name "joint_gated_modulator" -- kept
unchanged from the prior design specifically so ArmDConfig.get_freeze_filter's
`.*joint_gated_modulator.*` exemption and any future warm-start key renaming
keep matching without needing their own edits, even though the mechanism
underneath that name changed completely.
"""

import einops
import jax
import jax.numpy as jnp
import flax.linen as nn

from openpi.models.gemma import _apply_rope
import openpi.models.lora as lora
from mme_vla_suite.models.integration.history_gemma import MemoryRMSNorm
from mme_vla_suite.models.representation.utils import kernel_init_out_proj


class FusedMemoryAttention(nn.Module):
    """
    What it does:
        Cross-attention from the action-expert features into ONE combined
        memory sequence (both streams concatenated), forked from
        mme_vla_suite.models.integration.history_gemma.MemoryAttention (the
        released single-stream cross-attention) with one addition: a learned
        per-stream additive bias on the raw attention scores, applied before
        the single softmax over all combined memory tokens. Forked rather
        than reused unchanged (unlike this project's usual preference, see
        module docstring) because the released MemoryAttention has no hook
        for this -- adding it requires touching the score computation itself,
        which is exactly the "genuinely new" case this project's own stated
        convention (see prior versions of this file / symbolic_mem_encoder.py)
        already treats as justifying a fork rather than a silent edit to
        released code.

    Returns:
        n/a -- see __call__.

    Example input:
        FusedMemoryAttention(name="mem_attn_fused")

    Example output:
        a callable module; see __call__ below.
    """

    @nn.compact
    def __call__(
        self,
        x,           # jax.Array [b, t, d] -- action-expert features before the FFN sublayer
        mem_seq,     # jax.Array [b, s, d] -- the CONCATENATED memory (both streams, tagged)
        mem_mask,    # jax.Array [b, s] -- True where mem_seq is a real (non-padding) token
        is_sym,      # jax.Array [b, s], bool -- True where that position in mem_seq came from the symbolic stream
    ):
        """
        What it does:
            Identical attention math to the released MemoryAttention (same
            hardcoded num_heads/num_kv_heads/head_dim/width -- "same dim as
            the action expert in pi05", same RoPE positioning convention,
            same einsum shapes), with one inserted step: after computing raw
            attention logits and before masking/softmax, adds a learned
            scalar bias to every symbolic-stream position's logit and a
            (different) learned scalar bias to every perceptual-stream
            position's logit -- both start at exactly 0 (no correction),
            free to move during training. Also computes, as a read-only
            diagnostic (not fed back into the loss), how much of the
            resulting attention mass landed on symbolic vs. perceptual
            positions.

        Returns:
            tuple[jax.Array, jax.Array, jax.Array] -- (encoded, attn_mass_sym,
            attn_mass_perc). encoded has shape [b, t, d] (same as x).
            attn_mass_sym/attn_mass_perc have shape [b, t, 1] each (one value
            per query position per example, averaged over attention heads),
            and sum to 1 elementwise (every query's attention necessarily
            lands somewhere across the combined memory).

        Example input:
            attn(x=jnp.zeros((2, 20, 1024)),
                 mem_seq=jnp.zeros((2, 576, 1024)), mem_mask=jnp.ones((2, 576), dtype=bool),
                 is_sym=jnp.concatenate([jnp.ones((2, 64), dtype=bool), jnp.zeros((2, 512), dtype=bool)], axis=1))

        Example output:
            (Array of shape (2, 20, 1024), Array of shape (2, 20, 1), Array of shape (2, 20, 1))
        """
        b, mem_len, mem_width = mem_seq.shape  # int, int, int
        b, x_len, x_width = x.shape  # int, int, int
        num_heads, num_kv_heads, head_dim, width = (4, 1, 256, 1024)  # int, int, int, int -- same dim as the action expert in pi05, matches the released MemoryAttention exactly
        assert mem_width == x_width == width

        q_einsum = lora.Einsum(
            shape=(num_heads, width, head_dim), name="q_einsum_mem",
            init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
        )
        kv_einsum = lora.Einsum(
            shape=(2, num_kv_heads, width, head_dim), name="kv_einsum_mem",
            init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
        )
        rms_norm = MemoryRMSNorm(name="mem_rms_norm")
        x_normed = rms_norm(x)  # jax.Array [b, t, d]
        q = q_einsum("BTD,NDH->BTNH", x_normed)  # jax.Array [b, t, num_heads, head_dim]

        mem_normed = rms_norm(mem_seq)  # jax.Array [b, s, d]
        k, v = kv_einsum("BSD,2KDH->2BSKH", mem_normed)  # each jax.Array [b, s, num_kv_heads, head_dim]

        q_positions = einops.repeat(jnp.arange(mem_len, x_len + mem_len), "t -> b t", b=b)  # jax.Array [b, t]
        k_positions = einops.repeat(jnp.arange(mem_len), "t -> b t", b=b)  # jax.Array [b, s]

        q = _apply_rope(q, positions=q_positions)
        q *= head_dim**-0.5
        k = _apply_rope(k, positions=k_positions)
        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=num_kv_heads)

        logits = jnp.einsum(
            "BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32
        )  # jax.Array [b, num_kv_heads, num_heads/num_kv_heads, t, s], float32

        # The new step: a learned per-stream additive bias, applied before
        # masking/softmax so it competes on equal footing with content-based
        # scores rather than being layered on afterward. Zero-init: at step
        # 0 this is a no-op and the attention is pure, unmodified content-
        # based scoring, exactly as if this bias didn't exist.
        bias_sym = self.param("bias_sym", nn.initializers.zeros_init(), (), jnp.float32)  # jax.Array [], float32
        bias_perc = self.param("bias_perc", nn.initializers.zeros_init(), (), jnp.float32)  # jax.Array [], float32
        stream_bias = jnp.where(is_sym, bias_sym, bias_perc)  # jax.Array [b, s], float32
        logits = logits + stream_bias[:, None, None, None, :]  # broadcasts over (num_kv_heads, group, t)

        attn_mask = mem_mask[:, None, None, None, :]  # jax.Array [b, 1, 1, 1, s]
        masked_logits = jnp.where(attn_mask, logits, -2.3819763e38)
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(x.dtype)  # jax.Array [b, num_kv_heads, group, t, s]

        # Read-only diagnostic: how much of THIS forward pass's attention
        # mass actually landed on symbolic vs. perceptual positions, per
        # query position per example, averaged over heads. Not part of the
        # loss (see module docstring) -- sown one level up in
        # history_gemma_dual.DualMemoryHistoryBlock, same mechanism the
        # prior design used for gate_sym/gate_perc.
        is_sym_f = is_sym.astype(jnp.float32)[:, None, None, None, :]  # jax.Array [b, 1, 1, 1, s]
        attn_mass_sym = jnp.mean(jnp.sum(probs.astype(jnp.float32) * is_sym_f, axis=-1), axis=(1, 2))[..., None]  # jax.Array [b, t, 1]
        attn_mass_perc = jnp.mean(jnp.sum(probs.astype(jnp.float32) * (1.0 - is_sym_f), axis=-1), axis=(1, 2))[..., None]  # jax.Array [b, t, 1]

        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)  # jax.Array [b, t, num_kv_heads, group, head_dim]
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        out_einsum = lora.Einsum(
            shape=(num_heads, head_dim, width), name="out_einsum_mem",
            init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
        )
        return out_einsum("BTNH,NHD->BTD", encoded), attn_mass_sym, attn_mass_perc


class EarlyFusionModulator(nn.Module):
    """
    What it does:
        Tags each stream's tokens with a learned per-stream identity vector,
        concatenates the two (now-comparable, per unified_memory_encoder.py)
        streams into one memory sequence, runs ONE FusedMemoryAttention
        cross-attention against it, and produces the AdaLN-Zero-style
        (scale, shift) that modulates the (RMS-normalized) action features --
        the same overall shape (one attention, one modulation MLP) as the
        released single-stream design, just fed a fused two-stream memory
        instead of one stream.

    Returns:
        n/a -- see __call__.

    Example input:
        EarlyFusionModulator(name="joint_gated_modulator")

    Example output:
        a callable module; see __call__ below.
    """

    @nn.compact
    def __call__(
        self,
        x,           # jax.Array [b, t, d] -- action-expert features before the FFN sublayer
        mem_sym,     # jax.Array [b, s_sym, d] -- symbolic memory tokens (M_sym, post-UnifiedMemoryEncoder)
        mem_sym_mask,   # jax.Array [b, s_sym] -- True where mem_sym is a real (non-padding) token
        mem_perc,    # jax.Array [b, s_perc, d] -- perceptual memory tokens (M_perc, post-UnifiedMemoryEncoder)
        mem_perc_mask,  # jax.Array [b, s_perc] -- True where mem_perc is a real (non-padding) token
    ):
        """
        What it does:
            Runs the early-fusion mechanism (see module docstring) for one
            action-expert layer: tag both streams, concatenate into one
            memory sequence, one cross-attention, one modulation MLP,
            RMS-normalize x and apply the resulting modulation.

        Returns:
            tuple[jax.Array, dict] -- (modulated_x, stats). modulated_x has
            the same shape as x, [b, t, d]. stats is a dict with keys
            "attn_mass_sym" [b, t, 1] and "attn_mass_perc" [b, t, 1] (read-
            only diagnostics, see FusedMemoryAttention's docstring -- these
            sum to 1 elementwise, they are not trained targets themselves).

        Example input:
            modulator(x=jnp.zeros((2, 20, 1024)),
                      mem_sym=jnp.zeros((2, 64, 1024)), mem_sym_mask=jnp.ones((2, 64), dtype=bool),
                      mem_perc=jnp.zeros((2, 512, 1024)), mem_perc_mask=jnp.ones((2, 512), dtype=bool))

        Example output:
            (Array of shape (2, 20, 1024), {"attn_mass_sym": ..., "attn_mass_perc": ...})
        """
        dtype = x.dtype  # jax numpy dtype, e.g. bfloat16 during training
        width = x.shape[-1]  # int, action expert width (1024 for gemma_300m)

        # Modality tags: give attention an explicit, cheap "which stream is
        # this" signal, since after concatenation a token's position alone
        # carries no such information. Small-random-init (not zero) is fine
        # here -- unlike the bias below, these don't need to start inert:
        # the near-zero-init mlp_fused Dense below already guarantees the
        # whole modulator is an identity at init regardless of what these
        # tags contribute to the attention output.
        tag_sym = self.param("tag_sym", nn.initializers.normal(stddev=0.02), (width,), dtype)  # jax.Array [d]
        tag_perc = self.param("tag_perc", nn.initializers.normal(stddev=0.02), (width,), dtype)  # jax.Array [d]
        mem_sym_tagged = mem_sym + tag_sym  # jax.Array [b, s_sym, d]
        mem_perc_tagged = mem_perc + tag_perc  # jax.Array [b, s_perc, d]

        mem_fused = jnp.concatenate([mem_sym_tagged, mem_perc_tagged], axis=1)  # jax.Array [b, s_sym+s_perc, d]
        mem_fused_mask = jnp.concatenate([mem_sym_mask, mem_perc_mask], axis=1)  # jax.Array [b, s_sym+s_perc]
        is_sym = jnp.concatenate(
            [jnp.ones(mem_sym_mask.shape, dtype=bool), jnp.zeros(mem_perc_mask.shape, dtype=bool)], axis=1
        )  # jax.Array [b, s_sym+s_perc], bool -- True for the symbolic segment (first s_sym positions)

        r_fused, attn_mass_sym, attn_mass_perc = FusedMemoryAttention(name="mem_attn_fused")(
            x, mem_fused, mem_fused_mask, is_sym
        )  # jax.Array [b, t, d], jax.Array [b, t, 1], jax.Array [b, t, 1]

        # AdaLN-Zero-style modulation proposal from the single fused result.
        # kernel_init_out_proj (small non-zero noise, matching this
        # codebase's existing MemoryRMSNorm/JointGatedModulator convention)
        # keeps the modulator an (approximate) identity at init.
        mod = nn.Dense(
            width * 2, kernel_init=kernel_init_out_proj, dtype=dtype, name="mlp_fused"
        )(r_fused)  # jax.Array [b, t, 2*d]
        scale, shift = jnp.split(mod, 2, axis=-1)  # each jax.Array [b, t, d]

        variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)  # jax.Array [b, t, 1], float32
        normed_x = (x.astype(jnp.float32) * jax.lax.rsqrt(variance + 1e-6)).astype(dtype)  # jax.Array [b, t, d]
        modulated_x = normed_x * (1 + scale) + shift  # jax.Array [b, t, d]

        stats = {
            "attn_mass_sym": attn_mass_sym,
            "attn_mass_perc": attn_mass_perc,
        }  # dict[str, jax.Array]
        return modulated_x, stats
