"""
fusion_xattn.py

Gated cross-attention fusion block for XF (robomme-gated-xattn-fusion agent
-- see xattn_fusion/gated-fusion-agent.md section 4). Frame tokens (F, from
the unchanged PerceptualMemory/FeatureEncoder path) query event/caption
tokens (C_flat, from event_encoder.EventEncoder) through a same-event
attention mask, producing an enriched frame stream F'. Two learnable scalar
gates (tanh, zero-init) mean the block is an exact identity at
initialization -- fusion only starts contributing once training moves the
gates off zero.

Role in the system: xf_pi0.py's XFModel.embed_memory calls
GatedXAttnFusion(F, C_flat, static_token_event_idx, event_text_mask) to get
F', then hybrid_mem.assemble_memory concatenates F' with the event tokens E
before handing both to the UNCHANGED history_gemma.MemoryAttention/
MemoryRMSNorm modulator -- this module never touches that modulator.

q_proj FIX (2026-09-27, variant symroute_cond_qfix). Fusion's q_proj never left
init (0.9993x) because (1) the same-event mask left each frame only its own
~10 caption tokens to choose from, (2) its gradient was attenuated below
AdamW's eps, (3) frame tokens never saw the TASK. Three optional additions,
all OFF by default (old variants reproduce exactly):
  - open mask (build_open_fusion_mask): every frame may attend to EVERY real
    caption token in memory, plus a learned own-event logit bias (init +2)
    so it starts close to the old same-event behaviour. Not frame-causal on
    purpose: in VideoUnmask the demo frames belong to the demo event while
    the caption naming the target belongs to the LATER execution event; all
    captions in memory are already known at the current step, so there is
    no future leakage.
  - task/now-aware queries: a context vector (instruction + current subgoal)
    FiLMs the normalized frame features right before q_proj.
  - return_msgs: each block's PRE-gate message is returned so a grounding
    loss (fusion_grounding.py) can train q/kv/out_proj without the tanh(gate)
    attenuation.
"""

from __future__ import annotations

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at
from mme_vla_suite.models.representation.utils import kernel_init, kernel_init_out_proj
from xattn_fusion.mme_vla_suite.models.representation.xf_common import MASK_FILL, RMSNorm

# int, int, int, int -- hardcoded to match history_gemma.MemoryAttention's own hardcoded
# "same dim as the action expert in pi05" config (history_gemma.py:56), since F'+event tokens
# feed straight into that unchanged modulator afterward.
_HEADS = 4
_KV_HEADS = 1
_HEAD_DIM = 256
_WIDTH = 1024


def build_fusion_mask(
    static_token_event_idx: at.Int[at.Array, "b f"], event_text_mask: at.Bool[at.Array, "b e lc"]
) -> at.Bool[at.Array, "b f el1"]:
    """
    What it does:
        Builds the same-event cross-attention mask: frame token i may attend
        to caption token (k, l) only if frame i's event index equals k AND
        that caption token is real (not padding); the appended null column is
        always True so a frame whose event has no caption tokens (or was
        padded / overflow-dropped, static_token_event_idx < 0) still has a
        valid attention target and softmax never sees an all-False row.
        Built here from index arrays (never precomputed in the dataloader),
        matching the spec's requirement that this run inside the jitted
        model.

    Returns:
        at.Bool[at.Array, "b f el1"] -- mask of shape [b, num_frame_tokens,
        e*lc + 1].

    Example input:
        build_fusion_mask(static_token_event_idx, event_text_mask)

    Example output:
        Array of shape (2, 512, 385), dtype bool
    """
    b, e, lc = event_text_mask.shape  # int, int, int
    flat_mask = einops.rearrange(event_text_mask, "b e l -> b (e l)")  # bool[b, e*lc]
    event_id_per_key = jnp.repeat(jnp.arange(e), lc)  # int32[e*lc], which event each flattened key belongs to
    # static_token_event_idx: int32[b, f], -1 = padding, -2 = overflow-dropped (never matches
    # any real event id >= 0), >=0 = the frame's real event id.
    same_event = static_token_event_idx[:, :, None] == event_id_per_key[None, None, :]  # bool[b, f, e*lc]
    mask = same_event & flat_mask[:, None, :]  # bool[b, f, e*lc]
    null_col = jnp.ones((mask.shape[0], mask.shape[1], 1), dtype=jnp.bool_)  # bool[b, f, 1]
    return jnp.concatenate([mask, null_col], axis=-1)  # bool[b, f, e*lc+1]


def build_open_fusion_mask(
    static_token_event_idx: at.Int[at.Array, "b f"], event_text_mask: at.Bool[at.Array, "b e lc"]
) -> tuple[at.Bool[at.Array, "b f el1"], at.Bool[at.Array, "b f el1"]]:
    """
    What it does:
        Open (cross-event) variant of build_fusion_mask. A REAL frame token
        (event idx >= 0) may attend to every real caption token of every
        event in memory, plus the always-valid null column. A padding /
        overflow frame (idx < 0) keeps only the null column, exactly as in
        the same-event mask. Also returns which keys belong to the frame's
        OWN event, so the caller can add the learned own-event logit bias.

    Returns:
        tuple[jax.Array, jax.Array] -- (mask bool[b, f, e*lc+1],
        own_event bool[b, f, e*lc+1]; the null column is never "own").

    Example input:
        build_open_fusion_mask(static_token_event_idx, event_text_mask)

    Example output:
        (Array (2, 512, 309) bool, Array (2, 512, 309) bool)
    """
    b, e, lc = event_text_mask.shape  # int, int, int
    flat_mask = einops.rearrange(event_text_mask, "b e l -> b (e l)")  # bool[b, e*lc]
    event_id_per_key = jnp.repeat(jnp.arange(e), lc)  # int32[e*lc]
    real_frame = static_token_event_idx >= 0  # bool[b, f]
    mask = real_frame[:, :, None] & flat_mask[:, None, :]  # bool[b, f, e*lc]
    own = (static_token_event_idx[:, :, None] == event_id_per_key[None, None, :]) & mask  # bool[b, f, e*lc]
    null_true = jnp.ones((b, mask.shape[1], 1), dtype=jnp.bool_)  # bool[b, f, 1]
    null_false = jnp.zeros((b, mask.shape[1], 1), dtype=jnp.bool_)  # bool[b, f, 1]
    return jnp.concatenate([mask, null_true], axis=-1), jnp.concatenate([own, null_false], axis=-1)


class GatedXAttnBlock(nnx.Module):
    """One Flamingo-style tanh-gated cross-attention + FFW layer, zero-init so it starts as identity."""

    def __init__(self, rngs: nnx.Rngs, dtype, width: int = _WIDTH, ffw_dim: int = 2048, gate_init: float = 0.0):
        """
        What it does: builds one gated cross-attention + FFW block's
        parameters (RMSNorms, Q/KV/output projections sized for
        _HEADS/_KV_HEADS/_HEAD_DIM, a 2-layer FFW, and the two tanh gate
        scalars a_x/a_d, initialized to `gate_init`).

        gate_init defaults to 0.0 -- the original Flamingo-style "tanh_zero"
        behaviour, an exact identity at initialization. Set it non-zero to
        escape the AdamW epsilon floor documented below.

        Returns:
            None -- sets this module's attributes.

        Example input:
            GatedXAttnBlock(rngs=nnx.Rngs(0), dtype=jnp.float32, width=1024, ffw_dim=2048)

        Example output:
            <GatedXAttnBlock ...>
        """
        self.norm_f = RMSNorm(width, rngs=rngs, dtype=dtype)
        self.norm_kv = RMSNorm(width, rngs=rngs, dtype=dtype)
        self.q_proj = nnx.Linear(width, _HEADS * _HEAD_DIM, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.kv_proj = nnx.Linear(width, 2 * _KV_HEADS * _HEAD_DIM, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.out_proj = nnx.Linear(_HEADS * _HEAD_DIM, width, rngs=rngs, dtype=dtype, kernel_init=kernel_init_out_proj)
        self.norm_ff = RMSNorm(width, rngs=rngs, dtype=dtype)
        self.ffw_in = nnx.Linear(width, ffw_dim, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.ffw_out = nnx.Linear(ffw_dim, width, rngs=rngs, dtype=dtype, kernel_init=kernel_init_out_proj)
        # nnx.Param, scalar. At gate_init=0.0 these are the original "tanh_zero" gates --
        # tanh(0)==0, so both residual paths below start as an exact no-op and the block is a
        # bit-exact identity at initialization.
        #
        # WHY gate_init EXISTS (measured 2026-09-22, see RESEARCH_LOG.md and
        # XF_18k_eval/analysis/): zero-init gates deadlocked this block for a full 18,000-step
        # run. out_proj's gradient is scaled by tanh(a_x), so with the gate at ~3e-4 its
        # gradients measured 5e-11..1e-10 -- 100-200x BELOW AdamW's eps=1e-8. In that regime
        # `lr * m/(sqrt(v)+eps)` stops normalizing (sqrt(v) << eps) and the projection FREEZES:
        # out_proj's norm was still 2.046 against its 2.048 analytic init after 18,000 steps.
        # A frozen, near-zero-init out_proj then makes the message 3e-4 of the frame stream, so
        # the gate's own gradient is noise-dominated and it random-walks/decays instead of
        # opening. Each side holds the other down.
        #
        # Measured directly: forcing the gate to 0.1 lifts out_proj's gradients to 2e-8..3.6e-8,
        # i.e. 2-3.6x ABOVE eps, restoring Adam's normalization (per-step update ~2e-5 vs
        # ~2e-7). Note that raising out_proj's INIT instead does NOT work -- it scales the gate's
        # gradient mean and std equally, leaving Adam's update unchanged, since Adam is
        # approximately scale-invariant. Opening the gate slightly is the only lever measured to
        # break the deadlock.
        #
        # Cost to the warm start is negligible: with out_proj at its stddev=0.002 init the
        # message is ~3e-4 of the frame stream, so a 0.1 gate contributes ~3e-5 -- the same order
        # of perturbation the model already carried at step 18000.
        self.a_x = nnx.Param(jnp.full((), gate_init, dtype=jnp.float32))
        self.a_d = nnx.Param(jnp.full((), gate_init, dtype=jnp.float32))

    def __call__(self, f: jax.Array, c_flat: jax.Array, mask: jax.Array, q_film=None, logit_bias=None,
                 return_msg: bool = False):
        """
        What it does: cross-attends the frame stream `f` (query) into the
        caption keys/values `c_flat`, masked by `mask`, adds the result back
        through a tanh(a_x)-gated residual, then a tanh(a_d)-gated FFW
        residual -- both gates start at 0, so this returns `f` bit-exactly
        until training moves them.

        Optional (q_proj fix, default off): `q_film` = (scale, shift) each
        [B, D], applied to the normalized frame features right before q_proj;
        `logit_bias` [B, Tf, Tc] float added to the attention logits (the
        own-event bias); `return_msg` also returns the PRE-gate message.

        Returns:
            jax.Array -- same shape as `f`; or (f, msg) when return_msg.

        Example input:
            block(f, c_flat, mask)  # f: [B,Tf,D]  c_flat: [B,Tc,D]  mask: [B,Tf,Tc] bool

        Example output:
            Array of shape (B, Tf, D)
        """
        b, tf, _ = f.shape  # int, int, int
        f_normed = self.norm_f(f)  # [B, Tf, D]
        if q_film is not None:
            film_scale, film_shift = q_film  # each [B, D]
            f_normed = f_normed * (1 + film_scale[:, None, :].astype(f_normed.dtype)) + film_shift[:, None, :].astype(f_normed.dtype)
        q = self.q_proj(f_normed)  # [B, Tf, H*Hd]
        kv = self.kv_proj(self.norm_kv(c_flat))  # [B, Tc, 2*Kh*Hd]
        k, v = jnp.split(kv, 2, axis=-1)  # each [B, Tc, Kh*Hd]

        q = einops.rearrange(q, "b t (h d) -> b t h d", h=_HEADS) * (_HEAD_DIM**-0.5)
        k = einops.rearrange(k, "b s (h d) -> b s h d", h=_KV_HEADS)
        v = einops.rearrange(v, "b s (h d) -> b s h d", h=_KV_HEADS)
        group = _HEADS // _KV_HEADS  # int
        k = einops.repeat(k, "b s h d -> b s (h g) d", g=group)  # [B, Tc, H, Hd]
        v = einops.repeat(v, "b s h d -> b s (h g) d", g=group)  # [B, Tc, H, Hd]

        logits = jnp.einsum("bthd,bshd->bhts", q, k, preferred_element_type=jnp.float32)  # float32[B,H,Tf,Tc]
        if logit_bias is not None:
            logits = logits + logit_bias[:, None, :, :].astype(jnp.float32)
        masked_logits = jnp.where(mask[:, None, :, :], logits, MASK_FILL)
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(f.dtype)  # [B,H,Tf,Tc]
        out = jnp.einsum("bhts,bshd->bthd", probs, v)  # [B,Tf,H,Hd]
        out = einops.rearrange(out, "b t h d -> b t (h d)")
        msg = self.out_proj(out)  # [B,Tf,D]

        # a_x/a_d are kept as float32 scalars (better precision right at their 0 starting point
        # than bf16 would give), but the gate VALUE is cast to f's dtype before multiplying --
        # found by code review: multiplying a bf16 array by a bare float32 scalar promotes the
        # whole expression (and everything added to it) to float32 via JAX's dtype-promotion
        # rules, silently running the rest of this block's mixed-precision math in float32.
        f = f + (jnp.tanh(self.a_x.value).astype(f.dtype)) * msg
        ffw = self.ffw_out(nnx.silu(self.ffw_in(self.norm_ff(f))))
        f = f + (jnp.tanh(self.a_d.value).astype(f.dtype)) * ffw
        return (f, msg) if return_msg else f


class GatedXAttnFusion(nnx.Module):
    """Stack of GatedXAttnBlock layers -- the frame-stream side of XF's fusion mechanism."""

    def __init__(
        self, rngs: nnx.Rngs, dtype, num_layers: int = 2, width: int = _WIDTH, ffw_dim: int = 2048,
        gate_init: float = 0.0, qfix: bool = False, own_bias_init: float = 2.0,
    ):
        """
        What it does: builds `num_layers` independent GatedXAttnBlocks, each
        with its tanh gates initialized to `gate_init` (0.0 = the original
        exact-identity behaviour; see GatedXAttnBlock.__init__ for why a
        non-zero value exists and what it is for).

        Returns:
            None -- sets self.blocks.

        Example input:
            GatedXAttnFusion(rngs=nnx.Rngs(0), dtype=jnp.float32, num_layers=2, gate_init=0.1)

        Example output:
            <GatedXAttnFusion ...>
        """
        # list[GatedXAttnBlock], plain Python list -- NOT a string-keyed dict. A string-keyed
        # dict was tried 2026-09-20 to work around a real crash in the released, unmodified
        # openpi.training.weight_loaders._merge_params (flax.traverse_util.flatten_dict(params,
        # sep="/") crashes joining a non-str path component -- a plain list's submodules get
        # INTEGER indices internally) -- but that "fix" broke a DIFFERENT released function,
        # flax.nnx.State.replace_by_pure_dict (used for EVERY weight load, warm-start or resume,
        # not just _merge_params), which unconditionally converts numeric-looking pure_dict keys
        # BACK to int (`try_convert_int`) before comparing against the live model's own state --
        # i.e. it specifically EXPECTS a plain list here, not a string-keyed dict. That
        # incompatibility was invisible through the warm-start path (which never actually has
        # loaded values for XF's purely-additive new modules to compare) and only surfaced
        # 2026-09-21 on the first-ever checkpoint RESUME attempt (FileNotFoundError-adjacent
        # ValueError: "key in pure_dict not available in state"). Reverted back to a plain list
        # here; _merge_params is patched instead (launch_xf_training.py's
        # _patch_scripts_train_for_xf) to use tuple-keyed flatten/unflatten (no sep-join at all)
        # so it never hits the original int-path-component crash in the first place -- fixing
        # the actual broken function instead of changing this module to work around it.
        self.blocks = [
            GatedXAttnBlock(rngs=rngs, dtype=dtype, width=width, ffw_dim=ffw_dim, gate_init=gate_init)
            for i in range(num_layers)
        ]
        # q_proj fix (default off). qfix: bool. When on: one learned own-event logit bias (float32
        # scalar, init +2 -> starts close to the old same-event behaviour) and one FiLM projection
        # per block mapping the query context [B, width] to (scale, shift) [B, 2*width], init
        # stddev 0.002 (single small factor, like the released modulator) so the trained model
        # starts nearly unchanged. Names avoid "llm"/"img"/"lora" (freeze filter).
        self.qfix = qfix  # bool
        if qfix:
            self.own_event_bias = nnx.Param(jnp.full((), own_bias_init, dtype=jnp.float32))  # nnx.Param, scalar
            self.query_film = [
                nnx.Linear(width, 2 * width, rngs=rngs, dtype=dtype, kernel_init=kernel_init_out_proj)
                for i in range(num_layers)
            ]  # list[nnx.Linear], plain list (see self.blocks comment for why not a dict)

    def __call__(
        self,
        f: at.Float[at.Array, "b tf d"],
        c_flat: at.Float[at.Array, "b tc d"],
        static_token_event_idx: at.Int[at.Array, "b tf"],
        event_text_mask: at.Bool[at.Array, "b e lc"],
        query_ctx: jax.Array | None = None,
        return_msgs: bool = False,
    ):
        """
        What it does:
            Builds the same-event mask once, then runs the frame stream `f`
            through each GatedXAttnBlock in sequence, cross-attending into
            `c_flat` at every layer.

        Returns:
            at.Float[at.Array, "b tf d"] -- F', same shape as the input frame
            stream `f`. Bit-exact equal to `f` when every block's gates are
            at their zero-init value (checked by smoke_test.py's CHECK1).

        Example input:
            fusion(F, C_flat, static_token_event_idx, event_text_mask)

        Example output:
            Array of shape (2, 512, 1024)
        """
        if not self.qfix:
            mask = build_fusion_mask(static_token_event_idx, event_text_mask)  # bool[b, tf, e*lc+1]
            msgs = []  # list[jax.Array]
            for block in self.blocks:
                if return_msgs:
                    f, msg = block(f, c_flat, mask, return_msg=True)
                    msgs.append(msg)
                else:
                    f = block(f, c_flat, mask)
            return (f, msgs) if return_msgs else f

        # q_proj fix path: open mask + own-event bias + task/now-aware query FiLM.
        mask, own = build_open_fusion_mask(static_token_event_idx, event_text_mask)  # bool, bool [b, tf, e*lc+1]
        bias = own.astype(jnp.float32) * self.own_event_bias.value  # float32[b, tf, e*lc+1]
        msgs = []  # list[jax.Array]
        for block, film in zip(self.blocks, self.query_film):
            q_film = None  # tuple[jax.Array, jax.Array] or None
            if query_ctx is not None:
                q_film = tuple(jnp.split(film(query_ctx), 2, axis=-1))  # (scale [b, D], shift [b, D])
            f, msg = block(f, c_flat, mask, q_film=q_film, logit_bias=bias, return_msg=True)
            msgs.append(msg)
        return (f, msgs) if return_msgs else f
