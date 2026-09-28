"""
fusion_grounding.py

Two pieces of XF's q_proj fix (variant symroute_cond_qfix, 2026-09-27):

  QueryContext -- the context each remembered frame's fusion QUERY is
    conditioned on (via FiLM in fusion_xattn.GatedXAttnFusion): the pooled
    instruction embedding plus the current subgoal's event token E_open.
    Without it a frame's question depends only on the frame itself, so it
    cannot ask for task-relevant caption information.

  fusion_grounding_loss -- a direct training signal for the fusion
    cross-attention. For every real remembered frame, a tiny head reads its
    16 tokens' PRE-gate fusion messages and predicts which 4x4 image cell
    contains the CURRENT subgoal's target (y, x). The only way to answer is
    to read the caption's coordinates through the fusion attention and
    combine them with the token's own position -- the frame<->caption
    grounding link that was missing (VideoUnmask stays at the perceptual
    level: the caption names the target container but the policy does not
    map that location onto the scene). Reading the pre-gate message means
    the gradient reaches q_proj / kv_proj / out_proj without the tanh(gate)
    attenuation that kept q_proj at init (0.9993x) in every earlier run.

    Caveat (documented, accepted): the label assumes the target sits at the
    same image cell in the remembered frames. The front camera is fixed and
    containers mostly stay put, but in the *Swap tasks they move, so the
    label is a noisy proxy there. Weighted small (fusion.grounding_weight).

Token layout assumption (VERIFIED on real data before training by
diagnostics/smoke_test_symroute.py::layout_check): the frame-sampling
memory stores each frame as 16 consecutive tokens in row-major 4x4 order
(token j <-> cell j = 4*row + col, row = y // 64, col = x // 64 in the
256x256 front image), matching the PosEmb3D 4x4 position codes paired with
them.

Role in the system: owned/called by xf_pi0.XFModel when
history_config.fusion.qfix is true; the loss is added in compute_loss.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at
from mme_vla_suite.models.representation.utils import kernel_init
from xattn_fusion.mme_vla_suite.models.representation.xf_common import RMSNorm

TOKENS_PER_FRAME = 16  # int, 4x4 pooled SigLIP tokens per sampled frame (history_config.token_per_image)
GRID = 4  # int, cells per side
CELL_PX = 64  # int, 256 px / 4 cells


class QueryContext(nnx.Module):
    """MLP: [pooled instruction embedding ; E_open] -> query context vector."""

    def __init__(self, rngs: nnx.Rngs, dtype, embed_dim: int = 2048, width: int = 1024):
        """
        What it does:
            Builds the instruction projection (2048-d Gemma embedding ->
            width) and a 2-layer MLP over [instruction ; E_open].

        Returns:
            None -- sets this module's attributes.

        Example input:
            QueryContext(rngs=nnx.Rngs(0), dtype=jnp.bfloat16)

        Example output:
            <QueryContext ...>
        """
        self.prompt_proj = nnx.Linear(embed_dim, width, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.prompt_norm = RMSNorm(width, rngs=rngs, dtype=dtype)
        self.ctx_in = nnx.Linear(2 * width, width, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.ctx_out = nnx.Linear(width, width, rngs=rngs, dtype=dtype, kernel_init=kernel_init)

    def __call__(self, tokenized_prompt, tokenized_prompt_mask, e_tok, event_is_open, embed_fn) -> jax.Array:
        """
        What it does:
            Embeds the instruction tokens with the frozen PaliGemma embedder
            (stop_gradient, same as EventEncoder's captions), mean-pools the
            real tokens, projects, and combines with the open event's token.

        Returns:
            jax.Array -- [b, width] query context.

        Example input:
            ctx(obs.tokenized_prompt, obs.tokenized_prompt_mask, e_tok, obs.event_is_open, embed_fn)

        Example output:
            Array of shape (4, 1024)
        """
        emb = jax.lax.stop_gradient(embed_fn(tokenized_prompt))  # [b, L, 2048]
        m = tokenized_prompt_mask.astype(emb.dtype)[..., None]  # [b, L, 1]
        pooled = jnp.sum(emb * m, axis=1) / jnp.maximum(jnp.sum(m, axis=1), 1.0)  # [b, 2048]
        prompt_vec = self.prompt_norm(self.prompt_proj(pooled))  # [b, width]
        has_open = jnp.any(event_is_open, axis=-1)  # bool[b]
        k = jnp.argmax(event_is_open, axis=-1)  # int[b]
        e_open = jnp.take_along_axis(e_tok, k[:, None, None], axis=1)[:, 0, :]  # [b, width]
        e_open = e_open * has_open[:, None].astype(e_open.dtype)  # [b, width]
        h = nnx.silu(self.ctx_in(jnp.concatenate([prompt_vec, e_open.astype(prompt_vec.dtype)], axis=-1)))  # [b, width]
        return self.ctx_out(h)  # [b, width]


class GroundingHead(nnx.Module):
    """RMSNorm + Linear(width -> 1): per-token 'target is in my cell' logit from the pre-gate fusion message."""

    def __init__(self, rngs: nnx.Rngs, dtype, width: int = 1024):
        """
        What it does: builds the norm (makes the head insensitive to the
        message's small absolute scale) and the scoring layer.

        Returns:
            None -- sets this module's attributes.

        Example input:
            GroundingHead(rngs=nnx.Rngs(0), dtype=jnp.bfloat16)

        Example output:
            <GroundingHead ...>
        """
        self.norm = RMSNorm(width, rngs=rngs, dtype=dtype)
        self.score = nnx.Linear(width, 1, rngs=rngs, dtype=dtype, kernel_init=kernel_init)

    def __call__(self, msg: jax.Array) -> jax.Array:
        """
        What it does: scores every memory token.

        Returns:
            jax.Array -- float32[b, f] logits.

        Example input:
            head(msg)  # [4, 512, 1024]

        Example output:
            Array of shape (4, 512), float32
        """
        return self.score(self.norm(msg))[..., 0].astype(jnp.float32)


def target_cell(event_coords_open: jax.Array) -> jax.Array:
    """
    What it does: maps (y, x) in [0, 255] to its row-major 4x4 cell index.

    Returns:
        jax.Array -- int32[b] in 0..15.

    Example input:
        target_cell(jnp.array([[94, 162]]))

    Example output:
        Array([6])   # row 1, col 2
    """
    row = jnp.clip(event_coords_open[:, 0] // CELL_PX, 0, GRID - 1)  # int[b]
    col = jnp.clip(event_coords_open[:, 1] // CELL_PX, 0, GRID - 1)  # int[b]
    return (GRID * row + col).astype(jnp.int32)


def fusion_grounding_loss(
    token_logits: jax.Array,
    static_mask: at.Bool[at.Array, "b f"],
    event_is_open: at.Bool[at.Array, "b e"],
    event_coords: at.Int[at.Array, "b e 2"],
) -> tuple[jax.Array, dict]:
    """
    What it does:
        Per remembered frame, softmax over its 16 token logits and
        cross-entropy against the current subgoal's target cell. Counts only
        real frames (all 16 tokens present) in samples whose OPEN event has
        coordinates; returns a per-sample mean loss (0 where nothing counts).

    Returns:
        tuple[jax.Array, dict] -- (loss float32[b], metrics with scalar
        "grounding_ce", "grounding_acc" (chance 1/16 = 0.0625),
        "grounding_frac" (fraction of samples contributing)).

    Example input:
        fusion_grounding_loss(head(msg), obs.static_mask, obs.event_is_open, obs.event_coords)

    Example output:
        (Array (4,), {"grounding_ce": 2.77, "grounding_acc": 0.06, "grounding_frac": 0.8})
    """
    b, f = token_logits.shape  # int, int
    n_frames = f // TOKENS_PER_FRAME  # int
    logits = token_logits.reshape(b, n_frames, TOKENS_PER_FRAME)  # float32[b, n, 16]
    frame_ok = jnp.all(static_mask.reshape(b, n_frames, TOKENS_PER_FRAME), axis=-1)  # bool[b, n]

    has_open = jnp.any(event_is_open, axis=-1)  # bool[b]
    k = jnp.argmax(event_is_open, axis=-1)  # int[b]
    xy = jnp.take_along_axis(event_coords, k[:, None, None], axis=1)[:, 0, :]  # int[b, 2]
    sample_ok = has_open & jnp.all(xy >= 0, axis=-1)  # bool[b]
    cell = target_cell(jnp.maximum(xy, 0))  # int[b]

    logp = jax.nn.log_softmax(logits, axis=-1)  # float32[b, n, 16]
    ce = -jnp.take_along_axis(logp, jnp.broadcast_to(cell[:, None, None], (b, n_frames, 1)), axis=-1)[..., 0]  # float32[b, n]
    valid = frame_ok & sample_ok[:, None]  # bool[b, n]
    n_valid = jnp.sum(valid, axis=1)  # int[b]
    loss = jnp.sum(jnp.where(valid, ce, 0.0), axis=1) / jnp.maximum(n_valid, 1)  # float32[b]

    correct = (jnp.argmax(logits, axis=-1) == cell[:, None]) & valid  # bool[b, n]
    total = jnp.maximum(jnp.sum(valid), 1)  # int scalar
    metrics = {
        "grounding_ce": jnp.sum(jnp.where(valid, ce, 0.0)) / total,
        "grounding_acc": jnp.sum(correct) / total,
        "grounding_frac": jnp.mean((n_valid > 0).astype(jnp.float32)),
    }  # dict[str, jax.Array]
    return loss, metrics
