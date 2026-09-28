"""
event_encoder.py

Caption-side encoder for XF (robomme-gated-xattn-fusion agent -- see
xattn_fusion/gated-fusion-agent.md section 3). Turns the temporal aligner's
fixed-shape event/caption arrays (subgoal_table.py's pack_event_arrays
output) into two things the gated fusion block (fusion_xattn.py) consumes:
one pooled token per event (E_tok, used as part of the memory sequence
itself) and the flattened per-caption-token key/value sequence (C_flat, used
as the cross-attention keys/values), plus a null token so the fusion mask
never leaves a query row with nothing to attend to.

Role in the system: xf_pi0.py's XFModel.embed_memory calls EventEncoder once
per forward pass, alongside the unchanged PerceptualMemory encoder, before
GatedXAttnFusion fuses the two streams.
"""

from __future__ import annotations

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at
from mme_vla_suite.models.representation.utils import kernel_init, kernel_init_out_proj
from xattn_fusion.mme_vla_suite.models.representation.xf_common import MASK_FILL, RMSNorm


class _TinyTransformerLayer(nnx.Module):
    """One self-attention + FFW block, pre-norm, no RoPE -- used inside EventEncoder over the L_c axis."""

    def __init__(self, rngs: nnx.Rngs, dtype, width: int = 1024, heads: int = 4, ffw_dim: int = 2048):
        """
        What it does: builds one pre-norm self-attention + FFW block's
        parameters (RMSNorm, fused QKV projection, output projection, and a
        2-layer FFW).

        Returns:
            None -- sets this module's attributes.

        Example input:
            _TinyTransformerLayer(rngs=nnx.Rngs(0), dtype=jnp.float32, width=1024, heads=4)

        Example output:
            <_TinyTransformerLayer ...>
        """
        head_dim = width // heads  # int
        self.heads = heads  # int
        self.head_dim = head_dim  # int
        self.norm_attn = RMSNorm(width, rngs=rngs, dtype=dtype)
        self.qkv_proj = nnx.Linear(width, 3 * width, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.out_proj = nnx.Linear(width, width, rngs=rngs, dtype=dtype, kernel_init=kernel_init_out_proj)
        self.norm_ffw = RMSNorm(width, rngs=rngs, dtype=dtype)
        self.ffw_in = nnx.Linear(width, ffw_dim, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.ffw_out = nnx.Linear(ffw_dim, width, rngs=rngs, dtype=dtype, kernel_init=kernel_init_out_proj)

    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        """
        What it does: standard pre-norm self-attention (masked on the key
        axis) followed by a pre-norm 2-layer FFW, both as residual additions.

        Returns:
            jax.Array -- same shape as `x`.

        Example input:
            layer(x, mask)  # x: [N, L, D]  mask: [N, L] bool (True = valid token)

        Example output:
            Array of shape (N, L, D)
        """
        normed = self.norm_attn(x)  # [N, L, D]
        qkv = self.qkv_proj(normed)  # [N, L, 3D]
        q, k, v = jnp.split(qkv, 3, axis=-1)  # each [N, L, D]
        q = einops.rearrange(q, "n l (h d) -> n l h d", h=self.heads)
        k = einops.rearrange(k, "n l (h d) -> n l h d", h=self.heads)
        v = einops.rearrange(v, "n l (h d) -> n l h d", h=self.heads)
        q = q * (self.head_dim**-0.5)
        logits = jnp.einsum("nqhd,nkhd->nhqk", q, k, preferred_element_type=jnp.float32)  # float32[N,H,L,L]
        key_mask = mask[:, None, None, :]  # bool[N,1,1,L]
        masked_logits = jnp.where(key_mask, logits, MASK_FILL)
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(x.dtype)  # [N,H,L,L]
        out = jnp.einsum("nhqk,nkhd->nqhd", probs, v)  # [N,L,H,Hd]
        out = einops.rearrange(out, "n l h d -> n l (h d)")
        x = x + self.out_proj(out)
        ffw = self.ffw_out(nnx.silu(self.ffw_in(self.norm_ffw(x))))
        return x + ffw


class _AttnPool(nnx.Module):
    """Single learned query attention-pools a variable-length, masked token sequence to one vector."""

    def __init__(self, rngs: nnx.Rngs, dtype, width: int = 1024, heads: int = 4):
        """
        What it does: builds one learned query vector plus the Q/KV/output
        projections used to attention-pool a masked token sequence.

        Returns:
            None -- sets this module's attributes.

        Example input:
            _AttnPool(rngs=nnx.Rngs(0), dtype=jnp.float32, width=1024, heads=4)

        Example output:
            <_AttnPool ...>
        """
        head_dim = width // heads  # int
        self.heads = heads  # int
        self.head_dim = head_dim  # int
        self.query = nnx.Param(jax.random.normal(rngs.params(), (1, width)) * 0.02)  # nnx.Param, float32[1,D]
        self.kv_proj = nnx.Linear(width, 2 * width, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.q_proj = nnx.Linear(width, width, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.out_proj = nnx.Linear(width, width, rngs=rngs, dtype=dtype, kernel_init=kernel_init_out_proj)

    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        """
        What it does: the one learned query attends over `x`'s masked token
        axis (single query position, multi-head) and the pooled result is
        projected back to `width`.

        Returns:
            jax.Array -- pooled vector per row of `x`.

        Example input:
            pool(x, mask)  # x: [N, L, D]  mask: [N, L] bool

        Example output:
            Array of shape (N, D)
        """
        n = x.shape[0]  # int
        q = self.q_proj(jnp.broadcast_to(self.query.value, (n, self.query.value.shape[-1])))  # [N, D]
        kv = self.kv_proj(x)  # [N, L, 2D]
        k, v = jnp.split(kv, 2, axis=-1)  # each [N, L, D]
        q = einops.rearrange(q, "n (h d) -> n h d", h=self.heads) * (self.head_dim**-0.5)
        k = einops.rearrange(k, "n l (h d) -> n l h d", h=self.heads)
        v = einops.rearrange(v, "n l (h d) -> n l h d", h=self.heads)
        logits = jnp.einsum("nhd,nlhd->nhl", q, k, preferred_element_type=jnp.float32)  # float32[N,H,L]
        masked_logits = jnp.where(mask[:, None, :], logits, MASK_FILL)
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(x.dtype)  # [N,H,L]
        out = jnp.einsum("nhl,nlhd->nhd", probs, v)  # [N,H,Hd]
        out = einops.rearrange(out, "n h d -> n (h d)")
        return self.out_proj(out)  # [N, D]


class EventEncoder(nnx.Module):
    """Builds pooled event tokens (E) and flattened caption keys/values (C_flat) from aligner output."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        dtype,
        embed_dim: int = 2048,
        width: int = 1024,
        num_layers: int = 2,
        heads: int = 4,
        e_max: int = 16,
        spatial_dim: int = 512,
        temporal_dim: int = 256,
    ):
        self.width = width  # int
        self.e_max = e_max  # int
        self.embed_proj = nnx.Linear(embed_dim, width, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.embed_norm = RMSNorm(width, rngs=rngs, dtype=dtype)
        self.spatial_proj = nnx.Linear(spatial_dim, width, rngs=rngs, dtype=dtype, kernel_init=kernel_init_out_proj)
        self.temporal_proj = nnx.Linear(temporal_dim, width, rngs=rngs, dtype=dtype, kernel_init=kernel_init_out_proj)
        self.order_embed = nnx.Embed(e_max, width, rngs=rngs, dtype=dtype)
        self.flag_proj = nnx.Linear(3, width, rngs=rngs, dtype=dtype, kernel_init=kernel_init_out_proj)
        # list[_TinyTransformerLayer], plain Python list -- see fusion_xattn.py's
        # GatedXAttnFusion.blocks for the full story: a string-keyed dict was tried 2026-09-20 to
        # work around a real _merge_params crash, but that broke flax.nnx.State.
        # replace_by_pure_dict (used on every weight load, which expects a plain list here, not
        # a string-keyed dict) -- only surfaced 2026-09-21 on the first checkpoint resume
        # attempt. Reverted to a plain list; _merge_params is patched instead (see
        # launch_xf_training.py's _patch_scripts_train_for_xf) to avoid the original crash
        # without needing this module to work around it.
        self.layers = [_TinyTransformerLayer(rngs=rngs, dtype=dtype, width=width, heads=heads) for i in range(num_layers)]
        self.pool = _AttnPool(rngs=rngs, dtype=dtype, width=width, heads=heads)
        self.null_token = nnx.Param(jax.random.normal(rngs.params(), (1, 1, width)) * 0.02)  # nnx.Param, float32[1,1,D]

    def __call__(
        self,
        event_text_tokens: at.Int[at.Array, "b e lc"],
        event_text_mask: at.Bool[at.Array, "b e lc"],
        event_coords: at.Int[at.Array, "b e 2"],
        event_start: at.Int[at.Array, "b e"],
        event_is_demo: at.Bool[at.Array, "b e"],
        event_is_open: at.Bool[at.Array, "b e"],
        event_num_frames: at.Int[at.Array, "b e"],
        event_mask: at.Bool[at.Array, "b e"],
        embed_fn,
        pos_embedder,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        What it does:
            Embeds each event's caption tokens with the (frozen-gradient)
            PaliGemma embedder, adds spatial/temporal/order/flag positional
            codes, runs a small self-attention encoder over each caption's
            L_c tokens, attention-pools one token per event, and flattens the
            per-token sequence (plus a null token) into cross-attention keys.
            v1 simplification: the spatial code is broadcast-added to every
            token of a caption that has a bbox (not only a single located
            "<bbox>" sub-token span -- sentencepiece does not tokenize the
            literal string "<bbox>" as one token, so locating it precisely
            would need custom vocab or offset-tracking machinery out of scope
            for the v1 smoke test).

        Returns:
            tuple[jax.Array, jax.Array, jax.Array] -- (E_tok [b,e,width],
            C_flat [b, e*lc+1, width], C_mask [b, e*lc+1] bool).

        Example input:
            event_encoder(event_text_tokens, event_text_mask, event_coords,
                          event_start, event_is_demo, event_is_open,
                          event_num_frames, event_mask,
                          embed_fn=lambda t: model.PaliGemma.llm(t, method="embed"),
                          pos_embedder=pos_embedder)

        Example output:
            (Array (2, 16, 1024), Array (2, 385, 1024), Array (2, 385))
        """
        b, e, lc = event_text_tokens.shape  # int, int, int

        # history_gemma.Module.embed only accepts rank-2 [b, t] token arrays -- flatten the
        # event axis into the token axis for the embedding lookup, then restore it.
        flat_tokens = einops.rearrange(event_text_tokens, "b e l -> b (e l)")  # int32[b, e*lc]
        flat_x = jax.lax.stop_gradient(embed_fn(flat_tokens))  # [b, e*lc, 2048], PaliGemma's own Embedder.encode
        # already applies the sqrt(embed_dim) scale internally (openpi.models.gemma.Embedder.encode).
        x = einops.rearrange(flat_x, "b (e l) d -> b e l d", e=e)  # [b,e,lc,2048]
        x = self.embed_norm(self.embed_proj(x))  # [b,e,lc,width]

        # event_coords is packed with -1 in both slots when an event has no bounding box
        # (subgoal_table.pack_event_arrays), so a real box at (0, 0) is never mistaken for
        # "no box" the way a `!= 0` check would.
        has_coords = jnp.all(event_coords >= 0, axis=-1)  # bool[b,e]
        cell_row = jnp.clip(event_coords[..., 0] // 64, 0, 3)  # int32[b,e]
        cell_col = jnp.clip(event_coords[..., 1] // 64, 0, 3)  # int32[b,e]
        spatial_idx = 4 * cell_row + cell_col  # int32[b,e], index into PosEmb3D's 4x4 table (16 cells)
        # pos_embedder.spatial_pe4x4/.temporal_pe are plain numpy arrays (xf_common.XFPosEmb3D
        # builds them with numpy specifically so they're safe, concrete state across separate
        # jit traces -- see that class's docstring). Converting with jnp.asarray here, fresh on
        # every call, inside the SAME forward-pass trace that immediately indexes them with a
        # live tracer (spatial_idx/event_start), is what actually makes indexing them by a
        # tracer work -- plain numpy's own __getitem__ cannot accept a jax tracer as an index.
        spatial_pe = jnp.asarray(pos_embedder.spatial_pe4x4)[spatial_idx]  # float32[b,e,512]
        spatial_code = self.spatial_proj(spatial_pe) * has_coords[..., None]  # [b,e,width]
        x = x + spatial_code[:, :, None, :]  # broadcast over the lc axis

        temporal_table = jnp.asarray(pos_embedder.temporal_pe)  # float32[2048,256], see comment above
        temporal_pe = temporal_table[jnp.clip(event_start, 0, temporal_table.shape[0] - 1)]  # [b,e,256]
        temporal_code = self.temporal_proj(temporal_pe)  # [b,e,width]
        x = x + temporal_code[:, :, None, :]

        order_code = self.order_embed(jnp.clip(jnp.arange(e), 0, self.e_max - 1))  # [e,width]
        x = x + order_code[None, :, None, :]

        flags = jnp.stack(
            [event_is_demo.astype(jnp.float32), event_is_open.astype(jnp.float32), (event_num_frames == 0).astype(jnp.float32)],
            axis=-1,
        )  # float32[b,e,3]
        flag_code = self.flag_proj(flags)  # [b,e,width]
        x = x + flag_code[:, :, None, :]

        x = einops.rearrange(x, "b e l d -> (b e) l d")  # [b*e, lc, width]
        flat_mask = einops.rearrange(event_text_mask, "b e l -> (b e) l")  # bool[b*e, lc]
        # rows where every token is masked out (a fully-padded event slot) would otherwise give
        # softmax an all -inf row inside _TinyTransformerLayer/_AttnPool; force at least one
        # attendable position so those rows stay finite (their output is discarded downstream
        # via event_mask anyway).
        safe_mask = flat_mask | (~jnp.any(flat_mask, axis=-1, keepdims=True))  # bool[b*e, lc]

        for layer in self.layers:
            x = layer(x, safe_mask)

        pooled = self.pool(x, safe_mask)  # [b*e, width]
        pooled = einops.rearrange(pooled, "(b e) d -> b e d", b=b)  # [b,e,width]
        e_tok = pooled * event_mask[:, :, None].astype(pooled.dtype)  # [b,e,width]

        c_flat = einops.rearrange(x, "(b e) l d -> b (e l) d", b=b)  # [b, e*lc, width]
        c_mask = einops.rearrange(event_text_mask, "b e l -> b (e l)")  # bool[b, e*lc]

        null_tok = jnp.broadcast_to(self.null_token.value.astype(c_flat.dtype), (b, 1, self.width))  # [b,1,width]
        c_flat = jnp.concatenate([c_flat, null_tok], axis=1)  # [b, e*lc+1, width]
        null_mask = jnp.ones((b, 1), dtype=jnp.bool_)  # bool[b,1]
        c_mask = jnp.concatenate([c_mask, null_mask], axis=1)  # bool[b, e*lc+1]

        return e_tok, c_flat, c_mask
