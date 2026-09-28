"""
subgoal_cond.py

Current-subgoal conditioning for XF ("option B", 2026-09-26). Builds one
vector per sample from the CURRENT (open) event of the symbolic memory -- its
caption-encoder token E_open plus a full-resolution sinusoidal code of its
target (y, x) -- that xf_pi0.XFModel adds to the action expert's adaRMS
conditioning (the flow-matching time embedding). That conditioning sets the
scale, shift and residual gate of EVERY action-expert norm in all 18 layers,
so the current subgoal reaches the action computation through a channel the
action expert is built to use, not only through a cross-attention readout.

Why: the symroute short run (1,500 steps, RESEARCH_LOG 2026-09-26) showed the
action expert's output tokens encode the current subgoal (aux template
accuracy 37-62%, target error ~12-18 px) while the predicted ACTIONS stayed
independent of caption content (caption swap <= 0.65%, target move ~0.25%,
vs ~50% for swapping frames). The information was present but not coupled
into the action output.

Not the aux head's predictions: those are computed from the action expert's
own output tokens, so feeding them back into the same forward pass would be
circular. This module feeds what they are predicted from -- the current
subgoal's own representation -- directly.

Role in the system: owned by xf_pi0.XFModel when
history_config.symbolic_route.subgoal_cond is true; called once per forward
pass inside embed_memory's symbolic-route branch.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.shared.array_typing as at
from mme_vla_suite.models.representation.utils import kernel_init, kernel_init_out_proj


def coord_fourier_code(event_coords_open: jax.Array, has_coords: jax.Array, num_freqs: int = 16) -> jax.Array:
    """
    What it does:
        Encodes the open event's (y, x) in [0, 255] at full resolution as
        [sin, cos] of (coord / 255) * pi * 2^k for k = 0..num_freqs-1, for
        both axes, zeroed where the event has no bounding box. Unlike
        EventEncoder's 4x4 spatial cell (64-px bins), this keeps pixel-level
        position.

    Returns:
        jax.Array -- float32[b, 4 * num_freqs].

    Example input:
        coord_fourier_code(jnp.array([[94, 162]]), jnp.array([True]))

    Example output:
        Array of shape (1, 64), float32
    """
    freqs = jnp.asarray(np.pi * (2.0 ** np.arange(num_freqs)), dtype=jnp.float32)  # float32[F]
    xy = event_coords_open.astype(jnp.float32) / 255.0  # float32[b, 2]
    ang = xy[:, :, None] * freqs[None, None, :]  # float32[b, 2, F]
    code = jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1).reshape(xy.shape[0], -1)  # float32[b, 4F]
    return code * has_coords[:, None].astype(jnp.float32)


class SubgoalConditioner(nnx.Module):
    """MLP: [E_open ; coord code] -> additive term for the action expert's adaRMS conditioning."""

    def __init__(self, rngs: nnx.Rngs, dtype, width: int = 1024, hidden: int = 1024, num_freqs: int = 16):
        """
        What it does:
            Builds the input projection, hidden nonlinearity and the output
            projection. The output projection uses kernel_init_out_proj
            (normal stddev 0.002) -- ONE small factor on the path, the same
            init the released memory modulator trained from scratch with --
            so the warm-started model is barely perturbed at step 0 while
            the output projection still receives full-size gradients (its
            input is a normal-scale hidden activation). Deliberately NOT a
            zero-init gate stacked on a small projection: that pair froze
            under AdamW's eps in the 18k XF run.

        Returns:
            None -- sets this module's attributes.

        Example input:
            SubgoalConditioner(rngs=nnx.Rngs(0), dtype=jnp.bfloat16)

        Example output:
            <SubgoalConditioner ...>
        """
        self.num_freqs = num_freqs  # int
        in_dim = width + 4 * num_freqs  # int
        self.cond_in = nnx.Linear(in_dim, hidden, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.cond_out = nnx.Linear(hidden, width, rngs=rngs, dtype=dtype, kernel_init=kernel_init_out_proj)

    def __call__(
        self,
        e_tok: at.Float[at.Array, "b e d"],
        event_is_open: at.Bool[at.Array, "b e"],
        event_coords: at.Int[at.Array, "b e 2"],
    ) -> jax.Array:
        """
        What it does:
            Gathers the open event's token and coords, builds the coord code,
            and maps [E_open ; code] to a width-d conditioning term. Samples
            with no open event get E_open = 0 and a zero code.

        Returns:
            jax.Array -- [b, width], same dtype as e_tok.

        Example input:
            cond(e_tok, obs.event_is_open, obs.event_coords)  # e_tok: [4, 14, 1024]

        Example output:
            Array of shape (4, 1024)
        """
        has_open = jnp.any(event_is_open, axis=-1)  # bool[b]
        k = jnp.argmax(event_is_open, axis=-1)  # int[b], open event slot
        e_open = jnp.take_along_axis(e_tok, k[:, None, None], axis=1)[:, 0, :]  # [b, d]
        e_open = e_open * has_open[:, None].astype(e_open.dtype)  # [b, d]
        xy = jnp.take_along_axis(event_coords, k[:, None, None], axis=1)[:, 0, :]  # int[b, 2]
        has_coords = has_open & jnp.all(xy >= 0, axis=-1)  # bool[b]
        code = coord_fourier_code(jnp.maximum(xy, 0), has_coords, self.num_freqs).astype(e_open.dtype)  # [b, 4F]
        h = nnx.silu(self.cond_in(jnp.concatenate([e_open, code], axis=-1)))  # [b, hidden]
        return self.cond_out(h)  # [b, width]
