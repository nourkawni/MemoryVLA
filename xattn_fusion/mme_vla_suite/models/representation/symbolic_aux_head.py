"""
symbolic_aux_head.py

Auxiliary supervision that makes XF's action expert READ the symbolic
memory. A small head on the action expert's final hidden state predicts the
CURRENT subgoal's caption template (e.g. "pick up the green cube at <bbox>
for the fourth time" vs "press the button to stop") and its target (y, x).

Why: measured 2026-09-23 (measure_coord_usage.py) -- the 8k XF policy's
actions were independent of the caption stream (~0.2% change under any
caption intervention vs 34-45% for swapping frames). Behaviour cloning alone
gives the caption path almost no gradient: frames already predict most of
each action, and the caption-decisive moments are a few steps per episode.
The current subgoal is NOT in the prompt (symbolic_aux.in_prompt=false), so
the only way the action expert can predict it is by reading symbolic memory
through the modulator -- this loss creates exactly that gradient, without
removing any frame or caption information.

Labels are free: every cached subgoal_table.json already carries a real
task-global caption_id (1..123, verified 2026-09-23 on all 1,307 tables; one
template per id). UNK_CAPTION_ID (0) is masked out of the loss, so a
placeholder id can never become a training target.

Role in the system: xf_pi0.XFModel owns one SymbolicAuxHead when
history_config.symbolic_route.enabled is true, and compute_loss adds
aux_weight * symbolic_aux_loss(...) to the flow-matching loss (training
only; never used at inference).
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at
from mme_vla_suite.models.representation.utils import kernel_init

UNK_CAPTION_ID = 0  # int, must match subgoal_table.UNK_CAPTION_ID


class SymbolicAuxHead(nnx.Module):
    """Two-layer MLP: pooled action-expert state -> (caption-template logits, normalized (y, x))."""

    def __init__(self, rngs: nnx.Rngs, dtype, width: int = 1024, hidden: int = 512, num_captions: int = 128):
        """
        What it does:
            Builds the shared hidden layer and the two output layers
            (caption-template classifier and 2-d coordinate regressor).

        Returns:
            None -- sets this module's attributes.

        Example input:
            SymbolicAuxHead(rngs=nnx.Rngs(0), dtype=jnp.bfloat16, width=1024, num_captions=128)

        Example output:
            <SymbolicAuxHead ...>
        """
        self.num_captions = num_captions  # int
        self.hidden = nnx.Linear(width, hidden, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.caption_out = nnx.Linear(hidden, num_captions, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.coord_out = nnx.Linear(hidden, 2, rngs=rngs, dtype=dtype, kernel_init=kernel_init)

    def __call__(self, suffix_out: at.Float[at.Array, "b t d"]) -> tuple[jax.Array, jax.Array]:
        """
        What it does:
            Mean-pools the action expert's output tokens over time, then
            predicts caption-template logits and sigmoid-squashed (y, x) in
            [0, 1] (the 256x256 front-image frame divided by 255).

        Returns:
            tuple[jax.Array, jax.Array] -- (logits float32[b, num_captions],
            coords float32[b, 2]).

        Example input:
            head(suffix_out[:, -action_horizon:])  # [8, 50, 1024]

        Example output:
            (Array (8, 128) float32, Array (8, 2) float32)
        """
        pooled = jnp.mean(suffix_out.astype(jnp.float32), axis=1).astype(suffix_out.dtype)  # [b, d]
        h = nnx.silu(self.hidden(pooled))  # [b, hidden]
        logits = self.caption_out(h).astype(jnp.float32)  # float32[b, num_captions]
        coords = jax.nn.sigmoid(self.coord_out(h).astype(jnp.float32))  # float32[b, 2]
        return logits, coords


def symbolic_aux_loss(
    logits: jax.Array,
    pred_coords: jax.Array,
    event_is_open: at.Bool[at.Array, "b e"],
    event_caption_id: at.Int[at.Array, "b e"],
    event_coords: at.Int[at.Array, "b e 2"],
    coord_weight: float = 1.0,
) -> tuple[jax.Array, dict]:
    """
    What it does:
        Per-sample auxiliary loss for the OPEN (current) event: softmax
        cross-entropy on its caption_id plus coord_weight x L1 on its (y, x)
        / 255. The CE term is masked where there is no open event or the id
        is UNK (or out of range); the coord term is masked where the event
        has no bounding box (coords == -1).

    Returns:
        tuple[jax.Array, dict] -- (loss float32[b], metrics dict with
        scalar float32 "aux_ce", "aux_acc", "aux_coord_l1", "aux_label_frac").

    Example input:
        symbolic_aux_loss(logits, coords, obs.event_is_open, obs.event_caption_id, obs.event_coords)

    Example output:
        (Array (8,), {"aux_ce": 4.61, "aux_acc": 0.0, "aux_coord_l1": 0.21, "aux_label_frac": 1.0})
    """
    num_captions = logits.shape[-1]  # int
    has_open = jnp.any(event_is_open, axis=-1)  # bool[b]
    k = jnp.argmax(event_is_open, axis=-1)  # int[b], open event slot
    cap = jnp.take_along_axis(event_caption_id, k[:, None], axis=-1)[:, 0]  # int[b]
    xy = jnp.take_along_axis(event_coords, k[:, None, None], axis=1)[:, 0, :]  # int[b, 2]

    ce_valid = has_open & (cap != UNK_CAPTION_ID) & (cap < num_captions)  # bool[b]
    safe_cap = jnp.where(ce_valid, cap, 0)  # int[b]
    logp = jax.nn.log_softmax(logits, axis=-1)  # float32[b, C]
    ce = -jnp.take_along_axis(logp, safe_cap[:, None], axis=-1)[:, 0]  # float32[b]
    ce = jnp.where(ce_valid, ce, 0.0)  # float32[b]

    coord_valid = has_open & jnp.all(xy >= 0, axis=-1)  # bool[b]
    target = xy.astype(jnp.float32) / 255.0  # float32[b, 2]
    l1 = jnp.mean(jnp.abs(pred_coords - target), axis=-1)  # float32[b]
    l1 = jnp.where(coord_valid, l1, 0.0)  # float32[b]

    loss = ce + coord_weight * l1  # float32[b]

    n_ce = jnp.maximum(jnp.sum(ce_valid), 1)  # int scalar
    n_xy = jnp.maximum(jnp.sum(coord_valid), 1)  # int scalar
    correct = (jnp.argmax(logits, axis=-1) == cap) & ce_valid  # bool[b]
    metrics = {
        "aux_ce": jnp.sum(ce) / n_ce,
        "aux_acc": jnp.sum(correct) / n_ce,
        "aux_coord_l1": jnp.sum(l1) / n_xy,
        "aux_label_frac": jnp.mean(ce_valid.astype(jnp.float32)),
    }  # dict[str, jax.Array]
    return loss, metrics
