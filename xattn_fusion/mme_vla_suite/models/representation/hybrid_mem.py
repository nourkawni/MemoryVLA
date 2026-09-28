"""
hybrid_mem.py

Memory assembly for XF (robomme-gated-xattn-fusion agent -- see
xattn_fusion/gated-fusion-agent.md section 5). Concatenates the fused frame
stream F' with the pooled event tokens E into one memory sequence, so a
zero-frame subgoal event (one with no sampled frame inside it) still reaches
the modulator -- if only F' were used, that event's information would be
lost entirely, which is exactly the content that makes symbolic memory win on
counting tasks (temporal-alignment-agent.md's central point).

Role in the system: the last step of xf_pi0.py's XFModel.embed_memory, right
before handing (mem, mmask) to the UNCHANGED history_gemma.MemoryAttention/
MemoryRMSNorm modulator via HistoryPi0's existing mem_seq/mem_mask plumbing.
"""

import jax.numpy as jnp

import openpi.shared.array_typing as at


def assemble_memory(
    f_prime: at.Float[at.Array, "b f d"],
    e_tok: at.Float[at.Array, "b e d"],
    static_mask: at.Bool[at.Array, "b f"],
    event_mask: at.Bool[at.Array, "b e"],
    type_emb: at.Float[at.Array, "2 d"],
) -> tuple[at.Float[at.Array, "b fe d"], at.Bool[at.Array, "b fe"]]:
    """
    What it does:
        Concatenates the fused frame stream and the pooled event tokens into
        one memory sequence, tagging each half with a learned (zero-init)
        type embedding so the modulator can tell frame tokens from event
        tokens if it needs to. budget_mode="extend" (v1 default): the
        resulting sequence is strictly longer than the released budget
        (512 + E_max, not 512) -- "fixed" mode (trimming to stay at 512) is
        ablation A9, deferred past the v1 smoke test.

    Returns:
        tuple[at.Float, at.Bool] -- (mem [b, f+e, d], mmask [b, f+e]).

    Example input:
        assemble_memory(F_prime, E_tok, static_mask, event_mask, type_emb)

    Example output:
        (Array of shape (2, 528, 1024), Array of shape (2, 528))
    """
    mem = jnp.concatenate([f_prime + type_emb[0], e_tok + type_emb[1]], axis=1)  # [b, f+e, d]
    mmask = jnp.concatenate([static_mask, event_mask], axis=1)  # bool[b, f+e]
    return mem, mmask
