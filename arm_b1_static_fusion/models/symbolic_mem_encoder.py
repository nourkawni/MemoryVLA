"""
symbolic_mem_encoder.py

Builds Arm B1's symbolic memory stream M_sym: a dedicated width-2048 -> 1024
projection applied to PaliGemma token embeddings of the tokenized subgoal
history. Identical in role and implementation to
arm_d_dynamic_fusion.models.symbolic_mem_encoder.SymbolicMemoryEncoder --
kept as B1's own copy rather than imported from arm_d_dynamic_fusion so this
arm's diff stays self-contained (see README.md's isolation note). Mirrors
mme_vla_suite.models.representation.percep_mem.PerceptualMemory (which builds
M_perc) so b1_pi0.B1Model can hand both streams to
static_gated_modulator.StaticGatedModulator on equal footing.

Same architectural change described in cross_modal_gated_fusion_proposal.md
section 3.2 ("Unified memory interface") that Arm D relies on: in the
released implementation, symbolic memory bypasses the modulator entirely and
is concatenated into the VLM prompt as language tokens. Here both streams are
admitted as fixed-width token sequences M in R^(B x d) at the action-expert
modulator instead -- necessary for B1 same as it is for D, since B1 is only a
useful control for D if both arms share the same injection interface and
differ only in the gate.
"""

import flax.nnx as nnx

import openpi.shared.array_typing as at
from mme_vla_suite.models.representation.utils import kernel_init


class SymbolicMemoryEncoder(nnx.Module):
    """
    What it does:
        Owns the width-2048 -> width-1024 linear projection that turns
        PaliGemma subgoal-token embeddings into M_sym. Token embedding itself
        (subgoal token ids -> width-2048 vectors) is done by the caller via
        the shared PaliGemma embedder, since that embedder's weights live
        inside the main gemma expert stack, not here.

    Returns:
        n/a -- see __call__.

    Example input:
        SymbolicMemoryEncoder(rngs=nnx.Rngs(0), dtype=jnp.float32,
                               embed_dim=2048, output_dim=1024)

    Example output:
        a callable module; see __call__ below.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        dtype: at.DTypeLike,
        embed_dim: int,
        output_dim: int,
    ):
        self.dtype = dtype  # jax.numpy.dtype, compute dtype for the projection
        self.projector = nnx.Linear(
            embed_dim,
            output_dim,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init,
        )  # nnx.Linear, width-2048 -> width-1024

    @at.typecheck
    def __call__(
        self,
        subgoal_token_embeddings: at.Float[at.Array, "b l d_embed"],
        subgoal_token_mask: at.Bool[at.Array, "b l"],
    ):
        """
        What it does:
            Projects every subgoal token embedding independently into the
            shared width-1024 memory interface. No pooling across tokens --
            B1's symbolic budget (~64 tokens, same as Arm D) is exactly the
            tokenized-subgoal-history length, so every token is kept as its
            own memory token.

        Returns:
            tuple[jax.Array, jax.Array] -- (M_sym, mask). M_sym has shape
            [b, l, output_dim]; mask is the input mask, unchanged, forwarded
            for use as the cross-attention key/value mask downstream in
            static_gated_modulator.StaticGatedModulator.

        Example input:
            encoder(subgoal_token_embeddings=jnp.zeros((2, 64, 2048)),
                    subgoal_token_mask=jnp.ones((2, 64), dtype=bool))

        Example output:
            (Array of shape (2, 64, 1024), Array of shape (2, 64))
        """
        m_sym = self.projector(subgoal_token_embeddings)  # jax.Array [b, l, output_dim]
        return m_sym, subgoal_token_mask
