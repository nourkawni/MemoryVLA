"""
symbolic_mem_encoder.py

Builds Arm D's symbolic memory stream M_sym: a dedicated width-2048 -> 1024
projection applied to PaliGemma token embeddings of the tokenized subgoal
history. Mirrors the role of
mme_vla_suite.models.representation.percep_mem.PerceptualMemory (which builds
M_perc) so arm_d_pi0.ArmDModel can hand both streams to
joint_gated_modulator.JointGatedModulator on equal footing.

This is the architectural change described in cross_modal_gated_fusion_
proposal.md section 3.2 ("Unified memory interface"): in the released
implementation, symbolic memory bypasses the modulator entirely and is
concatenated into the VLM prompt as language tokens, while only perceptual
memory reaches the action-expert modulator. That asymmetry would confound any
comparison of fusion strategies with the injection interface itself, so here
both streams are admitted as fixed-width token sequences M in R^(B x d) at the
same site -- the action-expert modulator -- via a proposal-recommended shared
whole-MLP interface. See arm_d_dynamic_fusion/README.md for why this encoder
uses its own separately-parameterized projection rather than literally sharing
weights with PerceptualMemory's projector (a dimension mismatch caused by
position-embedding fusion in the released perceptual path, not a simplification
of convenience).
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
            Arm D's symbolic budget (~64 tokens, proposal section 3.2) is
            exactly the tokenized-subgoal-history length, so every token is
            kept as its own memory token.

        Returns:
            tuple[jax.Array, jax.Array] -- (M_sym, mask). M_sym has shape
            [b, l, output_dim]; mask is the input mask, unchanged, forwarded
            for use as the cross-attention key/value mask downstream in
            joint_gated_modulator.JointGatedModulator.

        Example input:
            encoder(subgoal_token_embeddings=jnp.zeros((2, 64, 2048)),
                    subgoal_token_mask=jnp.ones((2, 64), dtype=bool))

        Example output:
            (Array of shape (2, 64, 1024), Array of shape (2, 64))
        """
        m_sym = self.projector(subgoal_token_embeddings)  # jax.Array [b, l, output_dim]
        return m_sym, subgoal_token_mask
