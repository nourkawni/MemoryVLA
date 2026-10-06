"""
symbolic_mem_encoder.py

Builds this probe's symbolic memory stream M_sym: a dedicated width-2048 ->
1024 projection applied to PaliGemma token embeddings of the tokenized
subgoal history. Identical in role and implementation to
arm_b1_static_fusion.models.symbolic_mem_encoder.SymbolicMemoryEncoder (and
arm_d_dynamic_fusion's own copy) -- kept as this probe's own copy rather than
imported from either arm so its diff stays self-contained, same isolation
convention every arm in this project follows.

Unlike B1/D, this probe never combines M_sym with a perceptual stream -- it
is the ONLY memory input to symbolic_modulator_pi0.SymbolicModulatorModel's
action-expert modulator. See that file's docstring for why symbolic memory
needs a dedicated width-1024 token encoder at all, rather than being handed
to the modulator as raw PaliGemma embeddings: mme_vla_suite.models.
integration.history_gemma.MemoryAttention (reused unchanged here) expects
mem_seq at the action-expert's own width (1024), not PaliGemma's width
(2048).
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
            this probe's symbolic budget (~64 tokens, same as B1/D) is
            exactly the tokenized-subgoal-history length, so every token is
            kept as its own memory token.

        Returns:
            tuple[jax.Array, jax.Array] -- (M_sym, mask). M_sym has shape
            [b, l, output_dim]; mask is the input mask, unchanged, forwarded
            for use as the cross-attention key/value mask downstream in
            history_gemma.MemoryAttention (via symbolic_modulator_pi0's
            embed_memory).

        Example input:
            encoder(subgoal_token_embeddings=jnp.zeros((2, 64, 2048)),
                    subgoal_token_mask=jnp.ones((2, 64), dtype=bool))

        Example output:
            (Array of shape (2, 64, 1024), Array of shape (2, 64))
        """
        m_sym = self.projector(subgoal_token_embeddings)  # jax.Array [b, l, output_dim]
        return m_sym, subgoal_token_mask
