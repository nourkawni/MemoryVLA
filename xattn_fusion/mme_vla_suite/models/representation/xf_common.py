"""
xf_common.py

Small pieces shared by event_encoder.py and fusion_xattn.py: a scale-only
RMSNorm (the same formula history_gemma.MemoryRMSNorm uses in its cond=None
branch) and the finite masked-logit fill value history_gemma.MemoryAttention
uses. Factored out after a code review found the two call sites had drifted
into near-duplicate copies (one carried a dead, unused `self.dtype`
attribute the other never had) -- a single definition means a future fix to
either the epsilon or the masking constant can't accidentally apply to only
one of the two modules.

Role in the system: imported by event_encoder.EventEncoder's internal
sub-layers and fusion_xattn.GatedXAttnBlock.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from mme_vla_suite.shared.posemb_3d import PosEmb3D

# float, masked-logit fill value -- same finite constant history_gemma.MemoryAttention uses
# (jnp.where(mask, logits, -inf) can NaN the softmax under XLA; this doesn't).
MASK_FILL = -2.3819763e38


class XFPosEmb3D(PosEmb3D):
    """
    What it does:
        Subclass of the released, unmodified PosEmb3D that (1) adds
        value-based __eq__/__hash__ (based on `dim`, the only constructor
        argument XFModel ever varies) and (2) rebuilds every precomputed
        table with plain `numpy` instead of PosEmb3D's own `jax.numpy`
        construction. __call__ is inherited unchanged -- it is never
        actually invoked anywhere in this codebase (event_encoder.py reads
        the precomputed tables as raw attributes instead, see below).

        Why (1) is needed, found only by actually running run_tentative
        (2026-09-20, not caught by the smoke test): xf_pi0.py's XFModel
        holds its PosEmb3D instance as `self._pos_embedder`, deliberately
        NOT an nnx.Module/Variable (it's a pure, stateless lookup-table
        helper with no trainable params -- see xf_pi0.py's own comment).
        flax.nnx therefore treats it as a STATIC field, not a pytree leaf.
        A real training run constructs XFModel more than once (once for
        nnx.eval_shape's params_shape computation, once for the actual live
        model) -- plain PosEmb3D has no __eq__, so Python's default
        identity comparison applies, and two separately-built instances
        (even with byte-identical precomputed tables, same `dim`) always
        compare unequal. JAX's pytree-structure comparison (used when
        merging/validating params against the live model) then sees this
        as a genuine structural mismatch and raises a large TreeDef diff.

        Why (2) is needed, found immediately after fixing (1) by running
        run_tentative again (2026-09-20): XFModel.__init__ (and therefore
        PosEmb3D.__init__) runs inside scripts/train.py's jitted
        `init_train_state.<locals>.init`. Any `jax.numpy` operation
        executed while a jax.jit trace is active produces a
        DynamicJaxprTracer -- even when, as here, every input is a plain
        static Python int (JAX's DynamicJaxprTrace intercepts primitive
        calls for the whole active trace, not per-input-concreteness).
        PosEmb3D's compute_spatial_pe*/compute_temporal_pe therefore built
        their tables as tracers of THAT ONE construction-time trace, then
        stored them as plain (non-pytree, static) attributes on `self`. The
        tables get read again later inside a *different*, separate jit
        trace (the real forward pass, scripts/train.py's ptrain_step) --
        reusing a tracer across trace boundaries is exactly what
        jax.errors.UnexpectedTracerError guards against ("a reference to an
        intermediate value... escape the scope of the transformation"), and
        that is exactly the crash this caused: "The leaked intermediate
        value was created on line posemb_3d.py:89
        (PosEmb3D.compute_spatial_pe4x4)". Plain `numpy` functions are never
        intercepted by jax's trace stack at all, so tables built this way
        are genuine concrete host arrays from the moment they're created,
        regardless of whether __init__ happens inside or outside any trace
        -- safe to store as static state and reuse across arbitrarily many
        separate traces. event_encoder.py's two read sites
        (pos_embedder.spatial_pe4x4[spatial_idx],
        pos_embedder.temporal_pe[...]) wrap these numpy tables in
        jnp.asarray(...) right before indexing them with a live tracer, so
        the numpy->jax conversion happens fresh inside the SAME forward-pass
        trace that consumes it -- no cross-trace leak risk there either.

    Returns:
        n/a -- see PosEmb3D's own docstring for behavior; equality
        semantics and the numpy/jax split of table-building vs. __call__
        differ.

    Example input:
        XFPosEmb3D(dim=768) == XFPosEmb3D(dim=768)

    Example output:
        True (would be False for two plain PosEmb3D(dim=768) instances).
        self.spatial_pe4x4 is a numpy.ndarray, not a jax.Array.
    """

    def __init__(self, dim: int, temporal_base: int = 10_000, spatial_base: int = 1_000):
        self.dim = dim  # int
        assert dim % 6 == 0, "dim must be divisible by 6"
        width = dim // 6  # int

        omega = np.arange(width) / (width - 1)  # np.ndarray float64[width]
        self.temporal_omega = 1.0 / (temporal_base**omega)  # np.ndarray float64[width]
        self.spatial_omega = 1.0 / (spatial_base**omega)  # np.ndarray float64[width]

        self.spatial_pe8x8 = self.compute_spatial_pe8x8()  # np.ndarray float32[64, 4*width]
        self.spatial_pe4x4 = self.compute_spatial_pe4x4()  # np.ndarray float32[16, 4*width]
        self.spatial_pe2x2 = self.compute_spatial_pe2x2()  # np.ndarray float32[4, 4*width]
        self.temporal_pe = self.compute_temporal_pe(max_length=2048)  # np.ndarray float32[2048, 2*width]

    def compute_spatial_pe8x8(self) -> np.ndarray:
        y, x = np.mgrid[:8, :8]
        y = 2 * y + 1
        x = 2 * x + 1
        y = np.einsum("m,d->md", y.flatten(), self.spatial_omega)
        x = np.einsum("m,d->md", x.flatten(), self.spatial_omega)
        return np.concatenate([np.sin(y), np.cos(y), np.sin(x), np.cos(x)], axis=-1).astype(np.float32)

    def compute_spatial_pe4x4(self) -> np.ndarray:
        y, x = np.mgrid[:4, :4]
        y = 4 * y + 2
        x = 4 * x + 2
        y = np.einsum("m,d->md", y.flatten(), self.spatial_omega)
        x = np.einsum("m,d->md", x.flatten(), self.spatial_omega)
        return np.concatenate([np.sin(y), np.cos(y), np.sin(x), np.cos(x)], axis=-1).astype(np.float32)

    def compute_spatial_pe2x2(self) -> np.ndarray:
        y, x = np.mgrid[:2, :2]
        y = 8 * y + 4
        x = 8 * x + 4
        y = np.einsum("m,d->md", y.flatten(), self.spatial_omega)
        x = np.einsum("m,d->md", x.flatten(), self.spatial_omega)
        return np.concatenate([np.sin(y), np.cos(y), np.sin(x), np.cos(x)], axis=-1).astype(np.float32)

    def compute_temporal_pe(self, max_length: int = 2048) -> np.ndarray:
        pos = np.arange(max_length)
        sinusoid_input = np.einsum("m,d->md", pos, self.temporal_omega)
        return np.concatenate([np.sin(sinusoid_input), np.cos(sinusoid_input)], axis=-1).astype(np.float32)

    def __eq__(self, other) -> bool:
        return isinstance(other, PosEmb3D) and self.dim == other.dim

    def __hash__(self) -> int:
        return hash(("XFPosEmb3D", self.dim))


class RMSNorm(nnx.Module):
    """Scale-only RMSNorm (no conditioning) -- same formula as history_gemma.MemoryRMSNorm's cond=None branch."""

    def __init__(self, dim: int, rngs: nnx.Rngs, dtype=jnp.float32):
        """
        What it does: builds the learnable per-channel scale parameter.

        Returns:
            None -- sets self.scale.

        Example input:
            RMSNorm(1024, rngs=nnx.Rngs(0))

        Example output:
            <RMSNorm ...>
        """
        del dtype  # unused: __call__ always preserves the input's own dtype, never this one
        self.scale = nnx.Param(jnp.zeros((dim,), dtype=jnp.float32))  # nnx.Param, float32[dim], zero-init

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        What it does: root-mean-square normalizes the last axis of `x` in
        float32, then rescales by (1 + learned scale) before casting back to
        x's original dtype.

        Returns:
            jax.Array -- same shape and dtype as `x`.

        Example input:
            rms_norm(jnp.ones((2, 8, 1024)))

        Example output:
            Array of shape (2, 8, 1024)
        """
        dtype = x.dtype  # DTypeLike, preserve caller's dtype (e.g. bfloat16)
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)  # float32[..., 1]
        normed = x.astype(jnp.float32) * jax.lax.rsqrt(var + 1e-6)  # float32[..., dim]
        return (normed * (1 + self.scale.value)).astype(dtype)
