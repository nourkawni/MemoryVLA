"""
target_marker.py

Current-image target marker for XF (2026-09-27). Marks the CURRENT subgoal's
target location directly on the current front-camera image tokens, so the
pretrained vision-language model and the action expert see WHERE the target
is in what the robot sees now -- the grounding the paper's symbolic setup
gets for free by putting "pick up the container at <y, x>" in the prompt next
to the image.

Why: option B@8k and every earlier XF variant sit at the perceptual level on
VideoUnmask (34%): the caption names the target container, but its (y, x)
only reached the action expert as a number code (subgoal_cond) or through
past frames (fusion), never tied to the current image; the policy never
learned that mapping from ~220 VideoUnmask-type episodes.

Mechanism: a soft Gaussian spotlight over the 16x16 SigLIP patch grid of the
front image (224x224 input, So400m/14 -> 256 patch tokens, row-major),
centered on the open event's (y, x) (256x256 front-image pixels; relative
position is preserved by the resize), sigma = 1 patch; all-zero when the
current subgoal has no bounding box. XFModel.embed_prefix adds
weight * target_marker (a learned 2048-d vector, zero-init => the trained
model starts exactly unchanged) to those 256 tokens.

Caveat (accepted): training-time image augmentation (small random crop /
rotation of the base image) can shift the true target by up to ~1 patch
relative to the marker; the soft spotlight tolerates that.

Role in the system: used by xf_pi0.XFModel.embed_prefix when
history_config.symbolic_route.target_marker is true.
"""

import jax
import jax.numpy as jnp

PATCH_GRID = 16  # int, SigLIP So400m/14 at 224x224 -> 16x16 patches
IMAGE_PX = 256.0  # float, grounded-subgoal coordinate frame (256x256 front image)


def marker_weights(event_is_open: jax.Array, event_coords: jax.Array, sigma_patches: float = 1.0) -> jax.Array:
    """
    What it does:
        Builds the spotlight weights over the front image's 256 patch tokens
        for the OPEN event's (y, x): exp(-d^2 / (2 sigma^2)) with d the
        distance in patch units between each patch center and the target,
        normalized so the peak is 1; zero for samples with no open event or
        no bounding box.

    Returns:
        jax.Array -- float32[b, 256], row-major (patch index = 16*row + col).

    Example input:
        marker_weights(obs.event_is_open, obs.event_coords)

    Example output:
        Array of shape (4, 256), float32, max 1.0 near the target patch
    """
    has_open = jnp.any(event_is_open, axis=-1)  # bool[b]
    k = jnp.argmax(event_is_open, axis=-1)  # int[b]
    xy = jnp.take_along_axis(event_coords, k[:, None, None], axis=1)[:, 0, :]  # int[b, 2], (y, x)
    ok = has_open & jnp.all(xy >= 0, axis=-1)  # bool[b]
    target = xy.astype(jnp.float32) / IMAGE_PX * PATCH_GRID  # float32[b, 2], in patch units
    centers = jnp.arange(PATCH_GRID, dtype=jnp.float32) + 0.5  # float32[16]
    dy = centers[None, :, None] - target[:, 0, None, None]  # float32[b, 16, 1]
    dx = centers[None, None, :] - target[:, 1, None, None]  # float32[b, 1, 16]
    w = jnp.exp(-(dy**2 + dx**2) / (2.0 * sigma_patches**2))  # float32[b, 16, 16]
    return w.reshape(w.shape[0], -1) * ok[:, None].astype(jnp.float32)  # float32[b, 256]
