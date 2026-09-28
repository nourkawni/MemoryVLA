"""
measure_fusion_message.py

Measures what XF's fusion cross-attention actually produces at checkpoint
18000, on real training batches: how big the message is relative to the
frame stream it is added to, whether it would be usefully sized if the gate
ever opened, and -- the discriminating question -- whether the message
depends on caption CONTENT at all.

Why this exists. Two earlier checks in XF_18k_eval/analysis/ established:
  - `fusion_gate_findings.md`: all four gates sit at ~3e-4 at step 18000 and
    have DECAYED since step 4000, while the out_proj/ffw_out kernels behind
    them moved <0.3% in 14,000 steps.
  - `caption_plumbing_findings.md`: captions nonetheless arrive. Zero
    overflow across 199,264 real frame slots, zero empty captions across
    1,329 live events, median 10 caption tokens per event, no orphan events.

So the mechanism was fed real information for 18,000 steps and gated it off
anyway. The leading (UNMEASURED) explanation is mutual starvation: out_proj's
gradient is scaled by tanh(a_x) ~ 3e-4 so the message projection stays near
its random init, and a_x's own gradient is proportional to <dL/dF', msg> --
an inner product against that near-random message, hence near-zero-mean noise
with no consistent direction, while any nonzero alpha injects the noise into
F' and is penalised. Each side starved by the other.

That story makes a sharp, falsifiable prediction this file tests directly:
**if msg is still essentially a random projection, it should barely depend on
what the captions say.** Measurement 4 below rewrites the caption stream
while holding the attention pattern fixed; a near-zero change in msg confirms
the message carries no caption content, and a large change refutes the
starvation story and points instead at the training objective simply not
wanting the information.

It also finally produces the number the eval plan asked for in its pre-flight
("pull ||tanh(alpha)*message|| / ||F||"), which needs a real forward pass on
real data and so could not come from the params-only read.

What it measures, per fusion block, per batch:
  1. ||F||, ||msg||, ||ffw||  -- raw magnitudes entering and inside the block.
  2. ||tanh(alpha)*msg|| / ||F||  at the TRAINED gate values. The eval plan's
     ratio. >1 would mean F' is being driven out of the modulator's input
     distribution; at these gate values it is expected to be minuscule, and
     the point of measuring is to have the number rather than assume it.
  3. The same ratio with gates FORCED OPEN (tanh(3.0) ~= 0.995, the same
     "fully open" constant diagnostics/debug_fusion_routing.py already uses).
     This is a counterfactual on identical weights: it says whether the
     message would be usefully sized, or catastrophically oversized, if
     training ever did open the gate.
  4. Caption sensitivity: recompute msg with the caption stream ROLLED across
     the batch, so every sample attends to a different sample's captions
     through its OWN unchanged attention mask, then report
     ||msg_rolled - msg|| / ||msg||. Near 0 => the message ignores caption
     content entirely. Large => the message is genuinely caption-dependent.
  5. |dL/d alpha| for all four gate scalars on a real batch -- the direct
     reading of how much signal the gates receive. Guarded by try/except so a
     failure here cannot cost measurements 1-4, which are the robust ones.

Self-check. Measurements 1-4 need intermediates (msg, ffw) that
GatedXAttnBlock.__call__ does not return, so this file re-implements that
forward pass. A copied forward pass can silently drift from the real one, so
on every batch and every block, `_block_forward_with_intermediates` is checked
against the real `block(f, c_flat, mask)` before any of its numbers are used,
and the job aborts if they differ.

That check is run with the gates FORCED OPEN on both sides, not at the trained
values: with tanh(a) ~ 3e-4 the gated residuals are negligible, so
f_out ~= f_in ~= f_ref whether or not the msg/ffw math is right, and a
comparison at trained gates would pass almost regardless -- proving nothing
about the very path being measured. It compares against a tight RELATIVE
tolerance (1e-3) rather than bit-exactly, because XLA may fuse the two call
sites differently and perturb bf16 results by ~1 ULP, while genuine
arithmetic drift is orders of magnitude larger. The head/dim constants and
MASK_FILL are imported from the real modules rather than restated, so they
cannot drift either.

Cost. This is the only GPU job in XF_18k_eval/. A10G, matching training's own
tier -- the ~3B PaliGemma backbone has to be built and run. batch_size=2 and
4 batches are deliberately small: every quantity here is a per-batch norm
ratio that converges immediately, not a success rate needing episode counts.

Role in the system: read-only diagnostic. Writes nothing to any volume,
trains nothing, and never edits robomme_policy_learning/.

Run with:
    modal run XF_18k_eval/analysis/measure_fusion_message.py
    modal run XF_18k_eval/analysis/measure_fusion_message.py --num-batches 8 --batch-size 2
"""

import pathlib

import modal

# str, str -- released policy repo, and this project's XF code.
POLICY_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning")
XF_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "xattn_fusion")

# str -- mount points, MUST match launch_xf_training.py exactly (feature_shard_router.py
# hardcodes the shard paths; XFDataset's feature lookups silently resolve to nothing if they drift).
CKPT_VOLUME_PATH = "/ckpts"
MAIN_DATA_VOLUME_PATH = "/xf_data"
TRAINING_VOLUME_PATH = "/xf_training"
FEATURE_SHARD_PATHS = [f"/xf_features_shard_{i}" for i in range(4)]

# str, str -- must match launch_xf_training.py's own constants; together they locate
# ckpts/{REPO_ID}/{EXP_NAME}/{step} on the training volume.
REPO_ID = "xf_full_suite"
# str, int -- DEFAULTS point at the ORIGINAL zero-gate run so previously-reported numbers
# reproduce unchanged. Pass --exp-name / --ckpt-step to measure a different run, e.g. the
# gate-init-fixed one: --exp-name full-16task-xattn-fusion-gateinit0.1 --ckpt-step 7999
EXP_NAME = "full-16task-xattn-fusion"

CKPT_STEP = 18000  # int, the checkpoint this whole XF_18k_eval folder is about
DEFAULT_BATCH_SIZE = 2  # int, small on purpose -- these are norm ratios, not success rates
DEFAULT_NUM_BATCHES = 4  # int
OPEN_GATE_VALUE = 3.0  # float, tanh(3.0) ~= 0.995 -- same "fully open" constant debug_fusion_routing.py uses

app = modal.App("xf-measure-fusion-message")  # modal.App

ckpt_volume = modal.Volume.from_name("robomme-mme-vla-ckpts")  # modal.Volume
main_data_volume = modal.Volume.from_name("xf-full-suite-data")  # modal.Volume
training_volume = modal.Volume.from_name("xf-full-suite-training")  # modal.Volume
feature_shard_volumes = [modal.Volume.from_name(f"xf-features-shard-{i}") for i in range(4)]  # list[modal.Volume]

# dict[str, modal.Volume]. All 4 shards mounted simultaneously -- XFDataset routes features by
# episode_idx % 4, so a missing shard breaks a quarter of all episodes.
volumes = {
    CKPT_VOLUME_PATH: ckpt_volume,
    MAIN_DATA_VOLUME_PATH: main_data_volume,
    TRAINING_VOLUME_PATH: training_volume,
    **{path: vol for path, vol in zip(FEATURE_SHARD_PATHS, feature_shard_volumes)},
}

# modal.Image -- identical to launch_xf_training.py's, so layers are cache hits and the model /
# dataset code here is byte-identical to what trained.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "curl", "libgl1", "libglib2.0-0")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({
        "UV_LINK_MODE": "copy", "UV_PYTHON_DOWNLOADS": "automatic",
        "UV_PROJECT_ENVIRONMENT": "/usr/local",
    })
    .add_local_dir(POLICY_LOCAL_DIR, remote_path="/app", copy=True)
    .run_commands(
        r"""sed -i 's/members = \["packages\/\*", "sandbox2\/flash_attn_jax"\]/members = ["packages\/*"]/' /app/pyproject.toml"""
    )
    .run_commands("cd /app && /root/.local/bin/uv sync --no-dev --python 3.11")
    .run_commands("cd /app && /root/.local/bin/uv pip install --system pytest wandb")
    .add_local_dir(XF_LOCAL_DIR, remote_path="/xf_root/xattn_fusion", copy=True)
)


def _block_forward_with_intermediates(block, f, c_flat, mask, gate_override=None):
    """
    What it does:
        Runs one GatedXAttnBlock's forward pass and ALSO returns the two
        intermediates the real __call__ discards -- the cross-attention
        message `msg` and the FFW output `ffw` -- which are exactly the
        quantities this diagnostic needs. The arithmetic is a line-for-line
        copy of GatedXAttnBlock.__call__ (fusion_xattn.py); the caller must
        assert bit-exactness against the real method before trusting it.

        gate_override, when given, substitutes that raw scalar for BOTH
        a_x and a_d (pre-tanh), which is how the forced-open counterfactual
        is produced without mutating the loaded model.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array] -- (f_out, msg, ffw). f_out
        has the same shape as `f`; msg has f's shape; ffw has f's shape.

    Example input:
        _block_forward_with_intermediates(model.fusion.blocks[0], F, c_flat, mask)

    Example output:
        (Array (2, 512, 1024), Array (2, 512, 1024), Array (2, 512, 1024))
    """
    import einops
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp

    from xattn_fusion.mme_vla_suite.models.representation.fusion_xattn import (
        _HEAD_DIM,
        _HEADS,
        _KV_HEADS,
    )
    from xattn_fusion.mme_vla_suite.models.representation.xf_common import MASK_FILL

    q = block.q_proj(block.norm_f(f))  # [B, Tf, H*Hd]
    kv = block.kv_proj(block.norm_kv(c_flat))  # [B, Tc, 2*Kh*Hd]
    k, v = jnp.split(kv, 2, axis=-1)  # each [B, Tc, Kh*Hd]

    q = einops.rearrange(q, "b t (h d) -> b t h d", h=_HEADS) * (_HEAD_DIM**-0.5)
    k = einops.rearrange(k, "b s (h d) -> b s h d", h=_KV_HEADS)
    v = einops.rearrange(v, "b s (h d) -> b s h d", h=_KV_HEADS)
    group = _HEADS // _KV_HEADS  # int
    k = einops.repeat(k, "b s h d -> b s (h g) d", g=group)  # [B, Tc, H, Hd]
    v = einops.repeat(v, "b s h d -> b s (h g) d", g=group)  # [B, Tc, H, Hd]

    logits = jnp.einsum("bthd,bshd->bhts", q, k, preferred_element_type=jnp.float32)  # float32[B,H,Tf,Tc]
    masked_logits = jnp.where(mask[:, None, :, :], logits, MASK_FILL)
    probs = jax.nn.softmax(masked_logits, axis=-1).astype(f.dtype)  # [B,H,Tf,Tc]
    out = jnp.einsum("bhts,bshd->bthd", probs, v)  # [B,Tf,H,Hd]
    out = einops.rearrange(out, "b t h d -> b t (h d)")
    msg = block.out_proj(out)  # [B,Tf,D]

    a_x = jnp.asarray(gate_override) if gate_override is not None else block.a_x.value  # scalar
    a_d = jnp.asarray(gate_override) if gate_override is not None else block.a_d.value  # scalar

    f_mid = f + (jnp.tanh(a_x).astype(f.dtype)) * msg  # [B,Tf,D]
    ffw = block.ffw_out(nnx.silu(block.ffw_in(block.norm_ff(f_mid))))  # [B,Tf,D]
    f_out = f_mid + (jnp.tanh(a_d).astype(f.dtype)) * ffw  # [B,Tf,D]
    return f_out, msg, ffw


def _rel(a, b, jnp):
    """
    What it does:
        Computes ||a|| / ||b|| in float32, guarding a zero denominator so a
        degenerate batch reports as None instead of producing a NaN that
        would silently contaminate the averages.

    Returns:
        float or None -- the ratio, or None when ||b|| == 0.

    Example input:
        _rel(msg, F, jnp)

    Example output:
        0.000312
    """
    na = float(jnp.linalg.norm(a.astype(jnp.float32)))  # float
    nb = float(jnp.linalg.norm(b.astype(jnp.float32)))  # float
    return (na / nb) if nb > 0 else None


@app.function(image=image, gpu="A10G", timeout=3600, volumes=volumes)
def measure(num_batches: int = DEFAULT_NUM_BATCHES, batch_size: int = DEFAULT_BATCH_SIZE, seed: int = 0,
            exp_name: str = EXP_NAME, ckpt_step: int = CKPT_STEP) -> dict:
    """
    What it does:
        Loads checkpoint 18000 into a real XFModel, builds the XF data
        loader through launch_xf_training's own patch (so the pipeline is
        the one that trained), and for each batch reproduces embed_memory's
        fusion stage with instrumentation -- reporting message magnitudes,
        the gated ratio at trained and forced-open gates, and the message's
        sensitivity to caption content. Finally attempts a gradient read on
        the four gate scalars.

    Returns:
        dict -- {"gates": dict[str, float], "per_block": list[dict] (one per
        fusion block, each holding lists of per-batch floats), "grads":
        dict[str, float] or {"error": str}, "batches": int}.

    Example input:
        measure.remote(num_batches=4, batch_size=2)

    Example output:
        {"gates": {"blocks/0/a_x": 0.000356, ...},
         "per_block": [{"msg_over_f": [0.41, 0.39], "gated_over_f": [1.4e-4, ...]}],
         "grads": {"blocks/0/a_x": 2.7e-05, ...}, "batches": 4}
    """
    import os  # module
    import sys  # module

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/src")
    sys.path.insert(0, "/xf_root")
    # mme_vla_suite's config loader resolves paths against the process CWD, which Modal defaults
    # to /root (where the driver lands), not /app where the repo lives.
    os.chdir("/app")

    import jax  # module
    import jax.numpy as jnp  # module
    import numpy as np  # module

    for vol in [ckpt_volume, main_data_volume, training_volume, *feature_shard_volumes]:
        vol.reload()  # volumes are not live-synced into an already-running container

    import openpi.models.model as _model  # module
    import scripts.train as _train  # module

    # mme_vla_suite.training.dataloader, NOT openpi.training.data_loader. These are two
    # different modules with two different create_data_loader functions, and
    # _patch_scripts_train_for_xf rebinds the FORMER (launch_xf_training.py:554/629). Importing
    # the openpi one instead silently gets the unpatched, released function, whose signature has
    # no `history_config` parameter at all -- which is exactly how this was caught.
    import mme_vla_suite.training.dataloader as _dataloader  # module

    from xattn_fusion.mme_vla_suite.models.integration.xf_observation import preprocess_observation
    from xattn_fusion.mme_vla_suite.models.representation.fusion_xattn import build_fusion_mask
    from xattn_fusion.training.launch_xf_training import (
        _build_train_config,
        _patch_scripts_train_for_xf,
    )

    # Installs the XF data-loader/config patches training itself runs under, so the batches here
    # are produced by exactly the pipeline that trained -- not a reconstruction of it.
    _patch_scripts_train_for_xf(_train)

    train_config = _build_train_config(num_train_steps=40_000, batch_size=batch_size, resum_ckpt_id=ckpt_step)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)  # DataConfig

    ckpt_params_dir = pathlib.Path(TRAINING_VOLUME_PATH) / "ckpts" / REPO_ID / exp_name / str(ckpt_step) / "params"
    print(f"Measuring run={exp_name} step={ckpt_step}")
    if not ckpt_params_dir.exists():
        raise FileNotFoundError(f"No checkpoint params at {ckpt_params_dir}")
    print(f"Loading checkpoint params from {ckpt_params_dir} ...")
    raw_params = _model.restore_params(ckpt_params_dir, dtype=jnp.bfloat16)  # at.Params

    # Deliberately NOT train_config.model.load(raw_params). That helper runs a strict
    # at.check_pytree_equality BEFORE state.replace_by_pure_dict, and orbax returns list indices
    # as STRING keys ('0'/'1') while the live nnx state uses real INT keys for fusion.blocks and
    # event_encoder.layers (both plain Python lists). replace_by_pure_dict is precisely the
    # function that reconciles that -- flax's try_convert_int -- but the strict check fires
    # first, so .load() can never reach it. Normalizing the keys before calling .load() does not
    # help either (tried, 2026-09-22): its own `remove_extra_params` branch runs
    # ocp.transform_utils.intersect_trees, which flattens and rebuilds the tree and re-stringifies
    # the keys, undoing the normalization two lines before the check.
    #
    # So this replicates .load()'s last two lines directly and skips the check. That is the same
    # sequence the working resume path uses, and the keys are handed over RAW (string-keyed)
    # because replace_by_pure_dict is the thing that expects to convert them.
    #
    # This is the THIRD distinct manifestation of this project's int-vs-string key conflict; see
    # RESEARCH_LOG.md 2026-09-21 for the first two (_merge_params vs replace_by_pure_dict).
    import flax.nnx as nnx  # module

    abstract_model = nnx.eval_shape(train_config.model.create, jax.random.key(0))
    graphdef, state = nnx.split(abstract_model)
    state.replace_by_pure_dict(raw_params)
    model = nnx.merge(graphdef, state)
    print("Model loaded.")

    gates = {}  # dict[str, float]
    for i, blk in enumerate(model.fusion.blocks):
        gates[f"blocks/{i}/a_x"] = float(jnp.asarray(blk.a_x.value).reshape(()))
        gates[f"blocks/{i}/a_d"] = float(jnp.asarray(blk.a_d.value).reshape(()))
    print(f"Trained gate scalars: {gates}")

    loader = _dataloader.create_data_loader(
        train_config.dataset_path,
        data_config,
        history_config=train_config.model.history_config,
        action_horizon=train_config.model.action_horizon,
        batch_size=batch_size,
        shuffle=True,
        num_batches=num_batches,
        num_workers=0,
        seed=seed,
    )

    num_blocks = len(model.fusion.blocks)  # int
    per_block = [
        {
            "msg_over_f": [], "ffw_over_f": [], "gated_over_f": [], "gated_ffw_over_f": [],
            "open_gated_over_f": [], "caption_sensitivity": [],
        }
        for _ in range(num_blocks)
    ]  # list[dict[str, list[float]]]
    batches_done = 0  # int
    last_batch = None  # tuple or None, kept for the gradient read

    for b_i, (observation, actions) in enumerate(loader):
        obs = preprocess_observation(jax.random.key(b_i), observation, train=False)  # XFObservation
        # The RAW observation is kept for the gradient read below, not this preprocessed one:
        # XFModel.compute_loss calls preprocess_observation itself as its first step, so handing
        # it an already-preprocessed observation would normalize the images twice.
        last_batch = (observation, actions)

        f_stream, _, _ = model.mem_encoder(obs.static_image_emb, obs.static_pos_emb, obs.static_state_emb)
        _, c_flat, _ = model.event_encoder(
            obs.event_text_tokens, obs.event_text_mask, obs.event_coords, obs.event_start,
            obs.event_is_demo, obs.event_is_open, obs.event_num_frames, obs.event_mask,
            embed_fn=lambda t: model.PaliGemma.llm(t, method="embed"),
            pos_embedder=model._pos_embedder,
        )
        mask = build_fusion_mask(obs.static_token_event_idx, obs.event_text_mask)  # bool[b, tf, e*lc+1]
        # Rolling the caption stream by one along the batch axis makes each sample attend to a
        # DIFFERENT sample's captions through its OWN unchanged mask -- same attention pattern,
        # different caption content. Some rolled key positions will be padding embeddings where
        # this sample's mask says "real"; that is acceptable for a sensitivity probe, since a
        # genuinely content-dependent message would still move substantially, and a
        # content-independent one cannot move at all.
        c_flat_rolled = jnp.roll(c_flat, shift=1, axis=0)  # [b, e*lc+1, 1024]

        f_in = f_stream  # [b, budget, 1024], input to block 0
        f_in_rolled = f_stream  # [b, budget, 1024], parallel stream for the counterfactual
        for blk_i, blk in enumerate(model.fusion.blocks):
            f_out, msg, ffw = _block_forward_with_intermediates(blk, f_in, c_flat, mask)

            # Equivalence guard. The instrumented copy above MUST match the real module's own
            # forward, or every number derived from it describes different code than trained.
            #
            # Checked with the gates FORCED OPEN on BOTH sides, not at the trained values: with
            # tanh(a) ~ 3e-4 the gated residuals are negligible, so f_out ~= f_in ~= f_ref
            # whether or not the msg/ffw math is right -- comparing at trained gates would pass
            # almost regardless and prove nothing about the very path being measured. Forcing
            # both open makes the message dominate the output, so a drift in the attention math
            # actually shows up.
            #
            # Compared against a tight RELATIVE tolerance rather than bit-exactly: XLA may fuse
            # the two call sites differently, which perturbs bf16 results by ~1 ULP. A real
            # divergence in the arithmetic is orders of magnitude larger than that.
            saved_ax, saved_ad = blk.a_x.value, blk.a_d.value  # scalars
            try:
                blk.a_x.value = jnp.asarray(OPEN_GATE_VALUE, dtype=jnp.asarray(saved_ax).dtype)
                blk.a_d.value = jnp.asarray(OPEN_GATE_VALUE, dtype=jnp.asarray(saved_ad).dtype)
                f_ref_open = blk(f_in, c_flat, mask)
            finally:
                blk.a_x.value, blk.a_d.value = saved_ax, saved_ad
            f_mine_open, _, _ = _block_forward_with_intermediates(
                blk, f_in, c_flat, mask, gate_override=OPEN_GATE_VALUE
            )
            dev = float(jnp.max(jnp.abs(f_mine_open.astype(jnp.float32) - f_ref_open.astype(jnp.float32))))
            scale = float(jnp.max(jnp.abs(f_ref_open.astype(jnp.float32)))) or 1.0  # float
            if dev / scale > 1e-3:
                raise AssertionError(
                    f"Instrumented block forward diverged from GatedXAttnBlock.__call__ at "
                    f"batch {b_i}, block {blk_i}: max abs deviation {dev:.3e} vs scale {scale:.3e} "
                    f"(relative {dev / scale:.3e}, tolerance 1e-3) with gates forced open. "
                    f"_block_forward_with_intermediates has drifted from fusion_xattn.py."
                )
            if b_i == 0:
                print(f"  [guard] block {blk_i}: instrumented forward matches real module "
                      f"(relative deviation {dev / scale:.2e}, gates forced open)")

            tanh_ax = jnp.tanh(blk.a_x.value)  # scalar
            tanh_ad = jnp.tanh(blk.a_d.value)  # scalar
            _, msg_open, _ = _block_forward_with_intermediates(
                blk, f_in, c_flat, mask, gate_override=OPEN_GATE_VALUE
            )
            f_out_rolled, msg_rolled, _ = _block_forward_with_intermediates(
                blk, f_in_rolled, c_flat_rolled, mask
            )

            per_block[blk_i]["msg_over_f"].append(_rel(msg, f_in, jnp))
            per_block[blk_i]["ffw_over_f"].append(_rel(ffw, f_in, jnp))
            per_block[blk_i]["gated_over_f"].append(_rel(tanh_ax.astype(msg.dtype) * msg, f_in, jnp))
            per_block[blk_i]["gated_ffw_over_f"].append(_rel(tanh_ad.astype(ffw.dtype) * ffw, f_in, jnp))
            per_block[blk_i]["open_gated_over_f"].append(
                _rel(jnp.tanh(jnp.asarray(OPEN_GATE_VALUE)).astype(msg_open.dtype) * msg_open, f_in, jnp)
            )
            per_block[blk_i]["caption_sensitivity"].append(_rel(msg_rolled - msg, msg, jnp))

            f_in, f_in_rolled = f_out, f_out_rolled

        batches_done += 1
        print(f"  batch {b_i}: done")

    # --- gradient read on the four gate scalars -------------------------------------------
    # Guarded: nnx state mutation inside a traced function is the fragile part of this file, and
    # a failure here must not cost measurements 1-4, which are the robust ones.
    grads = {}  # dict[str, float] or dict[str, str]
    try:
        obs, actions = last_batch
        gate_init = jnp.array(
            [v for i in range(num_blocks) for v in (gates[f"blocks/{i}/a_x"], gates[f"blocks/{i}/a_d"])],
            dtype=jnp.float32,
        )  # float32[2*num_blocks]

        def loss_wrt_gates(gate_vec):
            """
            What it does: writes `gate_vec` into the model's gate scalars and
            returns the mean flow-matching training loss on the held batch,
            so jax.grad of this function is dL/d(alpha) for all four gates.

            Returns:
                jax.Array -- scalar mean loss.

            Example input:
                loss_wrt_gates(jnp.zeros((4,)))

            Example output:
                Array(0.0021, dtype=float32)
            """
            for i, blk in enumerate(model.fusion.blocks):
                blk.a_x.value = gate_vec[2 * i]
                blk.a_d.value = gate_vec[2 * i + 1]
            return jnp.mean(model.compute_loss(jax.random.key(1234), obs, actions, train=True))

        g = jax.grad(loss_wrt_gates)(gate_init)  # float32[2*num_blocks]
        g = np.asarray(g)
        for i in range(num_blocks):
            grads[f"blocks/{i}/a_x"] = float(g[2 * i])
            grads[f"blocks/{i}/a_d"] = float(g[2 * i + 1])
        # Restore the trained values -- the mutation above is in-place on a live module.
        for i, blk in enumerate(model.fusion.blocks):
            blk.a_x.value = jnp.asarray(gates[f"blocks/{i}/a_x"])
            blk.a_d.value = jnp.asarray(gates[f"blocks/{i}/a_d"])
        print(f"Gate gradients: {grads}")
    except Exception as err:  # noqa: BLE001 -- see comment above
        print(f"Gate-gradient read FAILED (measurements 1-4 are unaffected): {err!r}")
        grads = {"error": repr(err)}

    # Print the summary from INSIDE the remote function, not only from the local entrypoint.
    # Found the hard way 2026-09-23: `modal run --detach` lets the local client detach before
    # measure() returns, so a local_entrypoint that formats the results never runs and every
    # number is silently discarded even though the job succeeded. Printing here means the
    # output lands in the Modal app logs and survives regardless of how it was launched.
    import statistics as _st  # module

    def _avg(xs):
        vals = [x for x in xs if x is not None]  # list[float]
        return _st.fmean(vals) if vals else float("nan")

    print(f"===== SUMMARY: {exp_name} @ step {ckpt_step} ({batches_done} batches) =====")
    print(f"trained gates: {gates}")
    for i, pb in enumerate(per_block):
        print(f"--- fusion block {i} ---")
        print(f"  ||msg||/||F||                  : {_avg(pb['msg_over_f']):.6g}")
        print(f"  ||ffw||/||F||                  : {_avg(pb['ffw_over_f']):.6g}")
        print(f"  ||tanh(a_x)*msg||/||F||        : {_avg(pb['gated_over_f']):.6g}")
        print(f"  ||tanh(a_d)*ffw||/||F||        : {_avg(pb['gated_ffw_over_f']):.6g}")
        print(f"  ||tanh(3.0)*msg||/||F|| (open) : {_avg(pb['open_gated_over_f']):.6g}")
        print(f"  caption sensitivity            : {_avg(pb['caption_sensitivity']):.6g}")

    return {"gates": gates, "per_block": per_block, "grads": grads, "batches": batches_done}


@app.local_entrypoint()
def main(num_batches: int = DEFAULT_NUM_BATCHES, batch_size: int = DEFAULT_BATCH_SIZE, seed: int = 0,
         exp_name: str = EXP_NAME, ckpt_step: int = CKPT_STEP):
    """
    What it does:
        Triggers the remote measurement and prints, per fusion block, the
        message magnitudes, the eval plan's gated ratio at trained and
        forced-open gates, and the caption-sensitivity number that decides
        between "the message is a near-random projection" and "the message
        is real and the objective does not want it".

    Returns:
        None -- prints to stdout.

    Example input:
        modal run XF_18k_eval/analysis/measure_fusion_message.py --num-batches 4

    Example output:
        (stdout) per-block tables and a reading of the caption-sensitivity result
    """
    import statistics

    r = measure.remote(num_batches=num_batches, batch_size=batch_size, seed=seed,
                       exp_name=exp_name, ckpt_step=ckpt_step)  # dict

    def avg(xs):
        """
        What it does: means a per-batch list, ignoring None entries from
        degenerate batches.

        Returns:
            float or None.

        Example input:
            avg([0.4, None, 0.42])

        Example output:
            0.41
        """
        vals = [x for x in xs if x is not None]  # list[float]
        return statistics.fmean(vals) if vals else None

    print(f"\n===== XF fusion message -- {exp_name} @ step {ckpt_step} =====")
    print(f"batches: {r['batches']}  (batch_size={batch_size})")
    print(f"trained gate scalars: {r['gates']}\n")

    sens_all = []  # list[float]
    for i, pb in enumerate(r["per_block"]):
        print(f"--- fusion block {i} ---")
        for label, key in [
            ("||msg|| / ||F||                (ungated message vs frame stream)", "msg_over_f"),
            ("||ffw|| / ||F||                (ungated FFW vs frame stream)", "ffw_over_f"),
            ("||tanh(a_x)*msg|| / ||F||      <-- the eval plan's ratio, TRAINED gates", "gated_over_f"),
            ("||tanh(a_d)*ffw|| / ||F||      (the FFW residual's gated contribution)", "gated_ffw_over_f"),
            ("||tanh(3.0)*msg|| / ||F||      (counterfactual: gates forced OPEN)", "open_gated_over_f"),
            ("||msg_rolled - msg|| / ||msg|| <-- caption sensitivity", "caption_sensitivity"),
        ]:
            a = avg(pb[key])  # float or None
            print(f"  {label}: {a:.6g}" if a is not None else f"  {label}: n/a")
        s = avg(pb["caption_sensitivity"])  # float or None
        if s is not None:
            sens_all.append(s)
        print("")

    if "error" in r["grads"]:
        print(f"gate gradients: UNAVAILABLE ({r['grads']['error']})")
    else:
        print("gate gradients |dL/d alpha|:")
        for k, v in r["grads"].items():
            print(f"  {k}: {v:+.6g}")

    print("\n--------------------------------- reading ---------------------------------")
    s = statistics.fmean(sens_all) if sens_all else None  # float or None
    if s is None:
        print("INCONCLUSIVE: no caption-sensitivity numbers were produced.")
    elif s < 0.01:
        print(f"Caption sensitivity {s:.4g} -- the message barely moves when the captions are")
        print("replaced. msg carries essentially no caption content, consistent with out_proj")
        print("still sitting at its init. SUPPORTS the mutual-starvation account: the gate was")
        print("never offered information worth opening for.")
    elif s < 0.2:
        print(f"Caption sensitivity {s:.4g} -- weak but nonzero caption dependence.")
        print("Partially consistent with starvation; neither account is cleanly confirmed.")
    else:
        print(f"Caption sensitivity {s:.4g} -- the message genuinely depends on caption content.")
        print("REFUTES the simple starvation account: msg is not a content-blind random")
        print("projection, so the gate stayed shut despite real information being on offer.")
        print("That moves the question to the training objective, not the initialization.")
    print()
