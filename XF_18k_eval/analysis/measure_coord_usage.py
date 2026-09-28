"""
measure_coord_usage.py

Answers "does XF's action expert actually USE the caption stream -- and in
particular the current subgoal's target coordinates -- when it predicts an
action?" by direct counterfactual intervention on a trained checkpoint,
rather than by reading weights or attention maps.

Why this exists. The 8k gate-init-fixed checkpoint has a live fusion path
(message ~1e-3 of the frame stream, caption-sensitive), yet VideoUnmask
success did not move (34.0% vs 36.0%), with ~64% of episodes acting
confidently at the WRONG container. The execution caption there is
"pick up the container at <bbox> that hides the <color> cube": the text is
coordinate-stripped, so the target location reaches the model ONLY through
EventEncoder's 4x4 spatial cell code (event_encoder.py:728). The offline
check (dump_subgoal_captions.py, 2026-09-23) showed that 4x4 code still
separates the target from a real distractor in 88-95% of pairs -- so the
information mostly survives quantization, and the open question is whether
the model uses it at all.

What it measures. For each sample, actions are predicted with a FIXED noise
draw, then re-predicted with exactly one memory input changed; the reported
number is ||a_variant - a_base|| / ||a_base|| over the action chunk:
  coord_moved      the OPEN event's coords shifted 56 px (one container
                   spacing) in x toward the other image half -- changes the
                   4x4 cell. Directly: "do the actions follow the target?"
  coord_removed    the OPEN event's coords set to -1 (has_coords False).
  no_symbolic      gateinit: E hidden from the single modulator (mem_mask False on E).
                   symroute: the WHOLE symbolic stream (E and C) hidden from sym_mem_attn,
                   null token kept. (Named "no_event_tokens" in the 2026-09-23 gateinit run.)
  no_fusion        F used instead of F' (fusion bypassed), E kept.
  captions_rolled  event text/mask/coords taken from another sample of the
                   same category (frames and alignment kept).
  frames_rolled    REFERENCE: perceptual memory taken from another sample.
  noise_resampled  REFERENCE: same inputs, different noise draw.
Grouped by category of the OPEN event: "container" (template mentions
"container" and has coords -- the VideoUnmask family), "other_coords", and
"no_coords".

Also reports the param norms of EventEncoder's coordinate/flag projections
against their analytic init (normal stddev 0.002), and the fusion out_proj
norm as a provenance check (at init ~2.048 means the resume re-init bug hit
this checkpoint).

Self-checks: the hand-assembled memory must equal model.embed_memory
exactly, and the jitted sampler fed that memory must match
model.sample_actions on the same noise; the job aborts otherwise.

GPU: A10G. The ~3B backbone must run; T4's 16 GB is too tight for the
restored bf16 params plus the live model copy.

Role in the system: read-only diagnostic. Writes nothing to any volume,
trains nothing, never edits robomme_policy_learning/.

Run with:
    modal run XF_18k_eval/analysis/measure_coord_usage.py
"""

import pathlib

import modal

# str, str -- released policy repo, and this project's XF code.
POLICY_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning")
XF_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "xattn_fusion")

# str -- mount points, MUST match launch_xf_training.py exactly.
CKPT_VOLUME_PATH = "/ckpts"
MAIN_DATA_VOLUME_PATH = "/xf_data"
TRAINING_VOLUME_PATH = "/xf_training"
FEATURE_SHARD_PATHS = [f"/xf_features_shard_{i}" for i in range(4)]

REPO_ID = "xf_full_suite"  # str, must match launch_xf_training.py
EXP_NAME = "full-16task-xattn-fusion-gateinit0.1"  # str, the gate-init-fixed run
CKPT_STEP = 7999  # int, the checkpoint scored on VideoUnmask at 34.0%
VARIANT = "gateinit"  # str, launch_xf_training VARIANTS key -- "symroute" for the symbolic-route runs

# int, samples per counterfactual batch (all one category). 8 for the 2026-09-23 gateinit run; 4 since
# 2026-09-26 -- the symroute model (+sym head, 835-token memory) OOM'd at b=8 on the first counterfactual
# sample twice. Every metric is per-sample, so this does not change what is measured (24 per category).
BATCH_SIZE = 4
PER_CATEGORY = 24  # int, target samples per category (3 batches of 8)
LOADER_BATCH = 16  # int, loader batch size while scanning for samples
MAX_LOADER_BATCHES = 120  # int, scan cap (1920 samples)
COORD_SHIFT_PX = 56  # int, ~one container spacing (median measured 51-57 px)
CATEGORIES = ["container", "other_coords", "no_coords"]  # list[str]
VARIANTS = [
    "coord_moved", "coord_removed", "no_symbolic", "no_subgoal_cond", "no_fusion",
    "captions_rolled", "frames_rolled", "noise_resampled",
]  # list[str]

app = modal.App("xf-measure-coord-usage")  # modal.App

ckpt_volume = modal.Volume.from_name("robomme-mme-vla-ckpts")  # modal.Volume
main_data_volume = modal.Volume.from_name("xf-full-suite-data")  # modal.Volume
training_volume = modal.Volume.from_name("xf-full-suite-training")  # modal.Volume
feature_shard_volumes = [modal.Volume.from_name(f"xf-features-shard-{i}") for i in range(4)]  # list[modal.Volume]
# modal.Volume -- durable results store (2026-09-27): Modal log retention silently dropped whole
# containers' SUMMARY / aux lines twice, so every measure() call now also writes its full result JSON here.
results_store = modal.Volume.from_name("xf-measure-results", create_if_missing=True)
RESULTS_STORE_PATH = "/measure_results"  # str

# dict[str, modal.Volume]
volumes = {
    CKPT_VOLUME_PATH: ckpt_volume,
    MAIN_DATA_VOLUME_PATH: main_data_volume,
    TRAINING_VOLUME_PATH: training_volume,
    RESULTS_STORE_PATH: results_store,
    **{path: vol for path, vol in zip(FEATURE_SHARD_PATHS, feature_shard_volumes)},
}

# modal.Image -- identical to launch_xf_training.py's / measure_fusion_message.py's.
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


def _assemble_memory(model, obs, *, use_fusion=True, drop_symbolic=False, drop_cond=False):
    """
    What it does:
        Rebuilds XFModel.embed_memory step by step so single pieces can be
        switched off: fusion (use F instead of F') or the event tokens E
        (hidden from the modulator via its memory mask). With both flags at
        their defaults this must equal model.embed_memory exactly (checked
        by the caller).

    Returns:
        tuple[jax.Array, jax.Array] -- (mem [b, budget+e, 1024],
        mmask [b, budget+e] bool).

    Example input:
        _assemble_memory(model, obs, use_fusion=False)

    Example output:
        (Array (8, 526, 1024), Array (8, 526))
    """
    from xattn_fusion.mme_vla_suite.models.representation.hybrid_mem import assemble_memory

    tokens, _, _ = model.mem_encoder(obs.static_image_emb, obs.static_pos_emb, obs.static_state_emb)
    e_tok, c_flat, c_mask = model.event_encoder(
        obs.event_text_tokens, obs.event_text_mask, obs.event_coords, obs.event_start,
        obs.event_is_demo, obs.event_is_open, obs.event_num_frames, obs.event_mask,
        embed_fn=lambda t: model.PaliGemma.llm(t, method="embed"),
        pos_embedder=model._pos_embedder,
    )
    import jax.numpy as jnp

    # Mirrors xf_pi0.XFModel.embed_memory exactly (guarded by the caller), incl. the q_proj-fix
    # variant (2026-09-27): event tags on frames/E/C, then task-aware open-mask fusion.
    import jax.numpy as jnp

    if getattr(model, "use_event_tag", False):
        tag = model.event_tag.value  # [E, 1024]
        n_e = e_tok.shape[1]  # int
        lc = obs.event_text_mask.shape[-1]  # int
        idx = obs.static_token_event_idx  # int32[b, budget]
        tokens = tokens + jnp.where((idx >= 0)[..., None], tag[jnp.clip(idx, 0, n_e - 1)], 0).astype(tokens.dtype)
        e_tok = e_tok + tag[None, :n_e].astype(e_tok.dtype)
        c_tag = jnp.concatenate([jnp.repeat(tag[:n_e], lc, axis=0), jnp.zeros((1, tag.shape[-1]), tag.dtype)], axis=0)
        c_flat = c_flat + c_tag[None].astype(c_flat.dtype)
    if not use_fusion:
        f = tokens
    elif getattr(model, "use_qfix", False):
        query_ctx = model.query_context(
            obs.tokenized_prompt, obs.tokenized_prompt_mask, e_tok, obs.event_is_open,
            embed_fn=lambda t: model.PaliGemma.llm(t, method="embed"),
        )
        f = model.fusion(tokens, c_flat, obs.static_token_event_idx, obs.event_text_mask, query_ctx=query_ctx)
    else:
        f = model.fusion(tokens, c_flat, obs.static_token_event_idx, obs.event_text_mask)
    budget = tokens.shape[1]  # int
    if getattr(model, "symbolic_route", False):
        # Mirrors xf_pi0.XFModel.embed_memory's symroute branch exactly (guarded by the caller):
        # [F' | E + tag0 ; C + tag1], where XFHistoryBlock routes everything after `budget` to
        # sym_mem_attn. drop_symbolic hides the WHOLE symbolic stream (E and C) but keeps the
        # always-valid null token (last C position) so the symbolic softmax stays well-defined.
        tags = model.sym_type_emb.value  # [2, 1024]
        mem = jnp.concatenate([f, e_tok + tags[0], c_flat + tags[1]], axis=1)  # [b, budget+e+e*lc+1, 1024]
        mmask = jnp.concatenate([obs.static_mask, obs.event_mask, c_mask], axis=1)  # bool
        if drop_symbolic:
            mmask = mmask.at[:, budget:-1].set(False)
    else:
        mem, mmask = assemble_memory(f, e_tok, obs.static_mask, obs.event_mask, model.type_emb.value)
        if drop_symbolic:
            mmask = mmask.at[:, budget:].set(False)  # gateinit: E masked from the single modulator
    # Option B (symroute_cond): the current-subgoal conditioning term, recomputed from THIS obs so
    # coordinate/caption interventions also reach it. drop_cond zeroes it (the "no_subgoal_cond" test).
    cond = None  # jax.Array [b, 1024] or None
    if getattr(model, "use_subgoal_cond", False):
        cond = model.subgoal_cond(e_tok, obs.event_is_open, obs.event_coords)
        if drop_cond:
            cond = jnp.zeros_like(cond)
    return mem, mmask, cond


def _sample_with_memory(model, observation, noise, mem_seq, mem_mask, subgoal_cond=None, num_steps=10):
    """
    What it does:
        XFModel.sample_actions with the memory sequence supplied by the
        caller instead of computed inside, so the same prefix/noise can be
        paired with counterfactual memories. Line-for-line the Euler loop of
        xf_pi0.XFModel.sample_actions; `observation` must already be
        preprocessed.

    Returns:
        jax.Array -- actions [b, action_horizon, action_dim].

    Example input:
        _sample_with_memory(model, obs_pre, noise, mem, mmask)

    Example output:
        Array of shape (8, 50, 32)
    """
    import jax
    import jax.numpy as jnp

    from mme_vla_suite.models.integration.history_pi0 import make_attn_mask

    dt = -1.0 / num_steps  # float
    batch_size = observation.state.shape[0]  # int
    prefix_tokens, prefix_mask, prefix_ar_mask, _, _ = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    def step(carry):
        x_t, time = carry
        suffix_tokens, suffix_mask, suffix_ar_mask, _, adarms_cond = model.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        # NOT named `cond`: the while_loop's stop function below is `cond`, which shadowed it (crash
        # "'function' object has no attribute 'astype'", 2026-09-26 12:0x).
        if subgoal_cond is not None:
            adarms_cond = adarms_cond + subgoal_cond.astype(adarms_cond.dtype)  # option B, same as XFModel.sample_actions
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask_b = jnp.broadcast_to(
            prefix_mask[:, None, :], (batch_size, suffix_tokens.shape[1], prefix_mask.shape[1])
        )
        full_attn_mask = jnp.concatenate([prefix_attn_mask_b, suffix_attn_mask], axis=-1)
        step_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = model.PaliGemma.llm(
            [None, suffix_tokens], mask=full_attn_mask, positions=step_positions, kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond], mem_seq=[None, mem_seq], mem_mask=[None, mem_mask],
        )
        v_t = model.action_out_proj(suffix_out[:, -model.action_horizon:])
        return x_t + dt * v_t, time + dt

    def cond(carry):
        _, time = carry
        return time >= -dt / 2

    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x_0


# A10G ONLY (user, 2026-09-26: never A100 again -- too costly). A100-80GB was used twice that day after
# the A10G OOM'd (b=8 twice, then a slow per-batch leak at b=4 after 16 samples); fix the leak, not the GPU.
@app.function(image=image, gpu="A10G", timeout=3 * 3600, volumes=volumes)
def measure(exp_name: str = EXP_NAME, ckpt_step: int = CKPT_STEP, seed: int = 0, variant: str = VARIANT,
            aux_pass: int = 1, categories: str = ",".join(CATEGORIES)) -> dict:
    """
    What it does:
        Loads the checkpoint, scans the real training loader for samples of
        each category (by the OPEN event's template and coords), builds
        same-category batches, and for each batch predicts actions under
        the base memory and every counterfactual variant with fixed noise.

    Returns:
        dict -- {"param_norms": dict[str, dict], "results": {category:
        {variant: list[float]}}, "counts": dict[str, int],
        "guards": dict[str, float]}.

    Example input:
        measure.remote()

    Example output:
        {"results": {"container": {"coord_moved": [0.01, ...], ...}}, ...}
    """
    import os  # module
    import sys  # module

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/src")
    sys.path.insert(0, "/xf_root")
    os.chdir("/app")  # mme_vla_suite's config loader resolves paths against CWD
    # Must be set before jax is imported: JAX's default 75% preallocation left too little headroom
    # for two compiled samplers on the A10G (OOM twice, 2026-09-23).
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
    # "platform" allocator = direct cudaMalloc/cudaFree, no pooled arena. On the A10G the symroute
    # measurement died after 16 samples on a 15 KB allocation (2026-09-26) -- consistent with
    # fragmentation of JAX's default BFC pool (inference, not proven). Slower, irrelevant here.
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    import dataclasses  # module

    import flax.nnx as nnx  # module
    import jax  # module
    import jax.numpy as jnp  # module
    import numpy as np  # module

    for vol in [ckpt_volume, main_data_volume, training_volume, *feature_shard_volumes]:
        vol.reload()

    import openpi.models.model as _model  # module
    import scripts.train as _train  # module
    import mme_vla_suite.training.dataloader as _dataloader  # module (the one the XF patch rebinds)

    from xattn_fusion.mme_vla_suite.models.integration.xf_observation import preprocess_observation
    from xattn_fusion.mme_vla_suite.shared.subgoal_table import load_sentencepiece_tokenizer
    from xattn_fusion.training.launch_xf_training import _build_train_config, _patch_scripts_train_for_xf

    _patch_scripts_train_for_xf(_train)
    train_config = _build_train_config(num_train_steps=40_000, batch_size=LOADER_BATCH, resum_ckpt_id=ckpt_step,
                                       variant=variant)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)  # DataConfig

    ckpt_params_dir = pathlib.Path(TRAINING_VOLUME_PATH) / "ckpts" / REPO_ID / exp_name / str(ckpt_step) / "params"
    if not ckpt_params_dir.exists():
        raise FileNotFoundError(f"No checkpoint params at {ckpt_params_dir}")
    print(f"Loading {ckpt_params_dir} ...")
    raw_params = _model.restore_params(ckpt_params_dir, dtype=jnp.bfloat16)  # at.Params
    # Same load sequence as measure_fusion_message.py (skips .load()'s strict key check, which
    # cannot pass on this project's int-vs-string list keys; replace_by_pure_dict reconciles them).
    abstract_model = nnx.eval_shape(train_config.model.create, jax.random.key(0))
    graphdef, state = nnx.split(abstract_model)
    state.replace_by_pure_dict(raw_params)
    model = nnx.merge(graphdef, state)
    del raw_params

    # --- A. param norms vs analytic init --------------------------------------------------
    def _norm_entry(kernel, stddev):
        """
        What it does: Frobenius norm of a kernel next to its expected norm
        at a normal(stddev) init.

        Returns:
            dict -- {"norm": float, "init_expected": float, "ratio": float}.

        Example input:
            _norm_entry(model.event_encoder.spatial_proj.kernel.value, 0.002)

        Example output:
            {"norm": 1.52, "init_expected": 1.448, "ratio": 1.05}
        """
        k = np.asarray(kernel, dtype=np.float32)  # np.ndarray
        expected = stddev * float(np.sqrt(k.size))  # float
        n = float(np.linalg.norm(k))  # float
        return {"norm": n, "init_expected": expected, "ratio": n / expected}

    ee = model.event_encoder  # EventEncoder
    param_norms = {
        "event_encoder/spatial_proj": _norm_entry(ee.spatial_proj.kernel.value, 0.002),
        "event_encoder/temporal_proj": _norm_entry(ee.temporal_proj.kernel.value, 0.002),
        "event_encoder/flag_proj": _norm_entry(ee.flag_proj.kernel.value, 0.002),
        "event_encoder/embed_proj": _norm_entry(ee.embed_proj.kernel.value, 0.02),
        "fusion/blocks/0/out_proj (provenance)": _norm_entry(model.fusion.blocks[0].out_proj.kernel.value, 0.002),
        "fusion/blocks/0/q_proj": _norm_entry(model.fusion.blocks[0].q_proj.kernel.value, 0.02),
        "fusion/blocks/1/q_proj": _norm_entry(model.fusion.blocks[1].q_proj.kernel.value, 0.02),
    }
    te = np.asarray(model.type_emb.value, dtype=np.float32)  # np.ndarray [2, 1024]
    param_norms["type_emb"] = {"frame_row_norm": float(np.linalg.norm(te[0])), "event_row_norm": float(np.linalg.norm(te[1]))}
    if getattr(model, "symbolic_route", False):
        from flax import traverse_util  # module

        flat_params = traverse_util.flatten_dict(nnx.state(model, nnx.Param).to_pure_dict())  # dict[tuple, Any]

        def _by_path(substr):
            """
            What it does: returns the first param array whose "a/b/c" path
            contains `substr`.

            Returns:
                np.ndarray -- the parameter value.

            Example input:
                _by_path("sym_mem_mod_dense/kernel")

            Example output:
                np.ndarray of shape (18, 1024, 2048)
            """
            for k, v in flat_params.items():
                if substr in "/".join(map(str, k)):
                    return np.asarray(v, dtype=np.float32)
            raise KeyError(substr)

        # stddev 0.002 = kernel_init_out_proj, the analytic init of sym_mem_mod_dense
        param_norms["llm/sym_mem_mod_dense (init 0.002)"] = _norm_entry(_by_path("sym_mem_mod_dense/kernel"), 0.002)
        st = np.asarray(model.sym_type_emb.value, dtype=np.float32)  # np.ndarray [2, 1024]
        param_norms["sym_type_emb"] = {"E_row_norm": float(np.linalg.norm(st[0])), "C_row_norm": float(np.linalg.norm(st[1]))}
    if getattr(model, "use_qfix", False):
        # q_proj fix readouts: learned own-event logit bias (init +2.0) and the query FiLM layers
        # (init stddev 0.002) -- both show whether the new fusion path is actually being trained.
        param_norms["fusion/own_event_bias (init 2.0)"] = float(np.asarray(model.fusion.own_event_bias.value))
        for i, film in enumerate(model.fusion.query_film):
            param_norms[f"fusion/query_film/{i} (init 0.002)"] = _norm_entry(film.kernel.value, 0.002)
    if getattr(model, "use_event_tag", False):
        param_norms["event_tag (init 0)"] = float(np.linalg.norm(np.asarray(model.event_tag.value, dtype=np.float32)))
    if getattr(model, "use_target_marker", False):
        param_norms["target_marker (init 0)"] = float(np.linalg.norm(np.asarray(model.target_marker.value, dtype=np.float32)))
    print("param norms:", param_norms)

    # --- B. collect samples by category -----------------------------------------------------
    sp = load_sentencepiece_tokenizer()  # sentencepiece processor
    loader = _dataloader.create_data_loader(
        train_config.dataset_path, data_config,
        history_config=train_config.model.history_config,
        action_horizon=train_config.model.action_horizon,
        batch_size=LOADER_BATCH, shuffle=True, num_batches=MAX_LOADER_BATCHES, num_workers=0, seed=seed,
    )
    # Samples are stored as per-sample numpy LEAVES (host memory), not XFObservation objects: a
    # single-sample XFObservation would fail its own "b e"-shaped typecheck, and holding whole
    # loader batches would pin device memory. Batches are rebuilt with the loader's treedef.
    cats = [c for c in categories.split(",") if c]  # list[str], this container's categories only
    pools = {c: [] for c in cats}  # dict[str, list[list[np.ndarray]]]
    treedef = None  # jax.tree_util.PyTreeDef or None
    scanned = 0  # int
    action_pools = {c: [] for c in cats}  # dict[str, list[np.ndarray]], aligned with pools
    for observation, actions_b in loader:
        leaves, treedef = jax.tree_util.tree_flatten(observation)  # list[array], PyTreeDef
        actions_np = np.asarray(actions_b)  # np.ndarray [b, ah, ad]
        leaves_np = [np.asarray(l) for l in leaves]  # list[np.ndarray]
        is_open_b = np.asarray(observation.event_is_open)  # bool[b, e]
        tokens_b = np.asarray(observation.event_text_tokens)  # int32[b, e, lc]
        tmask_b = np.asarray(observation.event_text_mask)  # bool[b, e, lc]
        coords_b = np.asarray(observation.event_coords)  # int32[b, e, 2]
        for i in range(is_open_b.shape[0]):
            scanned += 1
            if not is_open_b[i].any():
                continue
            k = int(np.argmax(is_open_b[i]))  # int, open event slot
            toks = [int(t) for t, m in zip(tokens_b[i, k], tmask_b[i, k]) if m]  # list[int]
            text = sp.decode(toks)  # str
            has_coords = bool(np.all(coords_b[i, k] >= 0))  # bool
            cat = ("container" if "container" in text else "other_coords") if has_coords else "no_coords"
            if cat in pools and len(pools[cat]) < PER_CATEGORY:
                pools[cat].append([l[i].copy() for l in leaves_np])
                action_pools[cat].append(actions_np[i].copy())
        if all(len(p) >= PER_CATEGORY for p in pools.values()):
            break
    counts = {c: len(p) for c, p in pools.items()}  # dict[str, int]
    print(f"scanned {scanned} samples; collected {counts}")

    # --- B2. aux-head accuracy (symroute only), forward-only, BEFORE the sampler is compiled ---
    # One compiled program at a time, then jax.clear_caches(): two large compiled programs in one
    # process OOM'd the A10G repeatedly on 2026-09-23.
    aux_stats = {}  # dict[str, dict[str, float]]
    # aux_pass=0 skips this: running it and the counterfactual sampler in ONE container OOM'd the A10G
    # at the first counterfactual sample (2026-09-26 09:47, even after jax.clear_caches()).
    if aux_pass and getattr(model, "symbolic_route", False):
        loss_jit = nnx.jit(lambda m, r, o, a: m.compute_loss(r, o, a, train=False))
        for cat in cats:
            accs = {"aux_acc": [], "aux_ce": [], "aux_coord_l1": [], "aux_label_frac": [], "total_loss": [],
                    "grounding_acc": [], "grounding_ce": [], "grounding_frac": []}  # dict
            for start in range(0, len(pools[cat]) - BATCH_SIZE + 1, BATCH_SIZE):
                members = pools[cat][start:start + BATCH_SIZE]  # list[list[np.ndarray]]
                batch = jax.tree_util.tree_unflatten(treedef, [jnp.asarray(np.stack(col)) for col in zip(*members)])
                acts = jnp.asarray(np.stack(action_pools[cat][start:start + BATCH_SIZE]))  # jax.Array
                chunked, stats = loss_jit(model, jax.random.key(start), batch, acts)
                for k in [k2 for k2 in accs if k2 != "total_loss" and k2 in stats]:
                    accs[k].append(float(np.asarray(stats[k])))
                accs["total_loss"].append(float(np.mean(np.asarray(chunked))))
            aux_stats[cat] = {k: float(np.mean(v)) for k, v in accs.items() if v}
        print("aux stats:", aux_stats)
        del loss_jit
        jax.clear_caches()

    # aux_pass=2: aux/grounding statistics ONLY (own container). Running the aux pass and the
    # counterfactual sampler in one A10G container OOM'd on 2026-09-26.
    if aux_pass == 2:
        print(f"aux-only container done: {aux_stats}")
        _ret = {"param_norms": param_norms, "aux_stats": aux_stats, "results": {}, "counts": counts,
                "guards": {}, "scanned": scanned}  # dict
        _save = {"exp_name": exp_name, "ckpt_step": ckpt_step, "variant": variant, "categories": categories,
                 "aux_pass": aux_pass, **_ret}  # dict
        import json as _json  # module

        _fname = f"{RESULTS_STORE_PATH}/{exp_name}__{ckpt_step}__aux.json"  # str
        with open(_fname, "w") as _fh:
            _json.dump(_save, _fh)
        results_store.commit()
        print(f"saved results to volume xf-measure-results:{_fname}")
        return _ret

    # --- C. counterfactuals ----------------------------------------------------------------
    sample_jit = nnx.jit(_sample_with_memory)

    def _rel(a, b):
        """
        What it does: per-sample ||a-b|| / ||b|| over the action chunk.

        Returns:
            list[float] -- one value per sample.

        Example input:
            _rel(actions_variant, actions_base)

        Example output:
            [0.012, 0.009, ...]
        """
        a = np.asarray(a, dtype=np.float32).reshape(a.shape[0], -1)  # np.ndarray
        b = np.asarray(b, dtype=np.float32).reshape(b.shape[0], -1)  # np.ndarray
        return list(np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-8))

    def _roll(obs, names):
        """
        What it does: rolls the named observation fields by one along the
        batch axis (each sample gets the next sample's values).

        Returns:
            XFObservation -- copy with those fields rolled.

        Example input:
            _roll(obs, ["event_text_tokens", "event_text_mask", "event_coords"])

        Example output:
            XFObservation(...)
        """
        return dataclasses.replace(obs, **{n: jnp.roll(getattr(obs, n), 1, axis=0) for n in names})

    results = {c: {v: [] for v in VARIANTS} for c in cats}  # dict[str, dict[str, list[float]]]
    guards = {"memory_max_abs_diff": 0.0, "sampler_rel_diff": 0.0}  # dict[str, float]
    rng = jax.random.key(seed)  # jax PRNGKey

    for cat in cats:
        pool = pools[cat]  # list[XFObservation]
        for start in range(0, len(pool) - BATCH_SIZE + 1, BATCH_SIZE):
            members = pool[start:start + BATCH_SIZE]  # list[list[np.ndarray]]
            stacked = [jnp.asarray(np.stack(col)) for col in zip(*members)]  # list[jax.Array]
            batch = jax.tree_util.tree_unflatten(treedef, stacked)  # XFObservation
            obs = preprocess_observation(None, batch, train=False)  # XFObservation
            rng, k_noise, k_noise2 = jax.random.split(rng, 3)
            noise = jax.random.normal(k_noise, (BATCH_SIZE, model.action_horizon, model.action_dim))  # jax.Array
            noise2 = jax.random.normal(k_noise2, noise.shape)  # jax.Array

            mem, mmask, cond = _assemble_memory(model, obs)
            ref_mem, ref_mask, _, _, ref_stats = model.embed_memory(obs)
            if cond is not None:
                guards["cond_max_abs_diff"] = max(
                    guards.get("cond_max_abs_diff", 0.0),
                    float(jnp.max(jnp.abs(cond.astype(jnp.float32) - ref_stats["_subgoal_cond"].astype(jnp.float32)))),
                )
                if guards["cond_max_abs_diff"] > 1e-2:
                    raise RuntimeError(f"subgoal cond guard failed: {guards}")
            guards["memory_max_abs_diff"] = max(
                guards["memory_max_abs_diff"],
                float(jnp.max(jnp.abs(mem.astype(jnp.float32) - ref_mem.astype(jnp.float32)))),
            )
            if not bool(jnp.all(mmask == ref_mask)) or guards["memory_max_abs_diff"] > 1e-2:
                raise RuntimeError(f"memory assembly guard failed: {guards}")

            a_ref = None  # jax.Array or None
            if guards["sampler_rel_diff"] == 0.0 and start == 0:
                # Run ONCE, BEFORE sample_jit is ever compiled, then drop its executable: compiling
                # the real sample_actions alongside the compiled sampler OOM'd the A10G twice
                # (2026-09-23 17:32 at b=8 and 17:36 at b=2 -- a fixed ~1.2 GB allocation, not
                # batch-dependent). 2-sample slice; samples are independent along the batch axis.
                small = jax.tree.map(lambda x: x[:2], batch)  # XFObservation, b=2
                a_ref = np.asarray(nnx.jit(lambda m, o, n: m.sample_actions(None, o, noise=n))(model, small, noise[:2]))  # np.ndarray
                jax.clear_caches()
            a_base = sample_jit(model, obs, noise, mem, mmask, cond)  # jax.Array
            if a_ref is not None:
                rel = float(np.mean(_rel(np.asarray(a_base[:2]), a_ref)))  # float
                guards["sampler_rel_diff"] = max(rel, 1e-12)  # nonzero marks "guard ran"
                if rel > 1e-2:
                    raise RuntimeError(f"sampler guard failed: rel diff {rel}")

            variants = {}  # dict[str, tuple[XFObservation, dict]]
            coords = np.asarray(obs.event_coords).copy()  # np.ndarray int32[b, e, 2]
            removed = coords.copy()  # np.ndarray
            open_idx = np.argmax(np.asarray(obs.event_is_open), axis=1)  # np.ndarray int[b]
            for i, k in enumerate(open_idx):
                if np.all(coords[i, k] >= 0):
                    x = int(coords[i, k, 1])  # int
                    coords[i, k, 1] = min(x + COORD_SHIFT_PX, 255) if x < 128 else max(x - COORD_SHIFT_PX, 0)
                removed[i, k] = -1
            variants["coord_moved"] = (dataclasses.replace(obs, event_coords=jnp.asarray(coords)), {})
            variants["coord_removed"] = (dataclasses.replace(obs, event_coords=jnp.asarray(removed)), {})
            variants["no_symbolic"] = (obs, {"drop_symbolic": True})
            variants["no_subgoal_cond"] = (obs, {"drop_cond": True})
            variants["no_fusion"] = (obs, {"use_fusion": False})
            variants["captions_rolled"] = (_roll(obs, ["event_text_tokens", "event_text_mask", "event_coords"]), {})
            variants["frames_rolled"] = (_roll(obs, [
                "static_image_emb", "static_pos_emb", "static_state_emb", "static_mask", "static_token_event_idx",
            ]), {})

            for name, (v_obs, kw) in variants.items():
                v_mem, v_mask, v_cond = _assemble_memory(model, v_obs, **kw)
                # v_obs (not obs) for the prefix too: with the current-image target marker (2026-09-27)
                # the prefix depends on the open event's coords, so coordinate/caption interventions must
                # reach it. Identical for models without the marker (their prefix ignores event fields).
                a_v = sample_jit(model, v_obs, noise, v_mem, v_mask, v_cond)  # jax.Array
                results[cat][name] += [float(x) for x in _rel(a_v, a_base)]
                print(f"  {cat} @{start} {name}: {np.mean(results[cat][name][-BATCH_SIZE:]):.4f}", flush=True)
            a_n = sample_jit(model, obs, noise2, mem, mmask, cond)  # jax.Array
            results[cat]["noise_resampled"] += [float(x) for x in _rel(a_n, a_base)]
            print(f"{cat} batch @{start}: " + ", ".join(
                f"{v}={np.mean(results[cat][v][-BATCH_SIZE:]):.4f}" for v in VARIANTS))

    # Summary printed REMOTELY too, so a detached run's result is readable from `modal app logs`.
    for cat in cats:
        for v in VARIANTS:
            xs = results[cat][v]  # list[float]
            if xs:
                print(f"SUMMARY {cat:13s} {v:16s} mean={np.mean(xs):.4f} median={np.median(xs):.4f} n={len(xs)}")
    print(f"SUMMARY guards={guards} counts={counts}")
    _ret = {"param_norms": param_norms, "results": results, "counts": counts, "guards": guards, "scanned": scanned,
            "aux_stats": aux_stats}  # dict
    _save = {"exp_name": exp_name, "ckpt_step": ckpt_step, "variant": variant, "categories": categories,
             "aux_pass": aux_pass, **_ret}  # dict
    import json as _json  # module

    _fname = f"{RESULTS_STORE_PATH}/{exp_name}__{ckpt_step}__{'aux' if aux_pass == 2 else categories.replace(',', '+')}.json"  # str
    with open(_fname, "w") as _fh:
        _json.dump(_save, _fh)
    results_store.commit()
    print(f"saved results to volume xf-measure-results:{_fname}")
    return _ret


@app.local_entrypoint()
def main(exp_name: str = EXP_NAME, ckpt_step: int = CKPT_STEP, seed: int = 0, variant: str = VARIANT):
    """
    What it does:
        Runs the remote measurement, saves the raw result JSON next to this
        file, and prints a per-category table of mean/median relative
        action change for every variant.

    Returns:
        None -- prints to stdout and writes coord_usage_<exp>_<step>.json.

    Example input:
        modal run XF_18k_eval/analysis/measure_coord_usage.py

    Example output:
        (stdout) category x variant table
    """
    import json
    import statistics

    r = measure.remote(exp_name=exp_name, ckpt_step=ckpt_step, seed=seed, variant=variant)  # dict
    out = pathlib.Path(__file__).resolve().parent / f"coord_usage_{exp_name}_{ckpt_step}.json"  # pathlib.Path
    out.write_text(json.dumps(r, indent=1))

    print(f"\n===== coord usage -- {exp_name} @ {ckpt_step} =====")
    print(f"scanned {r['scanned']}, counts {r['counts']}, guards {r['guards']}")
    for k, v in r["param_norms"].items():
        print(f"  {k}: {v}")
    for cat, st in r.get("aux_stats", {}).items():
        print(f"  aux [{cat}]: {st}")
    for cat, per in r["results"].items():
        print(f"\n--- {cat} (n={len(per['noise_resampled'])}) --- rel action change: mean / median")
        for v, xs in per.items():
            if xs:
                print(f"  {v:18s} {statistics.fmean(xs):.4f} / {statistics.median(xs):.4f}")
    print(f"\nsaved {out}")


@app.local_entrypoint()
def launch_detached(exp_name: str = EXP_NAME, ckpt_step: int = CKPT_STEP, seed: int = 0, variant: str = VARIANT,
                    aux_pass: int = 1, split: int = 1):
    """
    What it does:
        Spawns `measure` so it survives the local client exiting (a blocking
        .remote() is cancelled with its caller even under --detach). MUST be
        run with `modal run --detach ...::launch_detached`. Results: the
        "param norms:", "aux stats:" and "SUMMARY" lines in `modal app logs`.

    Returns:
        None -- prints the spawned call id.

    Example input:
        modal run --detach XF_18k_eval/analysis/measure_coord_usage.py::launch_detached \
            --variant symroute --exp-name full-16task-xattn-fusion-symroute --ckpt-step 1499

    Example output:
        (stdout) spawned fc-...
    """
    # split=1: one A10G container PER CATEGORY, in parallel (same total GPU time, no single process
    # runs long enough to hit the slow memory creep seen 2026-09-26). The same seed makes every
    # container scan the same loader stream, so each category gets the same samples as an unsplit run.
    groups = [[c] for c in CATEGORIES] if split else [CATEGORIES]  # list[list[str]]
    if split and aux_pass:
        # aux statistics in their OWN container (aux_pass=2); counterfactual containers skip them.
        call = measure.spawn(exp_name=exp_name, ckpt_step=ckpt_step, seed=seed, variant=variant,
                             aux_pass=2, categories=",".join(CATEGORIES))  # modal.FunctionCall
        print(f"spawned {call.object_id} (aux-only, all categories)")
    for g in groups:
        call = measure.spawn(exp_name=exp_name, ckpt_step=ckpt_step, seed=seed, variant=variant,
                             aux_pass=0 if split else aux_pass, categories=",".join(g))  # modal.FunctionCall
        print(f"spawned {call.object_id} ({variant} {exp_name} @ {ckpt_step}, categories={g})")
