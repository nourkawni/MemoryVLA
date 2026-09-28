"""
smoke_test_symroute.py

One-shot pre-training verification of XF's SYMBOLIC ROUTE variant
(history_gemma_xf.py + symbolic_aux_head.py + xf_pi0.py's symbolic_route
branch, config xf-framesamp-modul-xattn-symroute.yaml), run BEFORE any GPU
time is spent training it.

Checks:
  CHECK1 warm start: loading perceptual-framesamp-modul@79999 through the
         launcher's own patched _merge_params leaves ONLY the intended new
         modules fresh (sym_mem_attn, sym_mem_mod_dense, sym_type_emb,
         sym_aux_head, event_encoder, fusion, type_emb) and loads every
         released param, including the perceptual modulator (mem_attn,
         mem_rms_norm_ffn/Dense_0).
  CHECK2 freeze filter: no new param matches the LoRA freeze filter
         (sym_mem_mod_dense would have been silently frozen under its
         original name "sym_mod_dense" -- caught by inspection 2026-09-23).
  CHECK3 real training step: on a real batch, compute_loss is finite, the
         aux metrics are finite, the open event's caption_id is a real id
         (not the UNK placeholder) for most samples, and gradients reaching
         every new module are finite and non-zero.
  CHECK4 identity: with sym_mem_mod_dense zeroed and all fusion gates at 0
         (F' == F), the symroute model's sample_actions equals the RELEASED
         FrameSamp+Modul HistoryPi0's on the same batch and noise -- proof
         the warm-started perceptual path is untouched.

GPU: A10G (full ~3B backbone forward/backward). The two models are built
one after the other, with caches cleared between, to fit in memory.

Role in the system: standalone diagnostic; writes nothing to any volume,
never edits robomme_policy_learning/.

Run with:
    modal run xattn_fusion/diagnostics/smoke_test_symroute.py
"""

import pathlib

import modal

POLICY_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning")  # str
XF_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)  # str, xattn_fusion/

CKPT_VOLUME_PATH = "/ckpts"  # str, must match launch_xf_training.py
MAIN_DATA_VOLUME_PATH = "/xf_data"  # str
TRAINING_VOLUME_PATH = "/xf_training"  # str
FEATURE_SHARD_PATHS = [f"/xf_features_shard_{i}" for i in range(4)]  # list[str]

BATCH_SIZE = 2  # int
# tuple[str, ...], top-level/leaf path fragments that are ALLOWED to be fresh-initialized
NEW_MODULE_MARKERS = ("sym_mem_attn", "sym_mem_mod_dense", "sym_type_emb", "sym_aux_head", "event_encoder", "fusion", "type_emb")
# tuple[str, ...], ALSO allowed fresh: the released FrameSamp+Modul@79999 checkpoint was trained
# WITHOUT LoRA, so XF's gemma_2b_lora adapters are always fresh on a warm start (lora_b is
# zero-init, so they are an exact no-op at step 0). Same as every earlier warm start in this project
# (RESEARCH_LOG "every 'Merging missing weight' line was a LoRA adapter"). First run of this smoke
# test (2026-09-23) flagged exactly 10 such paths, all LoRA, nothing else.
EXPECTED_FRESH_FROM_RELEASED = ("lora",)

app = modal.App("xf-smoke-test-symroute")  # modal.App

ckpt_volume = modal.Volume.from_name("robomme-mme-vla-ckpts")  # modal.Volume
main_data_volume = modal.Volume.from_name("xf-full-suite-data")  # modal.Volume
training_volume = modal.Volume.from_name("xf-full-suite-training")  # modal.Volume
feature_shard_volumes = [modal.Volume.from_name(f"xf-features-shard-{i}") for i in range(4)]  # list[modal.Volume]
volumes = {
    CKPT_VOLUME_PATH: ckpt_volume,
    MAIN_DATA_VOLUME_PATH: main_data_volume,
    TRAINING_VOLUME_PATH: training_volume,
    **{path: vol for path, vol in zip(FEATURE_SHARD_PATHS, feature_shard_volumes)},
}  # dict[str, modal.Volume]

# modal.Image -- identical to launch_xf_training.py's.
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


def _path_str(path) -> str:
    """
    What it does: joins a flattened pytree key path into "a/b/c".

    Returns:
        str -- slash-joined path.

    Example input:
        _path_str(("PaliGemma", "llm", "layers", "sym_mem_attn", "q_einsum_mem", "w"))

    Example output:
        "PaliGemma/llm/layers/sym_mem_attn/q_einsum_mem/w"
    """
    return "/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path)


@app.function(image=image, gpu="A10G", timeout=3600, memory=65536, volumes=volumes)
def smoke(mode: str = "route") -> dict:
    """
    What it does:
        Runs CHECK1-CHECK4 (see module docstring) and returns every
        measured quantity plus a pass/fail per check.

    Returns:
        dict -- {"CHECK1": {...,"ok": bool}, ..., "overall_ok": bool}.

    Example input:
        smoke.remote()

    Example output:
        {"CHECK1": {"fresh_unexpected": [], "ok": True}, ..., "overall_ok": True}
    """
    import os  # module
    import sys  # module

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/src")
    sys.path.insert(0, "/xf_root")
    os.chdir("/app")
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"  # before jax import; two models built in sequence

    import dataclasses  # module

    import flax.nnx as nnx  # module
    import jax  # module
    import jax.numpy as jnp  # module
    import numpy as np  # module
    from flax import traverse_util  # module
    from omegaconf import OmegaConf  # module

    for vol in [ckpt_volume, main_data_volume, training_volume, *feature_shard_volumes]:
        vol.reload()

    import mme_vla_suite.training.dataloader as _dataloader  # module
    import openpi.shared.nnx_utils as nnx_utils  # module
    import scripts.train as _train  # module
    from mme_vla_suite.models.integration.history_pi0 import HistoryPi0Config

    from xattn_fusion.training.launch_xf_training import _build_train_config, _patch_scripts_train_for_xf

    _patch_scripts_train_for_xf(_train)  # installs the XF _merge_params actually used in training
    train_config = _build_train_config(num_train_steps=10_000, batch_size=BATCH_SIZE, variant="symroute")
    model_config = train_config.model  # XFConfig
    results = {}  # dict[str, dict]

    # --- CHECK1: warm-start merge ---------------------------------------------------------------
    abstract_model = nnx.eval_shape(model_config.create, jax.random.key(0))
    graphdef, abstract_state = nnx.split(abstract_model)
    params_shape = abstract_state.to_pure_dict()  # dict, ShapeDtypeStruct leaves
    loaded = train_config.weight_loader.load(params_shape)  # dict, arrays where loaded, ShapeDtypeStruct where fresh
    flat = traverse_util.flatten_dict(loaded)  # dict[tuple, Any]
    fresh = ["/".join(map(str, k)) for k, v in flat.items() if isinstance(v, jax.ShapeDtypeStruct)]  # list[str]
    loaded_keys = ["/".join(map(str, k)) for k, v in flat.items() if not isinstance(v, jax.ShapeDtypeStruct)]  # list[str]
    fresh_unexpected = [
        p for p in fresh if not any(m in p for m in NEW_MODULE_MARKERS + EXPECTED_FRESH_FROM_RELEASED)
    ]  # list[str]
    must_load = ["layers/mem_attn/", "layers/mem_rms_norm_ffn/Dense_0/"]  # list[str]
    must_load_ok = {m: any(m in p for p in loaded_keys) for m in must_load}  # dict[str, bool]
    must_be_fresh = ["sym_mem_attn", "sym_mem_mod_dense", "sym_aux_head", "sym_type_emb"]  # list[str]
    must_fresh_ok = {m: any(m in p for p in fresh) for m in must_be_fresh}  # dict[str, bool]
    results["CHECK1"] = {
        "n_loaded": len(loaded_keys), "n_fresh": len(fresh), "fresh_unexpected": fresh_unexpected[:20],
        "must_load": must_load_ok, "must_be_fresh": must_fresh_ok,
        "ok": (not fresh_unexpected) and all(must_load_ok.values()) and all(must_fresh_ok.values()),
    }
    print("CHECK1", results["CHECK1"])

    # --- CHECK2: freeze filter -------------------------------------------------------------------
    freeze_filter = model_config.get_freeze_filter()  # nnx filter
    frozen_state = abstract_state.filter(freeze_filter)  # nnx.State
    frozen_paths = ["/".join(map(str, k)) for k in traverse_util.flatten_dict(frozen_state.to_pure_dict()).keys()]  # list[str]
    frozen_new = [p for p in frozen_paths if any(m in p for m in NEW_MODULE_MARKERS)]  # list[str]
    frozen_perc_mod = [p for p in frozen_paths if "mem_attn" in p or "mem_rms_norm_ffn" in p]  # list[str]
    results["CHECK2"] = {
        "n_frozen": len(frozen_paths), "frozen_new_params": frozen_new[:20],
        "frozen_modulator_params": frozen_perc_mod[:5], "ok": not frozen_new and not frozen_perc_mod,
    }
    print("CHECK2", results["CHECK2"])

    # --- real batch -----------------------------------------------------------------------------
    data_config = train_config.data.create(train_config.assets_dirs, model_config)  # DataConfig
    loader = _dataloader.create_data_loader(
        train_config.dataset_path, data_config, history_config=model_config.history_config,
        action_horizon=model_config.action_horizon, batch_size=BATCH_SIZE, shuffle=True,
        num_batches=1, num_workers=0, seed=0,
    )
    observation, actions = next(iter(loader))  # XFObservation, jax.Array
    # float, deterministic loader (seed=0, num_workers=0) -- lets the orchestrator confirm the "route"
    # and "ref" containers saw the SAME batch before comparing their actions.
    batch_checksum = float(np.sum(np.asarray(observation.static_image_emb, dtype=np.float64))) + float(
        np.sum(np.asarray(observation.state, dtype=np.float64)))
    noise = jax.random.normal(jax.random.key(2), (BATCH_SIZE, model_config.action_horizon, model_config.action_dim))  # jax.Array

    if mode == "ref":
        # CHECK4 reference half, in its OWN container: two full models in one process OOM'd the
        # A10G twice (2026-09-23) because JAX did not release the first model's buffers.
        released_hc = OmegaConf.load("/app/src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul.yaml")  # DictConfig
        released_cfg = HistoryPi0Config(
            pi05=True, action_horizon=model_config.action_horizon, use_history=True, history_config=released_hc,
            discrete_state_input=False, paligemma_variant=model_config.paligemma_variant,
            action_expert_variant=model_config.action_expert_variant,
        )  # HistoryPi0Config
        ref_model = nnx.eval_shape(lambda: released_cfg.create(jax.random.key(0)))
        g_ref, s_ref = nnx.split(ref_model)
        import openpi.models.model as _model  # module

        ref_params = _model.restore_params(pathlib.Path(CKPT_VOLUME_PATH) / "perceptual-framesamp-modul/79999/params", dtype=jnp.bfloat16)
        s_ref.replace_by_pure_dict(ref_params)
        # The released checkpoint has no LoRA adapters but this reference uses the same
        # gemma_2b_lora variant as XF, so those leaves stay ShapeDtypeStruct placeholders (first
        # "ref" run, 2026-09-23 18:1x: TypeError on a (18, 8, 256, 32) lora leaf). Zero-filled LoRA is
        # an exact no-op -- the same effective state as the route model's fresh LoRA (lora_b = 0).
        flat_ref = traverse_util.flatten_dict(s_ref.to_pure_dict())  # dict[tuple, Any]
        missing = ["/".join(map(str, k)) for k, v in flat_ref.items() if isinstance(v, jax.ShapeDtypeStruct)]  # list[str]
        assert missing and all("lora" in p for p in missing), f"non-LoRA params missing from released ckpt: {missing[:10]}"
        flat_ref = {k: (jnp.zeros(v.shape, v.dtype) if isinstance(v, jax.ShapeDtypeStruct) else v) for k, v in flat_ref.items()}
        s_ref.replace_by_pure_dict(traverse_util.unflatten_dict(flat_ref))
        ref_model = nnx.merge(g_ref, s_ref)
        a_ref = np.asarray(nnx.jit(lambda m, o, n: m.sample_actions(None, o, noise=n))(ref_model, observation, noise), dtype=np.float32)  # np.ndarray
        return {"actions": a_ref.tolist(), "batch_checksum": batch_checksum}

    # --- build the live symroute model with the merged warm-start params -------------------------
    # Fresh values are produced INSIDE jax.jit and only the fresh leaves are returned, so XLA
    # dead-code-eliminates every other param init -- including the released llm that
    # HistoryPi0.__init__ builds before XFModel replaces it. The first version created the model
    # eagerly, which materialized two full ~3B llms at once and OOM'd the A10G (2026-09-23);
    # train.py's own init is jitted, so training never had that problem.
    key_by_str = {"/".join(map(str, k)): k for k in flat}  # dict[str, tuple]
    fresh_set = set(fresh)  # set[str]

    def _fresh_init():
        """
        What it does: builds the model and returns ONLY the fresh-param leaves.

        Returns:
            dict[str, jax.Array] -- "a/b/c" path -> freshly initialized value.

        Example input:
            jax.jit(_fresh_init)()

        Example output:
            {"sym_aux_head/hidden/kernel": Array(...), ...}
        """
        m = model_config.create(jax.random.key(0))
        fl = traverse_util.flatten_dict(nnx.state(m).to_pure_dict())  # dict[tuple, jax.Array]
        return {"/".join(map(str, k)): v for k, v in fl.items() if "/".join(map(str, k)) in fresh_set}

    fresh_vals = jax.jit(_fresh_init)()  # dict[str, jax.Array]
    merged = {k: v for k, v in flat.items() if not isinstance(v, jax.ShapeDtypeStruct)}  # dict[tuple, Any]
    for p, v in fresh_vals.items():
        merged[key_by_str[p]] = v
    graphdef, state = nnx.split(abstract_model)
    state.replace_by_pure_dict(traverse_util.unflatten_dict(merged))
    # Frozen params -> bf16, exactly as scripts/train.py:152-158 does. Without it the model sat in
    # float32 (~12 GB) and the first sample_actions compile OOM'd trying to allocate a float32 copy of
    # the 257152x2048 embedding table (2,106,589,184 bytes, 2026-09-23 18:04).
    state = nnx_utils.state_map(state, freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
    model = nnx.merge(graphdef, state)
    del loaded, flat, merged, fresh_vals

    # --- CHECK4 (route half): sym path zeroed + gates at 0 => must equal released model --------
    # Runs BEFORE CHECK3: after CHECK3's full backward pass the A10G no longer had room to compile
    # sample_actions (OOM, 2026-09-23 17:58). Built on a zeroed COPY of the state -- only the small
    # zeroed leaves are new arrays, every other leaf is the same buffer -- so CHECK3 below still
    # runs on the untouched model.
    graphdef, state = nnx.split(model)
    flat_p = dict(traverse_util.flatten_dict(state.to_pure_dict()))  # dict[tuple, Any], shallow copy
    zeroed = []  # list[str]
    for k in list(flat_p.keys()):
        p = "/".join(map(str, k))  # str
        if "sym_mem_mod_dense" in p or p.endswith("a_x") or p.endswith("a_d"):
            flat_p[k] = jnp.zeros_like(flat_p[k])
            zeroed.append(p)
    zero_state = nnx.split(model)[1]
    zero_state.replace_by_pure_dict(traverse_util.unflatten_dict(flat_p))
    zero_model = nnx.merge(graphdef, zero_state)
    a_route = np.asarray(nnx.jit(lambda m, o, n: m.sample_actions(None, o, noise=n))(zero_model, observation, noise), dtype=np.float32)  # np.ndarray
    results["CHECK4_route"] = {"actions": a_route.tolist(), "batch_checksum": batch_checksum, "n_zeroed": len(zeroed)}

    del zero_model, zero_state, flat_p
    jax.clear_caches()

    # --- CHECK3: real training step ------------------------------------------------------------
    trainable_filter = nnx.All(nnx.Param, nnx.Not(freeze_filter))  # nnx filter

    def loss_fn(m, rng, obs, act):
        """
        What it does: mean training loss exactly as train.py computes it.

        Returns:
            tuple[jax.Array, dict] -- (scalar loss, stats dict).

        Example input:
            loss_fn(model, jax.random.key(0), observation, actions)

        Example output:
            (Array(0.07), {"aux_ce": ..., ...})
        """
        chunked, stats = m.compute_loss(rng, obs, act, train=True)
        return jnp.mean(chunked), stats

    (loss, stats), grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, trainable_filter), has_aux=True)(
        model, jax.random.key(1), observation, actions
    )
    flat_g = traverse_util.flatten_dict(grads.to_pure_dict())  # dict[tuple, jax.Array]
    gnorm = {}  # dict[str, float]
    for marker in ["sym_mem_attn", "sym_mem_mod_dense", "sym_aux_head", "sym_type_emb", "event_encoder", "fusion", "mem_attn/"]:
        vals = [np.asarray(v, dtype=np.float32) for k, v in flat_g.items() if marker in "/".join(map(str, k)) + "/"]  # list
        gnorm[marker] = float(np.sqrt(sum(float(np.sum(x * x)) for x in vals))) if vals else -1.0
    stats_f = {k: float(np.asarray(v)) for k, v in (stats or {}).items()}  # dict[str, float]
    grads_ok = all(np.isfinite(v) and v > 0 for v in gnorm.values())  # bool
    results["CHECK3"] = {
        "loss": float(loss), "stats": stats_f, "grad_norms": gnorm,
        "ok": bool(np.isfinite(float(loss))) and all(np.isfinite(v) for v in stats_f.values())
        and stats_f.get("aux_label_frac", 0.0) >= 0.5 and grads_ok,
    }
    print("CHECK3", results["CHECK3"])
    del grads, flat_g

    results["overall_ok"] = all(r["ok"] for r in results.values() if isinstance(r, dict) and "ok" in r)
    del nnx_utils, dataclasses
    return results


@app.function(image=image, gpu=None, cpu=4.0, memory=32768, timeout=7200, volumes=volumes)
def check_all() -> dict:
    """
    What it does:
        CPU orchestrator: runs smoke("route") (CHECK1-3 + the route half of
        CHECK4) and smoke("ref") (released model) in SEPARATE GPU
        containers, compares their actions on the same batch/noise
        (CHECK4), and measures on 256 real samples the fraction whose open
        event carries a real (non-UNK) caption_id -- the aux loss's label
        coverage (CHECK5; the 2-sample CHECK3 batch read 0.5).

    Returns:
        dict -- all checks, each with "ok", plus "overall_ok".

    Example input:
        check_all.remote()

    Example output:
        {"CHECK1": {...}, "CHECK4": {"rel_diff": 0.001, "ok": True}, "CHECK5": {...}, "overall_ok": True}
    """
    import os  # module
    import sys  # module

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/src")
    sys.path.insert(0, "/xf_root")
    os.chdir("/app")
    os.environ["JAX_PLATFORMS"] = "cpu"
    import numpy as np  # module

    route = smoke.remote("route")  # dict
    for k in ["CHECK1", "CHECK2", "CHECK3"]:
        print(k, route[k])
    ref = smoke.remote("ref")  # dict
    a_route = np.asarray(route["CHECK4_route"]["actions"], dtype=np.float32)  # np.ndarray
    a_ref = np.asarray(ref["actions"], dtype=np.float32)  # np.ndarray
    same_batch = abs(route["CHECK4_route"]["batch_checksum"] - ref["batch_checksum"]) < 1e-3  # bool
    rel = float(np.linalg.norm(a_route - a_ref) / max(np.linalg.norm(a_ref), 1e-8))  # float
    out = {k: route[k] for k in ["CHECK1", "CHECK2", "CHECK3"]}  # dict
    out["CHECK4"] = {"rel_diff": rel, "same_batch": same_batch, "n_zeroed": route["CHECK4_route"]["n_zeroed"],
                     "ok": same_batch and rel < 1e-2}
    print("CHECK4", out["CHECK4"])

    out["CHECK5"] = _label_coverage()
    print("CHECK5", out["CHECK5"])
    out["overall_ok"] = all(v["ok"] for v in out.values() if isinstance(v, dict))
    print("SMOKE_SYMROUTE_OVERALL_OK" if out["overall_ok"] else "SMOKE_SYMROUTE_FAILED")
    return out


def _label_coverage() -> dict:
    """
    What it does:
        CHECK5: over 256 real training samples (through the real XFDataset,
        including its stochastic chunk-grid snapping), the fraction whose
        OPEN event carries a real caption_id (not UNK) -- the aux loss's
        label coverage. First measured 53% on 2026-09-23 because snapping
        dropped caption_id; must be ~100% after the subgoal_logger fix.
        Must run inside a container with the data volumes mounted.

    Returns:
        dict -- {"samples", "open_frac", "real_caption_id_frac",
        "open_with_coords_frac", "ok"}.

    Example input:
        _label_coverage()

    Example output:
        {"samples": 256, "open_frac": 1.0, "real_caption_id_frac": 1.0, "open_with_coords_frac": 0.81, "ok": True}
    """
    import numpy as np  # module

    for vol in [main_data_volume, *feature_shard_volumes]:
        vol.reload()
    import mme_vla_suite.training.dataloader as _dataloader  # module
    import scripts.train as _train  # module
    from xattn_fusion.training.launch_xf_training import _build_train_config, _patch_scripts_train_for_xf

    _patch_scripts_train_for_xf(_train)
    tc = _build_train_config(num_train_steps=10_000, batch_size=64, variant="symroute")  # TrainConfig
    dc = tc.data.create(tc.assets_dirs, tc.model)  # DataConfig
    loader = _dataloader.create_data_loader(
        tc.dataset_path, dc, history_config=tc.model.history_config, action_horizon=tc.model.action_horizon,
        batch_size=64, shuffle=True, num_batches=4, num_workers=0, seed=1,
    )
    n = n_open = n_real = n_coords = 0  # int, int, int, int
    for obs, _ in loader:
        is_open = np.asarray(obs.event_is_open)  # bool[b, e]
        cap = np.asarray(obs.event_caption_id)  # int[b, e]
        xy = np.asarray(obs.event_coords)  # int[b, e, 2]
        for i in range(is_open.shape[0]):
            n += 1
            if not is_open[i].any():
                continue
            n_open += 1
            k = int(np.argmax(is_open[i]))  # int
            n_real += int(cap[i, k] != 0)
            n_coords += int(np.all(xy[i, k] >= 0))
    return {"samples": n, "open_frac": n_open / max(n, 1), "real_caption_id_frac": n_real / max(n, 1),
                     "open_with_coords_frac": n_coords / max(n, 1), "ok": n_real / max(n, 1) >= 0.8}


@app.function(image=image, gpu=None, cpu=4.0, memory=32768, timeout=3600, volumes=volumes)
def label_check() -> dict:
    """
    What it does: CPU-only rerun of CHECK5 alone (no model, no GPU).

    Returns:
        dict -- see _label_coverage.

    Example input:
        label_check.remote()

    Example output:
        {"samples": 256, "real_caption_id_frac": 1.0, "ok": True, ...}
    """
    import os  # module
    import sys  # module

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/src")
    sys.path.insert(0, "/xf_root")
    os.chdir("/app")
    os.environ["JAX_PLATFORMS"] = "cpu"
    r = _label_coverage()  # dict
    print("CHECK5", r)
    return r


@app.local_entrypoint()
def run_label_check():
    """
    What it does: local trigger for label_check (short, blocking).

    Returns:
        None -- prints the CHECK5 dict.

    Example input:
        modal run xattn_fusion/diagnostics/smoke_test_symroute.py::run_label_check

    Example output:
        (stdout) CHECK5 {...}
    """
    print("CHECK5", label_check.remote())

@app.local_entrypoint()
def main():
    """
    What it does: runs the smoke test and prints each check plus the
    overall verdict.

    Returns:
        None -- prints to stdout.

    Example input:
        modal run xattn_fusion/diagnostics/smoke_test_symroute.py

    Example output:
        (stdout) CHECK1 ... ok=True ... SMOKE_SYMROUTE_OVERALL_OK
    """
    r = smoke.remote()  # dict
    for k, v in r.items():
        print(f"{k}: {v}")
    print("SMOKE_SYMROUTE_OVERALL_OK" if r.get("overall_ok") else "SMOKE_SYMROUTE_FAILED")


@app.local_entrypoint()
def launch_detached():
    """
    What it does:
        Spawns the smoke test so it survives the local client dying (the
        2026-09-23 runs were lost to a local DNS drop and a local memory
        kill). MUST be run with `modal run --detach ...::launch_detached`.
        Each CHECK line is printed remotely; read them with
        `modal app logs <app id>`.

    Returns:
        None -- prints the spawned call id.

    Example input:
        modal run --detach xattn_fusion/diagnostics/smoke_test_symroute.py::launch_detached

    Example output:
        (stdout) spawned fc-...
    """
    call = check_all.spawn()  # modal.FunctionCall
    print(f"spawned {call.object_id}")


@app.function(image=image, gpu=None, cpu=4.0, memory=65536, timeout=3600, volumes=volumes)
def warmstart_check(variant: str = "symroute_cond", expected_fresh_marker: str = "subgoal_cond") -> dict:
    """
    What it does:
        CPU-only (no GPU spend) version of CHECK1 + CHECK2 for any variant:
        loads the variant's own warm start (launch_xf_training.
        VARIANT_WARM_START, e.g. symroute/1499 for symroute_cond) through
        the real patched _merge_params, and checks that EVERY fresh-
        initialized leaf contains `expected_fresh_marker` (for option B:
        only the new SubgoalConditioner), and that no such leaf is frozen.

    Returns:
        dict -- {"n_loaded", "n_fresh", "fresh_unexpected", "fresh_expected",
        "frozen_new", "ok"}.

    Example input:
        warmstart_check.remote("symroute_cond", "subgoal_cond")

    Example output:
        {"n_loaded": 153, "n_fresh": 4, "fresh_unexpected": [], "fresh_expected": 4, "frozen_new": [], "ok": True}
    """
    import os  # module
    import sys  # module

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/src")
    sys.path.insert(0, "/xf_root")
    os.chdir("/app")
    os.environ["JAX_PLATFORMS"] = "cpu"

    import flax.nnx as nnx  # module
    import jax  # module
    from flax import traverse_util  # module

    for vol in [ckpt_volume, training_volume]:
        vol.reload()
    import scripts.train as _train  # module
    from xattn_fusion.training.launch_xf_training import _build_train_config, _patch_scripts_train_for_xf

    _patch_scripts_train_for_xf(_train)
    tc = _build_train_config(num_train_steps=10_000, batch_size=BATCH_SIZE, variant=variant)  # TrainConfig
    abstract_model = nnx.eval_shape(tc.model.create, jax.random.key(0))
    _, abstract_state = nnx.split(abstract_model)
    loaded = tc.weight_loader.load(abstract_state.to_pure_dict())  # dict
    flat = traverse_util.flatten_dict(loaded)  # dict[tuple, Any]
    fresh = ["/".join(map(str, k)) for k, v in flat.items() if isinstance(v, jax.ShapeDtypeStruct)]  # list[str]
    markers = [m for m in expected_fresh_marker.split(",") if m]  # list[str], comma-separated allowed
    unexpected = [p for p in fresh if not any(m in p for m in markers)]  # list[str]
    frozen = traverse_util.flatten_dict(abstract_state.filter(tc.model.get_freeze_filter()).to_pure_dict())  # dict
    frozen_new = ["/".join(map(str, k)) for k in frozen if any(m in "/".join(map(str, k)) for m in markers)]  # list[str]
    r = {"n_loaded": len(flat) - len(fresh), "n_fresh": len(fresh), "fresh_unexpected": unexpected[:20],
         "fresh_expected": len(fresh) - len(unexpected), "frozen_new": frozen_new[:10],
         "ok": len(fresh) > 0 and not unexpected and not frozen_new}  # dict
    print("WARMSTART_CHECK", r)
    return r


@app.local_entrypoint()
def run_warmstart_check(variant: str = "symroute_cond", expected_fresh_marker: str = "subgoal_cond"):
    """
    What it does: local trigger for the CPU-only warmstart_check (short, blocking).

    Returns:
        None -- prints the result dict.

    Example input:
        modal run xattn_fusion/diagnostics/smoke_test_symroute.py::run_warmstart_check

    Example output:
        (stdout) WARMSTART_CHECK {...}
    """
    print("WARMSTART_CHECK", warmstart_check.remote(variant, expected_fresh_marker))



@app.function(image=image, gpu=None, cpu=4.0, memory=32768, timeout=3600, volumes=volumes)
def qfix_checks() -> dict:
    """
    What it does:
        CPU-only pre-training checks for the q_proj fix (no GPU spend):
          LAYOUT: on real training samples, every real memory token t carries
            the 4x4 spatial position code of cell t % 16 (row-major, y then
            x) -- the assumption fusion_grounding's labels rely on.
          MASK: build_open_fusion_mask on a synthetic case -- padding frames
            see only the null key; real frames see every real caption token
            of every event; "own" marks exactly the frame's own event's real
            tokens; the null key is never "own".

    Returns:
        dict -- {"LAYOUT": {...,"ok"}, "MASK": {...,"ok"}, "ok": bool}.

    Example input:
        qfix_checks.remote()

    Example output:
        {"LAYOUT": {"max_abs_diff": 0.0, "tokens_checked": 12000, "ok": True}, "MASK": {"ok": True}, "ok": True}
    """
    import os  # module
    import sys  # module

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/src")
    sys.path.insert(0, "/xf_root")
    os.chdir("/app")
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax.numpy as jnp  # module
    import numpy as np  # module

    for vol in [main_data_volume, *feature_shard_volumes]:
        vol.reload()
    import mme_vla_suite.training.dataloader as _dataloader  # module
    import scripts.train as _train  # module
    from xattn_fusion.mme_vla_suite.models.representation.fusion_xattn import build_open_fusion_mask
    from xattn_fusion.mme_vla_suite.models.representation.xf_common import XFPosEmb3D
    from xattn_fusion.training.launch_xf_training import _build_train_config, _patch_scripts_train_for_xf

    out = {}  # dict[str, dict]

    # --- LAYOUT ---
    _patch_scripts_train_for_xf(_train)
    tc = _build_train_config(num_train_steps=10_000, batch_size=16, variant="symroute_cond_qfix")  # TrainConfig
    dc = tc.data.create(tc.assets_dirs, tc.model)  # DataConfig
    loader = _dataloader.create_data_loader(
        tc.dataset_path, dc, history_config=tc.model.history_config, action_horizon=tc.model.action_horizon,
        batch_size=16, shuffle=True, num_batches=2, num_workers=0, seed=3,
    )
    table = np.asarray(XFPosEmb3D(dim=768).spatial_pe4x4, dtype=np.float32)  # np.ndarray [16, 512]
    max_diff = 0.0  # float
    checked = 0  # int
    for obs, _ in loader:
        pos = np.asarray(obs.static_pos_emb, dtype=np.float32)  # [b, 512, 768]
        msk = np.asarray(obs.static_mask)  # bool[b, 512]
        spatial = pos[..., 256:]  # [b, 512, 512] -- PosEmb3D concatenates [temporal 256 | spatial 512]
        expected = table[np.arange(pos.shape[1]) % 16]  # [512, 512]
        diff = np.abs(spatial - expected[None])  # [b, 512, 512]
        max_diff = max(max_diff, float(diff[msk].max()) if msk.any() else 0.0)
        checked += int(msk.sum())
    out["LAYOUT"] = {"max_abs_diff": max_diff, "tokens_checked": checked, "ok": checked > 0 and max_diff < 1e-3}
    print("LAYOUT", out["LAYOUT"])

    # --- MASK (synthetic) ---
    # 1 sample, 3 frame tokens: event 0, event 1, padding(-1); 2 events x 3 caption tokens,
    # event 0 has 2 real tokens, event 1 has 1 real token.
    idx = jnp.array([[0, 1, -1]], dtype=jnp.int32)  # int32[1, 3]
    tmask = jnp.array([[[True, True, False], [True, False, False]]])  # bool[1, 2, 3]
    m, own = build_open_fusion_mask(idx, tmask)  # bool[1, 3, 7] each
    m, own = np.asarray(m[0]), np.asarray(own[0])  # np.ndarray [3, 7]
    real_keys = np.array([1, 1, 0, 1, 0, 0, 1], dtype=bool)  # e0t0 e0t1 e0t2 e1t0 e1t1 e1t2 null
    checks = {
        "real_frames_see_all_real_keys": bool((m[0] == real_keys).all() and (m[1] == real_keys).all()),
        "pad_frame_only_null": bool((m[2] == np.array([0, 0, 0, 0, 0, 0, 1], dtype=bool)).all()),
        "own_event0": bool((own[0] == np.array([1, 1, 0, 0, 0, 0, 0], dtype=bool)).all()),
        "own_event1": bool((own[1] == np.array([0, 0, 0, 1, 0, 0, 0], dtype=bool)).all()),
        "pad_owns_nothing": bool(not own[2].any()),
    }  # dict[str, bool]
    out["MASK"] = {**checks, "ok": all(checks.values())}
    print("MASK", out["MASK"])
    out["ok"] = all(v["ok"] for v in out.values())
    print("QFIX_CHECKS_OK" if out["ok"] else "QFIX_CHECKS_FAILED")
    return out


@app.local_entrypoint()
def run_qfix_checks():
    """
    What it does: local trigger for the CPU-only q_proj-fix checks, then the
    CPU-only warm-start check for the symroute_cond_qfix variant.

    Returns:
        None -- prints both results.

    Example input:
        modal run xattn_fusion/diagnostics/smoke_test_symroute.py::run_qfix_checks

    Example output:
        (stdout) QFIX {...} / WARMSTART_CHECK {...}
    """
    print("QFIX", qfix_checks.remote())
    print("WARMSTART_CHECK", warmstart_check.remote(
        "symroute_cond_qfix", "query_context,grounding_head,own_event_bias,query_film,event_tag"))


@app.function(image=image, gpu=None, cpu=4.0, memory=32768, timeout=3600, volumes=volumes)
def marker_checks() -> dict:
    """
    What it does:
        CPU-only checks for the current-image target marker (no GPU):
          SPOTLIGHT: marker_weights peaks (weight 1-ish) on the patch that
            contains the target -- (y, x) = (94, 162) in the 256x256 frame is
            patch row 94//16 = 5, col 162//16 = 10 -> index 90 -- and is all
            zero when the open event has no bounding box.
          KEY_ORDER: in real training batches the first image key is
            "base_0_rgb" (the front camera the (y, x) refer to), which
            XFModel.embed_prefix asserts at trace time.

    Returns:
        dict -- {"SPOTLIGHT": {...}, "KEY_ORDER": {...}, "ok": bool}.

    Example input:
        marker_checks.remote()

    Example output:
        {"SPOTLIGHT": {"argmax": 90, "ok": True}, "KEY_ORDER": {"keys": [...], "ok": True}, "ok": True}
    """
    import os  # module
    import sys  # module

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/src")
    sys.path.insert(0, "/xf_root")
    os.chdir("/app")
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax.numpy as jnp  # module
    import numpy as np  # module

    from xattn_fusion.mme_vla_suite.models.representation.target_marker import marker_weights

    out = {}  # dict[str, dict]
    is_open = jnp.array([[False, True], [False, True]])  # bool[2, 2]
    coords = jnp.array([[[-1, -1], [94, 162]], [[10, 10], [-1, -1]]], dtype=jnp.int32)  # int32[2, 2, 2]
    w = np.asarray(marker_weights(is_open, coords))  # np.ndarray [2, 256]
    out["SPOTLIGHT"] = {
        "argmax": int(w[0].argmax()), "peak": float(w[0].max()), "no_box_max": float(w[1].max()),
        "ok": int(w[0].argmax()) == 90 and w[0].max() > 0.5 and float(w[1].max()) == 0.0,
    }
    print("SPOTLIGHT", out["SPOTLIGHT"])

    for vol in [main_data_volume, *feature_shard_volumes]:
        vol.reload()
    import mme_vla_suite.training.dataloader as _dataloader  # module
    import scripts.train as _train  # module
    from xattn_fusion.training.launch_xf_training import _build_train_config, _patch_scripts_train_for_xf

    _patch_scripts_train_for_xf(_train)
    tc = _build_train_config(num_train_steps=10_000, batch_size=4, variant="symroute_cond_qfix")  # TrainConfig
    dc = tc.data.create(tc.assets_dirs, tc.model)  # DataConfig
    loader = _dataloader.create_data_loader(
        tc.dataset_path, dc, history_config=tc.model.history_config, action_horizon=tc.model.action_horizon,
        batch_size=4, shuffle=True, num_batches=1, num_workers=0, seed=5,
    )
    obs, _ = next(iter(loader))
    keys = list(obs.images)  # list[str]
    out["KEY_ORDER"] = {"keys": keys, "ok": bool(keys) and keys[0] == "base_0_rgb"}
    print("KEY_ORDER", out["KEY_ORDER"])
    out["ok"] = all(v["ok"] for v in out.values())
    print("MARKER_CHECKS_OK" if out["ok"] else "MARKER_CHECKS_FAILED")
    return out


@app.local_entrypoint()
def run_marker_checks():
    """
    What it does: local trigger for the CPU-only marker checks.

    Returns:
        None -- prints the result.

    Example input:
        modal run xattn_fusion/diagnostics/smoke_test_symroute.py::run_marker_checks

    Example output:
        (stdout) MARKER {...}
    """
    print("MARKER", marker_checks.remote())
