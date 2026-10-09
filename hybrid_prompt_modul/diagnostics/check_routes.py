"""
check_routes.py

Measures, on REAL training batches, whether each memory route actually moves the hybrid
policy's actions. Two modes:

  --mode init     Step-0 check of the warm start. Builds the model exactly as training does
                  (scripts/train.py init_train_state with TwoCheckpointWeightLoader: GroundSG
                  base + FrameSamp+Modul's trained memory modules + fresh LoRA). Reports
                  caption-swap and frame-swap effects AT STEP 0, plus "frame route off": the same
                  model with the modulator's scale/shift Dense ZEROED. With scale = shift = 0 the
                  modulator reduces to an extra RMSNorm right before the pre-FFN RMSNorm, i.e. the
                  GroundSG computation up to the norm's 1e-6 epsilon -- so that row is how far
                  the transplanted frame path moves GroundSG's actions at step 0. Expected:
                  caption swap clearly > floor (GroundSG reads captions); frame swap and
                  "frame route off" > floor (the transplanted modulator is active). If those two
                  are ~0, the transplant did not take.
  --mode trained  On a trained hybrid checkpoint (all params required): swaps captions between
                  samples of the batch (symbolic_tokenized_prompt rolled by one), and separately
                  swaps frame memory (static_* rolled by one), and reports how much the actions
                  change. XF's failure signature was caption-swap ~0.2% vs frame-swap 34-45%;
                  here BOTH should be far above the floor.

  --swing-ahead   (either mode) also measures the timing-shift fix's target behaviour on 16 real
                  SwingXtimes samples from the LAST TWO SWING subgoals (before "put the cube on
                  the table" / "press the button"): the same sample with its true caption vs the
                  caption one subgoal ahead, and vs the terminal "press the button". Large =
                  the policy obeys a premature caption (the 2026-10-05 failure); the fine-tune
                  should make both shrink. Captions are taken noise-free from the episode
                  timeline (load_episode_timelines), identical for all three variants.
  --per-task      (either mode) for each listed task, e.g. "VideoUnmask,SwingXtimes": real samples of
                  that task only (8 per task), and the action change when (a) the caption is
                  REMOVED, (b) the frame memory is REMOVED (static_mask all False), (c) the caption
                  is swapped with another sample of the same task, (d) the frame memory is swapped
                  likewise. (a) large => the task follows the caption; (b) large => it follows the
                  frames. Answers "which memory does each task use".
  --variant       which training variant's checkpoint dir --mode trained reads (base |
                  timingshift). The data pipeline is ALWAYS the base config (same model, caption
                  shifting OFF), so measurements are never perturbed by the augmentation itself.

Metric: per sample, ||a_cf - a|| / ||a|| over the full (horizon x action_dim) chunk, same flow
noise for both calls; averaged over samples whose swapped input actually differs. The "floor"
row re-runs the unmodified inputs through the same jitted function (expected exactly 0).

Runs on ONE A10G with ONE model (preflight item 6): the model comes from init_train_state, the
same code path that already fits on an A10G during training, and the zeroed variant shares every
array except the few modulator kernels.

Run (training account, after stage_checkpoints; trained mode after a checkpoint exists):
    modal run hybrid_prompt_modul/diagnostics/check_routes.py --mode init
    modal run hybrid_prompt_modul/diagnostics/check_routes.py --mode trained --step 2000
    modal run hybrid_prompt_modul/diagnostics/check_routes.py --mode trained --step 9999 --swing-ahead
    modal run hybrid_prompt_modul/diagnostics/check_routes.py --mode trained --variant timingshift --step 1999 --swing-ahead
    modal run hybrid_prompt_modul/diagnostics/check_routes.py --mode trained --step 9999 --per-task VideoUnmask,SwingXtimes
"""

import pathlib
import sys

import modal

# Make `hybrid_prompt_modul` importable both locally (repo root) and in the container (/hyb_root).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, "/hyb_root")

from hybrid_prompt_modul.training.launch_hybrid_training import (  # noqa: E402
    CKPT_BASE_DIR,
    REPO_ID,
    VARIANTS,
    image,
    training_volumes,
)

app = modal.App("hybrid-prompt-modul-diagnostics")  # modal.App

MODULATOR_REGEX = ".*mem_rms_norm_ffn.*"  # str, the modulator's scale/shift Dense (history_gemma.MemoryRMSNorm)


@app.function(image=image, gpu="A10G", timeout=3600, volumes=training_volumes)
def check_routes_remote(mode: str = "init", step: int | None = None, num_batches: int = 4, batch_size: int = 4,
                        variant: str = "base", swing_ahead: bool = False, swing_samples: int = 16,
                        per_task: str = "", per_task_samples: int = 8) -> dict:
    """
    What it does:
        Builds the model (see module docstring), draws `num_batches` real
        batches, and reports relative action change for each intervention.

    Returns:
        dict -- {"mode": str, "n_samples": int, "rows": {name: {"mean": float,
        "max": float, "n": int}}}.

    Example input:
        check_routes_remote.remote(mode="trained", step=2000)

    Example output:
        {"mode": "trained", "n_samples": 16, "rows": {"floor": {"mean": 0.0, ...},
         "caption_swap": {"mean": 0.21, ...}, "frame_swap": {"mean": 0.33, ...}}}
    """
    import dataclasses
    import os
    import re

    from hybrid_prompt_modul.training.launch_hybrid_training import (
        DATASET_PATH,
        _build_train_config,
        _patch_train_module,
        _setup_paths,
        training_volume,
    )

    _setup_paths()
    os.chdir("/app")
    training_volume.reload()

    import flax.nnx as nnx
    import flax.traverse_util as traverse_util
    import jax
    import jax.numpy as jnp
    import numpy as np

    import mme_vla_suite.training.dataloader as _dataloader
    import openpi.shared.nnx_utils as nnx_utils
    import openpi.training.sharding as sharding
    import scripts.train as _train

    if mode == "init":
        config = _build_train_config(10_000, batch_size)  # TrainConfig, two-checkpoint warm start
        _patch_train_module(_train)
    elif mode == "trained":
        if step is None:
            raise ValueError("--mode trained needs --step")
        if variant not in VARIANTS:
            raise ValueError(f"--variant must be one of {sorted(VARIANTS)}")
        params_path = f"{CKPT_BASE_DIR}/{REPO_ID}/{VARIANTS[variant]['exp_name']}/{step}/params"  # str
        if not pathlib.Path(params_path).is_dir():
            raise FileNotFoundError(params_path)
        # base config on purpose: same model, caption shifting OFF in the data pipeline
        config = _build_train_config(10_000, batch_size, trained_params_path=params_path)  # TrainConfig
        _patch_train_module(_train)  # patched merge: every param must come from the checkpoint
    else:
        raise ValueError(f"unknown mode {mode!r}")

    mesh = sharding.make_mesh(1)  # jax.sharding.Mesh
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))  # NamedSharding
    data_config = config.data.create(config.assets_dirs, config.model)  # DataConfig
    if data_config.norm_stats is None:
        raise FileNotFoundError("norm stats missing from the training assets dir -- run ::stage_checkpoints first")
    loader = _dataloader.create_data_loader(
        DATASET_PATH, data_config, history_config=config.model.history_config, action_horizon=config.model.action_horizon,
        batch_size=batch_size, sharding=data_sharding, shuffle=True, num_batches=num_batches, num_workers=2, seed=0,
    )  # DataLoaderImpl, yields (HistAugObservation, actions)

    train_state, _ = _train.init_train_state(config, jax.random.key(config.seed), mesh, resume=False)
    graphdef = train_state.model_def  # nnx.GraphDef
    state = train_state.params  # nnx.State

    @jax.jit
    def _sample(params_state, observation, noise):
        """Flow-matching sample with fixed noise; params passed in so the zeroed variant reuses the compile."""
        model = nnx.merge(graphdef, params_state)  # HybridPi0
        return model.sample_actions(jax.random.key(0), observation, noise=noise)

    variants = {}  # dict[str, nnx.State]
    if mode == "init":
        # Only the matched modulator leaves get new (zero) arrays; every other leaf is shared.
        zeroed = nnx_utils.state_map(
            state, nnx_utils.PathRegex(MODULATOR_REGEX), lambda p: p.replace(jnp.zeros_like(p.value))
        )  # nnx.State
        flat_names = ["/".join(map(str, k)) for k in traverse_util.flatten_dict(state.to_pure_dict())]  # list[str]
        matched = [n for n in flat_names if re.fullmatch(MODULATOR_REGEX, n)]  # list[str]
        if not matched:
            raise ValueError(f"modulator regex {MODULATOR_REGEX!r} matched no params -- cannot build the zeroed variant")
        print(f"[check_routes] zeroed {len(matched)} modulator params, e.g. {matched[:3]}")
        variants["frame route off (modulator zeroed ~= GroundSG)"] = zeroed

    def _rel(a, b) -> np.ndarray:
        """Per-sample ||a - b|| / ||b|| over the whole action chunk."""
        a = np.asarray(a, dtype=np.float64).reshape(a.shape[0], -1)  # float64[B, H*D]
        b = np.asarray(b, dtype=np.float64).reshape(b.shape[0], -1)  # float64[B, H*D]
        return np.linalg.norm(a - b, axis=1) / (np.linalg.norm(b, axis=1) + 1e-8)

    rows = {}  # dict[str, list[float]]
    n_samples = 0  # int
    for batch_idx, (obs, _actions) in enumerate(loader):
        bsz = obs.state.shape[0]  # int
        noise = jax.random.normal(jax.random.key(100 + batch_idx), (bsz, config.model.action_horizon, config.model.action_dim))  # jax.Array
        base = _sample(state, obs, noise)  # jax.Array[B, H, D]
        n_samples += bsz

        rows.setdefault("floor (same inputs, rerun)", []).extend(_rel(_sample(state, obs, noise), base).tolist())

        cap = np.asarray(obs.symbolic_tokenized_prompt)  # int[B, L]
        cap_rolled = np.roll(cap, 1, axis=0)  # int[B, L]
        cap_differs = np.any(cap != cap_rolled, axis=1)  # bool[B]
        obs_cap = dataclasses.replace(
            obs,
            symbolic_tokenized_prompt=jnp.roll(obs.symbolic_tokenized_prompt, 1, axis=0),
            symbolic_tokenized_prompt_mask=jnp.roll(obs.symbolic_tokenized_prompt_mask, 1, axis=0),
        )  # HistAugObservation
        rows.setdefault("caption_swap", []).extend(_rel(_sample(state, obs_cap, noise), base)[cap_differs].tolist())

        obs_frames = dataclasses.replace(
            obs,
            static_image_emb=jnp.roll(obs.static_image_emb, 1, axis=0),
            static_pos_emb=jnp.roll(obs.static_pos_emb, 1, axis=0),
            static_state_emb=jnp.roll(obs.static_state_emb, 1, axis=0),
            static_mask=jnp.roll(obs.static_mask, 1, axis=0),
        )  # HistAugObservation
        rows.setdefault("frame_swap", []).extend(_rel(_sample(state, obs_frames, noise), base).tolist())

        for name, variant_state in variants.items():  # str, nnx.State
            rows.setdefault(name, []).extend(_rel(_sample(variant_state, obs, noise), base).tolist())
        print(f"[check_routes] batch {batch_idx + 1}/{num_batches} done")

    if swing_ahead:
        rows.update(_swing_ahead_rows(config, data_config, state, _sample, batch_size, swing_samples))
    if per_task:
        tasks = [t.strip() for t in per_task.split(",") if t.strip()]  # list[str]
        rows.update(_per_task_rows(config, data_config, state, _sample, batch_size, tasks, per_task_samples))

    summary = {  # dict[str, dict]
        name: {"mean": float(np.mean(vals)) if vals else float("nan"),
               "max": float(np.max(vals)) if vals else float("nan"), "n": len(vals)}
        for name, vals in rows.items()
    }
    print(f"\n[check_routes] mode={mode} variant={variant} step={step} samples={n_samples}")
    for name, s in summary.items():  # str, dict
        print(f"  {name:55s} mean {100 * s['mean']:8.3f}%   max {100 * s['max']:8.3f}%   n={s['n']}")
    return {"mode": mode, "variant": variant, "step": step, "n_samples": n_samples, "rows": summary}


def _swing_ahead_rows(config, data_config, state, sample_fn, batch_size: int, num_samples: int) -> dict:
    """
    What it does:
        Container-side. Finds `num_samples` real SwingXtimes samples whose
        caption is one of the last two swing subgoals (k in n-4..n-3 of the
        n execution events), builds each sample three times -- true caption,
        caption +1 ahead, terminal caption -- through the real transform
        pipeline, runs the policy with fixed noise, and returns the per-sample
        relative action change of each premature caption vs the true one.

    Returns:
        dict[str, list[float]] -- {"swing_end: caption +1 ahead": [...],
        "swing_end: caption -> terminal (press)": [...]}.

    Example input:
        _swing_ahead_rows(config, data_config, state, _sample, 4, 16)

    Example output:
        {"swing_end: caption +1 ahead": [0.41, ...], "swing_end: caption -> terminal (press)": [0.66, ...]}
    """
    import random

    import jax
    import jax.numpy as jnp
    import numpy as np
    from openpi.training.data_loader import transform_dataset

    from mme_vla_suite.models.integration.history_observation import HistAugObservation

    from hybrid_prompt_modul.training.caption_shift import find_true_index
    from hybrid_prompt_modul.training.hybrid_dataset import HybridDataset, _scalar, load_episode_timelines
    from hybrid_prompt_modul.training.launch_hybrid_training import DATASET_PATH

    if num_samples % batch_size:
        raise ValueError("swing_samples must be a multiple of batch_size (reuses the compiled batch shape)")
    dataset = HybridDataset(DATASET_PATH, data_config, config.model.history_config, config.model.action_horizon)  # HybridDataset
    timelines = load_episode_timelines(DATASET_PATH)  # dict[int, tuple[str, list[dict]]]
    rng = random.Random(0)  # random.Random
    picked, tries = [], 0  # list[tuple[int, int, list[dict]]], int
    while len(picked) < num_samples and tries < 40000:
        tries += 1
        idx = rng.randrange(len(dataset.dataset))  # int
        raw = dataset.dataset[idx]  # dict, raw sample
        entry = timelines.get(_scalar(raw["epis_idx"]))  # tuple | None
        if entry is None or entry[0] != "SwingXtimes":
            continue
        ivs = entry[1]  # list[dict]
        k = find_true_index(ivs, raw["grounded_subgoal"], _scalar(raw["step_idx"]))  # int | None
        if k is not None and len(ivs) - 4 <= k <= len(ivs) - 3:
            picked.append((idx, k, ivs))
    print(f"[check_routes] swing-ahead: {len(picked)} SwingXtimes last-swing samples after {tries} random draws")
    if len(picked) < num_samples:
        raise RuntimeError(f"only found {len(picked)} last-swing SwingXtimes samples")

    variants = {"true": [], "ahead": [], "terminal": []}  # dict[str, list[dict]]
    for idx, k, ivs in picked:  # int, int, list[dict]
        item = dataset[idx]  # dict, full sample incl. frame memory
        for name, cap in (("true", ivs[k]["caption"]), ("ahead", ivs[k + 1]["caption"]), ("terminal", ivs[-1]["caption"])):  # str, str
            variants[name].append({**item, "grounded_subgoal": cap, "simple_subgoal": cap})

    class _ListDataset:
        """Minimal map-style dataset over prepared raw samples."""

        def __init__(self, items):
            self.items = items  # list[dict]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return dict(self.items[i])

    def _obs(items: list[dict]):
        """Raw samples -> one batched HistAugObservation via the real transform pipeline."""
        tds = transform_dataset(_ListDataset(items), data_config)  # transformed dataset
        rows = [tds[i] for i in range(len(items))]  # list[dict]
        batch = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *rows)  # dict
        batch = {k: v for k, v in batch.items() if not (isinstance(v, np.ndarray) and v.dtype.kind in "UO")}  # drop strings
        return HistAugObservation.from_dict(jax.tree.map(jnp.asarray, batch))

    out = {"swing_end: caption +1 ahead": [], "swing_end: caption -> terminal (press)": []}  # dict[str, list[float]]
    for start in range(0, num_samples, batch_size):  # int
        chunk = slice(start, start + batch_size)  # slice
        noise = jax.random.normal(jax.random.key(500 + start), (batch_size, config.model.action_horizon, config.model.action_dim))  # jax.Array
        base = np.asarray(sample_fn(state, _obs(variants["true"][chunk]), noise), dtype=np.float64).reshape(batch_size, -1)  # float64[B, H*D]
        for key, name in (("swing_end: caption +1 ahead", "ahead"), ("swing_end: caption -> terminal (press)", "terminal")):  # str, str
            alt = np.asarray(sample_fn(state, _obs(variants[name][chunk]), noise), dtype=np.float64).reshape(batch_size, -1)  # float64[B, H*D]
            out[key].extend((np.linalg.norm(alt - base, axis=1) / (np.linalg.norm(base, axis=1) + 1e-8)).tolist())
    return out


def _per_task_rows(config, data_config, state, sample_fn, batch_size: int, tasks: list[str], n_per_task: int) -> dict:
    """
    What it does:
        Container-side. For each task, draws `n_per_task` random real samples
        of that task (any step), and measures the relative action change vs
        the unmodified sample when the caption is removed, the frame memory is
        removed, the caption is swapped with another sample of the same task,
        and the frame memory is swapped likewise (rolled by one within the
        task's samples). Same fixed noise for every variant of a sample.

    Returns:
        dict[str, list[float]] -- {"<task>: caption removed": [...], "<task>: frames removed": [...],
        "<task>: caption swapped (same task)": [...], "<task>: frames swapped (same task)": [...]}.

    Example input:
        _per_task_rows(config, data_config, state, _sample, 4, ["VideoUnmask", "SwingXtimes"], 8)

    Example output:
        {"VideoUnmask: caption removed": [0.52, ...], "VideoUnmask: frames removed": [0.08, ...], ...}
    """
    import random

    import jax
    import jax.numpy as jnp
    import numpy as np
    from openpi.training.data_loader import transform_dataset

    from mme_vla_suite.models.integration.history_observation import HistAugObservation

    from hybrid_prompt_modul.training.hybrid_dataset import HybridDataset, _scalar, load_episode_timelines
    from hybrid_prompt_modul.training.launch_hybrid_training import DATASET_PATH

    if n_per_task % batch_size:
        raise ValueError("per_task_samples must be a multiple of batch_size (reuses the compiled batch shape)")
    dataset = HybridDataset(DATASET_PATH, data_config, config.model.history_config, config.model.action_horizon)  # HybridDataset
    timelines = load_episode_timelines(DATASET_PATH)  # dict[int, tuple[str, list[dict]]]
    task_of = {e: t for e, (t, _) in timelines.items()}  # dict[int, str]

    class _ListDataset:
        """Minimal map-style dataset over prepared raw samples."""

        def __init__(self, items):
            self.items = items  # list[dict]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return dict(self.items[i])

    def _obs(items: list[dict]):
        """Raw samples -> one batched HistAugObservation via the real transform pipeline."""
        tds = transform_dataset(_ListDataset(items), data_config)  # transformed dataset
        rows_t = [tds[i] for i in range(len(items))]  # list[dict]
        batch = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *rows_t)  # dict
        batch = {k: v for k, v in batch.items() if not (isinstance(v, np.ndarray) and v.dtype.kind in "UO")}  # drop strings
        return HistAugObservation.from_dict(jax.tree.map(jnp.asarray, batch))

    out = {}  # dict[str, list[float]]
    frame_keys = ("static_image_emb", "static_pos_emb", "static_state_emb", "static_mask")  # tuple[str, ...]
    for task in tasks:  # str
        rng = random.Random(7)  # random.Random, same draws for every checkpoint
        picked, tries = [], 0  # list[int], int
        while len(picked) < n_per_task and tries < 40000:
            tries += 1
            idx = rng.randrange(len(dataset.dataset))  # int
            if task_of.get(_scalar(dataset.dataset[idx]["epis_idx"])) == task:
                picked.append(idx)
        if len(picked) < n_per_task:
            raise RuntimeError(f"only found {len(picked)} {task} samples")
        items = [dataset[i] for i in picked]  # list[dict]
        rolled = items[1:] + items[:1]  # list[dict], another sample of the same task
        variants = {
            "caption removed": [{**it, "grounded_subgoal": "", "simple_subgoal": ""} for it in items],
            "frames removed": [{**it, "static_mask": np.zeros_like(it["static_mask"])} for it in items],
            "caption swapped (same task)": [{**it, "grounded_subgoal": ro["grounded_subgoal"], "simple_subgoal": ro["simple_subgoal"]}
                                            for it, ro in zip(items, rolled)],
            "frames swapped (same task)": [{**it, **{k: ro[k] for k in frame_keys}} for it, ro in zip(items, rolled)],
        }  # dict[str, list[dict]]
        print(f"[check_routes] per-task {task}: {n_per_task} samples after {tries} draws")
        for start in range(0, n_per_task, batch_size):  # int
            chunk = slice(start, start + batch_size)  # slice
            noise = jax.random.normal(jax.random.key(900 + start), (batch_size, config.model.action_horizon, config.model.action_dim))  # jax.Array
            base = np.asarray(sample_fn(state, _obs(items[chunk]), noise), dtype=np.float64).reshape(batch_size, -1)  # float64[B, H*D]
            for name, var_items in variants.items():  # str, list[dict]
                alt = np.asarray(sample_fn(state, _obs(var_items[chunk]), noise), dtype=np.float64).reshape(batch_size, -1)  # float64[B, H*D]
                out.setdefault(f"{task}: {name}", []).extend(
                    (np.linalg.norm(alt - base, axis=1) / (np.linalg.norm(base, axis=1) + 1e-8)).tolist())
    return out


@app.local_entrypoint()
def main(mode: str = "init", step: int | None = None, num_batches: int = 4, variant: str = "base",
         swing_ahead: bool = False, per_task: str = ""):
    """
    What it does: runs check_routes_remote and prints the result dict.

    Returns:
        None.

    Example input:
        modal run hybrid_prompt_modul/diagnostics/check_routes.py --mode trained --step 2000

    Example output:
        (stdout) {"mode": "trained", ..., "rows": {...}}
    """
    print(check_routes_remote.remote(mode=mode, step=step, num_batches=num_batches, variant=variant,
                                     swing_ahead=swing_ahead, per_task=per_task))
