"""
build_full_suite_dataset.py

Prepares the training data for ALL 16 RoboMME tasks (4 suites: Counting,
Permanence, Reference, Imitation) -- NOT just the 4-task Counting-suite
subset symbolic_as_modulator's current checkpoint was trained on. Per the
user's explicit direction: this is preparation for LATER use (comparing
symbolic-as-modulator against symbolic-alone/memory-as-context across the
full suite, and informing arm_b1_static_fusion/arm_d_dynamic_fusion's own
future work), not an immediate retrain -- no training is triggered by this
file.

Why a NEW, lighter preprocessing path instead of reusing arm_d_dynamic_
fusion/training/build_pilot_dataset.py's approach at 4x the scale: verified
(not assumed) that mme_vla_suite.training.dataset.RoboMMEDataset's
representation_type=="symbolic" branch never reads anything from a
preprocessed dataset's "features/" directory (the per-frame SigLIP
embeddings mme_vla_suite.dataset_builder.build_robomme_dataset.DatasetProcessor
always computes, GPU-bound, ~65 min per 100 episodes on an A10G) -- only
"perceptual"-representation training needs those. This probe only ever needs
representation_type=="symbolic", so LightweightDatasetProcessor below
reproduces DatasetProcessor._process_episode exactly, MINUS the MemoryBuffer/
feature-embedding computation (no MemoryBuffer instantiated, no
mem_buffer.add_buffer()/get_history_feats() calls, no token_emb_*.npy or
kept_indices.json written) -- CPU-only, no GPU needed at all. Trade-off, made
explicitly rather than silently: the resulting preprocessed dataset is usable
for symbolic-memory training only, not perceptual/recurrent -- acceptable
since that's this project's current scope, but worth remembering if a later
arm wants perceptual features from this same 16-task raw data (it would need
the released DatasetProcessor's GPU-bound pass instead, same as Arm D's own).

Data reuse: the 4 Counting-suite tasks' raw HDF5 files already exist on
`robomme-arm-d-pilot-data` (downloaded by arm_d_dynamic_fusion/training/
build_pilot_dataset.py) -- copied from there (read-only source) rather than
re-downloaded, avoiding ~13.6GB of redundant HF Hub traffic. Only the
remaining 12 tasks are freshly downloaded from the HF dataset repo. Output
lands on this probe's OWN new volume (robomme-symbolic-modulator-full-suite-
data), never written back to robomme-arm-d-pilot-data.

robomme_policy_learning/ is not edited: get_action_chunk and
first_execution_step are imported and reused unchanged; only the
MemoryBuffer-dependent parts of _process_episode are reimplemented, in this
file, minus that dependency.

Run with (in this order -- each step checked before the next, per the user's
explicit request to review this project's own prior data-prep mistakes first;
see RESEARCH_LOG.md's 2026-08-20/2026-08-24 entries and
[[project_modal_image_gotchas]] for what those were and how they're avoided
here: never guess a preprocessing timeout -- calibrate first; `.spawn()`
alone does not survive local disconnect -- always pair with `--detach`):
    modal run --detach symbolic_as_modulator/training/build_full_suite_dataset.py::download_raw_data
    modal run symbolic_as_modulator/training/build_full_suite_dataset.py::check_raw_data
    modal run symbolic_as_modulator/training/build_full_suite_dataset.py::run_calibration
    modal run --detach symbolic_as_modulator/training/build_full_suite_dataset.py::build_preprocessed_dataset
    modal run --detach symbolic_as_modulator/training/build_full_suite_dataset.py::compute_full_suite_norm_stats
"""

import pathlib

import modal

# All 16 RoboMME tasks (4 suites), exact task-id strings -- matches
# modal_reproduction/full_eval.py's own TASKS list verbatim (the paper's own
# reproduction script, so these are confirmed-correct task ids).
ALL_TASKS = [  # list[str]
    "BinFill", "StopCube", "PickXtimes", "SwingXtimes",
    "ButtonUnmask", "VideoUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap",
    "PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder",
    "MoveCube", "InsertPeg", "PatternLock", "RouteStick",
]
EXISTING_TASKS = ["BinFill", "PickXtimes", "SwingXtimes", "StopCube"]  # list[str], already on robomme-arm-d-pilot-data
NEW_TASKS = [t for t in ALL_TASKS if t not in EXISTING_TASKS]  # list[str], the remaining 12 -- freshly downloaded here

HF_DATASET_REPO = "Yinpei/robomme_data_h5"  # str, HF dataset repo id (public, no auth needed) -- same as arm_d's

POLICY_LOCAL_DIR = str(  # str
    pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning"
)
PROBE_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)  # str, this symbolic_as_modulator/ directory -- needed for compute_full_suite_norm_stats_remote's history_config yaml

app = modal.App("robomme-symbolic-modulator-full-suite-dataset")  # modal.App

data_volume = modal.Volume.from_name("robomme-symbolic-modulator-full-suite-data", create_if_missing=True)  # modal.Volume, this probe's OWN
arm_d_data_volume = modal.Volume.from_name("robomme-arm-d-pilot-data")  # modal.Volume, READ-ONLY source for the 4 already-downloaded tasks' raw h5

DATA_VOLUME_PATH = "/full_suite_data"  # str
ARM_D_DATA_VOLUME_PATH = "/arm_d_pilot_data"  # str, read-only mount, only for copying existing raw h5 files
RAW_DATA_PATH = f"{DATA_VOLUME_PATH}/raw_h5"  # str
PREPROCESSED_DATA_PATH = f"{DATA_VOLUME_PATH}/preprocessed"  # str
ASSETS_BASE_DIR = f"{DATA_VOLUME_PATH}/assets"  # str

image = (  # modal.Image
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "curl", "libgl1", "libglib2.0-0", "xz-utils")
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
    .run_commands("cd /app && /root/.local/bin/uv pip install --system pytest huggingface_hub")
    .add_local_dir(PROBE_LOCAL_DIR, remote_path="/sym_mod_root/symbolic_as_modulator", copy=True)
)


@app.function(
    image=image, gpu=None, timeout=6 * 3600,
    volumes={DATA_VOLUME_PATH: data_volume, ARM_D_DATA_VOLUME_PATH: arm_d_data_volume},
)
def download_raw_data_remote() -> dict:
    """
    What it does:
        Populates RAW_DATA_PATH with all 16 tasks' raw HDF5 files: the 4
        EXISTING_TASKS are copied from robomme-arm-d-pilot-data (read-only,
        already downloaded there by arm_d_dynamic_fusion), the 12 NEW_TASKS
        are freshly downloaded+extracted from the HF dataset repo. Idempotent:
        skips any task whose FINAL h5 path already exists -- and extraction
        is done into a per-task temp path, then atomically renamed into place
        only on success, specifically so a killed-mid-extraction run (hit for
        real: the first attempt's 3600s timeout cut off `tar` mid-write on
        VideoRepick, leaving a same-named but truncated .h5 file that a naive
        `if h5_path.exists(): skip` would have wrongly treated as done on
        retry) can never leave a corrupt-but-"complete-looking" file behind.
        Per-task volume commits (not one commit at the very end) so a later
        timeout/crash doesn't risk losing already-finished tasks' progress.
        Timeout sized at 6h (not the first attempt's guessed 1h) -- measured
        after the fact that individual tasks' DECOMPRESSED sizes run
        13-37 GiB each (the "56.4GB full dataset" figure in arm_d's own
        docstring refers to compressed .tar.xz sizes, not decompressed --
        confirmed by checking actual file sizes on the volume, not assumed).

    Returns:
        dict -- {"copied": list[str], "downloaded": list[str], "already_present": list[str]}

    Example input:
        download_raw_data_remote.remote()

    Example output:
        {"copied": ["BinFill", ...], "downloaded": ["InsertPeg", ...], "already_present": []}
    """
    import shutil  # module
    import subprocess  # module

    import huggingface_hub  # module

    raw_dir = pathlib.Path(RAW_DATA_PATH)  # Path
    raw_dir.mkdir(parents=True, exist_ok=True)
    arm_d_raw_dir = pathlib.Path(ARM_D_DATA_VOLUME_PATH) / "raw_h5"  # Path

    copied = []  # list[str]
    downloaded = []  # list[str]
    already_present = []  # list[str]

    for task in EXISTING_TASKS:
        h5_name = f"record_dataset_{task}.h5"  # str
        h5_path = raw_dir / h5_name  # Path
        if h5_path.exists():
            already_present.append(task)
            continue
        src_path = arm_d_raw_dir / h5_name  # Path
        tmp_path = h5_path.with_suffix(".h5.copying")  # Path, atomic-rename target
        print(f"[build_full_suite_dataset] copying {h5_name} from robomme-arm-d-pilot-data ...")
        shutil.copy2(src_path, tmp_path)
        tmp_path.rename(h5_path)  # atomic: h5_path only ever appears once fully copied
        copied.append(task)
        data_volume.commit()

    for task in NEW_TASKS:
        h5_name = f"record_dataset_{task}.h5"  # str
        h5_path = raw_dir / h5_name  # Path
        if h5_path.exists():
            already_present.append(task)
            continue

        archive_name = f"{h5_name}.tar.xz"  # str
        print(f"[build_full_suite_dataset] downloading {archive_name} ...")
        archive_path = huggingface_hub.hf_hub_download(
            repo_id=HF_DATASET_REPO, repo_type="dataset", filename=archive_name,
            local_dir=str(raw_dir),
        )  # str

        extract_tmp_dir = raw_dir / f".extracting_{task}"  # Path, per-task scratch dir
        if extract_tmp_dir.exists():
            shutil.rmtree(extract_tmp_dir)  # leftover from a prior killed attempt
        extract_tmp_dir.mkdir()
        print(f"[build_full_suite_dataset] extracting {archive_name} ...")
        subprocess.run(["tar", "-xJf", archive_path, "-C", str(extract_tmp_dir)], check=True)
        (extract_tmp_dir / h5_name).rename(h5_path)  # atomic: h5_path only ever appears once fully extracted
        shutil.rmtree(extract_tmp_dir)
        pathlib.Path(archive_path).unlink()  # free the compressed copy once extracted
        downloaded.append(task)
        data_volume.commit()

    print(f"[build_full_suite_dataset] copied={copied}, downloaded={downloaded}, already_present={already_present}")
    return {"copied": copied, "downloaded": downloaded, "already_present": already_present}


@app.function(image=image, gpu=None, volumes={DATA_VOLUME_PATH: data_volume}, timeout=60)
def check_raw_data() -> list[str]:
    """Lists which of the 16 tasks' raw .h5 files are present on RAW_DATA_PATH -- sanity check before preprocessing."""
    raw_dir = pathlib.Path(RAW_DATA_PATH)  # Path
    if not raw_dir.exists():
        return []
    present = sorted(p.stem.replace("record_dataset_", "") for p in raw_dir.glob("record_dataset_*.h5"))  # list[str]
    print(f"Present ({len(present)}/16): {present}")
    missing = [t for t in ALL_TASKS if t not in present]  # list[str]
    if missing:
        print(f"Missing: {missing}")
    return present


@app.function(
    image=image, gpu=None, timeout=6 * 3600,
    volumes={DATA_VOLUME_PATH: data_volume},
)
def build_preprocessed_dataset_remote(max_episodes: int | None = None, tasks: list[str] | None = None) -> dict:
    """
    What it does:
        Converts RAW_DATA_PATH's .h5 files into the same per-timestep pickle
        format mme_vla_suite.training.dataset.SampleDataset reads (data/*.pkl
        + meta/stats.json) -- structurally identical to the released
        DatasetProcessor's output, MINUS the features/ directory (see module
        docstring for why that's correct and safe for this probe's
        representation_type=="symbolic" use). CPU-only: no MemoryBuffer, no
        JAX vision model, no GPU. Like the released DatasetProcessor, this
        WIPES PREPROCESSED_DATA_PATH on every call and restarts sample
        numbering from 0 -- not incremental, so `tasks` (a raw_data_path
        SUBSET filter, new -- the released class has no equivalent) exists
        specifically to let a first calibration call measure per-task/
        per-episode throughput on a small slice (e.g. one new task) before
        committing to a single all-16-tasks call and its timeout budget,
        rather than guessing (see RESEARCH_LOG.md's 2026-08-20 00:52 entry --
        this project has paid for a wrong timeout guess here before).

    Returns:
        dict -- {"execution_samples": int, "total_samples": int}, read back
        from the preprocessed dataset's meta/stats.json.

    Example input:
        build_preprocessed_dataset_remote.remote(max_episodes=5, tasks=["InsertPeg"])  # calibration
        build_preprocessed_dataset_remote.remote()  # the real, full-16-task run

    Example output:
        {"execution_samples": 5820, "total_samples": 23100}
    """
    import json  # module
    import os  # module
    import pickle  # module
    import sys  # module
    import time  # module

    import h5py  # module
    import numpy as np  # module

    sys.path.insert(0, "/app/src")
    from mme_vla_suite.dataset_builder.build_robomme_dataset import ACTION_CHUNK_HORIZON, get_action_chunk
    from mme_vla_suite.dataset_builder.robomme_h5_utils import first_execution_step

    preprocessed_dir = pathlib.Path(PREPROCESSED_DATA_PATH)  # Path
    if preprocessed_dir.exists():
        import shutil as _shutil  # module
        _shutil.rmtree(preprocessed_dir)
    data_dir = preprocessed_dir / "data"  # Path
    meta_dir = preprocessed_dir / "meta"  # Path
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = pathlib.Path(RAW_DATA_PATH)  # Path
    task_filter = set(tasks) if tasks is not None else None  # set[str] | None

    global_episode_idx = 0  # int
    exec_sample_id = 0  # int
    total_sample_id = 0  # int
    start_time = time.monotonic()  # float

    fnames = sorted(f for f in os.listdir(raw_dir) if f.endswith(".h5"))  # list[str]
    if task_filter is not None:
        fnames = [f for f in fnames if f.replace("record_dataset_", "").replace(".h5", "") in task_filter]
    print(f"[build_preprocessed_dataset] processing {len(fnames)} task file(s): {fnames}")

    for fname in fnames:
        print(f"[build_preprocessed_dataset] Processing file: {fname} (elapsed {time.monotonic()-start_time:.0f}s)")
        path = raw_dir / fname  # Path
        with h5py.File(path, "r") as data:
            episode_indices = sorted(
                int(k.split("_")[1]) for k in data.keys() if k.startswith("episode_")
            )  # list[int]
            if max_episodes is not None:
                episode_indices = episode_indices[:max_episodes]

            for episode_idx in episode_indices:
                episode_data = data[f"episode_{episode_idx}"]
                task_goal = episode_data["setup"]["task_goal"][()][0].decode()  # str
                num_timesteps = sum(1 for k in episode_data.keys() if k.startswith("timestep_"))  # int
                exec_start_idx = first_execution_step(episode_data)  # int

                for step_idx in range(num_timesteps):
                    ts = episode_data[f"timestep_{step_idx}"]
                    action_chunk = get_action_chunk(episode_data, step_idx, horizon=ACTION_CHUNK_HORIZON)  # np.ndarray
                    joint_state = ts["obs"]["joint_state"][()]  # np.ndarray
                    gripper_state = ts["obs"]["gripper_state"][()]  # np.ndarray
                    state = np.concatenate([joint_state, gripper_state[:1]], axis=0, dtype=np.float32)  # np.ndarray
                    image_arr = ts["obs"]["front_rgb"][()]  # np.ndarray
                    wrist_image = ts["obs"]["wrist_rgb"][()]  # np.ndarray
                    is_video_demo = step_idx < exec_start_idx  # bool
                    assert ts["info"]["is_video_demo"][()] == is_video_demo, "is_video_demo mismatch"

                    if not ts["info"]["is_completed"][()]:
                        simple_subgoal = ts["info"]["simple_subgoal"][()].decode()
                        grounded_subgoal = ts["info"]["grounded_subgoal"][()].decode()
                        simple_subgoal_online = ts["info"]["simple_subgoal_online"][()].decode()
                        grounded_subgoal_online = ts["info"]["grounded_subgoal_online"][()].decode()

                    frame_dict = {  # dict
                        "image": image_arr,
                        "wrist_image": wrist_image,
                        "state": state,
                        "actions": action_chunk,
                        "is_demo": np.array([is_video_demo], dtype=np.bool_),
                        "exec_start_idx": np.array([exec_start_idx], dtype=np.int32),
                        "step_idx": np.array([step_idx], dtype=np.int32),
                        "epis_idx": np.array([global_episode_idx], dtype=np.int32),
                        "prompt": task_goal.lower(),
                        "simple_subgoal": simple_subgoal.lower(),
                        "grounded_subgoal": grounded_subgoal.lower(),
                        "simple_subgoal_online": simple_subgoal_online.lower(),
                        "grounded_subgoal_online": grounded_subgoal_online.lower(),
                    }
                    # NO mem_buffer.add_buffer()/get_history_feats() here, unlike the
                    # released DatasetProcessor -- see module docstring for why this
                    # probe's representation_type=="symbolic" never needs those.

                    if not is_video_demo:
                        pkl_path = data_dir / f"{exec_sample_id}.pkl"  # Path
                        assert not pkl_path.exists(), f"Collision: {pkl_path}"
                        with open(pkl_path, "wb") as f:
                            pickle.dump(frame_dict, f)
                        exec_sample_id += 1
                    total_sample_id += 1

                global_episode_idx += 1

        data_volume.commit()  # commit after each task file, so a later timeout doesn't lose finished tasks' pkls
        print(f"[build_preprocessed_dataset] {fname} done. exec_samples={exec_sample_id} total_samples={total_sample_id}")

    stats = {"execution_samples": exec_sample_id, "total_samples": total_sample_id}  # dict
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    data_volume.commit()

    elapsed = time.monotonic() - start_time  # float
    print(f"[build_preprocessed_dataset] DONE in {elapsed:.0f}s: {stats}")
    return stats


@app.function(image=image, gpu=None, timeout=2 * 3600, volumes={DATA_VOLUME_PATH: data_volume})
def compute_full_suite_norm_stats_remote(repo_id: str = "symbolic_modulator_full_suite") -> dict:
    """
    What it does:
        Computes state/action norm_stats over the full 16-task preprocessed
        dataset, representation_type="symbolic" (no GPU, no mem_buffer --
        same reasoning as symbolic_as_modulator/training/compute_norm_stats.py's
        own docstring). Kept separate from that script/repo_id (own
        "symbolic_modulator_full_suite" identity) since this is a DIFFERENT
        dataset (16 tasks, not 4) with its own state/action distribution --
        NOT a drop-in replacement for the current checkpoint's Counting-suite
        norm_stats.

    Returns:
        dict -- {"state": {...}, "actions": {...}}, also written to
        ASSETS_BASE_DIR/<repo_id>/<repo_id>.

    Example input:
        compute_full_suite_norm_stats.remote()

    Example output:
        {"state": {"mean": [...], ...}, "actions": {...}}
    """
    import sys  # module

    import numpy as np  # module
    import tqdm  # module

    sys.path.insert(0, "/app/src")

    import openpi.shared.normalize as normalize
    from openpi.models.pi0_config import Pi0Config
    from openpi.training.data_loader import TorchDataLoader, TransformedDataset

    from mme_vla_suite.training.config import RoboMMEDataConfig, DataConfig
    from mme_vla_suite.training.dataset import RoboMMEDataset
    from mme_vla_suite.models.config.utils import get_history_config

    history_config_path = "/sym_mod_root/symbolic_as_modulator/config/symbolic-modulator-only.yaml"  # str
    history_config = get_history_config(history_config_path)  # omegaconf.DictConfig, representation_type="symbolic"

    data_config_factory = RoboMMEDataConfig(
        repo_id=repo_id, base_config=DataConfig(prompt_from_task=True)
    )  # RoboMMEDataConfig, unmodified released class
    assets_dirs = pathlib.Path(ASSETS_BASE_DIR)  # Path
    data_config = data_config_factory.create(
        assets_dirs=assets_dirs, model_config=Pi0Config(action_horizon=20)
    )

    dataset = RoboMMEDataset(
        dataset_path=PREPROCESSED_DATA_PATH,
        data_config=data_config,
        history_config=history_config,
        action_horizon=20,
        compute_norm_stats=True,
    )
    dataset = TransformedDataset(
        dataset,
        [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs, RemoveStrings()],
    )

    # Systematic stride across the FULL index range, not a sequential prefix
    # or the full 476,857-sample dataset. Two things learned the hard way on
    # a real run (RESEARCH_LOG 2026-09-14): (1) sample indices are laid out
    # task-by-task in the order build_preprocessed_dataset_remote processed
    # them (all BinFill first, then ButtonUnmask, ...) -- capping num_batches
    # with shuffle=False would silently compute norm_stats from only the
    # first task or two, not a representative cross-task sample. (2) reading
    # the FULL dataset measured ~2.2-2.76s/it (a real run, not guessed) --
    # projecting to 9-11h against this function's 1h timeout, almost
    # certainly the same "cold storage first-read" penalty this project's
    # own history has hit before for freshly-written files. Neither problem
    # needs the full 476,857 samples anyway: mean/std/quantiles converge with
    # a much smaller sample. TARGET_SAMPLES=20_000 (stride computed from the
    # real dataset length) keeps every task proportionally represented while
    # cutting wall-clock by >20x.
    import torch.utils.data as _torch_data  # module

    TARGET_SAMPLES = 20_000  # int
    total_len = len(dataset)  # int
    stride = max(1, total_len // TARGET_SAMPLES)  # int
    indices = list(range(0, total_len, stride))  # list[int]
    dataset = _torch_data.Subset(dataset, indices)  # torch.utils.data.Subset
    print(f"[compute_full_suite_norm_stats] sampling {len(indices)}/{total_len} examples (stride={stride}) across all tasks")

    batch_size = 32  # int
    num_batches = max(1, len(dataset) // batch_size)  # int
    data_loader = TorchDataLoader(
        dataset, local_batch_size=batch_size, sharding=None,
        num_batches=num_batches, num_workers=4, seed=0, shuffle=False, framework="jax",
    )

    keys = ["state", "actions"]  # list[str]
    stats = {key: normalize.RunningStats() for key in keys}
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing full-suite norm stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}  # dict
    output_path = pathlib.Path(ASSETS_BASE_DIR) / repo_id / repo_id  # Path
    normalize.save(output_path, norm_stats)
    data_volume.commit()
    print(f"[compute_full_suite_norm_stats] wrote norm_stats to {output_path}")
    return norm_stats


class RemoveStrings:
    """Drops string-typed fields (prompt/subgoal text) before JAX device_put -- module-level
    (not nested), see symbolic_as_modulator/training/compute_norm_stats.py's identical class
    for why (TorchDataLoader's multiprocessing workers need to pickle this)."""

    def __call__(self, x: dict) -> dict:
        import numpy as np  # module

        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


@app.local_entrypoint()
def download_raw_data():
    """
    What it does:
        Fire-and-forget trigger for download_raw_data_remote, following this
        project's established .spawn()-based convention for anything that
        shouldn't depend on a local process/laptop staying connected --
        ~43GB of new download (12 tasks) plus a copy of the 4 existing
        tasks' files, easily long enough to be worth this even though a
        single earlier 13.6GB download "completed in a few minutes" (Arm D's
        own precedent, RESEARCH_LOG 2026-08-20 00:52).

        IMPORTANT -- .spawn() alone is NOT enough to survive this local
        process/terminal exiting: this file's `modal run` invocation MUST
        include `--detach` (`-d`) too, or the whole app (including the
        spawned call) gets torn down the moment the local entrypoint
        returns, silently. This exact gotcha has hit this project's dataset-
        prep scripts twice already (RESEARCH_LOG 2026-08-20 12:17 and
        2026-08-24 12:49 entries -- both "`.spawn()` without `--detach`
        insufficient") -- see [[feedback_modal_unattended_jobs]].

    Returns:
        None -- prints the spawned call ID to stdout.

    Example input:
        modal run --detach symbolic_as_modulator/training/build_full_suite_dataset.py::download_raw_data

    Example output:
        (stdout) "Spawned fc-abc123. This keeps running on Modal's servers..."
    """
    call = download_raw_data_remote.spawn()  # modal.FunctionCall
    print(f"Spawned {call.object_id}. This keeps running on Modal's servers ONLY IF this was")
    print("launched with `modal run --detach` -- otherwise it's torn down when this exits.")
    print("Check progress with: modal run .../build_full_suite_dataset.py::check_raw_data")


@app.local_entrypoint()
def run_calibration(max_episodes: int = 5, task: str = "InsertPeg"):
    """
    What it does:
        Blocking (no spawn/detach needed -- short by design) calibration run
        of build_preprocessed_dataset_remote over ONE new task's first few
        episodes, to MEASURE real CPU throughput before committing to a
        timeout for the full 16-task run. Per this project's own established
        lesson (RESEARCH_LOG 2026-08-20 00:52 / [[project_modal_image_gotchas]]
        item 10): never guess a preprocessing timeout, and remember
        build_preprocessed_dataset_remote (like the released DatasetProcessor
        it's adapted from) WIPES its output on every call, so a killed-by-
        timeout run means a full redo, not a resume -- measure first.

    Returns:
        None -- prints the measured stats and implied full-run time estimate.

    Example input:
        modal run symbolic_as_modulator/training/build_full_suite_dataset.py::run_calibration

    Example output:
        (stdout) "5 episodes of InsertPeg: ... seconds. Extrapolated 16-task estimate: ..."
    """
    import time  # module

    start = time.monotonic()  # float
    stats = build_preprocessed_dataset_remote.remote(max_episodes=max_episodes, tasks=[task])  # dict
    elapsed = time.monotonic() - start  # float
    print(f"[run_calibration] {max_episodes} episodes of {task}: {stats}, {elapsed:.0f}s wall-clock (includes container cold-start).")
    print("Use this to sanity-check build_preprocessed_dataset_remote's 6h timeout before running the full 16-task job.")


@app.local_entrypoint()
def build_preprocessed_dataset(max_episodes: int | None = None):
    """
    What it does:
        Fire-and-forget trigger for the real, full-16-task preprocessing run
        (or a max_episodes-capped version of it) -- MUST run run_calibration
        first and sanity-check the timeout, per this project's own established
        practice. Same --detach requirement as download_raw_data above (same
        two RESEARCH_LOG precedents apply) -- this step is expected to be the
        single longest-running one.

    Returns:
        None -- prints the spawned call ID to stdout.

    Example input:
        modal run --detach symbolic_as_modulator/training/build_full_suite_dataset.py::build_preprocessed_dataset

    Example output:
        (stdout) "Spawned fc-abc123. This keeps running on Modal's servers..."
    """
    call = build_preprocessed_dataset_remote.spawn(max_episodes=max_episodes)  # modal.FunctionCall
    print(f"Spawned {call.object_id}. This keeps running on Modal's servers ONLY IF this was")
    print("launched with `modal run --detach` -- otherwise it's torn down when this exits.")
    print("This WIPES any previous partial preprocessed output and rebuilds from scratch (not resumable).")


@app.local_entrypoint()
def compute_full_suite_norm_stats(repo_id: str = "symbolic_modulator_full_suite"):
    """
    What it does:
        Fire-and-forget trigger for compute_full_suite_norm_stats_remote.
        Same --detach requirement as the other long steps above -- untested
        at this 16-task scale (4x the data compute_norm_stats.py's own
        4-task run processed), so treated as potentially long-running out of
        caution even though the underlying operation is CPU-only and simple.

    Returns:
        None -- prints the spawned call ID to stdout.

    Example input:
        modal run --detach symbolic_as_modulator/training/build_full_suite_dataset.py::compute_full_suite_norm_stats

    Example output:
        (stdout) "Spawned fc-abc123. This keeps running on Modal's servers..."
    """
    call = compute_full_suite_norm_stats_remote.spawn(repo_id=repo_id)  # modal.FunctionCall
    print(f"Spawned {call.object_id}. This keeps running on Modal's servers ONLY IF this was")
    print("launched with `modal run --detach` -- otherwise it's torn down when this exits.")
