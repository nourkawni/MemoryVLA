"""
build_pilot_dataset.py

Prepares the training data for Arm D's Counting-suite pilot (see
arm_d_dynamic_fusion/README.md, "Pilot: does the gate actually arbitrate?").
Scopes the RoboMME dataset down to exactly the 4 pilot tasks (BinFill,
PickXtimes, SwingXtimes, StopCube) by only downloading and preprocessing
those 4 tasks' raw HDF5 files -- 13.6 GB combined vs. the full dataset's
56.4 GB -- rather than building a runtime task-filter over an already-built
16-task dataset. mme_vla_suite.dataset_builder.build_robomme_dataset's
DatasetProcessor already scopes itself to whatever *.h5 files exist in its
raw_data_path directory (confirmed by reading its source, not guessed), so
this is the entire mechanism needed: point it at a directory containing only
these 4 tasks' files.

Role in the system: the first stage of the pilot's data pipeline. Produces a
preprocessed pickle dataset (mme_vla_suite.training.dataset.SampleDataset
format) on a persistent Modal Volume; the training launcher (a follow-up
piece, not yet built) reads from that volume's preprocessed_data_path.
Does NOT compute norm_stats or wire the result into mme_vla_suite's config
registry -- both depend on how the training launcher resolves its DataConfig,
which is the next piece to build.

robomme_policy_learning/ is not edited: this script only imports and calls
its existing DatasetProcessor class with different (narrower) input data,
exactly the pattern arm_d_dynamic_fusion/README.md documents for the rest of
this arm.

Run with:
    modal run arm_d_dynamic_fusion/training/build_pilot_dataset.py::download_raw_data
    modal run arm_d_dynamic_fusion/training/build_pilot_dataset.py::build_preprocessed_dataset
"""

import pathlib

import modal

POLICY_LOCAL_DIR = str(  # str
    pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning"
)
ARM_D_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)  # str, the arm_d_dynamic_fusion/ dir

class RemoveStrings:
    """
    What it does:
        A data-transform step (structurally matches openpi.transforms.
        DataTransformFn's Protocol -- a plain __call__(dict)->dict, no
        inheritance needed) that drops any string-typed field from a sample
        dict. Defined at module level, not nested in compute_pilot_norm_stats,
        because TorchDataLoader's multiprocessing workers pickle the full
        transform list to hand to worker subprocesses -- a class defined
        inside a function isn't picklable ("Can't pickle local object").
        numpy-only (no openpi import) so this file's module-level code stays
        loadable by the local `modal run` CLI, which doesn't have openpi/jax
        installed -- matches this project's established convention of only
        importing heavy remote-only deps inside function bodies, never at
        module scope (see modal_reproduction/full_eval.py).

        Reproduces scripts/compute_norm_stats.py's own RemoveStrings class
        (not importable directly, that script isn't a package): string-typed
        fields (prompt/subgoal text) survive repack/data transforms
        unchanged, and JAX's device_put (TorchDataLoader's framework="jax"
        path) rejects any non-numeric dtype outright.

    Returns:
        n/a -- see __call__.

    Example input:
        RemoveStrings()

    Example output:
        a callable transform; see __call__ below.
    """

    def __call__(self, x: dict) -> dict:
        """
        What it does:
            Drops every key whose value is a string-dtype array.

        Returns:
            dict -- same as input minus any string-typed entries.

        Example input:
            {"state": np.zeros(8), "prompt": np.array("pick up the cube")}

        Example output:
            {"state": np.zeros(8)}
        """
        import numpy as np  # module

        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


PILOT_TASKS = ["BinFill", "PickXtimes", "SwingXtimes", "StopCube"]  # list[str], the Counting suite
HF_DATASET_REPO = "Yinpei/robomme_data_h5"  # str, HF dataset repo id (public, no auth needed)
HF_SIGLIP_REPO = "Yinpei/pi05_vision_encoder"  # str, HF model repo id for the frozen feature-extraction SigLIP subset

app = modal.App("robomme-arm-d-pilot-dataset")  # modal.App

data_volume = modal.Volume.from_name("robomme-arm-d-pilot-data", create_if_missing=True)  # modal.Volume
DATA_VOLUME_PATH = "/pilot_data"  # str
RAW_DATA_PATH = f"{DATA_VOLUME_PATH}/raw_h5"  # str
PREPROCESSED_DATA_PATH = f"{DATA_VOLUME_PATH}/preprocessed"  # str
OPENPI_DATA_HOME = f"{DATA_VOLUME_PATH}/openpi_data_home"  # str, holds pi05_vision_encoder/siglip_params.pkl

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
    .add_local_dir(ARM_D_LOCAL_DIR, remote_path="/arm_d_root/arm_d_dynamic_fusion", copy=True)
)


@app.function(image=image, volumes={DATA_VOLUME_PATH: data_volume}, timeout=3600)
def download_raw_data() -> dict:
    """
    What it does:
        Downloads exactly the 4 pilot tasks' compressed HDF5 archives from
        the RoboMME HF dataset repo (skipping the other 12 tasks entirely --
        13.6 GB instead of 56.4 GB), extracts them into RAW_DATA_PATH, and
        separately downloads the frozen SigLIP feature-extraction subset
        (pi05_vision_encoder) into OPENPI_DATA_HOME, since
        mme_vla_suite.shared.siglip_tokenizer.SigLipTokenizer hard-requires
        $OPENPI_DATA_HOME/pi05_vision_encoder/siglip_params.pkl to exist
        before it can compute perceptual memory features -- needed by the
        next step (build_preprocessed_dataset) regardless of task scope,
        since Arm D's perceptual stream depends on these features.
        Idempotent/resumable: skips any file already present and extracted
        on the volume, so re-running after an interruption only fetches
        what's missing.

    Returns:
        dict -- {"downloaded": list[str], "already_present": list[str]},
        the pilot task archives handled this call.

    Example input:
        download_raw_data.remote()

    Example output:
        {"downloaded": ["BinFill", "PickXtimes", "SwingXtimes", "StopCube"], "already_present": []}
    """
    import subprocess  # module

    import huggingface_hub  # module

    raw_dir = pathlib.Path(RAW_DATA_PATH)  # Path
    raw_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []  # list[str]
    already_present = []  # list[str]

    for task in PILOT_TASKS:
        h5_name = f"record_dataset_{task}.h5"  # str
        h5_path = raw_dir / h5_name  # Path
        if h5_path.exists():
            already_present.append(task)
            continue

        archive_name = f"{h5_name}.tar.xz"  # str
        print(f"[build_pilot_dataset] downloading {archive_name} ...")
        archive_path = huggingface_hub.hf_hub_download(
            repo_id=HF_DATASET_REPO, repo_type="dataset", filename=archive_name,
            local_dir=str(raw_dir),
        )  # str
        print(f"[build_pilot_dataset] extracting {archive_name} ...")
        subprocess.run(["tar", "-xJf", archive_path, "-C", str(raw_dir)], check=True)
        pathlib.Path(archive_path).unlink()  # free the compressed copy once extracted
        downloaded.append(task)

    siglip_marker = pathlib.Path(OPENPI_DATA_HOME) / "pi05_vision_encoder" / "siglip_params.pkl"  # Path
    if not siglip_marker.exists():
        print("[build_pilot_dataset] downloading pi05_vision_encoder (SigLIP feature-extraction subset) ...")
        siglip_marker.parent.mkdir(parents=True, exist_ok=True)
        huggingface_hub.snapshot_download(
            repo_id=HF_SIGLIP_REPO, local_dir=str(siglip_marker.parent),
        )

    data_volume.commit()
    print(f"[build_pilot_dataset] downloaded={downloaded}, already_present={already_present}")
    return {"downloaded": downloaded, "already_present": already_present}


@app.function(image=image, gpu="A10G", volumes={DATA_VOLUME_PATH: data_volume}, timeout=6 * 3600)
def build_preprocessed_dataset() -> dict:
    """
    What it does:
        Runs mme_vla_suite.dataset_builder.build_robomme_dataset.
        DatasetProcessor against RAW_DATA_PATH -- unmodified released code,
        called with narrower input rather than edited. DatasetProcessor.run()
        only ever processes *.h5 files it finds in raw_data_path, so with
        only the 4 pilot tasks' files present, the resulting preprocessed
        dataset naturally contains only those 4 tasks -- no separate
        task-filter code needed. Requires a GPU: this step computes and
        caches SigLIP perceptual-memory token embeddings per frame via
        mme_vla_suite.shared.mem_buffer.MemoryBuffer(prepare_buffer=True),
        which loads a JAX vision model (see download_raw_data's
        pi05_vision_encoder step).

        Timing note (measured, not estimated): a first real run on A10G took
        ~65 minutes to process BinFill's 100 episodes alone -- ~4+ hours for
        all 4 tasks, well past this function's original 1-hour timeout,
        which killed that run partway through the second task. Not
        resumable across separate calls: DatasetProcessor.__init__
        unconditionally `shutil.rmtree`s preprocessed_data_path and
        run()'s per-sample numbering restarts from 0 every call, with no
        parameter to opt out -- so a second call after any partial failure
        (timeout, crash, quota) redoes all already-finished tasks from
        scratch, not just the remaining ones. Sized generously (6h) rather
        than tightly to the observed ~4h, since a second timeout would mean
        a second full redo.

    Returns:
        dict -- {"execution_samples": int, "total_samples": int}, read back
        from the preprocessed dataset's meta/stats.json.

    Example input:
        build_preprocessed_dataset.remote()

    Example output:
        {"execution_samples": 380, "total_samples": 1450}
    """
    import json  # module
    import os  # module
    import sys  # module

    os.environ["OPENPI_DATA_HOME"] = OPENPI_DATA_HOME
    os.chdir("/app")  # some released code resolves config paths relative to cwd, not __file__
    sys.path.insert(0, "/app/src")
    from mme_vla_suite.dataset_builder.build_robomme_dataset import DatasetProcessor

    processor = DatasetProcessor(
        raw_data_path=RAW_DATA_PATH,
        preprocessed_data_path=PREPROCESSED_DATA_PATH,
        visualize=False,
    )
    processor.run()
    data_volume.commit()

    stats_path = pathlib.Path(PREPROCESSED_DATA_PATH) / "meta" / "stats.json"  # Path
    stats = json.loads(stats_path.read_text())  # dict
    print(f"[build_pilot_dataset] preprocessing done: {stats}")
    return stats


@app.function(image=image, volumes={DATA_VOLUME_PATH: data_volume}, timeout=120)
def check_preprocessed_size() -> str:
    """Runs `du -sh` on the preprocessed dataset's features/data dirs -- checks whether a bulk
    local-disk copy (the fix for diagnose_io_speed's cold-read finding) is feasible size-wise."""
    import subprocess  # module
    result = subprocess.run(
        ["du", "-sh", f"{PREPROCESSED_DATA_PATH}/features", f"{PREPROCESSED_DATA_PATH}/data"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    return result.stdout


@app.function(image=image, volumes={DATA_VOLUME_PATH: data_volume}, timeout=300)
def diagnose_bulk_copy_speed() -> dict:
    """
    What it does:
        Times a bulk `cp -r` of a handful of complete episode feature
        directories (not a random sample-by-sample walk like diagnose_io_speed)
        from the Modal Volume mount to the container's local (non-volume)
        disk -- tests whether a single streaming/sequential transfer is
        meaningfully faster per-byte than the many-small-random-reads pattern
        the dataloader uses, before committing to copying the full 107GB
        features dir (see check_preprocessed_size). Also confirms the local
        disk actually has room for that, via `df -h /local_copy_test`.

    Returns:
        dict -- {"bytes_copied": int, "seconds": float, "mbps": float,
                 "local_disk_free_gb_before": float, "extrapolated_full_107gb_minutes": float}

    Example input:
        diagnose_bulk_copy_speed.remote()

    Example output:
        {"bytes_copied": 2147483648, "seconds": 12.3, "mbps": 174.6, ...}
    """
    import pathlib as _pathlib  # module
    import shutil  # module
    import subprocess  # module
    import time  # module

    df_before = subprocess.run(["df", "-h", "/tmp"], capture_output=True, text=True).stdout  # str
    print(f"[diagnose_bulk_copy_speed] local disk before copy:\n{df_before}")

    feature_dir = _pathlib.Path(PREPROCESSED_DATA_PATH) / "features"  # Path
    episode_dirs = sorted(feature_dir.iterdir(), key=lambda p: p.stat().st_mtime)[:5]  # list[Path], 5 episodes' worth
    dest = _pathlib.Path("/tmp/local_copy_test")  # Path, NOT under the volume mount -- genuinely local disk
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    t0 = time.perf_counter()
    total_bytes = 0  # int
    for ep_dir in episode_dirs:
        dst_ep = dest / ep_dir.name
        shutil.copytree(ep_dir, dst_ep)
        total_bytes += sum(f.stat().st_size for f in dst_ep.rglob("*") if f.is_file())
    elapsed = time.perf_counter() - t0  # float

    mbps = (total_bytes / 1024 / 1024) / elapsed if elapsed > 0 else 0.0  # float
    extrapolated_minutes = (107 * 1024 / mbps) / 60 if mbps > 0 else float("inf")  # float, 107GB features dir

    df_after = subprocess.run(["df", "-h", "/tmp"], capture_output=True, text=True).stdout  # str
    print(f"[diagnose_bulk_copy_speed] local disk after copy:\n{df_after}")

    result = {
        "episodes_copied": [p.name for p in episode_dirs],
        "bytes_copied": total_bytes,
        "seconds": elapsed,
        "mbps": mbps,
        "extrapolated_full_107gb_minutes": extrapolated_minutes,
    }
    print(f"[diagnose_bulk_copy_speed] {result}")
    return result


@app.function(image=image, volumes={DATA_VOLUME_PATH: data_volume}, timeout=300)
def diagnose_io_speed(num_samples: int = 8) -> dict:
    """
    What it does:
        Isolates whether Modal Volume small-file read latency (not GPU
        compute, not the DataLoader's multiprocessing overhead) is why
        compute_pilot_norm_stats crawled at ~69 sec/batch instead of the
        expected few-minutes-total pace. Times raw ArmDDataset.__getitem__
        calls directly, single-process, no TorchDataLoader/multiprocessing
        involved -- each call internally reads ~32 small feature files
        (token_budget=512 / token_per_image=16) via a 36-thread pool (see
        mem_buffer.py's _gather_history_feat), so this isolates per-sample
        file-I/O cost from every other layer. No GPU needed for this (only
        the build step's prepare_buffer=True path touches SigLIP).

    Returns:
        dict -- {"cold_sample_times_sec": list[float], "warm_sample_times_sec": list[float],
                 "mean_cold_sec": float, "extrapolated_full_pass_hours_at_1_worker": float}

    Example input:
        diagnose_io_speed.remote(num_samples=8)

    Example output:
        {"cold_sample_times_sec": [2.1, 1.9, ...], "mean_cold_sec": 2.0, ...}
    """
    import os  # module
    import sys  # module
    import time  # module

    os.chdir("/app")
    sys.path.insert(0, "/app/src")
    sys.path.insert(0, "/arm_d_root")

    from arm_d_dynamic_fusion.training.arm_d_data import ArmDDataset
    from mme_vla_suite.models.config.utils import get_history_config

    history_config_path = "/arm_d_root/arm_d_dynamic_fusion/config/dynamic-fusion-arm-d.yaml"  # str
    history_config = get_history_config(history_config_path)  # omegaconf.DictConfig

    class _FakeDataConfig:
        norm_stats = None
        use_quantile_norm = False

    dataset = ArmDDataset(
        dataset_path=PREPROCESSED_DATA_PATH,
        data_config=_FakeDataConfig(),
        history_config=history_config,
        action_horizon=20,
        compute_norm_stats=True,
    )
    dataset_len = len(dataset)  # int
    print(f"[diagnose_io_speed] dataset length: {dataset_len}")

    indices = [int(i * dataset_len / num_samples) for i in range(num_samples)]  # list[int], spread across the dataset

    cold_times = []  # list[float]
    for idx in indices:
        t0 = time.perf_counter()
        _ = dataset[idx]
        cold_times.append(time.perf_counter() - t0)
        print(f"[diagnose_io_speed] sample {idx}: {cold_times[-1]:.2f}s")

    warm_times = []  # list[float], re-reading the SAME indices to check for any caching effect
    for idx in indices:
        t0 = time.perf_counter()
        _ = dataset[idx]
        warm_times.append(time.perf_counter() - t0)
        print(f"[diagnose_io_speed] sample {idx} (warm): {warm_times[-1]:.2f}s")

    mean_cold = sum(cold_times) / len(cold_times)  # float
    mean_warm = sum(warm_times) / len(warm_times)  # float
    # A single-process, single-sample-at-a-time extrapolation -- the real
    # DataLoader parallelizes across num_workers processes, so real
    # throughput should be faster than this by roughly that factor if the
    # bottleneck is CPU-bound; if it's instead a shared I/O ceiling (Modal
    # Volume backend throughput, not per-request latency), more workers
    # won't help proportionally, which is exactly the thing this number is
    # meant to help distinguish once compared against the real batch rate.
    extrapolated_hours = (mean_cold * dataset_len) / 3600  # float

    result = {
        "dataset_length": dataset_len,
        "cold_sample_times_sec": cold_times,
        "warm_sample_times_sec": warm_times,
        "mean_cold_sec": mean_cold,
        "mean_warm_sec": mean_warm,
        "extrapolated_full_pass_hours_at_1_worker": extrapolated_hours,
    }
    print(f"[diagnose_io_speed] {result}")

    # Locality hypothesis: build_robomme_dataset.py writes samples in
    # episode order (outer loop over episode_indices, inner loop over that
    # episode's timesteps), so consecutive dataset indices should mostly be
    # consecutive timesteps within the same episode -- and frame_sampling's
    # even_sampling_indices(step_idx, ...) for nearby step_idx values should
    # select mostly-overlapping frame windows. If true, reading in dataset
    # order (not shuffled) should hit mostly-warm files after the first
    # sample in each run, unlike diagnose_io_speed's spread-out indices
    # above (each far enough apart to guarantee a cold miss). 177GB of
    # preprocessed features/data (see check_preprocessed_size) makes a bulk
    # local-disk copy an expensive fix to reach for first -- this is much
    # cheaper to test, and RunningStats doesn't care about sample order.
    start_idx = indices[len(indices) // 2]  # int, pick a point already touched once above (still a fair first-touch test for its neighbors)
    consecutive_times = []  # list[float]
    for offset in range(20):
        idx = start_idx + offset
        if idx >= dataset_len:
            break
        t0 = time.perf_counter()
        _ = dataset[idx]
        consecutive_times.append(time.perf_counter() - t0)
        print(f"[diagnose_io_speed] consecutive idx {idx} (offset {offset}): {consecutive_times[-1]:.2f}s")

    result["consecutive_times_sec"] = consecutive_times
    result["mean_consecutive_after_first_sec"] = (
        sum(consecutive_times[1:]) / len(consecutive_times[1:]) if len(consecutive_times) > 1 else None
    )
    print(f"[diagnose_io_speed] locality test: {result['consecutive_times_sec']}")
    return result


@app.function(image=image, volumes={DATA_VOLUME_PATH: data_volume}, timeout=6 * 3600)
def consolidate_episode_features(episode_names_csv: str = "") -> dict:
    """
    What it does:
        Packs each episode's many small per-timestep `token_emb_<idx>.npy`
        files (the root cause of diagnose_io_speed's cold-read finding --
        ~32 separate small-file opens per training sample) into ONE combined
        file per episode under features_consolidated/. Doesn't touch or
        re-derive the underlying SigLIP embeddings themselves -- purely a
        repackaging of already-computed values, so this does NOT redo the
        GPU preprocessing step (build_preprocessed_dataset). Original small
        files are left untouched (not deleted), so this is safe to re-run or
        abandon without risk to the already-working data.

        Resumable and interruption-safe: skips any episode whose consolidated
        output file already exists, and commits the volume after EVERY
        episode (not just at the end) -- unlike this file's earlier mistakes
        (DatasetProcessor wiping everything on retry, single end-of-run
        commits), an interruption here only costs whatever the
        currently-in-progress episode's work was, not everything done so far.

    Returns:
        dict -- {"episodes_consolidated": list[str], "skipped_already_done": list[str],
                 "details": list[dict]} -- details has one {"episode", "num_files", "seconds"} per episode processed this call.

    Example input:
        consolidate_episode_features.remote(episode_names_csv="episode_0,episode_1")

    Example output:
        {"episodes_consolidated": ["episode_0"], "skipped_already_done": ["episode_1"], "details": [...]}
    """
    import pathlib as _pathlib  # module
    import time  # module

    import numpy as np  # module

    feature_dir = _pathlib.Path(PREPROCESSED_DATA_PATH) / "features"  # Path
    out_dir = _pathlib.Path(PREPROCESSED_DATA_PATH) / "features_consolidated"  # Path
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_names = [n.strip() for n in episode_names_csv.split(",") if n.strip()] or None  # list[str] | None

    all_episode_dirs = sorted(  # list[Path]
        (d for d in feature_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")),
        key=lambda p: int(p.name.split("_")[1]),
    )
    episode_dirs = (  # list[Path]
        [d for d in all_episode_dirs if d.name in episode_names]
        if episode_names is not None
        else all_episode_dirs
    )
    print(f"[consolidate] {len(episode_dirs)} episode(s) to process")

    consolidated = []  # list[str]
    skipped = []  # list[str]
    details = []  # list[dict]

    for ep_dir in episode_dirs:
        out_path = out_dir / f"{ep_dir.name}.npy"  # Path
        if out_path.exists():
            skipped.append(ep_dir.name)
            print(f"[consolidate] {ep_dir.name}: already consolidated, skipping")
            continue

        t0 = time.perf_counter()
        npy_files = sorted(  # list[Path]
            ep_dir.glob("token_emb_*.npy"), key=lambda p: int(p.stem.split("_")[-1])
        )
        combined = {}  # dict[int, dict]
        for f in npy_files:
            idx = int(f.stem.split("_")[-1])  # int
            with open(f, "rb") as fh:
                combined[idx] = np.load(fh, allow_pickle=True).item()

        np.save(out_path, combined, allow_pickle=True)
        elapsed = time.perf_counter() - t0  # float
        data_volume.commit()  # after every episode, not just at the end

        consolidated.append(ep_dir.name)
        details.append({"episode": ep_dir.name, "num_files": len(npy_files), "seconds": elapsed})
        print(f"[consolidate] {ep_dir.name}: {len(npy_files)} files -> {out_path} in {elapsed:.1f}s")

    result = {"episodes_consolidated": consolidated, "skipped_already_done": skipped, "details": details}
    print(f"[consolidate] {result}")
    return result


@app.function(image=image, volumes={DATA_VOLUME_PATH: data_volume}, timeout=300)
def verify_consolidated_read_speed(episode_name: str = "episode_0") -> dict:
    """
    What it does:
        Times reading ALL of one episode's features from its consolidated
        file (one open, one np.load) vs. what the same data cost via the
        original many-small-files layout (comparable to diagnose_io_speed's
        per-sample numbers) -- the actual before/after proof this fix works,
        not another extrapolation, before committing to consolidating and
        re-pointing the dataloader at all ~400 episodes.

    Returns:
        dict -- {"episode": str, "num_timesteps": int, "cold_read_seconds": float,
                 "warm_read_seconds": float}

    Example input:
        verify_consolidated_read_speed.remote(episode_name="episode_0")

    Example output:
        {"episode": "episode_0", "num_timesteps": 550, "cold_read_seconds": 0.8, "warm_read_seconds": 0.1}
    """
    import pathlib as _pathlib  # module
    import time  # module

    import numpy as np  # module

    consolidated_path = _pathlib.Path(PREPROCESSED_DATA_PATH) / "features_consolidated" / f"{episode_name}.npy"  # Path
    if not consolidated_path.exists():
        return {"error": f"{consolidated_path} does not exist -- run consolidate_episode_features first"}

    t0 = time.perf_counter()
    with open(consolidated_path, "rb") as fh:
        combined = np.load(fh, allow_pickle=True).item()
    cold_seconds = time.perf_counter() - t0  # float

    t0 = time.perf_counter()
    with open(consolidated_path, "rb") as fh:
        _ = np.load(fh, allow_pickle=True).item()
    warm_seconds = time.perf_counter() - t0  # float

    result = {
        "episode": episode_name,
        "num_timesteps": len(combined),
        "cold_read_seconds": cold_seconds,
        "warm_read_seconds": warm_seconds,
    }
    print(f"[verify_consolidated_read_speed] {result}")
    return result


@app.function(image=image, volumes={DATA_VOLUME_PATH: data_volume}, timeout=1800)
def test_parallel_consolidation(episode_names_csv: str) -> dict:
    """
    What it does:
        Dispatches consolidate_episode_features once PER episode in
        episode_names_csv, concurrently, via .map() -- calling .map() from
        inside a Modal function (not a local_entrypoint) keeps the fan-out
        server-side, avoiding the local-dispatch-loop problem this file hit
        earlier (see run_all_remote's docstring / project memory on
        unattended Modal jobs). Times total wall-clock for the whole batch,
        to compare against the already-known sequential rate (~0.2 sec/file,
        measured on episode_0/episode_1) -- the actual evidence for whether
        parallelizing across episodes helps here, before committing to
        firing off all ~400 at once.

    Returns:
        dict -- {"episodes_requested": list[str], "wall_clock_seconds": float,
                 "total_files": int, "per_episode_results": list[dict]}

    Example input:
        test_parallel_consolidation.remote(episode_names_csv="episode_2,episode_3,...,episode_9")

    Example output:
        {"episodes_requested": [...], "wall_clock_seconds": 95.3, "total_files": 3800, "per_episode_results": [...]}
    """
    import time  # module

    episode_names = [n.strip() for n in episode_names_csv.split(",") if n.strip()]  # list[str]
    print(f"[test_parallel_consolidation] dispatching {len(episode_names)} episodes concurrently via .map()")

    t0 = time.perf_counter()
    per_episode_results = list(
        consolidate_episode_features.map([n for n in episode_names])
    )  # list[dict], one consolidate_episode_features() result per episode
    wall_clock = time.perf_counter() - t0  # float

    total_files = sum(
        d["num_files"] for r in per_episode_results for d in r.get("details", [])
    )  # int
    result = {
        "episodes_requested": episode_names,
        "wall_clock_seconds": wall_clock,
        "total_files": total_files,
        "per_episode_results": per_episode_results,
    }
    print(f"[test_parallel_consolidation] {result}")
    return result


@app.function(image=image, volumes={DATA_VOLUME_PATH: data_volume}, timeout=10 * 3600)
def consolidate_all_episodes() -> dict:
    """
    What it does:
        Discovers every episode directory under features/, filters out ones
        already consolidated (features_consolidated/<episode>.npy already
        exists -- covers episode_0/1 from the first small test and
        episode_2-9 from the parallel-speedup test), and dispatches the rest
        via consolidate_episode_features.map(), same as
        test_parallel_consolidation but for the full ~400-episode dataset
        instead of a manually-listed test batch. The .map() call happens
        inside this Modal function, not a local_entrypoint, so the whole
        dispatch stays server-side -- meant to be launched with
        `modal run --detach`, surviving both this local process exiting and
        (per direct testing earlier this session) the local machine
        sleeping/disconnecting entirely, unlike this file's very first
        (unfixed) version of run_all.

        From test_parallel_consolidation's measured 8-way aggregate rate
        (~9.3 files/sec, sub-linear vs the ~5 files/sec single-container
        rate -- real but limited benefit from parallelism, likely a shared
        backend bottleneck) and an estimated ~249,600 total files across all
        400 episodes: expect roughly 7-8 hours wall-clock. Original small
        per-timestep files are left untouched throughout -- this only adds
        features_consolidated/, so nothing about the already-working
        (if slow) original data path is at risk.

    Returns:
        dict -- {"total_episodes": int, "processed_this_run": int,
                 "wall_clock_seconds": float, "results": list[dict]}

    Example input:
        consolidate_all_episodes.remote()

    Example output:
        {"total_episodes": 400, "processed_this_run": 390, "wall_clock_seconds": 27000.0, "results": [...]}
    """
    import pathlib as _pathlib  # module
    import time  # module

    feature_dir = _pathlib.Path(PREPROCESSED_DATA_PATH) / "features"  # Path
    out_dir = _pathlib.Path(PREPROCESSED_DATA_PATH) / "features_consolidated"  # Path
    out_dir.mkdir(parents=True, exist_ok=True)

    all_episode_names = sorted(  # list[str]
        (d.name for d in feature_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")),
        key=lambda n: int(n.split("_")[1]),
    )
    remaining = [n for n in all_episode_names if not (out_dir / f"{n}.npy").exists()]  # list[str]
    print(f"[consolidate_all] {len(all_episode_names)} total episodes, {len(remaining)} remaining to consolidate")

    t0 = time.perf_counter()
    results = list(consolidate_episode_features.map(remaining)) if remaining else []  # list[dict]
    elapsed = time.perf_counter() - t0  # float

    result = {
        "total_episodes": len(all_episode_names),
        "processed_this_run": len(remaining),
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    print(f"[consolidate_all] done: {len(remaining)} episodes in {elapsed / 3600:.2f}h")
    return result


@app.function(image=image, volumes={DATA_VOLUME_PATH: data_volume}, timeout=6 * 3600)
def compute_pilot_norm_stats(repo_id: str = "arm_d_pilot") -> dict:
    """
    What it does:
        Computes state/action normalization statistics over the pilot's
        preprocessed dataset and saves them under ASSETS_PATH/<repo_id>, the
        exact location mme_vla_suite.training.config.DataConfigFactory.
        create_base_config looks for norm_stats (assets_dir / asset_id, and
        asset_id defaults to repo_id) -- required before training: without
        it, RoboMMEDataset.__init__ crashes on `data_config.norm_stats['state']`
        (norm_stats is None, not just under-populated, if this step hasn't
        run). Reimplements scripts/compute_norm_stats.py's logic rather than
        calling its main() directly, since that function imports and
        constructs RoboMMEDataset by name with no injection point -- same
        reason the training launcher monkeypatches it for the actual
        training run.

        NOT actually a "single, short-lived stats pass" in practice, per
        diagnose_io_speed/diagnose_bulk_copy_speed's findings: Modal Volume
        read latency for a file that's never been touched before is ~100x
        higher than for one that has (confirmed durable across separate
        containers, not just within one), and neither reordering reads
        (locality test, disproven) nor bulk-copying to local disk (hit its
        own 300s timeout on just 5 episodes) sidesteps it -- it looks like a
        server-side cold-storage-to-cache promotion cost, paid once per file
        regardless of access pattern. This run doubles as that one-time
        "warming" pass for the whole ~240k-file dataset, estimated (from the
        best throughput actually measured, ~15 files/sec aggregate) at
        4-5 hours -- hence the generous 6h timeout and no GPU (this step
        never needed one; an earlier version wastefully carried gpu="A10G"
        over from build_preprocessed_dataset's config). Unverified: whether
        the warm state persists indefinitely or could evict over time: if
        training happens long after this run, the same cold-read cost could
        recur.

    Returns:
        dict -- {"state": {...}, "actions": {...}}, the computed norm_stats
        (mean/std/q01/q99 per dimension) -- also the return value of
        openpi.shared.normalize.RunningStats.get_statistics() for each key.

    Example input:
        compute_pilot_norm_stats.remote()

    Example output:
        {"state": {"mean": [...], "std": [...], "q01": [...], "q99": [...]}, "actions": {...}}
    """
    import os  # module
    import sys  # module

    import numpy as np  # module
    import tqdm  # module

    os.chdir("/app")  # some released code resolves config paths relative to cwd, not __file__
    sys.path.insert(0, "/app/src")
    sys.path.insert(0, "/arm_d_root")
    import openpi.shared.normalize as normalize
    from openpi.training.data_loader import TorchDataLoader, TransformedDataset

    from arm_d_dynamic_fusion.training.arm_d_data import ArmDDataConfig, ArmDDataset
    from mme_vla_suite.models.config.utils import get_history_config

    history_config_path = "/arm_d_root/arm_d_dynamic_fusion/config/dynamic-fusion-arm-d.yaml"  # str
    history_config = get_history_config(history_config_path)  # omegaconf.DictConfig

    data_config_factory = ArmDDataConfig(repo_id=repo_id)  # ArmDDataConfig
    assets_dirs = pathlib.Path(DATA_VOLUME_PATH) / "assets"  # Path, doesn't need to exist yet --
    # create_base_config's norm_stats lookup catches FileNotFoundError and
    # returns None, which is fine here since compute_norm_stats=True below
    # means RoboMMEDataset never reads data_config.norm_stats in the first
    # place. model_transforms isn't used by this stats pass either, so a
    # minimal stand-in model_config is enough.
    from openpi.models.pi0_config import Pi0Config
    data_config = data_config_factory.create(
        assets_dirs=assets_dirs, model_config=Pi0Config(action_horizon=20)
    )

    dataset = ArmDDataset(
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

    batch_size = 32  # int
    num_batches = max(1, len(dataset) // batch_size)  # int
    # shuffle=False deliberately: RunningStats doesn't care about sample
    # order, and ArmDDataset._gather_history_feat's per-episode LRU cache
    # (see arm_d_data.py, Fix 3) only pays off when consecutive samples
    # stay within the same episode -- true for the dataset's natural
    # (episode-then-timestep) write order, false under full shuffling,
    # which spreads each batch's 32 samples across ~30 different episodes
    # and forces ~30 cold episode-loads per batch regardless of caching.
    # Confirmed by a real run: shuffle=True measured ~38s/batch even with
    # consolidation; this is the fix for that, not yet measured.
    data_loader = TorchDataLoader(
        dataset, local_batch_size=batch_size, sharding=None,
        num_batches=num_batches, num_workers=4, seed=0, shuffle=False, framework="jax",
    )

    keys = ["state", "actions"]  # list[str]
    stats = {key: normalize.RunningStats() for key in keys}
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing norm stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}  # dict
    # Path must match launch_pilot_training.py's TrainConfig.assets_dirs
    # property exactly: (assets_base_dir / name) / asset_id, where asset_id
    # defaults to repo_id. Both scripts use "arm_d_pilot" for both `name`
    # and `repo_id`, and ASSETS_BASE_DIR below must equal that launcher's
    # TrainConfig.assets_base_dir.
    ASSETS_BASE_DIR = f"{DATA_VOLUME_PATH}/assets"  # str
    output_path = pathlib.Path(ASSETS_BASE_DIR) / "arm_d_pilot" / repo_id  # Path
    normalize.save(output_path, norm_stats)
    data_volume.commit()
    print(f"[build_pilot_dataset] wrote norm_stats to {output_path}")
    return norm_stats


@app.function(image=image, timeout=7 * 3600)
def run_all_remote() -> dict:
    """
    What it does:
        The full data-prep sequence (download+extract, then build, then
        norm_stats), run entirely server-side -- calling .remote() on
        another Modal function from inside a Modal function stays on
        Modal's infrastructure regardless of the local client, unlike
        calling .remote() from a local_entrypoint (see run_all's docstring
        for why that distinction matters here: a first attempt at this
        exact job died twice from local-client disconnects with no error on
        Modal's side, confirmed by "Stopping app - local client
        disconnected" in the logs, despite each individual step being well
        inside its own generous timeout).

    Returns:
        dict -- {"download": dict, "build": dict, "norm_stats": dict}, the
        three stages' own return values.

    Example input:
        run_all_remote.spawn()

    Example output:
        {"download": {...}, "build": {"execution_samples": 380, "total_samples": 1450}, "norm_stats": {...}}
    """
    download_result = download_raw_data.remote()  # dict
    print(f"[build_pilot_dataset] download stage done: {download_result}")
    build_result = build_preprocessed_dataset.remote()  # dict
    print(f"[build_pilot_dataset] build stage done: {build_result}")
    norm_stats_result = compute_pilot_norm_stats.remote()  # dict
    print(f"[build_pilot_dataset] norm_stats stage done: {norm_stats_result}")
    return {"download": download_result, "build": build_result, "norm_stats": norm_stats_result}


@app.local_entrypoint()
def run_all():
    """
    What it does:
        Fire-and-forget trigger for the full data-prep sequence: spawns
        run_all_remote() and returns immediately, printing the function call
        ID. Does NOT block on it and does NOT depend on this local process
        staying alive or connected -- the whole multi-hour sequence runs
        entirely server-side (see run_all_remote's docstring). Matches this
        project's established convention for anything that shouldn't depend
        on a local process/laptop staying connected (see
        modal_reproduction/full_eval.py's run_batch for the original
        precedent and reasoning, and [[feedback_modal_unattended_jobs]]).

    Returns:
        None -- prints the spawned call ID to stdout.

    Example input:
        modal run arm_d_dynamic_fusion/training/build_pilot_dataset.py::run_all

    Example output:
        (stdout) "Spawned fc-abc123. This keeps running on Modal's servers..."
    """
    call = run_all_remote.spawn()  # modal.FunctionCall
    print(f"Spawned {call.object_id}. This keeps running on Modal's servers regardless of")
    print("this local process - no need to keep a terminal/session open.")
    print("Check progress anytime with: modal app list / modal app logs <app-id>")
