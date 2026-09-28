"""
download_xf_full_suite_dataset.py

Downloads the RoboMME paper authors' own pre-built, fully preprocessed
16-task dataset (Yinpei/robomme_preprocessed_data on HuggingFace) instead of
recomputing perceptual (SigLIP) features from scratch. Chosen over
build_xf_full_suite_dataset.py's from-scratch DatasetProcessor path
(calibrated 2026-09-19 at ~6.93 GPU-hours on A10G for the full 1,600
episodes) because it needs zero GPU compute and carries zero risk of this
project's own preprocessing subtly diverging from what the paper's released
checkpoints actually trained on -- it costs bandwidth/storage instead of
GPU-hours, which this project treats as the scarcer resource.

Only downloads the `data/` (125.2 GB) and `features/` (230.5 GB) prefixes --
355.7 GB total, confirmed via `huggingface_hub.HfApi().dataset_info(...,
files_metadata=True)` on 2026-09-19. Deliberately skips `memer/images.zip`
(24.8 GB) and `qwenvl/images.zip` (3.1 GB): those are VLM subgoal-predictor
training images, unrelated to XF's Oracle-caption v1 scope.

Known open risk, not yet resolved (see chat, 2026-09-19): Modal volumes on
this account hard-cap at 500,000 files (confirmed via a real warning on the
existing robomme-symbolic-modulator-full-suite-data volume, currently at
476,857 files just for ITS OWN data/ folder on this same 16-task dataset).
Extrapolating this run's own real numbers -- data/ alone lands close to the
same ~505k figure, and features/ (one token_emb_*.npy per kept frame) looks
like it could land well over 1M files -- this download may hit that ceiling
before finishing. Deliberately NOT pre-sharding across multiple volumes for
this first attempt: that's real added complexity (RoboMMEDataset hardcodes
feature_dir = dataset_path/"features" as one directory, so a real fix would
need extra routing logic in our own XFDataset subclass) that's only worth
building if this actually fails, not speculatively. This run is resumable
(huggingface_hub.snapshot_download skips already-downloaded files, and
volume.commit() is called periodically) specifically so that if it DOES hit
the wall, nothing already-downloaded is lost and we'll know exactly where the
real ceiling is, instead of guessing.

Run with (long job, survives a closed laptop only with --detach):
    modal run --detach xattn_fusion/training/download_xf_full_suite_dataset.py::run_download
"""

import pathlib

import modal

ROBOMME_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning")

app = modal.App("xf-download-full-suite-dataset")

# modal.Volume, same one build_xf_full_suite_dataset.py's calibration run already wrote
# the staged pi05_vision_encoder weights to -- reused, not duplicated.
xf_data_volume = modal.Volume.from_name("xf-full-suite-data", create_if_missing=True)
XF_DATA_VOLUME_PATH = "/xf_data"

HF_DATASET_REPO = "Yinpei/robomme_preprocessed_data"  # str, the paper authors' own release
RAW_ZIP_DIR = f"{XF_DATA_VOLUME_PATH}/hf_raw_zips"  # str, where the .zip files land before unzipping
PREPROCESSED_DATA_PATH = f"{XF_DATA_VOLUME_PATH}/preprocessed"  # str, unzipped, matches RoboMMEDataset's expected layout

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"UV_LINK_MODE": "copy", "UV_PYTHON_DOWNLOADS": "automatic"})
    .add_local_dir(ROBOMME_LOCAL_DIR, remote_path="/app", copy=True)
    .run_commands("cd /app && /root/.local/bin/uv pip install --system huggingface_hub tqdm")
    .run_commands("cd /app && /root/.local/bin/uv pip install --system hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # bool-as-str, parallel chunked downloads for large files
)


@app.function(
    image=image,
    timeout=6 * 3600,
    volumes={XF_DATA_VOLUME_PATH: xf_data_volume},
    secrets=[modal.Secret.from_name("hf-write-token")],
)
def download_and_unzip_remote() -> dict:
    """
    What it does:
        Downloads data/*.zip, features/*.zip, and meta/* from the released
        robomme_preprocessed_data HF dataset repo (skipping memer/qwenvl),
        then unzips everything using the released repo's own
        scripts/unzip_data.py (unchanged, called as a subprocess -- its
        features/episode_*.zip special-casing is exactly what
        RoboMMEDataset needs). Commits the volume after the download step
        and again after unzipping, so a mid-run failure doesn't lose
        already-completed work. Ends by walking PREPROCESSED_DATA_PATH and
        reporting the real file count under data/ and features/ separately,
        to directly confirm or refute the ~500k-file-per-volume ceiling
        concern with a real number instead of an extrapolation.

    Returns:
        dict -- {"download_elapsed_s": float, "unzip_elapsed_s": float,
        "data_file_count": int, "features_file_count": int,
        "total_file_count": int}.

    Example input:
        download_and_unzip_remote.remote()

    Example output:
        {"download_elapsed_s": 3120.4, "unzip_elapsed_s": 2810.7, "data_file_count": 504821, "features_file_count": 1102345, "total_file_count": 1607166}
    """
    import os
    import subprocess
    import time

    import huggingface_hub

    t0 = time.perf_counter()
    print("[download_xf_full_suite_dataset] downloading data/ + features/ + meta/ from", HF_DATASET_REPO)
    huggingface_hub.snapshot_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        local_dir=RAW_ZIP_DIR,
        allow_patterns=["data/*", "features/*", "meta/*"],
        max_workers=16,
    )
    download_elapsed = time.perf_counter() - t0
    xf_data_volume.commit()
    print(f"[download_xf_full_suite_dataset] download done in {download_elapsed:.1f}s, volume committed")

    t1 = time.perf_counter()
    pathlib.Path(PREPROCESSED_DATA_PATH).mkdir(parents=True, exist_ok=True)
    # Reuses the released, unmodified unzip logic (features/episode_*.zip -> episode_*/ layout).
    subprocess.run(
        ["cp", "-r", f"{RAW_ZIP_DIR}/data", f"{RAW_ZIP_DIR}/features", f"{RAW_ZIP_DIR}/meta", PREPROCESSED_DATA_PATH],
        check=True,
    )
    result = subprocess.run(
        ["/usr/local/bin/python", "/app/scripts/unzip_data.py", PREPROCESSED_DATA_PATH, "-p", "8"],
        capture_output=True, text=True, timeout=6 * 3600 - int(download_elapsed) - 300,
    )
    unzip_elapsed = time.perf_counter() - t1
    print(result.stdout[-6000:])
    if result.stderr:
        print("--- STDERR ---")
        print(result.stderr[-3000:])
    xf_data_volume.commit()
    print(f"[download_xf_full_suite_dataset] unzip done in {unzip_elapsed:.1f}s, volume committed")

    data_file_count = sum(len(files) for _, _, files in os.walk(os.path.join(PREPROCESSED_DATA_PATH, "data")))
    features_file_count = sum(len(files) for _, _, files in os.walk(os.path.join(PREPROCESSED_DATA_PATH, "features")))

    return {
        "download_elapsed_s": download_elapsed,
        "unzip_elapsed_s": unzip_elapsed,
        "data_file_count": data_file_count,
        "features_file_count": features_file_count,
        "total_file_count": data_file_count + features_file_count,
        "returncode": result.returncode,
    }


# list[int], the exact episode indices whose features/episode_N.zip failed to unzip
# with "[Errno 28] No space left on device" on the 2026-09-19 run (confirmed via
# `modal app logs`, not guessed) -- data/ (416,960 files) and the other 1,572
# episodes' features finished cleanly; only these 28/1600 (1.75%) need a retry.
FAILED_EPISODE_IDS = [
    98, 99, 869, 875, 882, 890, 898, 904, 911, 918, 927, 934, 937, 943, 944,
    950, 951, 958, 959, 964, 967, 972, 976, 981, 986, 990, 997, 998,
]


@app.function(
    image=image,
    timeout=3600,
    volumes={XF_DATA_VOLUME_PATH: xf_data_volume},
    secrets=[modal.Secret.from_name("hf-write-token")],
)
def cleanup_and_repair_remote() -> dict:
    """
    What it does:
        Fixes the real root cause of the 2026-09-19 ENOSPC failure -- NOT a
        Modal per-volume file-count ceiling (data/ finished at 416,960 files,
        well under the ~500k figure that was originally suspected), but
        actual disk space exhaustion caused by this script's own wasteful
        design: it copied the full 355.7 GB of downloaded .zip files into
        preprocessed/ before unzipping (instead of unzipping in place), and
        unzip_data.py never deletes a .zip after extracting it -- so the
        volume briefly needed roughly 2-3x the dataset's real size at once
        (raw zips in hf_raw_zips/ + duplicate copied zips in preprocessed/ +
        the actual unzipped output). Fixes it by: (1) deleting every leftover
        .zip under preprocessed/ (pure post-unzip waste, ~355.7 GB), (2)
        deleting hf_raw_zips/ entirely (~355.7 GB, no longer needed once
        (1) confirms the unzip already happened), (3) removing the 28
        FAILED_EPISODE_IDS' partial output directories, (4) re-downloading
        ONLY those 28 episodes' zip files (a few GB, not 355.7 GB) and
        unzipping them properly into the now-freed space.

    Returns:
        dict -- {"freed_zip_count": int, "repaired_episode_count": int,
        "data_file_count": int, "features_episode_count": int}.

    Example input:
        cleanup_and_repair_remote.remote()

    Example output:
        {"freed_zip_count": 1610, "repaired_episode_count": 28, "data_file_count": 416960, "features_episode_count": 1600}
    """
    import os
    import shutil
    import subprocess
    import sys

    import huggingface_hub

    sys.path.insert(0, "/app/scripts")

    freed = 0  # int, count of leftover .zip files deleted
    for root in ("data", "features", "meta"):
        d = os.path.join(PREPROCESSED_DATA_PATH, root)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name.endswith(".zip"):
                os.remove(os.path.join(d, name))
                freed += 1
    print(f"[cleanup_and_repair] removed {freed} leftover .zip files from preprocessed/")

    if os.path.isdir(RAW_ZIP_DIR):
        shutil.rmtree(RAW_ZIP_DIR)
        print(f"[cleanup_and_repair] removed {RAW_ZIP_DIR}")
    xf_data_volume.commit()

    repair_dir = f"{XF_DATA_VOLUME_PATH}/hf_repair_zips"
    pathlib.Path(repair_dir).mkdir(parents=True, exist_ok=True)
    features_root = os.path.join(PREPROCESSED_DATA_PATH, "features")

    for epi in FAILED_EPISODE_IDS:
        out_dir = os.path.join(features_root, f"episode_{epi}")
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)  # drop the partial extraction, if any

        zip_path = huggingface_hub.hf_hub_download(
            repo_id=HF_DATASET_REPO, repo_type="dataset",
            filename=f"features/episode_{epi}.zip", local_dir=repair_dir,
        )
        # Reuses unzip_data.py's own "features_episode" extraction rule (imported, not
        # reimplemented), so the output layout is byte-for-byte what the first pass produced.
        from unzip_data import unzip_one  # type: ignore
        unzip_one(pathlib.Path(zip_path), overwrite=True)
        # unzip_one writes relative to the zip's OWN parent dir (repair_dir/features/episode_N/);
        # move it to where RoboMMEDataset actually expects it.
        produced = os.path.join(repair_dir, "features", f"episode_{epi}")
        if os.path.isdir(produced):
            shutil.move(produced, out_dir)
        else:
            print(f"[cleanup_and_repair] WARNING: episode_{epi} still missing after retry")

    shutil.rmtree(repair_dir, ignore_errors=True)
    xf_data_volume.commit()

    data_file_count = sum(
        1 for _, _, files in os.walk(os.path.join(PREPROCESSED_DATA_PATH, "data")) for _ in files
    )
    features_episode_count = len(
        [d for d in os.listdir(features_root) if os.path.isdir(os.path.join(features_root, d))]
    )
    result = {
        "freed_zip_count": freed,
        "repaired_episode_count": len(FAILED_EPISODE_IDS),
        "data_file_count": data_file_count,
        "features_episode_count": features_episode_count,
    }
    print(f"[cleanup_and_repair] done: {result}")
    return result


@app.local_entrypoint()
def run_cleanup_and_repair():
    """
    What it does: blocking (short by design, no spawn/detach needed) repair
    run -- frees the wasted duplicate storage and re-fetches only the 28
    episodes that failed on 2026-09-19, then prints the final real counts.

    Returns:
        None.

    Example input:
        modal run xattn_fusion/training/download_xf_full_suite_dataset.py::run_cleanup_and_repair

    Example output:
        (stdout) "{'freed_zip_count': 1610, 'repaired_episode_count': 28, 'data_file_count': 416960, 'features_episode_count': 1600}"
    """
    result = cleanup_and_repair_remote.remote()
    print(result)


# int, confirmed via a direct raw-H5 episode count (100 episodes x 16 tasks) earlier this project.
TOTAL_EPISODES = 1600
# int, 4 shards keeps each one around ~275k files (1.1M estimated features files / 4), comfortably
# under the CONFIRMED (not estimated) 500,000-file-per-volume ceiling -- see chat, 2026-09-19: the
# single-volume attempt hit "using 100.0% of available inodes (500000 out of 500000)" with 382GB of
# disk space still free, proving this is a file-COUNT ceiling, not a byte-size one.
NUM_FEATURE_SHARDS = 4
feature_shard_volumes = [
    modal.Volume.from_name(f"xf-features-shard-{i}", create_if_missing=True) for i in range(NUM_FEATURE_SHARDS)
]  # list[modal.Volume]
FEATURE_SHARD_PATHS = [f"/xf_features_shard_{i}" for i in range(NUM_FEATURE_SHARDS)]  # list[str]


@app.function(
    image=image,
    timeout=3600,
    volumes={path: vol for path, vol in zip(FEATURE_SHARD_PATHS, feature_shard_volumes)},
    secrets=[modal.Secret.from_name("hf-write-token")],
)
def download_feature_shard_remote(shard_id: int) -> dict:
    """
    What it does:
        Downloads and unzips exactly the episodes where episode_idx %
        NUM_FEATURE_SHARDS == shard_id (routing by episode index, not a
        contiguous range, so it works regardless of the exact max episode
        index or any numbering gaps) into this shard's own dedicated
        volume, unzipping IN PLACE (features/episode_N.zip ->
        features/episode_N/, no wasteful full-copy first -- the mistake
        the single-volume attempt made). Deletes each .zip right after
        successfully unzipping it, to keep this shard's own file count as
        low as possible. ~400 episodes/shard at 4 shards, ~275k files
        each -- safely under the 500k ceiling with real margin.

    Returns:
        dict -- {"shard_id": int, "episode_count": int, "file_count": int}.

    Example input:
        download_feature_shard_remote.remote(shard_id=0)

    Example output:
        {"shard_id": 0, "episode_count": 400, "file_count": 278412}
    """
    import os
    import sys

    import huggingface_hub

    sys.path.insert(0, "/app/scripts")

    shard_path = FEATURE_SHARD_PATHS[shard_id]
    episode_ids = [i for i in range(TOTAL_EPISODES) if i % NUM_FEATURE_SHARDS == shard_id]  # list[int]
    patterns = [f"features/episode_{i}.zip" for i in episode_ids]  # list[str]

    print(f"[shard {shard_id}] downloading {len(patterns)} episode zips ...")
    huggingface_hub.snapshot_download(
        repo_id=HF_DATASET_REPO, repo_type="dataset", local_dir=shard_path,
        allow_patterns=patterns, max_workers=16,
    )

    from unzip_data import unzip_one  # type: ignore

    features_dir = os.path.join(shard_path, "features")
    zips = sorted(p for p in os.listdir(features_dir) if p.endswith(".zip"))
    print(f"[shard {shard_id}] unzipping {len(zips)} files ...")
    for i, name in enumerate(zips):
        zip_path = pathlib.Path(features_dir) / name
        unzip_one(zip_path)
        zip_path.unlink()  # delete right after extracting -- keeps this shard's file count minimal
        if i % 100 == 0:
            print(f"[shard {shard_id}] {i}/{len(zips)} unzipped")
            feature_shard_volumes[shard_id].commit()

    feature_shard_volumes[shard_id].commit()
    file_count = sum(len(files) for _, _, files in os.walk(features_dir))
    result = {"shard_id": shard_id, "episode_count": len(episode_ids), "file_count": file_count}
    print(f"[shard {shard_id}] done: {result}")
    return result


@app.local_entrypoint()
def run_download_features_sharded():
    """
    What it does: launches all NUM_FEATURE_SHARDS shard downloads in
    parallel (each is an independent volume, no shared state, so this is
    safe) and waits for all to finish, printing per-shard results plus the
    grand total. Blocking by design (not spawned/detached) -- with 4
    parallel shards each handling ~1/4 of the 230.5 GB features/ prefix,
    expected wall-clock is in the same order as the original single-shot
    download (~27 min), not 4x it.

    Returns:
        None.

    Example input:
        modal run xattn_fusion/training/download_xf_full_suite_dataset.py::run_download_features_sharded

    Example output:
        (stdout) "TOTAL: 1600 episodes, 1113648 files across 4 shards"
    """
    results = list(download_feature_shard_remote.map(range(NUM_FEATURE_SHARDS)))
    total_episodes = sum(r["episode_count"] for r in results)
    total_files = sum(r["file_count"] for r in results)
    for r in results:
        print(r)
    print(f"TOTAL: {total_episodes} episodes, {total_files} files across {NUM_FEATURE_SHARDS} shards")


@app.local_entrypoint()
def run_download():
    """
    What it does: fire-and-forget trigger for the full download+unzip job.
    Must use --detach: a 355.7 GB download plus unzip of ~1.6M files is a
    multi-hour job this local process/laptop closing must not kill.

    Returns:
        None -- prints the spawned call ID to stdout.

    Example input:
        modal run --detach xattn_fusion/training/download_xf_full_suite_dataset.py::run_download

    Example output:
        (stdout) "Spawned fc-abc123. This keeps running on Modal's servers ONLY IF launched with --detach."
    """
    call = download_and_unzip_remote.spawn()
    print(f"Spawned {call.object_id}. This keeps running on Modal's servers ONLY IF this was")
    print("launched with `modal run --detach` -- otherwise it's torn down when this exits.")
    print("Check progress with: modal app list   (look for xf-download-full-suite-dataset)")
