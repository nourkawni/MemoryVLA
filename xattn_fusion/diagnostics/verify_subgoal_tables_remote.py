"""
verify_subgoal_tables_remote.py

Ground-truth check for the episode_33 subgoal_table.json mystery (2026-09-20:
run_training crashed twice on this exact file; a local-machine investigation
gave contradictory answers across runs and directly caught real local DNS/
network failures during testing, casting doubt on every local-machine-based
check made that day -- see RESEARCH_LOG.md's 2026-09-20 entries and
project_xf_xattn_fusion_arm.md's "UPDATE" section for the full history).

This script runs ENTIRELY inside a Modal container (a real remote function,
not a local script), eliminating the local-network confound that undermined
yesterday's local checks. It cross-validates TWO independent access methods
for every one of the 1,307 real, mapped episodes, both executed from within
the SAME remote container:
  1. pathlib.Path(...).exists() through the container's mounted volume (the
     same access method XFDataset._load_subgoal_table itself uses).
  2. modal.Volume.listdir() called directly on the volume object (the same
     underlying API the `modal volume ls` CLI command uses).
If these two agree for every episode, that's strong, trustworthy evidence
of the real state (mount-based reads are being fairly compared against the
direct API for the first time without a local-network variable in the way).
If they disagree anywhere, that disagreement is real and worth chasing
further, not an artifact of local network flakiness.

Run with:
    modal run xattn_fusion/diagnostics/verify_subgoal_tables_remote.py
"""

import pathlib

import modal

XF_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
ROBOMME_SRC_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning" / "src")

app = modal.App("xf-verify-subgoal-tables-remote")

main_data_volume = modal.Volume.from_name("xf-full-suite-data")
feature_shard_volumes = [modal.Volume.from_name(f"xf-features-shard-{i}") for i in range(4)]

MAIN_DATA_VOLUME_PATH = "/xf_data"
FEATURE_SHARD_PATHS = [f"/xf_features_shard_{i}" for i in range(4)]

volumes = {
    MAIN_DATA_VOLUME_PATH: main_data_volume,
    **{path: vol for path, vol in zip(FEATURE_SHARD_PATHS, feature_shard_volumes)},
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "h5py", "omegaconf", "flax", "jax[cpu]", "einops", "sentencepiece")
    .add_local_dir(ROBOMME_SRC_DIR, remote_path="/app_src", copy=True)
    .add_local_dir(XF_LOCAL_DIR, remote_path="/xf_root/xattn_fusion", copy=True)
)


@app.function(image=image, gpu=None, timeout=3600, volumes=volumes)
def verify() -> dict:
    """
    What it does:
        For every one of the 1,307 real, mapped episodes, checks
        subgoal_table.json's presence TWO independent ways, both from
        inside this same remote container: (1) Path.exists() through the
        mounted volume, (2) modal.Volume.listdir() called directly on the
        matching shard's Volume object. Reports full agreement/disagreement
        detail, not just a pass/fail count.

    Returns:
        dict -- {"total": int, "mount_missing": list[int],
        "listdir_missing": list[int], "agree": bool,
        "disagreements": list[int]}.

    Example input:
        verify.remote()

    Example output:
        {"total": 1307, "mount_missing": [], "listdir_missing": [],
         "agree": True, "disagreements": []}
    """
    import sys
    sys.path.insert(0, "/app_src")
    sys.path.insert(0, "/xf_root")
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    from xattn_fusion.mme_vla_suite.dataset_builder.xf_subgoal_table_builder import load_episode_mapping
    from xattn_fusion.mme_vla_suite.shared.feature_shard_router import feature_episode_dir

    mapping = load_episode_mapping(f"{MAIN_DATA_VOLUME_PATH}/episode_mapping.json")
    all_ids = sorted(mapping.keys())  # list[int]

    shard_volumes = {i: vol for i, vol in enumerate(feature_shard_volumes)}  # dict[int, modal.Volume]

    mount_missing = []  # list[int]
    listdir_missing = []  # list[int]
    lock = threading.Lock()  # threading.Lock, guards the two lists above across worker threads
    progress_count = [0]  # list[int], single-element mutable counter (closures can't rebind an int)

    def check_one(epis_idx: int) -> None:
        shard = epis_idx % 4  # int, must match feature_episode_dir's own routing rule

        # Method 1: mount-based Path.exists() -- the same access XFDataset itself uses.
        mount_path = Path(feature_episode_dir(epis_idx)) / "subgoal_table.json"
        mount_ok = mount_path.exists()  # bool

        # Method 2: direct SDK listdir() -- same underlying API `modal volume ls` uses.
        # listdir_error tracks the exact exception (type+message) separately from a genuine
        # "file not in this directory's listing" result -- found necessary after a first version
        # of this script bare-`except Exception: listdir_ok = False`'d here, which could have
        # been silently misclassifying real API errors (rate limits, transient failures) as
        # false "missing" results rather than genuine absence.
        listdir_error = None  # str | None
        try:
            entries = list(shard_volumes[shard].listdir(f"features/episode_{epis_idx}"))
            names = {e.path.split("/")[-1] for e in entries}
            listdir_ok = "subgoal_table.json" in names  # bool
        except Exception as e:
            listdir_ok = False
            listdir_error = f"{type(e).__name__}: {e}"

        with lock:
            if not mount_ok:
                mount_missing.append(epis_idx)
            if not listdir_ok:
                listdir_missing.append((epis_idx, listdir_error))
            progress_count[0] += 1
            if progress_count[0] % 200 == 0:
                print(f"...checked {progress_count[0]}/{len(all_ids)}")

    # I/O-bound network calls, not CPU work -- a thread pool cuts this from ~30min sequential
    # to a couple minutes (same pattern xf_dataset.py's _gather_history_feat already uses).
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(check_one, epis_idx) for epis_idx in all_ids]
        for future in as_completed(futures):
            future.result()  # re-raises any worker exception instead of silently swallowing it

    listdir_missing_ids = {epis_idx for epis_idx, _ in listdir_missing}  # set[int]
    listdir_error_count = sum(1 for _, err in listdir_missing if err is not None)  # int, real API errors, not genuine "not found"
    listdir_genuine_not_found = sorted(epis_idx for epis_idx, err in listdir_missing if err is None)  # list[int]
    disagreements = sorted(set(mount_missing) ^ listdir_missing_ids)  # list[int], symmetric difference
    return {
        "total": len(all_ids),
        "mount_missing": mount_missing,
        "listdir_missing": listdir_missing,
        "listdir_error_count": listdir_error_count,
        "listdir_genuine_not_found": listdir_genuine_not_found,
        "agree": len(disagreements) == 0,
        "disagreements": disagreements,
    }


@app.local_entrypoint()
def main():
    """
    What it does: runs verify() remotely and prints a clear summary of
    whether the two independent methods agree, and any real gaps found.

    Returns:
        None.

    Example input:
        modal run xattn_fusion/diagnostics/verify_subgoal_tables_remote.py

    Example output:
        (stdout) "AGREE: both methods found 0 missing out of 1307 episodes."
    """
    result = verify.remote()
    print(f"\ntotal={result['total']}")
    print(f"mount (Path.exists()) missing: {result['mount_missing']}")
    print(f"listdir API errors (NOT genuine 'not found', see type/message): {result['listdir_error_count']}")
    print(f"listdir GENUINE 'not in directory listing' results: {result['listdir_genuine_not_found']}")
    if result["listdir_error_count"] > 0:
        sample_errors = [f"epis_idx={i}: {e}" for i, e in result["listdir_missing"] if e is not None][:10]
        print(f"sample listdir errors (first 10): {sample_errors}")
    if result["mount_missing"] == [] and result["listdir_genuine_not_found"] == []:
        print(f"\nAGREE: both methods found 0 GENUINE missing files out of {result['total']} episodes -- no real gap. "
              f"({result['listdir_error_count']} listdir() calls hit real API errors, not missing-file results.)")
    else:
        print(f"\nREAL DISCREPANCY -- mount_missing={result['mount_missing']}, "
              f"listdir_genuine_not_found={result['listdir_genuine_not_found']}")
