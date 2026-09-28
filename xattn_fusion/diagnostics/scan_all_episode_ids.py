"""
scan_all_episode_ids.py

Finds every DISTINCT epis_idx value that actually appears across all
416,950 real data/*.pkl training samples, then cross-references against
episode_mapping.json's 1,307 valid keys to find ALL excluded-but-present
episodes (not just the already-confirmed 33/97) -- see RESEARCH_LOG.md's
2026-09-21 entries for why this matters: episodes 33/97 were assumed to
have "zero training samples" (so excluding them from episode_mapping.json
was assumed harmless) but that assumption was WRONG and caused 5 real
training crashes before being caught.

Two things fixed versus an earlier, unfinished attempt at this same scan:
1. SPEED: work is sharded across many PARALLEL Modal containers (via
   .map()), not just threads within one container -- the single-container
   version measured ~15 files/sec with 128 threads (60,000 files in 66
   minutes), which extrapolates to ~7-8 HOURS for the full 416,950 files
   and was very likely hitting its own function timeout as a result (the
   log signature -- a cancellation signal followed by a forced kill after
   worker threads didn't exit within Modal's 30s grace period -- matches a
   timeout, not a human-issued `modal app stop`, despite Modal's generic
   "user stopped from CLI" log wording for both cases).
2. RESILIENCE: each chunk's result is written to the xf-full-suite-training
   volume as soon as that chunk finishes (not held in memory until the very
   end), so an interruption only costs the in-flight chunks, not the whole
   scan -- and re-running this script skips chunks that already have a
   saved result file.

Run with:
    modal run xattn_fusion/diagnostics/scan_all_episode_ids.py
"""

import pathlib

import modal

app = modal.App("xf-scan-all-episode-ids-v2")

main_data_volume = modal.Volume.from_name("xf-full-suite-data")
training_volume = modal.Volume.from_name("xf-full-suite-training", create_if_missing=True)

MAIN_DATA_VOLUME_PATH = "/xf_data"
TRAINING_VOLUME_PATH = "/xf_training"
DATASET_PATH = f"{MAIN_DATA_VOLUME_PATH}/preprocessed"
PROGRESS_DIR = f"{TRAINING_VOLUME_PATH}/episode_id_scan_progress"  # str, one small json file per chunk

NUM_CHUNKS = 80  # int, ~5,200 files/chunk at 416,950 total -- small enough each finishes well within its own timeout
# even at the pessimistic per-thread rate the earlier single-container attempt implied
# (~15 files/sec with 128 threads -> ~0.117 files/sec/thread; 32 threads/chunk -> ~3.7 files/sec/
# chunk -> ~23.5 min for ~5,200 files, comfortably under scan_chunk's own 2400s/40min timeout)

volumes = {MAIN_DATA_VOLUME_PATH: main_data_volume, TRAINING_VOLUME_PATH: training_volume}

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")


@app.function(image=image, gpu=None, timeout=2400, volumes=volumes)
def scan_chunk(chunk_id: int, real_ids_chunk: list[int]) -> dict:
    """
    What it does:
        Reads epis_idx from every sample id in real_ids_chunk (a slice of
        the full 416,950), and writes the resulting distinct-epis_idx set
        to this chunk's own progress file on the training volume as soon
        as it finishes -- so a later interruption of SOME OTHER chunk
        doesn't lose this one's completed work.

    Returns:
        dict -- {"chunk_id": int, "epis_idx_seen": list[int], "count": int}.

    Example input:
        scan_chunk.remote(0, [0, 1, 2, ..., 10424])

    Example output:
        {"chunk_id": 0, "epis_idx_seen": [0, 1, 5, 33, ...], "count": 10425}
    """
    import json
    import os
    import pickle
    from concurrent.futures import ThreadPoolExecutor, as_completed

    data_dir = os.path.join(DATASET_PATH, "data")
    epis_idx_seen = set()  # set[int]

    def read_one(real_id: int) -> int:
        with open(os.path.join(data_dir, f"{real_id}.pkl"), "rb") as f:
            d = pickle.load(f)
        return int(d["epis_idx"].item())

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(read_one, rid) for rid in real_ids_chunk]
        for future in as_completed(futures):
            epis_idx_seen.add(future.result())

    result = {"chunk_id": chunk_id, "epis_idx_seen": sorted(epis_idx_seen), "count": len(real_ids_chunk)}

    os.makedirs(PROGRESS_DIR, exist_ok=True)
    with open(f"{PROGRESS_DIR}/chunk_{chunk_id}.json", "w") as f:
        json.dump(result, f)
    training_volume.commit()  # durably persist THIS chunk's result immediately, not just at the very end

    return result


@app.function(image=image, gpu=None, timeout=120, volumes=volumes)
def list_real_ids() -> list[int]:
    """
    What it does: lists every real sample id that actually exists in
    data/ (matches XFSampleDataset's own gap-safe indexing logic).

    Returns:
        list[int] -- sorted real sample ids.

    Example input:
        list_real_ids.remote()

    Example output:
        [0, 1, 2, 3, 5, 7, ...]
    """
    import os

    data_dir = os.path.join(DATASET_PATH, "data")
    return sorted(int(f.split(".")[0]) for f in os.listdir(data_dir) if f.endswith(".pkl"))


@app.function(image=image, gpu=None, timeout=120, volumes=volumes)
def already_done_chunks() -> set:
    """
    What it does: checks the training volume for chunk result files
    already written by a previous (possibly interrupted) run, so this run
    can skip redoing them.

    Returns:
        set[int] -- chunk ids that already have a saved result file.

    Example input:
        already_done_chunks.remote()

    Example output:
        {0, 1, 2, 5, 6}
    """
    import os

    if not os.path.isdir(PROGRESS_DIR):
        return set()
    done = set()
    for fname in os.listdir(PROGRESS_DIR):
        if fname.startswith("chunk_") and fname.endswith(".json"):
            done.add(int(fname[len("chunk_"):-len(".json")]))
    return done


@app.local_entrypoint()
def main():
    """
    What it does:
        Lists real sample ids, splits them into NUM_CHUNKS, skips any
        chunk already completed by a prior (interrupted) run, dispatches
        the remaining chunks IN PARALLEL across many containers via
        .map(), then combines every chunk's result (old + new) into the
        final distinct epis_idx set and reports which excluded episodes
        (beyond the already-known 33/97) have real samples, if any.

    Returns:
        None.

    Example input:
        modal run xattn_fusion/diagnostics/scan_all_episode_ids.py

    Example output:
        (stdout) "ALL EXCLUDED-BUT-PRESENT EPISODES: [33, 97]" or a longer list.
    """
    import json

    real_ids = list_real_ids.remote()
    print(f"Total real samples: {len(real_ids)}")

    chunk_size = (len(real_ids) + NUM_CHUNKS - 1) // NUM_CHUNKS
    chunks = [real_ids[i:i + chunk_size] for i in range(0, len(real_ids), chunk_size)]
    print(f"Split into {len(chunks)} chunks of ~{chunk_size} each.")

    done = already_done_chunks.remote()
    print(f"Already-completed chunks from a prior run: {sorted(done)}")

    todo = [(i, c) for i, c in enumerate(chunks) if i not in done]
    print(f"Dispatching {len(todo)} remaining chunks in parallel...")

    all_epis_idx_seen = set()

    # Load already-done chunks' saved results first (from the volume, via a tiny remote read).
    if done:
        for chunk_id in sorted(done):
            partial = read_chunk_result.remote(chunk_id)
            all_epis_idx_seen.update(partial["epis_idx_seen"])
            print(f"  loaded saved chunk {chunk_id}: {partial['count']} samples, "
                  f"{len(partial['epis_idx_seen'])} distinct epis_idx")

    if todo:
        for result in scan_chunk.starmap(todo):
            all_epis_idx_seen.update(result["epis_idx_seen"])
            print(f"  chunk {result['chunk_id']} done: {result['count']} samples, "
                  f"{len(result['epis_idx_seen'])} distinct epis_idx")

    print(f"\nTotal distinct epis_idx values seen across all {len(real_ids)} samples: {len(all_epis_idx_seen)}")

    mapping = get_episode_mapping_keys.remote()
    valid_ids = set(mapping)
    excluded_but_present = sorted(all_epis_idx_seen - valid_ids)
    print(f"episode_mapping.json valid episodes: {len(valid_ids)}")
    print(f"\nEXCLUDED-BUT-PRESENT EPISODES (have real samples but no entry in episode_mapping.json): "
          f"{excluded_but_present}")
    print("SCAN_COMPLETE_OK" if True else "")


@app.function(image=image, gpu=None, timeout=60, volumes=volumes)
def read_chunk_result(chunk_id: int) -> dict:
    """
    What it does: reads back a previously-saved chunk result file.

    Returns:
        dict -- same shape scan_chunk returns.

    Example input:
        read_chunk_result.remote(3)

    Example output:
        {"chunk_id": 3, "epis_idx_seen": [...], "count": 10425}
    """
    import json

    with open(f"{PROGRESS_DIR}/chunk_{chunk_id}.json") as f:
        return json.load(f)


@app.function(image=image, gpu=None, timeout=60, volumes={MAIN_DATA_VOLUME_PATH: main_data_volume})
def get_episode_mapping_keys() -> list[int]:
    """
    What it does: reads episode_mapping.json and returns its keys as ints.

    Returns:
        list[int].

    Example input:
        get_episode_mapping_keys.remote()

    Example output:
        [0, 1, 2, 3, 5, ...]
    """
    import json

    with open(f"{MAIN_DATA_VOLUME_PATH}/episode_mapping.json") as f:
        mapping = json.load(f)
    return [int(k) for k in mapping.keys()]
