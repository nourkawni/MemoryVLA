"""
dump_subgoal_captions.py

Dumps every episode's cached XF subgoal table (the temporal aligner's
(start, end, caption, coords) intervals) together with the episode's task
name, so the coordinate-quantization question can be answered locally:
how much of the grounded (y, x) target location survives EventEncoder's
4x4 spatial cell (event_encoder.py:728), given that the caption TEXT the
model sees is the coordinate-stripped template ("at <bbox>").

Role in the system: read-only diagnostic for the XF coordinate-path
investigation (step 1 of the plan agreed 2026-09-23). CPU-only, no model,
no checkpoint, writes nothing to any volume. Output is written locally to
XF_18k_eval/analysis/subgoal_captions_dump.json for offline analysis by
analyze_coord_quantization.py.

Run with:
    modal run XF_18k_eval/analysis/dump_subgoal_captions.py
"""

import json
import pathlib

import modal

MAIN_DATA_VOLUME_PATH = "/xf_data"  # str, must match launch_xf_training.py
FEATURE_SHARD_PATHS = [f"/xf_features_shard_{i}" for i in range(4)]  # list[str], must match feature_shard_router.py

# pathlib.Path, local output file
OUTPUT_PATH = pathlib.Path(__file__).resolve().parent / "subgoal_captions_dump.json"

app = modal.App("xf-dump-subgoal-captions")  # modal.App

main_data_volume = modal.Volume.from_name("xf-full-suite-data")  # modal.Volume
feature_shard_volumes = [modal.Volume.from_name(f"xf-features-shard-{i}") for i in range(4)]  # list[modal.Volume]

# dict[str, modal.Volume]
volumes = {
    MAIN_DATA_VOLUME_PATH: main_data_volume,
    **{path: vol for path, vol in zip(FEATURE_SHARD_PATHS, feature_shard_volumes)},
}

# modal.Image -- plain JSON reading only, no project code or dependencies needed.
image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=image, gpu=None, cpu=2.0, memory=4096, timeout=1800, volumes=volumes)
def dump_tables() -> dict:
    """
    What it does:
        Loads episode_mapping.json (global_episode_idx -> (h5 filename, h5
        key)), derives each episode's task from the h5 filename, then reads
        that episode's cached subgoal_table.json from its feature shard
        (episode_idx % 4, the feature_shard_router rule) and collects the
        raw intervals.

    Returns:
        dict -- {"episodes": list[dict], "missing": list[int]}; each episode
        dict has keys "episode_idx", "task", "h5_key", "episode_len",
        "exec_start_idx", "intervals" (the table's own JSON interval dicts,
        which include "caption", "template", "coords", "start", "end",
        "is_demo", "is_open").

    Example input:
        dump_tables.remote()

    Example output:
        {"episodes": [{"episode_idx": 0, "task": "PatternLock", "intervals": [...], ...}], "missing": []}
    """
    import os  # module

    with open(f"{MAIN_DATA_VOLUME_PATH}/episode_mapping.json") as f:
        mapping = json.load(f)  # dict[str, list[str]]

    episodes = []  # list[dict]
    missing = []  # list[int]
    for key, (h5_file, h5_key) in sorted(mapping.items(), key=lambda kv: int(kv[0])):
        epis_idx = int(key)  # int
        task = h5_file.replace("record_dataset_", "").replace(".h5", "")  # str
        table_path = os.path.join(
            FEATURE_SHARD_PATHS[epis_idx % 4], "features", f"episode_{epis_idx}", "subgoal_table.json"
        )  # str
        if not os.path.exists(table_path):
            missing.append(epis_idx)
            continue
        with open(table_path) as f:
            payload = json.load(f)  # dict
        episodes.append({
            "episode_idx": epis_idx,
            "task": task,
            "h5_key": h5_key,
            "episode_len": payload.get("episode_len"),
            "exec_start_idx": payload.get("exec_start_idx"),
            "intervals": payload["intervals"],
        })
    print(f"read {len(episodes)} tables, {len(missing)} missing")
    return {"episodes": episodes, "missing": missing}


@app.local_entrypoint()
def main():
    """
    What it does:
        Runs dump_tables remotely (short, blocking -- a few minutes at most)
        and writes the result to OUTPUT_PATH.

    Returns:
        None -- writes a JSON file and prints a one-line summary.

    Example input:
        modal run XF_18k_eval/analysis/dump_subgoal_captions.py

    Example output:
        wrote 1307 episodes (0 missing) to .../subgoal_captions_dump.json
    """
    result = dump_tables.remote()  # dict
    OUTPUT_PATH.write_text(json.dumps(result))
    print(f"wrote {len(result['episodes'])} episodes ({len(result['missing'])} missing) to {OUTPUT_PATH}")
