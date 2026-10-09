"""
check_corruption_coverage.py

Read-only, CPU-only Modal check of the general caption corruption on the REAL training data,
before any GPU is spent. Applies the exact training rule (training/caption_corruption.py:
build_windows + corrupt_caption) to every execution step of all mapped episodes' timelines and
reports, per task, the share of steps that would be shown a held-shift caption, no caption, or
left unchanged because a target-naming caption is protected. Training samples are drawn per
step, so these step-weighted shares are what training will actually see.

Added 2026-10-07 after the first "general" run's own logs showed 36-91% corruption on most tasks
instead of ~20% (windows were not scaled to short episodes) -- a bug a toy-episode unit test had
not caught.

Run (training account, which owns the xf-* volumes):
    modal run hybrid_prompt_modul/diagnostics/check_corruption_coverage.py
"""

import json
import os
import pathlib

import modal

HYBRID_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)  # str
YAML_NAME = "hybrid-groundsg-prompt-framesamp-modul-general.yaml"  # str, the config whose caption_corruption block is checked

app = modal.App("hybrid-prompt-modul-coverage-check")  # modal.App
image = (  # modal.Image, pure-Python check plus omegaconf for the yaml
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("omegaconf")
    .add_local_dir(HYBRID_LOCAL_DIR, remote_path="/hyb_root/hybrid_prompt_modul", copy=True)
)
main_data_volume = modal.Volume.from_name("xf-full-suite-data")  # modal.Volume
shard_volumes = [modal.Volume.from_name(f"xf-features-shard-{i}") for i in range(4)]  # list[modal.Volume]
MOUNTS = {"/xf_data": main_data_volume, **{f"/xf_features_shard_{i}": v for i, v in enumerate(shard_volumes)}}  # dict[str, modal.Volume]


@app.function(image=image, volumes=MOUNTS, timeout=1800, gpu=None)
def check_coverage() -> dict:
    """
    What it does:
        Loads every mapped episode's subgoal_table.json (execution intervals
        only), builds its corruption windows with the training config, and
        tallies per task how many execution steps end up held_shift /
        dropout / protected / none.

    Returns:
        dict -- {task: {"steps": int, "held_shift": float, "dropout": float, "protected": float, "corrupted": float}}.

    Example input:
        check_coverage.remote()

    Example output:
        {"SwingXtimes": {"steps": 41230, "held_shift": 0.09, "dropout": 0.10, "protected": 0.0, "corrupted": 0.19}, ...}
    """
    import sys

    sys.path.insert(0, "/hyb_root")
    import omegaconf

    from hybrid_prompt_modul.training.caption_corruption import build_windows, corrupt_caption, corruption_config_from_yaml

    cfg = corruption_config_from_yaml(
        omegaconf.OmegaConf.load(f"/hyb_root/hybrid_prompt_modul/config/{YAML_NAME}").get("caption_corruption")
    )  # CorruptionConfig
    print(f"[coverage] {cfg}")
    with open("/xf_data/episode_mapping.json") as f:
        mapping = {int(k): v for k, v in json.load(f).items()}  # dict[int, list[str]]
    tally = {}  # dict[str, dict[str, int]]
    for epis_idx, (h5_name, _local) in mapping.items():  # int, str, str
        path = f"/xf_features_shard_{epis_idx % 4}/features/episode_{epis_idx}/subgoal_table.json"  # str
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            intervals = [iv for iv in json.load(f)["intervals"] if not iv["is_demo"]]  # list[dict]
        if not intervals:
            continue
        task = h5_name.replace("record_dataset_", "").replace(".h5", "")  # str
        windows = build_windows(epis_idx, intervals, cfg)  # list[dict]
        counts = tally.setdefault(task, {"steps": 0, "held_shift": 0, "dropout": 0, "protected": 0, "none": 0})  # dict[str, int]
        for iv in intervals:  # dict
            for step in range(iv["start"], iv["end"]):  # int
                _shown, outcome = corrupt_caption(intervals, windows, step, iv["caption"])  # str, str
                counts[outcome] += 1
                counts["steps"] += 1
    report = {}  # dict[str, dict]
    for task in sorted(tally):  # str
        c = tally[task]  # dict[str, int]
        n = max(c["steps"], 1)  # int
        report[task] = {"steps": c["steps"], "held_shift": round(c["held_shift"] / n, 3), "dropout": round(c["dropout"] / n, 3),
                        "protected": round(c["protected"] / n, 3), "corrupted": round((c["held_shift"] + c["dropout"]) / n, 3)}
        r = report[task]  # dict
        print(f"[coverage] {task:18s} steps {r['steps']:>7}  corrupted {r['corrupted']:.1%}  (held {r['held_shift']:.1%}, "
              f"dropout {r['dropout']:.1%})  protected {r['protected']:.1%}")
    return report


@app.local_entrypoint()
def main():
    """CLI: runs the coverage check and prints the per-task table."""
    check_coverage.remote()
