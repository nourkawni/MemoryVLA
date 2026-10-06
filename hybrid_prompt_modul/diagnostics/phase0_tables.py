"""
phase0_tables.py

Phase 0 of the timing-shift fix plan (read-only, CPU-only Modal check). Answers, from the real
training data, the questions the caption-shift augmentation depends on:

  0.1  Does XF's per-episode subgoal_table.json give, at a sample's step, the SAME caption the
       training sample carries (grounded_subgoal / grounded_subgoal_online)? Reported as exact,
       case-insensitive, and coordinate-stripped agreement over ~400 random samples. Also counts
       how many mapped episodes have a table.
  0.2  For each of the 16 tasks: an example episode's ordered caption sequence (video-demo
       interval excluded), the terminal caption, events per episode, and the share of execution
       steps covered by the terminal caption.
  fake-terminal ratio  Simulates the plan's shift rates over every execution step of up to 20
       episodes per task and reports, among samples whose SHOWN caption is the terminal one, the
       fraction that would be fake (true caption earlier). Run for terminal-jump shares 0.25
       (plan) and 0.40 (suggested). If most terminal captions are fake, the model would learn to
       ignore the terminal caption entirely -- the rates must then be capped.

Nothing is written to any volume. The xf-* volumes are only read.

Run (training account, which owns the xf-* volumes):
    modal run hybrid_prompt_modul/diagnostics/phase0_tables.py
"""

import json
import os
import random
import re

import modal

app = modal.App("hybrid-prompt-modul-phase0")  # modal.App

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")  # modal.Image, numpy needed to unpickle samples
main_data_volume = modal.Volume.from_name("xf-full-suite-data")  # modal.Volume
shard_volumes = [modal.Volume.from_name(f"xf-features-shard-{i}") for i in range(4)]  # list[modal.Volume]
MOUNTS = {"/xf_data": main_data_volume, **{f"/xf_features_shard_{i}": v for i, v in enumerate(shard_volumes)}}  # dict[str, modal.Volume]

DATA_DIR = "/xf_data/preprocessed/data"  # str
MAPPING_PATH = "/xf_data/episode_mapping.json"  # str

COUNTING_TASKS = {"SwingXtimes", "PickXtimes", "BinFill", "StopCube"}  # set[str]
UNMASK_TASKS = {"VideoUnmask", "VideoUnmaskSwap", "ButtonUnmask", "ButtonUnmaskSwap"}  # set[str]


def table_path(epis_idx: int) -> str:
    """
    What it does: subgoal_table.json path for an episode (same epis_idx % 4 shard routing XF used).

    Returns:
        str -- absolute path.

    Example input:
        table_path(802)

    Example output:
        "/xf_features_shard_2/features/episode_802/subgoal_table.json"
    """
    return f"/xf_features_shard_{epis_idx % 4}/features/episode_{epis_idx}/subgoal_table.json"


def strip_coords(caption: str) -> str:
    """
    What it does: removes "at <y, x>" coordinates and normalizes case/whitespace.

    Returns:
        str -- coordinate-free lowercase caption.

    Example input:
        strip_coords("Pick up the red cube at <128, 64>")

    Example output:
        "pick up the red cube"
    """
    return re.sub(r"\s*at <\d+, \d+>", "", caption or "").strip().lower()


def task_of(h5_filename: str) -> str:
    """
    What it does: task name from the raw H5 filename recorded in episode_mapping.json.

    Returns:
        str -- task name.

    Example input:
        task_of("record_dataset_BinFill.h5")

    Example output:
        "BinFill"
    """
    return h5_filename.replace("record_dataset_", "").replace(".h5", "")


def exec_intervals(table: dict) -> list[dict]:
    """
    What it does: the table's intervals excluding the video-demo prefix.

    Returns:
        list[dict] -- intervals with is_demo False, in time order.

    Example input:
        exec_intervals({"intervals": [{"is_demo": True, ...}, {"is_demo": False, ...}]})

    Example output:
        [{"is_demo": False, ...}]
    """
    return [iv for iv in table["intervals"] if not iv["is_demo"]]


def simulate_shift(task: str, ivs: list[dict], k: int, rng: random.Random, terminal_share: float) -> int:
    """
    What it does:
        Applies the plan's shift rule to one sample whose true caption is
        execution interval k, and returns which interval's caption is SHOWN.
        Rates: counting tasks p=0.40 (doubled in the last 2 events), unmask
        tasks p=0.10 (first execution caption never shifted), others 0.15.
        Kinds: early+1 0.35, early+k 0.25 (k in 2..remaining), terminal jump
        `terminal_share` (counting only, needs >=2 remaining, else early+k),
        late-1 the rest. An impossible kind means no shift.

    Returns:
        int -- index of the shown interval.

    Example input:
        simulate_shift("SwingXtimes", ivs, 2, random.Random(0), 0.25)

    Example output:
        6
    """
    n = len(ivs)  # int
    remaining = n - 1 - k  # int, events after the true one
    if task in COUNTING_TASKS:
        p = 0.40 * (2 if k >= n - 2 else 1)  # float
    elif task in UNMASK_TASKS:
        if k == 0:
            return k
        p = 0.10  # float
    else:
        p = 0.15  # float
    if rng.random() >= min(p, 1.0):
        return k
    late_share = 1.0 - 0.35 - 0.25 - terminal_share  # float
    r = rng.random()  # float
    if r < 0.35:
        return k + 1 if remaining >= 1 else k
    if r < 0.60:
        return k + rng.randint(2, remaining) if remaining >= 2 else k
    if r < 0.60 + terminal_share:
        if task in COUNTING_TASKS and remaining >= 2:
            return n - 1
        return k + rng.randint(2, remaining) if remaining >= 2 else k
    assert late_share > 0
    return k - 1 if k >= 1 else k


@app.function(image=image, volumes=MOUNTS, timeout=3600, gpu=None)
def run_phase0(num_samples: int = 400, episodes_per_task: int = 20, seed: int = 0) -> dict:
    """
    What it does:
        Runs checks 0.1, 0.2 and the fake-terminal simulation (see module
        docstring) and prints a report.

    Returns:
        dict -- {"tables_found": int, "mapped": int, "agreement": dict, "tasks": dict}.

    Example input:
        run_phase0.remote()

    Example output:
        {"tables_found": 1307, "mapped": 1307, "agreement": {"exact_online": 0.99, ...}, "tasks": {...}}
    """
    import pickle

    import numpy as np

    rng = random.Random(seed)  # random.Random
    with open(MAPPING_PATH) as f:
        mapping = {int(k): v for k, v in json.load(f).items()}  # dict[int, list[str]]
    tables = {}  # dict[int, dict]
    for epis_idx in mapping:  # int
        path = table_path(epis_idx)  # str
        if os.path.isfile(path):
            with open(path) as f:
                tables[epis_idx] = json.load(f)
    print(f"[0.1] mapped episodes: {len(mapping)}; with subgoal_table.json: {len(tables)}")

    # ---- 0.1 caption agreement on random real samples ----
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pkl")]  # list[str]
    print(f"[0.1] data/*.pkl files: {len(files)}")
    counts = {"checked": 0, "no_table": 0, "no_interval": 0, "exact_online": 0, "exact_recorded": 0,
              "lower_online": 0, "text_only_online": 0}  # dict[str, int]
    mismatches = []  # list[tuple]
    for name in rng.sample(files, min(num_samples, len(files))):  # str
        with open(os.path.join(DATA_DIR, name), "rb") as f:
            d = pickle.load(f)  # dict
        # stored as 1-element arrays; the released dataset reads them with .item()
        epis_idx, step_idx = int(np.asarray(d["epis_idx"]).reshape(-1)[0]), int(np.asarray(d["step_idx"]).reshape(-1)[0])  # int, int
        if epis_idx not in tables:
            counts["no_table"] += 1
            continue
        iv = next((iv for iv in tables[epis_idx]["intervals"] if iv["start"] <= step_idx < iv["end"]), None)  # dict | None
        if iv is None:
            counts["no_interval"] += 1
            continue
        counts["checked"] += 1
        online, recorded, cap = d["grounded_subgoal_online"], d["grounded_subgoal"], iv["caption"]  # str, str, str
        counts["exact_online"] += cap == online
        counts["exact_recorded"] += cap == recorded
        counts["lower_online"] += cap.strip().lower() == online.strip().lower()
        counts["text_only_online"] += strip_coords(cap) == strip_coords(online)
        if cap.strip().lower() != online.strip().lower() and len(mismatches) < 8:
            mismatches.append((epis_idx, step_idx, cap, online, recorded))
    c = max(counts["checked"], 1)  # int
    agreement = {k: round(counts[k] / c, 4) for k in ("exact_online", "exact_recorded", "lower_online", "text_only_online")}  # dict
    print(f"[0.1] counts {counts}")
    print(f"[0.1] agreement (of {counts['checked']} checked): {agreement}")
    for m in mismatches:  # tuple
        print(f"[0.1] mismatch ep{m[0]} step{m[1]}: table='{m[2]}' | online='{m[3]}' | recorded='{m[4]}'")

    # ---- 0.2 per-task sequences + fake-terminal simulation ----
    by_task = {}  # dict[str, list[int]]
    for epis_idx in tables:  # int
        by_task.setdefault(task_of(mapping[epis_idx][0]), []).append(epis_idx)
    report = {}  # dict[str, dict]
    for task in sorted(by_task):  # str
        eps = sorted(by_task[task])[:episodes_per_task]  # list[int]
        n_events, term_share, term_texts = [], [], {}  # list[int], list[float], dict[str, int]
        sims = {0.25: [0, 0], 0.40: [0, 0]}  # dict[float, list[int]], share -> [fake, total] shown-terminal
        for epis_idx in eps:  # int
            ivs = exec_intervals(tables[epis_idx])  # list[dict]
            if not ivs:
                continue
            n_events.append(len(ivs))
            exec_len = ivs[-1]["end"] - ivs[0]["start"]  # int
            term_share.append((ivs[-1]["end"] - ivs[-1]["start"]) / max(exec_len, 1))
            t = strip_coords(ivs[-1]["caption"])  # str
            term_texts[t] = term_texts.get(t, 0) + 1
            for share in sims:  # float
                for k, iv in enumerate(ivs):  # int, dict
                    for _ in range(iv["end"] - iv["start"]):  # one draw per execution step
                        shown = simulate_shift(task, ivs, k, rng, share)  # int
                        if shown == len(ivs) - 1:
                            sims[share][1] += 1
                            sims[share][0] += shown != k
        example = [strip_coords(iv["caption"]) for iv in exec_intervals(tables[eps[0]])]  # list[str]
        fake = {share: round(f / t, 3) if t else None for share, (f, t) in sims.items()}  # dict[float, float]
        report[task] = {"episodes": len(by_task[task]), "events_min": min(n_events), "events_max": max(n_events),
                        "terminal_captions": term_texts,
                        "terminal_step_share_mean": round(sum(term_share) / len(term_share), 3),
                        "fake_terminal_frac": fake, "example_sequence": example}
        print(f"\n[0.2] {task}: {len(by_task[task])} episodes, events per episode {min(n_events)}-{max(n_events)}, "
              f"terminal caption covers {report[task]['terminal_step_share_mean']:.1%} of execution steps")
        print(f"[0.2]   terminal captions: {term_texts}")
        print(f"[0.2]   example sequence: {' -> '.join(example)}")
        print(f"[sim]   fake share of SHOWN-terminal samples: terminal-jump 0.25 -> {fake[0.25]}, 0.40 -> {fake[0.40]}")
    return {"tables_found": len(tables), "mapped": len(mapping), "agreement": agreement, "counts": counts, "tasks": report}


@app.local_entrypoint()
def main(num_samples: int = 400, episodes_per_task: int = 20):
    """
    What it does: runs run_phase0 remotely and saves the returned report locally as JSON.

    Returns:
        None.

    Example input:
        modal run hybrid_prompt_modul/diagnostics/phase0_tables.py

    Example output:
        (stdout) report + "saved hybrid_prompt_modul/diagnostics/phase0_report.json"
    """
    report = run_phase0.remote(num_samples=num_samples, episodes_per_task=episodes_per_task)  # dict
    out = "hybrid_prompt_modul/diagnostics/phase0_report.json"  # str
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"saved {out}")
