"""
check_caption_plumbing.py

Answers one question the fusion-gate read could not: when XF trained, did
caption information actually REACH the cross-attention, or was the mechanism
being fed nothing?

Why this matters: `fusion_gate_findings.md` established that all four fusion
gates sit at ~3e-4 at step 18000 and have decayed since step 4000. A gate
pinned at zero looks identical whether (a) captions arrived and training
judged them unhelpful, or (b) captions never arrived, so the cross-attention
had nothing to route and zero was the correct answer. Those call for opposite
responses -- (a) is a real negative result about the mechanism, (b) is a data
bug -- and the plan's remedies (aux heads on F', causal mask, caption
dropout) only make sense for (a).

The mechanism being tested, from build_fusion_mask (fusion_xattn.py): frame
token i may attend to caption token (k, l) only if
static_token_event_idx[i] == k AND event_text_mask[k, l] is True. A null
column is always appended as True, backed by event_encoder's learned
null_token. So if a frame's event index is PAD_EVENT_IDX (-1) or
OVERFLOW_EVENT_IDX (-2), every real key in its row is masked off, softmax
puts 100% of the mass on the null token, and the message that frame receives
is a CONSTANT vector -- identical for every frame, every sample, every batch,
carrying no caption information whatsoever.

What it measures, per real training sample:
  1. Of the frame slots holding a REAL frame (static_mask True), what
     fraction carry an event id >= 0. This is the headline number. A large
     count of -1 is NOT by itself a problem: the memory budget is 512 slots
     and early-in-episode samples legitimately leave most of them empty, so
     the denominator must be real frames, not all slots.
  2. How that splits between -1 (pad) and -2 (overflow -- the frame's true
     event was dropped because the table held more than fusion.max_events).
  3. How many live events each sample has, and how many NON-PAD caption
     tokens each live event carries. An event whose caption tokenized to
     nothing fails the mask's second condition even when the index is right.
  4. How many samples are FULLY degenerate -- zero real frames with a real
     event id -- i.e. samples where fusion provably saw only the null token.

Deliberately reads the dataset BEFORE openpi's transform_dataset, since the
packed event arrays are what the model's fusion path consumes and the
downstream transforms neither create nor repair them. The TrainConfig comes
from launch_xf_training._build_train_config itself rather than being
reconstructed here, so this cannot silently diverge from the config the real
run used.

Role in the system: read-only diagnostic for the XF 18k eval. Writes nothing
to any volume, never edits robomme_policy_learning/, and builds no model --
this is a dataloader-level check, so it needs no GPU and no checkpoint.

Run with:
    modal run XF_18k_eval/analysis/check_caption_plumbing.py
    modal run XF_18k_eval/analysis/check_caption_plumbing.py --num-samples 600
"""

import pathlib

import modal

# str, str -- the two local dirs the container needs: the released policy repo (openpi +
# mme_vla_suite) and this project's XF code (XFDataset and the launcher's config builder).
POLICY_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning")
XF_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "xattn_fusion")

# str -- mount points. These MUST match launch_xf_training.py's own constants exactly:
# feature_shard_router.py hardcodes the shard paths, and XFDataset's feature lookups resolve
# to nothing (silently) if they drift.
CKPT_VOLUME_PATH = "/ckpts"
MAIN_DATA_VOLUME_PATH = "/xf_data"
TRAINING_VOLUME_PATH = "/xf_training"
FEATURE_SHARD_PATHS = [f"/xf_features_shard_{i}" for i in range(4)]

DEFAULT_NUM_SAMPLES = 400  # int, random real training samples to inspect

app = modal.App("xf-check-caption-plumbing")  # modal.App

ckpt_volume = modal.Volume.from_name("robomme-mme-vla-ckpts")  # modal.Volume
main_data_volume = modal.Volume.from_name("xf-full-suite-data")  # modal.Volume
training_volume = modal.Volume.from_name("xf-full-suite-training")  # modal.Volume
feature_shard_volumes = [modal.Volume.from_name(f"xf-features-shard-{i}") for i in range(4)]  # list[modal.Volume]

# dict[str, modal.Volume]. All 4 shards must be mounted simultaneously: XFDataset routes each
# episode's features by episode_idx % 4, so a missing shard breaks a quarter of all episodes.
volumes = {
    CKPT_VOLUME_PATH: ckpt_volume,
    MAIN_DATA_VOLUME_PATH: main_data_volume,
    TRAINING_VOLUME_PATH: training_volume,
    **{path: vol for path, vol in zip(FEATURE_SHARD_PATHS, feature_shard_volumes)},
}

# modal.Image -- identical to launch_xf_training.py's, including the xattn_fusion mount, so
# every layer is a cache hit and the dataset code here is byte-identical to what trained.
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


def _percentiles(values, np):
    """
    What it does:
        Summarizes a list of numbers as a small percentile spread, guarding
        against the empty case so a degenerate result reports as empty
        rather than raising and losing every other number in the run.

    Returns:
        dict -- keys "n", "min", "p25", "p50", "p75", "max", "mean"; all
        values float except "n" (int). Returns {"n": 0} when given nothing.

    Example input:
        _percentiles([3, 5, 5, 9], np)

    Example output:
        {"n": 4, "min": 3.0, "p25": 4.5, "p50": 5.0, "p75": 6.0, "max": 9.0, "mean": 5.5}
    """
    if not len(values):
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)  # np.ndarray
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


@app.function(image=image, gpu=None, timeout=5400, memory=32768, cpu=4.0, volumes=volumes)
def check_plumbing(num_samples: int = DEFAULT_NUM_SAMPLES, seed: int = 0) -> dict:
    """
    What it does:
        Builds a real XFDataset using launch_xf_training's own TrainConfig
        (so this check cannot diverge from the training config), draws
        `num_samples` random sample indices, and tallies, across all of
        them, how many real frame slots carry a real event id versus the
        -1/-2 sentinels, plus per-event caption-token counts.

        Reads the dataset directly rather than through transform_dataset:
        the packed event arrays are what the fusion path consumes, and the
        downstream transforms neither create nor repair them.

    Returns:
        dict -- aggregate counters and percentile summaries; see the
        "Example output" below for the shape. Every count is over real
        frame slots (static_mask True) unless its name says otherwise.

    Example input:
        check_plumbing.remote(num_samples=400, seed=0)

    Example output:
        {"samples_read": 400, "real_frame_slots": 81234, "with_event_idx": 79001,
         "pad_event_idx": 2233, "overflow_event_idx": 0,
         "frac_real_frames_with_event": 0.9725, "fully_degenerate_samples": 3,
         "events_per_sample": {...}, "caption_tokens_per_live_event": {...}}
    """
    import os  # module
    import random  # module
    import sys  # module

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/src")
    sys.path.insert(0, "/xf_root")
    # mme_vla_suite's own config loader resolves paths relative to the process CWD, which Modal
    # defaults to /root (where the driver script lands), not /app where the repo lives.
    os.chdir("/app")

    import numpy as np  # module

    for vol in [main_data_volume, training_volume, *feature_shard_volumes]:
        vol.reload()  # volumes are not live-synced into an already-running container

    from xattn_fusion.mme_vla_suite.models.config.xf_config_utils import get_xf_history_config
    from xattn_fusion.mme_vla_suite.training.xf_dataset import XFDataset
    from xattn_fusion.training.launch_xf_training import _build_train_config

    # Same call the real run makes. resum_ckpt_id is irrelevant to the data pipeline (it only
    # flips overwrite/resume), but is passed as 18000 so the config matches the trained state.
    train_config = _build_train_config(num_train_steps=40_000, batch_size=4, resum_ckpt_id=18000)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)  # DataConfig
    history_config = get_xf_history_config(train_config.model.history_config)  # omegaconf.DictConfig

    print("Constructing XFDataset (this eagerly preloads every episode's subgoal table)...")
    dataset = XFDataset(
        dataset_path=train_config.dataset_path,
        data_config=data_config,
        history_config=history_config,
        action_horizon=train_config.model.action_horizon,
    )
    print(f"XFDataset built. len(dataset) = {len(dataset)}")
    print(f"fusion.max_events = {dataset.fusion_cfg.max_events}, caption_len = {dataset.fusion_cfg.caption_len}")

    rng = random.Random(seed)  # random.Random
    indices = [rng.randrange(len(dataset)) for _ in range(num_samples)]  # list[int]

    real_frame_slots = 0  # int, frame slots with static_mask True
    with_event_idx = 0  # int, of those, ones carrying an event id >= 0
    pad_event_idx = 0  # int, of those, ones carrying PAD_EVENT_IDX (-1)
    overflow_event_idx = 0  # int, of those, ones carrying OVERFLOW_EVENT_IDX (-2)
    total_slots = 0  # int, ALL frame slots including empty padding (context only)
    fully_degenerate_samples = 0  # int, samples where no real frame had a real event id
    samples_read = 0  # int
    events_per_sample = []  # list[int], live events (event_mask True) per sample
    distinct_events_attended = []  # list[int], distinct event ids the real frames point at
    caption_tokens_per_live_event = []  # list[int], non-pad caption tokens per live event
    empty_caption_events = 0  # int, live events whose caption tokenized to zero real tokens
    live_events_seen = 0  # int

    for n, idx in enumerate(indices):
        try:
            data = dataset[idx]  # dict
        except Exception as err:  # noqa: BLE001 -- one bad sample must not lose the whole run
            print(f"[sample {n} idx={idx}] read failed: {err!r}")
            continue

        if samples_read == 0:
            print(f"[first sample] keys available: {sorted(data.keys())}")

        ev_idx = np.asarray(data["static_token_event_idx"]).reshape(-1)  # int32[f]
        static_mask = np.asarray(data["static_mask"]).reshape(-1).astype(bool)  # bool[f]
        event_mask = np.asarray(data["event_mask"]).reshape(-1).astype(bool)  # bool[E]
        text_mask = np.asarray(data["event_text_mask"])  # bool[E, L_c]
        text_mask = text_mask.reshape(event_mask.shape[0], -1).astype(bool)

        if static_mask.shape != ev_idx.shape:
            print(
                f"[sample {n} idx={idx}] SHAPE MISMATCH static_mask{static_mask.shape} "
                f"vs static_token_event_idx{ev_idx.shape} -- skipping"
            )
            continue

        samples_read += 1
        total_slots += int(ev_idx.size)

        real = ev_idx[static_mask]  # int32[num_real_frames]
        real_frame_slots += int(real.size)
        n_with = int((real >= 0).sum())  # int
        with_event_idx += n_with
        pad_event_idx += int((real == -1).sum())
        overflow_event_idx += int((real == -2).sum())
        if real.size > 0 and n_with == 0:
            fully_degenerate_samples += 1
        distinct_events_attended.append(int(np.unique(real[real >= 0]).size))

        n_live = int(event_mask.sum())  # int
        events_per_sample.append(n_live)
        live_events_seen += n_live
        for k in np.nonzero(event_mask)[0]:
            n_tok = int(text_mask[k].sum())  # int
            caption_tokens_per_live_event.append(n_tok)
            if n_tok == 0:
                empty_caption_events += 1

        if (n + 1) % 50 == 0:
            print(f"  ...{n + 1}/{len(indices)} sampled, {samples_read} read OK")

    frac = (with_event_idx / real_frame_slots) if real_frame_slots else None  # float or None
    return {
        "samples_read": samples_read,
        "samples_requested": num_samples,
        "total_frame_slots": total_slots,
        "real_frame_slots": real_frame_slots,
        "with_event_idx": with_event_idx,
        "pad_event_idx": pad_event_idx,
        "overflow_event_idx": overflow_event_idx,
        "frac_real_frames_with_event": frac,
        "fully_degenerate_samples": fully_degenerate_samples,
        "live_events_seen": live_events_seen,
        "empty_caption_events": empty_caption_events,
        "events_per_sample": _percentiles(events_per_sample, np),
        "distinct_events_attended_per_sample": _percentiles(distinct_events_attended, np),
        "caption_tokens_per_live_event": _percentiles(caption_tokens_per_live_event, np),
        "overflow_warnings_this_worker": dataset._overflow_count,
    }


@app.local_entrypoint()
def main(num_samples: int = DEFAULT_NUM_SAMPLES, seed: int = 0):
    """
    What it does:
        Triggers the remote check and prints its verdict: whether caption
        information reaches the fusion cross-attention on real training
        samples, with the supporting counts.

    Returns:
        None -- prints to stdout.

    Example input:
        modal run XF_18k_eval/analysis/check_caption_plumbing.py --num-samples 400

    Example output:
        (stdout) counts, percentile tables, and a CAPTIONS ARRIVE / DO NOT ARRIVE verdict
    """
    r = check_plumbing.remote(num_samples=num_samples, seed=seed)  # dict

    print("\n=========== XF caption plumbing on real training samples ===========")
    print(f"samples read                     : {r['samples_read']} / {r['samples_requested']}")
    print(f"frame slots total (incl. padding): {r['total_frame_slots']}")
    print(f"frame slots holding a REAL frame : {r['real_frame_slots']}")
    print("")
    print("Of the REAL frame slots:")
    print(f"  carrying a real event id (>=0) : {r['with_event_idx']}")
    print(f"  PAD_EVENT_IDX  (-1)            : {r['pad_event_idx']}")
    print(f"  OVERFLOW_EVENT_IDX (-2)        : {r['overflow_event_idx']}")
    frac = r["frac_real_frames_with_event"]  # float or None
    print(f"  ==> fraction with a real event : {frac:.4f}" if frac is not None else "  ==> no real frames at all")
    print("")
    print(f"samples where NO real frame had a real event id: {r['fully_degenerate_samples']}")
    print(f"live events seen                 : {r['live_events_seen']}")
    print(f"live events with an EMPTY caption: {r['empty_caption_events']}")
    print(f"event_overflow warnings           : {r['overflow_warnings_this_worker']}")

    for label, key in [
        ("live events per sample", "events_per_sample"),
        ("distinct events attended per sample", "distinct_events_attended_per_sample"),
        ("non-pad caption tokens per live event", "caption_tokens_per_live_event"),
    ]:
        d = r[key]  # dict
        if d.get("n"):
            print(
                f"\n{label}: n={d['n']}  min={d['min']:.0f}  p25={d['p25']:.1f}  "
                f"p50={d['p50']:.1f}  p75={d['p75']:.1f}  max={d['max']:.0f}  mean={d['mean']:.2f}"
            )
        else:
            print(f"\n{label}: EMPTY")

    print("\n---------------------------------- verdict ----------------------------------")
    if frac is None:
        print("INCONCLUSIVE: no real frame slots were seen at all -- investigate static_mask.")
    elif frac < 0.05:
        print("CAPTIONS DO NOT ARRIVE. Nearly every real frame attends only to the null token,")
        print("so the fusion message is a constant carrying no caption information. The dead")
        print("gate is a DATA-PLUMBING failure; the plan's remedies would target the wrong thing.")
    elif frac < 0.5:
        print("CAPTIONS ARRIVE ONLY PARTIALLY. A large share of real frames sees the null token")
        print("only -- worth fixing before concluding anything about the mechanism itself.")
    else:
        print("CAPTIONS ARRIVE. Most real frames are aligned to a real event with real caption")
        print("tokens, so the fusion path was genuinely fed information and still gated it off.")
        print("That points at the mechanism/objective, not at data plumbing.")
    print()
