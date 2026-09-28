"""
xf_subgoal_table_builder.py

Offline driver for the temporal aligner's Task A (training-time) -- see
xattn_fusion/temporal-alignment-agent.md. Runs a SECOND pass over the same
raw .h5 files build_robomme_dataset.py already processed, building one
subgoal_table.SubgoalTable per episode and writing it to
episode_{global_episode_idx}/subgoal_table.json inside whichever feature
shard directory that episode's precomputed features already live in
(feature_shard_router.feature_episode_dir -- see that module's docstring:
one Modal volume hard-caps at 500,000 files, so the full 16-task dataset's
features/ is split across NUM_FEATURE_SHARDS separate volumes/directories
instead of build_robomme_dataset.py's single preprocessed_data_path/
features/ directory). This script never touches build_robomme_dataset.py
itself and never writes anywhere else.

IMPORTANT: episode numbering is NOT derived from os.listdir order (an
earlier version of this script assumed it was, mirroring
build_robomme_dataset.py's own iteration -- that assumption is WRONG for
the pre-built HF dataset (Yinpei/robomme_preprocessed_data) this project
actually trains on, confirmed 2026-09-19: our local raw_h5's os.listdir
order does not match the numbering the dataset's authors used when they
built it -- episode 0 means a completely different recording on each side).
Instead, this script loads an explicit (raw H5 file, h5_episode_key) ->
global_episode_idx mapping from episode_mapping.json, built separately by
content-matching robot state vectors between the two (see chat/
RESEARCH_LOG.md, 2026-09-19). Only episodes present in that mapping are
processed -- 1,309 of the "1,600" raw episodes, confirmed exhaustively (not
estimated): 291 episodes have fully-populated features but zero training
samples in data/, which the paper itself explains is expected ("we discard
episodes in which the built-in planner fails, retaining only successful
rollouts for training") -- this script correctly skips those 291 rather
than writing a subgoal_table.json nothing will ever read.

Role in the system: run once, before training. Also computes and prints the
coverage report (max/p50/p99 events per truncated table, max/p99 tokenized-
caption length) that xf-framesamp-modul-xattn.yaml's fusion.max_events/
caption_len placeholders need real values for, before any real training run
(not required to pass for the v1 smoke test, which uses the placeholders
as-is).
"""

from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np

from mme_vla_suite.dataset_builder.robomme_h5_utils import first_execution_step

from xattn_fusion.mme_vla_suite.shared.feature_shard_router import feature_episode_dir
from xattn_fusion.mme_vla_suite.shared.subgoal_table import (
    apply_caption_vocab,
    build_caption_vocab,
    build_subgoal_table,
    load_sentencepiece_tokenizer,
    save_subgoal_table,
    truncate_table,
)


def load_episode_mapping(path: str) -> dict[int, tuple[str, str]]:
    """
    What it does:
        Reads the JSON mapping built separately (content-matching robot
        state vectors between our local raw H5 and the downloaded dataset --
        see chat/RESEARCH_LOG.md, 2026-09-19) from global_episode_idx to
        which (h5 filename, h5 episode key) it actually corresponds to.

    Returns:
        dict[int, tuple[str, str]] -- {global_episode_idx: (h5_filename,
        h5_episode_key)}.

    Example input:
        load_episode_mapping("/xf_data/episode_mapping.json")

    Example output:
        {0: ("record_dataset_PatternLock.h5", "episode_12"), ...}
    """
    with open(path) as f:
        raw = json.load(f)  # dict[str, list[str]]
    return {int(k): (v[0], v[1]) for k, v in raw.items()}


def build_all_subgoal_tables(raw_data_path: str, episode_mapping: dict[int, tuple[str, str]], max_episodes: int | None = None):
    """
    What it does:
        Builds one SubgoalTable per episode in episode_mapping (NOT by
        iterating raw_data_path's own os.listdir order -- see module
        docstring for why that numbering doesn't match the downloaded
        dataset), applies a caption_id vocabulary built across ALL episodes
        first (build_caption_vocab needs every table before any caption_id
        can be assigned), and writes each to
        feature_episode_dir(global_episode_idx)/subgoal_table.json (sharded
        across NUM_FEATURE_SHARDS volumes, not one fixed directory).

    Returns:
        list[tuple[int, SubgoalTable]] -- (global_episode_idx, table) pairs,
        for build_coverage_report to summarize.

    Example input:
        build_all_subgoal_tables("data/raw", {0: ("record_dataset_BinFill.h5", "episode_3")})

    Example output:
        [(0, SubgoalTable(...)), (1, SubgoalTable(...)), ...]
    """
    tables: list[tuple[int, object]] = []
    h5_cache: dict[str, h5py.File] = {}

    def get_h5(fname: str) -> h5py.File:
        if fname not in h5_cache:
            h5_cache[fname] = h5py.File(os.path.join(raw_data_path, fname), "r")
        return h5_cache[fname]

    items = sorted(episode_mapping.items())  # list[tuple[int, tuple[str,str]]]
    if max_episodes is not None:
        items = items[:max_episodes]

    for global_episode_idx, (fname, h5_key) in items:
        episode_data = get_h5(fname)[h5_key]  # h5py.Group
        exec_start_idx = first_execution_step(episode_data)  # int
        table = build_subgoal_table(episode_data, exec_start_idx)
        tables.append((global_episode_idx, table))

    for f in h5_cache.values():
        f.close()

    vocab = build_caption_vocab([t for _, t in tables])  # dict[str,int]
    print(f"Caption vocabulary size (excluding UNK/DEMO reserved ids): {len(vocab)}")

    for global_episode_idx, table in tables:
        table = apply_caption_vocab(table, vocab)
        episode_dir = feature_episode_dir(global_episode_idx)  # str
        assert os.path.isdir(episode_dir), (
            f"{episode_dir} does not exist -- run the feature preprocessing/download step for "
            f"this episode first (see feature_shard_router.py for the sharded volume layout)"
        )
        save_subgoal_table(table, os.path.join(episode_dir, "subgoal_table.json"))

    return [(i, apply_caption_vocab(t, vocab)) for i, t in tables]


def build_coverage_report(tables) -> dict:
    """
    What it does:
        Computes the statistics xf-framesamp-modul-xattn.yaml's
        fusion.max_events/caption_len placeholders need real values for:
        events-per-truncated-table (sampled at a stride across each episode
        for the percentile distribution, PLUS episode_len-1 explicitly --
        see below for why) and tokenized-caption-length (exhaustive, every
        interval of every table), both summarized as max/p50/p99.

        Fixed 2026-09-20 (independent review, see RESEARCH_LOG.md): the
        original version only strided-sampled `now` (~20 points/episode),
        which systematically misses `now=episode_len-1` unless the stride
        happens to divide evenly -- and since truncate_table's kept-interval
        set only ever GROWS as `now` increases (an interval, once its
        start<=now, is never later excluded), len(truncated.intervals) is
        monotonically non-decreasing in `now`, so the true per-episode
        maximum is ALWAYS achieved at the last real timestep. The old
        version's reported "max" was therefore a provable lower bound, not
        the true worst case -- explicitly including episode_len-1 for every
        episode (not just a stride sample) closes this gap exactly, without
        needing a full dense per-timestep scan.

    Returns:
        dict -- {"events_per_table": {"max":.., "p50":.., "p99":..},
        "caption_token_len": {"max":.., "p50":.., "p99":..}}.

    Example input:
        build_coverage_report(tables)

    Example output:
        {"events_per_table": {"max": 19, "p50": 4, "p99": 14}, "caption_token_len": {"max": 21, "p50": 9, "p99": 18}}
    """
    sp = load_sentencepiece_tokenizer()
    event_counts: list[int] = []
    caption_lens: list[int] = []

    for _, table in tables:
        episode_len = table.episode_len  # int
        stride = max(1, episode_len // 20)  # int, ~20 truncation points per episode
        now_points = set(range(0, episode_len, stride))  # set[int]
        now_points.add(episode_len - 1)  # the true per-episode max is always here -- see docstring
        for now in now_points:
            truncated = truncate_table(table, now)
            event_counts.append(len(truncated.intervals))
        for iv in table.intervals:
            caption_lens.append(len(sp.encode(iv.template, add_bos=False)))

    def _summary(values: list[int]) -> dict:
        arr = np.array(values)  # int64[n]
        return {"max": int(arr.max()), "p50": int(np.percentile(arr, 50)), "p99": int(np.percentile(arr, 99))}

    return {"events_per_table": _summary(event_counts), "caption_token_len": _summary(caption_lens)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", type=str, default="data/raw")
    parser.add_argument("--episode_mapping_path", type=str, default="data/episode_mapping.json")
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()

    mapping = load_episode_mapping(args.episode_mapping_path)
    print(f"loaded {len(mapping)} episode mappings")

    built_tables = build_all_subgoal_tables(args.raw_data_path, mapping, args.max_episodes)
    report = build_coverage_report(built_tables)
    print("Coverage report (use p99 + margin to set fusion.max_events / fusion.caption_len):")
    print(report)
