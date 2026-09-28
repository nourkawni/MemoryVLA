"""
test_subgoal_table.py

Unit tests for subgoal_table.py's Task A pipeline (build_subgoal_table ->
apply_caption_vocab -> truncate_table -> assign_events_to_frames ->
pack_event_arrays). Pure Python, no JAX -- runs anywhere pytest + numpy are
installed.

Two kinds of coverage:
  1. Synthetic-episode tests (_FakeEpisode below) -- always run, no external
     data needed. Cover RLE correctness, the 10 invariants, truncation
     renumbering, and E_max overflow handling.
  2. Real-H5 tests -- only run if the ROBOMME_H5_PATH environment variable
     points at a directory of raw .h5 files (the same raw_data_path
     build_robomme_dataset.py reads); otherwise skipped with a clear reason,
     since no sample .h5 data is checked into this repo.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from xattn_fusion.mme_vla_suite.shared.subgoal_table import (
    CAPTION_DEMO,
    DEMO_CAPTION_ID,
    UNK_CAPTION_ID,
    apply_caption_vocab,
    assign_events_to_frames,
    build_caption_vocab,
    build_subgoal_table,
    pack_event_arrays,
    truncate_table,
)


class _FakeLeaf:
    """Mimics one h5py leaf dataset: supports `leaf[()]` -> the stored value."""

    def __init__(self, value):
        self.value = value  # Any, the stored value (bytes for strings, bool for flags)

    def __getitem__(self, _key):
        return self.value


class _FakeTokenizer:
    """Trivial whitespace tokenizer standing in for sentencepiece.SentencePieceProcessor."""

    def encode(self, text: str, add_bos: bool = False) -> list[int]:
        # list[int], one id per character (deterministic, no external vocab needed for shape tests)
        return [ord(c) for c in text][:64]


def _fake_episode(captions: list[str], exec_start_idx: int) -> dict:
    """
    What it does: builds a dict that satisfies build_subgoal_table's H5
    access pattern (episode_data[f"timestep_{i}"]["info"][field][()]).

    Returns:
        dict -- fake h5py.Group-like episode.

    Example input:
        _fake_episode(["a", "a", "b"], exec_start_idx=1)

    Example output:
        {"timestep_0": {"info": {...}}, "timestep_1": {...}, "timestep_2": {...}}
    """
    episode = {}
    for step, cap in enumerate(captions):
        episode[f"timestep_{step}"] = {
            "info": {
                "grounded_subgoal_online": _FakeLeaf(cap.encode()),
                "is_video_demo": _FakeLeaf(step < exec_start_idx),
            }
        }
    return episode


def _full_pipeline(captions: list[str], exec_start_idx: int, now: int, max_size: int = 8, E: int = 4, L_c: int = 8):
    """Runs build -> vocab -> truncate -> assign -> pack for one synthetic episode."""
    episode = _fake_episode(captions, exec_start_idx)
    table = build_subgoal_table(episode, exec_start_idx)
    vocab = build_caption_vocab([table])
    table = apply_caption_vocab(table, vocab)
    truncated = truncate_table(table, now)
    sampled = list(range(0, now + 1, max(1, (now + 1) // max_size))) or [0]
    sampled = sampled[:max_size]
    aligned = assign_events_to_frames(truncated, sampled, max_size=max_size)
    packed = pack_event_arrays(truncated, aligned, _FakeTokenizer(), E=E, L_c=L_c)
    return table, truncated, aligned, packed


def test_rle_merges_consecutive_equal_captions():
    captions = ["pick up the cube"] * 5 + ["place the cube"] * 3
    table = build_subgoal_table(_fake_episode(captions, 0), exec_start_idx=0)
    assert len(table.intervals) == 2
    assert table.intervals[0].start == 0 and table.intervals[0].end == 5
    assert table.intervals[1].start == 5 and table.intervals[1].end == 8


def test_video_demo_prefix_produces_sentinel_interval():
    captions = ["watching"] * 3 + ["pick up the cube"] * 4
    table = build_subgoal_table(_fake_episode(captions, 3), exec_start_idx=3)
    assert table.intervals[0].is_demo
    assert table.intervals[0].caption == CAPTION_DEMO
    assert table.intervals[0].start == 0 and table.intervals[0].end == 3


def test_zero_frame_event_is_kept_not_dropped():
    # a very short-lived subgoal ("b") that no sampled frame lands inside must still
    # appear as a real (event_mask=True, event_num_frames=0) row, never be dropped.
    captions = ["a", "a", "a", "b", "c", "c", "c", "c"]
    _, truncated, aligned, packed = _full_pipeline(captions, exec_start_idx=0, now=7, max_size=4)
    # sample only frames that land in "a" and "c", skipping the single "b" step (idx 3)
    sampled = [0, 1, 4, 7]
    aligned = assign_events_to_frames(truncated, sampled, max_size=4)
    packed = pack_event_arrays(truncated, aligned, _FakeTokenizer(), E=4, L_c=8)
    b_rows = [i for i in range(4) if packed["event_mask"][i] and not packed["event_is_demo"][i]
              and packed["event_start"][i] == 3 and packed["event_end"][i] == 4]
    assert len(b_rows) == 1, "the zero-frame 'b' event must still be a real, kept row"
    assert packed["event_num_frames"][b_rows[0]] == 0


def test_truncation_renumbers_event_idx_and_keeps_caption_id_stable():
    captions = ["a", "a", "b", "b", "c", "c"]
    episode = _fake_episode(captions, 0)
    table = build_subgoal_table(episode, exec_start_idx=0)
    vocab = build_caption_vocab([table])
    table = apply_caption_vocab(table, vocab)

    t1 = truncate_table(table, now=1)  # only "a" visible
    assert [iv.event_idx for iv in t1.intervals] == [0]
    assert t1.intervals[-1].is_open and t1.intervals[-1].end == 2

    t3 = truncate_table(table, now=3)  # "a" and "b" visible
    assert [iv.event_idx for iv in t3.intervals] == [0, 1]
    assert t3.intervals[0].caption_id == vocab["a"]
    assert t3.intervals[1].caption_id == vocab["b"]

    t5 = truncate_table(table, now=5)  # all three visible
    assert [iv.event_idx for iv in t5.intervals] == [0, 1, 2]
    # caption_id for the SAME template stays the same across different truncations
    assert t3.intervals[0].caption_id == t5.intervals[0].caption_id


def test_invariants_hold_across_several_truncation_points():
    captions = ["watching"] * 2 + ["a", "a", "a", "b", "c", "c", "c", "c", "d"]
    exec_start_idx = 2
    episode = _fake_episode(captions, exec_start_idx)
    table = build_subgoal_table(episode, exec_start_idx)
    vocab = build_caption_vocab([table])
    table = apply_caption_vocab(table, vocab)

    for now in [2, 4, 7, len(captions) - 1]:
        truncated = truncate_table(table, now)
        intervals = truncated.intervals
        assert intervals[0].start == 0
        assert intervals[-1].is_open
        assert intervals[-1].end == now + 1
        for i in range(len(intervals) - 1):
            assert intervals[i].end == intervals[i + 1].start
        sampled = list(range(now + 1))[:8]
        aligned = assign_events_to_frames(truncated, sampled, max_size=8)
        packed = pack_event_arrays(truncated, aligned, _FakeTokenizer(), E=8, L_c=8)
        assert not packed["event_overflow"]
        assert int(packed["event_num_frames"].sum()) == len(sampled)


def test_overflow_keeps_demo_and_most_recent_events():
    captions = ["watching"] + [f"step{i}" for i in range(10)]
    episode = _fake_episode(captions, exec_start_idx=1)
    table = build_subgoal_table(episode, exec_start_idx=1)
    vocab = build_caption_vocab([table])
    table = apply_caption_vocab(table, vocab)
    truncated = truncate_table(table, now=len(captions) - 1)
    assert len(truncated.intervals) == 11  # demo + 10 distinct steps, no merging possible (all unique)

    sampled = list(range(len(captions)))
    aligned = assign_events_to_frames(truncated, sampled, max_size=len(captions))
    packed = pack_event_arrays(truncated, aligned, _FakeTokenizer(), E=4, L_c=8)

    assert packed["event_overflow"]
    assert packed["event_mask"].sum() == 4  # E=4 slots filled
    assert packed["event_is_demo"][0]  # demo kept in slot 0
    # dropped-event frame slots must be tagged OVERFLOW_EVENT_IDX, not silently misassigned
    from xattn_fusion.mme_vla_suite.shared.subgoal_table import OVERFLOW_EVENT_IDX
    assert OVERFLOW_EVENT_IDX in packed["static_token_event_idx"]


def test_unseen_eval_template_maps_to_unk():
    train_table = build_subgoal_table(_fake_episode(["pick up the cube"] * 3, 0), exec_start_idx=0)
    vocab = build_caption_vocab([train_table])
    eval_table = build_subgoal_table(_fake_episode(["an entirely new instruction"] * 3, 0), exec_start_idx=0)
    eval_table = apply_caption_vocab(eval_table, vocab)
    assert eval_table.intervals[0].caption_id == UNK_CAPTION_ID


# ---------------------------------------------------------------------------
# Real-H5 coverage -- only runs when ROBOMME_H5_PATH is set to a directory of
# raw .h5 files (matching build_robomme_dataset.py's raw_data_path). No such
# data is checked into this repo, so this is skipped by default rather than
# failing CI/local runs that don't have the dataset downloaded.
# ---------------------------------------------------------------------------

def _find_sample_h5():
    root = os.environ.get("ROBOMME_H5_PATH")
    if not root or not os.path.isdir(root):
        return None
    for fname in os.listdir(root):
        if fname.endswith(".h5"):
            return os.path.join(root, fname)
    return None


@pytest.mark.skipif(_find_sample_h5() is None, reason="set ROBOMME_H5_PATH to a directory of raw .h5 files to run this")
def test_real_episode_no_is_completed_carry_over():
    import h5py

    from mme_vla_suite.dataset_builder.robomme_h5_utils import first_execution_step, get_episode_indices

    path = _find_sample_h5()
    with h5py.File(path, "r") as f:
        episode_idx = get_episode_indices(f, max_episodes=1)[0]
        episode_data = f[f"episode_{episode_idx}"]
        exec_start_idx = first_execution_step(episode_data)
        table = build_subgoal_table(episode_data, exec_start_idx)

        intervals = table.intervals
        assert intervals[0].start == 0
        for i in range(len(intervals) - 1):
            assert intervals[i].end == intervals[i + 1].start
        # spot-check: no interval's raw caption is literally "complete" (resolve_subgoal
        # must have substituted the last valid caption for every "complete" sentinel step)
        assert all("complete" not in iv.caption for iv in intervals)
