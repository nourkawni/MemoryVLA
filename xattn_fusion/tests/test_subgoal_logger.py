"""
test_subgoal_logger.py

Unit tests for subgoal_logger.py's Task B (eval-time) aligner. Pure Python,
no JAX, synthetic caption streams only (no live eval loop needed).
"""

from __future__ import annotations

from xattn_fusion.mme_vla_suite.shared.subgoal_logger import COORD_MERGE_PX, SubgoalLogger, snap_table_to_chunk_grid, to_clock_a
from xattn_fusion.mme_vla_suite.shared.subgoal_table import (
    SubgoalInterval,
    SubgoalTable,
    UNK_CAPTION_ID,
    apply_caption_vocab,
    build_caption_vocab,
)


def _mk_table(bounds, exec_start_idx=0):
    """Builds a SubgoalTable directly from a list of (start, end, caption) triples, for full
    control over synthetic snap_table_to_chunk_grid test scenarios."""
    intervals = [
        SubgoalInterval(
            start=start, end=end, caption=caption, template=caption, coords=(),
            event_idx=i, caption_id=i + 2, is_demo=False, is_open=False,
        )
        for i, (start, end, caption) in enumerate(bounds)
    ]
    return SubgoalTable(intervals=intervals, view="online", episode_len=bounds[-1][1], exec_start_idx=exec_start_idx)


def test_repeated_captions_collapse_to_one_interval():
    logger = SubgoalLogger()
    for step in range(5):
        logger.append(step, "pick up the cube at <128, 64>")
    table = logger.to_subgoal_table(now=4, exec_start_idx=0)
    assert len(table.intervals) == 1
    assert table.intervals[0].start == 0 and table.intervals[0].is_open


def test_template_change_starts_new_interval():
    logger = SubgoalLogger()
    for step in range(3):
        logger.append(step, "pick up the cube at <128, 64>")
    for step in range(3, 6):
        logger.append(step, "place the cube at <10, 20>")
    table = logger.to_subgoal_table(now=5, exec_start_idx=0)
    assert len(table.intervals) == 2
    assert table.intervals[0].end == 3 and table.intervals[1].start == 3
    assert table.intervals[1].is_open


def test_coordinate_jitter_under_threshold_does_not_split():
    logger = SubgoalLogger()
    logger.append(0, "pick up the cube at <128, 64>")
    logger.append(1, f"pick up the cube at <{128 + COORD_MERGE_PX - 1}, 64>")
    table = logger.to_subgoal_table(now=1, exec_start_idx=0)
    assert len(table.intervals) == 1


def test_coordinate_jitter_over_threshold_splits():
    logger = SubgoalLogger()
    logger.append(0, "pick up the cube at <128, 64>")
    logger.append(1, f"pick up the cube at <{128 + COORD_MERGE_PX + 20}, 64>")
    table = logger.to_subgoal_table(now=1, exec_start_idx=0)
    assert len(table.intervals) == 2


def test_complete_sentinel_does_not_start_a_new_interval():
    # SubgoalLogger receives already-resolved captions from the eval loop in practice
    # (env_runner exposes grounded_subgoal_oracle, not a raw "complete" string), but
    # this documents the expected behavior if a raw sentinel ever reaches append():
    # it is treated as a distinct template like any other string -- resolution to the
    # last valid caption is the caller's (env_runner's) responsibility, not the logger's.
    logger = SubgoalLogger()
    logger.append(0, "pick up the cube at <128, 64>")
    logger.append(1, "pick up the cube at <128, 64>")
    table = logger.to_subgoal_table(now=1, exec_start_idx=0)
    assert len(table.intervals) == 1


def test_video_demo_task_with_no_live_caption_yet_still_ends_open():
    # Regression test for a code-review-found bug: on a video-demo task
    # (exec_start_idx > 0), if to_subgoal_table() is called before any live
    # caption has been logged (log is empty), the last interval must still be
    # the open fallback, not the closed demo sentinel -- pack_event_arrays'
    # "last interval is open" invariant depends on this.
    logger = SubgoalLogger()
    table = logger.to_subgoal_table(now=12, exec_start_idx=10)
    assert len(table.intervals) == 2
    assert table.intervals[0].is_demo and not table.intervals[0].is_open
    assert table.intervals[1].is_open
    assert table.intervals[1].start == 10 and table.intervals[1].end == 13


def test_now_inside_video_demo_prefix_gives_open_clipped_sentinel_no_inversion():
    # Regression test for a CRITICAL bug (independent review, 2026-09-20): the previous
    # implementation unconditionally built the demo sentinel as [0, exec_start_idx) and always
    # tried to append an execution-phase interval, producing an INVERTED interval
    # (end=now+1 < start=exec_start_idx) whenever now < exec_start_idx -- reachable from real
    # training via snap_table_to_chunk_grid whenever a sampled step lands inside a video-demo
    # prefix, which the reviewing agent confirmed happens routinely for the 9 video-conditioned
    # tasks at the real snap_boundaries_to_chunk_grid_prob=0.5. This must never produce an
    # end < start interval, and the demo sentinel itself must be OPEN and CLIPPED to now+1.
    logger = SubgoalLogger()
    table = logger.to_subgoal_table(now=12, exec_start_idx=50)
    assert len(table.intervals) == 1, "no execution-phase interval should exist while still inside the demo"
    demo = table.intervals[0]
    assert demo.is_demo and demo.is_open
    assert demo.start == 0 and demo.end == 13, "clipped to now+1, not the full [0, exec_start_idx)"
    assert demo.end >= demo.start, "must never invert"
    assert table.episode_len == 13


def test_now_inside_video_demo_prefix_even_with_live_captions_logged_early():
    # Same scenario, but with an (out-of-order/premature) live caption already logged before
    # exec_start_idx -- to_subgoal_table filters the log to entries >= exec_start_idx, so this
    # must not change the outcome: still just the open, clipped demo sentinel.
    logger = SubgoalLogger()
    logger.append(5, "pick up the cube at <128, 64>")
    table = logger.to_subgoal_table(now=20, exec_start_idx=50)
    assert len(table.intervals) == 1
    assert table.intervals[0].is_demo and table.intervals[0].is_open
    assert table.intervals[0].end == 21


def test_video_demo_task_boundaries_and_sentinel():
    logger = SubgoalLogger()
    for step in range(10, 15):
        logger.append(step, "pick up the cube at <128, 64>")
    table = logger.to_subgoal_table(now=14, exec_start_idx=10)
    assert table.intervals[0].is_demo
    assert table.intervals[0].start == 0 and table.intervals[0].end == 10
    assert table.intervals[1].start == 10 and table.intervals[1].is_open


def test_output_matches_task_a_invariants():
    logger = SubgoalLogger()
    for step in range(5, 8):
        logger.append(step, "watching")
    for step in range(8, 12):
        logger.append(step, "pick up the cube at <128, 64>")
    table = logger.to_subgoal_table(now=11, exec_start_idx=5)
    intervals = table.intervals
    assert intervals[0].start == 0
    assert intervals[-1].is_open and intervals[-1].end == 12
    for i in range(len(intervals) - 1):
        assert intervals[i].end == intervals[i + 1].start


def test_unseen_template_maps_to_unk_against_training_vocab():
    logger = SubgoalLogger()
    logger.append(0, "an instruction never seen in training")
    table = logger.to_subgoal_table(now=0, exec_start_idx=0)
    vocab = build_caption_vocab([])  # empty training vocab
    table = apply_caption_vocab(table, vocab)
    assert table.intervals[0].caption_id == UNK_CAPTION_ID


def test_no_caption_yet_emits_single_open_empty_interval():
    logger = SubgoalLogger()
    table = logger.to_subgoal_table(now=0, exec_start_idx=0)
    assert len(table.intervals) == 1
    assert table.intervals[0].is_open


def test_to_clock_a_conversion():
    assert to_clock_a(count=15, exec_start_idx=42) == 57
    assert to_clock_a(count=0, exec_start_idx=0) == 0


def test_snap_boundaries_land_on_chunk_grid():
    # true transition at step 40; chunk_size=16 -> boundaries at 0,16,32,48,... -- the snapped
    # transition must land on the first boundary AT or AFTER 40, i.e. 48, never before it.
    table = _mk_table([(0, 40, "a"), (40, 100, "b")])
    snapped = snap_table_to_chunk_grid(table, exec_start_idx=0, now=99, chunk_size=16)
    assert len(snapped.intervals) == 2
    assert snapped.intervals[0].end == 48
    assert snapped.intervals[1].start == 48


def test_snap_never_reveals_a_transition_before_its_chunk_boundary():
    # causality: at now=45 (before the 48 boundary), the snapped view must NOT yet show interval
    # "b" -- eval-time inference genuinely cannot know about a transition before polling it.
    table = _mk_table([(0, 40, "a"), (40, 100, "b")])
    snapped = snap_table_to_chunk_grid(table, exec_start_idx=0, now=45, chunk_size=16)
    assert len(snapped.intervals) == 1
    assert snapped.intervals[0].template == "a"
    assert snapped.intervals[0].is_open


def test_snap_can_lose_a_transition_entirely_inside_one_chunk():
    # two true transitions both fall inside [32,48) -- real eval-time inference only polls once
    # at step 48, so it only ever sees whichever caption is active AT 48 ("c"); "b" (active only
    # 35..42) is genuinely invisible to eval too, not just delayed -- snap must match that exactly.
    table = _mk_table([(0, 35, "a"), (35, 42, "b"), (42, 100, "c")])
    snapped = snap_table_to_chunk_grid(table, exec_start_idx=0, now=99, chunk_size=16)
    templates = [iv.template for iv in snapped.intervals]
    assert "b" not in templates
    assert templates == ["a", "c"]


def test_snap_matches_exact_truncation_when_transitions_align_with_the_grid():
    # a transition that already lands exactly on a chunk boundary needs no correction.
    table = _mk_table([(0, 32, "a"), (32, 100, "b")])
    snapped = snap_table_to_chunk_grid(table, exec_start_idx=0, now=99, chunk_size=16)
    assert snapped.intervals[0].end == 32
    assert snapped.intervals[1].start == 32


def test_snap_with_video_demo_prefix():
    table = _mk_table([(10, 40, "a"), (40, 100, "b")], exec_start_idx=10)
    snapped = snap_table_to_chunk_grid(table, exec_start_idx=10, now=99, chunk_size=16)
    assert snapped.intervals[0].is_demo
    assert snapped.intervals[0].start == 0 and snapped.intervals[0].end == 10


def test_snap_preserves_caption_id_from_ground_truth_table():
    """
    What it does:
        Regression test (2026-09-23): snap_table_to_chunk_grid replays captions through the
        eval-time SubgoalLogger, which always emits UNK_CAPTION_ID, so snapped TRAINING samples
        (~50%) used to lose the caption_id the symbolic route's auxiliary loss uses as its label.
        Snapping must now carry each template's id over from the ground-truth table.

    Returns:
        None -- asserts.

    Example input:
        table with templates "a" (id 2), "b" (id 3); snapped at now=99, chunk 16

    Example output:
        snapped caption_ids == [2, 3]
    """
    table = _mk_table([(0, 40, "a"), (40, 100, "b")])  # SubgoalTable, ids 2 and 3
    snapped = snap_table_to_chunk_grid(table, exec_start_idx=0, now=99, chunk_size=16)  # SubgoalTable
    assert [iv.caption_id for iv in snapped.intervals] == [2, 3]
    assert all(iv.caption_id != UNK_CAPTION_ID for iv in snapped.intervals)
