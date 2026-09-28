"""
subgoal_logger.py

Temporal aligner, Task B (eval-time) -- see xattn_fusion/temporal-alignment-
agent.md. The released eval loop has no stored subgoal history (eval.py just
overwrites a scalar `subgoal`/`last_subgoal` each step); SubgoalLogger fixes
that by RLE-ing the live caption stream as it arrives during a rollout, into
the same SubgoalTable/SubgoalInterval structures subgoal_table.py's Task A
(training-time) builds from H5 logs -- so truncate_table()/
assign_events_to_frames()/pack_event_arrays() run identically for both.

Role in the system: xf_policy.py's XFPolicy owns one SubgoalLogger instance,
calling .append(self.step_idx, obs["grounded_subgoal"]) once per step inside
_prepare_history (self.step_idx is already Clock A -- the policy/memory-
buffer clock -- so no clock conversion is needed at that call site) and
.reset() inside XFPolicy.reset().
"""

from __future__ import annotations

from mme_vla_suite.dataset_builder.robomme_h5_utils import preprocess_grounded_subgoal
from xattn_fusion.mme_vla_suite.shared.subgoal_table import (
    CAPTION_DEMO,
    DEMO_CAPTION_ID,
    UNK_CAPTION_ID,
    SubgoalInterval,
    SubgoalTable,
    apply_caption_vocab,
    find_interval_at_step,
)

# int, pixel tolerance (in 256x256 image space) for treating two consecutive grounded-subgoal
# coordinate readings as "the same event" despite small predictor jitter, rather than splitting
# a new interval on every frame. Spec's own suggested starting value, unvalidated -- flagged as
# an open tunable in the v1 plan, not blocking the smoke test.
COORD_MERGE_PX = 16


def _coord_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    """
    What it does: Euclidean distance between two (row, col) bbox points.

    Returns:
        float -- pixel distance.

    Example input:
        _coord_distance((128, 64), (130, 70))

    Example output:
        6.32...
    """
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class SubgoalLogger:
    """Streams live grounded_subgoal captions into an RLE'd SubgoalTable during eval rollout."""

    def __init__(self) -> None:
        self._log: list[tuple[int, str, str, tuple[tuple[int, int], ...]]] = []  # list[(step, raw, template, coords)]

    def reset(self) -> None:
        """
        What it does: clears logged history -- call at the start of every episode.

        Returns:
            None.

        Example input:
            logger.reset()

        Example output:
            None
        """
        self._log = []

    def append(self, step_idx: int, caption: str | None) -> None:
        """
        What it does:
            Appends one live observation, starting a new interval whenever
            the coordinate-stripped template changes OR the bbox coordinate
            drifts more than COORD_MERGE_PX from the current interval's first
            reading (predictor jitter guard) -- otherwise treated as a
            continuation of the current interval (natural RLE as the stream
            arrives, no separate merge pass needed).

        Returns:
            None -- mutates internal log state.

        Example input:
            logger.append(57, "pick up the cube at <128, 64>")

        Example output:
            None
        """
        if caption is None:
            return
        raw = caption.strip().lower()  # str
        template, coords_raw = preprocess_grounded_subgoal(raw)  # str, list[list[int]]
        coords = tuple(tuple(c) for c in coords_raw)  # tuple[tuple[int,int],...]

        if not self._log:
            self._log.append((step_idx, raw, template, coords))
            return

        _, _, last_template, last_coords = self._log[-1]  # int, str, str, tuple[tuple[int,int],...]
        same_template = template == last_template  # bool
        same_coords = True  # bool
        if same_template and coords and last_coords:
            same_coords = _coord_distance(coords[0], last_coords[0]) <= COORD_MERGE_PX

        if same_template and same_coords:
            return  # continuation of the current interval, nothing new to record
        self._log.append((step_idx, raw, template, coords))

    def to_subgoal_table(self, now: int, exec_start_idx: int) -> SubgoalTable:
        """
        What it does:
            Converts the running log into a SubgoalTable spanning [0, now+1),
            inserting the same CAPTION_DEMO sentinel interval Task A uses
            when the task has a video-demo prefix. caption_id is left at the
            placeholder UNK id here -- eval-time templates are matched
            against the training vocabulary (apply_caption_vocab) by the
            caller, since an eval-only template should map to UNK, not a
            fabricated new id.

            Fixed 2026-09-20 (independent review, see RESEARCH_LOG.md): the
            previous version unconditionally built the demo sentinel as
            [0, exec_start_idx) and, separately, always tried to append an
            execution-phase interval/fallback -- both wrong whenever
            `now < exec_start_idx` (i.e. still inside the video-demo prefix,
            reachable from real training via snap_table_to_chunk_grid
            whenever a sampled step_idx lands inside a demo prefix, which
            happens routinely for the 9 video-conditioned tasks at the
            configured snap_boundaries_to_chunk_grid_prob=0.5). That
            produced either a genuinely INVERTED interval (end < start, from
            end=now+1 < start=exec_start_idx) or a duplicate event_idx=0 (the
            fallback was hardcoded to 0 instead of len(intervals)), both of
            which crash pack_event_arrays. Fixed by clipping the demo
            sentinel to min(exec_start_idx, now+1) and marking it open
            (is_open=(now < exec_start_idx)) -- when still inside the demo,
            that clipped-open sentinel IS the correct "as of now" table on
            its own, no execution-phase interval should exist yet at all, so
            the method returns right after appending it. The remaining
            fallback (still needed for `now >= exec_start_idx` with no live
            caption logged yet, e.g. the very first post-demo inference
            call) now correctly uses event_idx=len(intervals) instead of a
            hardcoded 0, matching the RLE loop's own convention below.

        Returns:
            SubgoalTable -- intervals=[demo?, ...RLE'd live intervals...],
            view="live", episode_len=now+1, exec_start_idx=exec_start_idx.
            The last interval is marked is_open=True directly (this table is
            already "as of now" by construction -- unlike Task A's full-
            episode table, no separate truncate_table() call is required,
            though calling it is harmless/idempotent).

        Example input:
            logger.to_subgoal_table(now=57, exec_start_idx=42)

        Example output:
            SubgoalTable(intervals=[SubgoalInterval(0,42,CAPTION_DEMO,...,is_demo=True), SubgoalInterval(42,58,...,is_open=True)], view="live", episode_len=58, exec_start_idx=42)
        """
        intervals: list[SubgoalInterval] = []
        if exec_start_idx > 0:
            demo_end = min(exec_start_idx, now + 1)  # int, clip to `now` if still inside the demo
            intervals.append(
                SubgoalInterval(
                    start=0,
                    end=demo_end,
                    caption=CAPTION_DEMO,
                    template=CAPTION_DEMO,
                    coords=(),
                    event_idx=0,
                    caption_id=DEMO_CAPTION_ID,
                    is_demo=True,
                    is_open=(now < exec_start_idx),
                )
            )
            if now < exec_start_idx:
                # Still inside the video-demo prefix as of `now` -- no execution-phase caption
                # can exist yet. The (open) demo sentinel above already satisfies "last interval
                # is open"; returning here avoids building an execution-phase interval that
                # would start AFTER `now` (the exact inversion bug this fix closes).
                return SubgoalTable(intervals=intervals, view="live", episode_len=now + 1, exec_start_idx=exec_start_idx)

        log = [entry for entry in self._log if entry[0] >= exec_start_idx]  # list, execution-phase only
        for i, (step, raw, template, coords) in enumerate(log):
            end = log[i + 1][0] if i + 1 < len(log) else now + 1  # int
            intervals.append(
                SubgoalInterval(
                    start=step,
                    end=end,
                    caption=raw,
                    template=template,
                    coords=coords,
                    event_idx=len(intervals),
                    caption_id=UNK_CAPTION_ID,
                    is_demo=False,
                    is_open=(i + 1 == len(log)),
                )
            )

        if not log:
            # No live caption observed yet (e.g. the very first post-demo inference call, before
            # any subgoal predictor reading has arrived at this exact step) -- emit an open,
            # empty-caption interval so the LAST interval is always open. event_idx=len(intervals)
            # (not a hardcoded 0) so this correctly becomes 1 when a closed demo sentinel already
            # occupies event_idx=0 above, avoiding a duplicate event_idx.
            intervals.append(
                SubgoalInterval(
                    start=exec_start_idx,
                    end=now + 1,
                    caption="",
                    template="",
                    coords=(),
                    event_idx=len(intervals),
                    caption_id=UNK_CAPTION_ID,
                    is_demo=False,
                    is_open=True,
                )
            )

        return SubgoalTable(intervals=intervals, view="live", episode_len=now + 1, exec_start_idx=exec_start_idx)


def to_clock_a(count: int, exec_start_idx: int) -> int:
    """
    What it does:
        Converts the eval-loop clock (epstate.count, starting at 0 at the
        first execution step) to Clock A (step_idx, the policy/memory-buffer
        clock, counting every frame including the video-demo prefix). Not
        needed at XFPolicy._prepare_history's own call site (self.step_idx
        already IS Clock A there), but provided for any other eval-side
        caller that only has `count`.

    Returns:
        int -- the Clock A step index.

    Example input:
        to_clock_a(count=15, exec_start_idx=42)

    Example output:
        57
    """
    return count + exec_start_idx


def snap_table_to_chunk_grid(
    table_full: SubgoalTable, exec_start_idx: int, now: int, chunk_size: int
) -> SubgoalTable:
    """
    What it does:
        temporal-alignment-agent.md's SNAP_BOUNDARIES_TO_CHUNK_GRID: rebuilds
        the causally-truncated table as of `now`, but quantized onto the
        SAME coarse grid real eval-time inference actually observes subgoal
        transitions on -- confirmed (not assumed) from examples/robomme/
        eval.py's own defaults: the client only calls client.infer() (the
        only place a fresh subgoal reading reaches the policy) once per
        `chunk_size`-step action chunk, at steps `exec_start_idx,
        exec_start_idx + chunk_size, exec_start_idx + 2*chunk_size, ...`.

        Implemented by literally replaying `table_full`'s ground-truth
        per-step captions through a fresh SubgoalLogger, calling .append()
        only at those same chunk-boundary steps (never in between) -- i.e.
        this is not an approximation of what eval-time inference would see,
        it IS the same computation eval-time inference performs, just run
        offline against a pre-recorded H5 log instead of a live rollout.
        That also means it inherits eval's real failure mode exactly: if two
        true transitions fall inside the same chunk, only the one active AT
        the boundary step survives -- the other is genuinely invisible to
        eval too, not just delayed.

        Call with probability p during training (a config knob, not a fixed
        behavior -- see xf_dataset.XFDataset) so the model sees a MIX of
        exact-to-step and eval-realistic boundaries, rather than always one
        or the other.

    Returns:
        SubgoalTable -- same shape/contract as truncate_table's output
        (intervals covering [0, now], last interval is_open=True), but with
        internal boundaries snapped to the chunk grid.

    Example input:
        snap_table_to_chunk_grid(table_full, exec_start_idx=0, now=340, chunk_size=16)

    Example output:
        SubgoalTable(intervals=[SubgoalInterval(0,96,...), SubgoalInterval(96,144,...), SubgoalInterval(144,341,...,is_open=True)], view="live", episode_len=341, exec_start_idx=0)
    """
    logger = SubgoalLogger()
    boundary = exec_start_idx  # int, next chunk-boundary step to poll
    while boundary <= now:
        gt_iv = find_interval_at_step(table_full, boundary)
        logger.append(boundary, gt_iv.caption if gt_iv is not None else None)
        boundary += chunk_size
    snapped = logger.to_subgoal_table(now=now, exec_start_idx=exec_start_idx)  # SubgoalTable, caption_id all UNK
    # Restore caption_id from the ground-truth table being replayed (template -> id). SubgoalLogger is
    # the eval-time aligner and always emits UNK_CAPTION_ID, so before 2026-09-23 every snapped
    # TRAINING sample (~snap_boundaries_to_chunk_grid_prob = 50%) silently lost its caption_id --
    # measured 53% real ids on 256 real samples (smoke_test_symroute CHECK5). Harmless until the
    # symbolic route's auxiliary loss started using caption_id as its label. Eval never calls this.
    vocab = {iv.template: iv.caption_id for iv in table_full.intervals if iv.caption_id != UNK_CAPTION_ID}  # dict[str, int]
    return apply_caption_vocab(snapped, vocab) if vocab else snapped
