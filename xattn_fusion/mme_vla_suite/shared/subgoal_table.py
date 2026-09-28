"""
subgoal_table.py

Temporal aligner, Task A (training-time) + the shared data structures used by
both Task A and Task B (robomme-temporal-aligner agent -- see
xattn_fusion/temporal-alignment-agent.md). Converts a raw H5 episode's
per-timestep caption stream into a causally-truncatable table of subgoal
"events", each tagged with which sampled perceptual frames fall inside it,
for XF's symbolic+perceptual gated cross-attention fusion.

Role in the system: xf_subgoal_table_builder.py calls build_subgoal_table()
offline, once per episode, and caches the result to
features/episode_i/subgoal_table.json. xf_dataset.py's XFDataset loads that
cache per training sample and calls truncate_table() ->
assign_events_to_frames() -> pack_event_arrays(). subgoal_logger.py's
SubgoalLogger (Task B, eval-time) builds the same SubgoalTable/
SubgoalInterval structures from a live caption stream and reuses
truncate_table()/assign_events_to_frames()/pack_event_arrays() unchanged --
this module owns the one implementation of both, so training and eval can
never silently diverge on the alignment logic itself.
"""

from __future__ import annotations

import dataclasses
import json
import random
import time

import numpy as np

from mme_vla_suite.dataset_builder.robomme_h5_utils import (
    get_timestep_indices,
    preprocess_grounded_subgoal,
    resolve_subgoal,
)


# str, this arm's fixed symbolic-memory source (pinned per temporal-alignment-agent.md:
# GroundedSG+Oracle 84.08 vs SimpleSG+Oracle 49.58 on Table 3)
SYMBOLIC_TYPE = "grounded_subgoal"
# str, read the "online" (early-change) caption view -- matches eval-time timing
CAPTION_VIEW = "online"
# str, this arm's fixed integration type (memory modulator, not VLM-prefix concat)
INTEGRATION = "modulation"
# str, this arm's fixed perceptual memory source
PERCEPTUAL_TYPE = "frame_sampling"
# int, PLACEHOLDER max event slots (E_max) -- needs xf_subgoal_table_builder.py's
# coverage report (p99 + margin) before any real training run; fine for the v1 smoke test
MAX_EVENTS = 16
# str, sentinel caption text for the video-demo-prefix interval
CAPTION_DEMO = "watching the demonstration video"
# int, caption_id reserved for the video-demo sentinel (never assigned by build_caption_vocab)
DEMO_CAPTION_ID = 1
# int, caption_id for an unseen-at-eval template (build_caption_vocab reserves this)
UNK_CAPTION_ID = 0
# int, padding sentinel for a frame slot with no event at all (right-padded, unused slot)
PAD_EVENT_IDX = -1
# int, sentinel for a frame slot whose true event was dropped by E_max overflow
OVERFLOW_EVENT_IDX = -2

# tuple[str,...], the 11 event/aligner keys pack_event_arrays produces (event_overflow excluded --
# that one is a diagnostic, not a fixed-shape model input). Single source of truth for every place
# that must thread these names through: xf_observation.py's from_dict/to_dict, xf_robomme_policy.
# py's XFInputs, and xf_config.py's repack mapping -- found by code review to have drifted into
# independently hand-retyped lists in each of those files, a silent-None risk if one is ever missed.
EVENT_FIELD_NAMES = (
    "event_mask",
    "event_start",
    "event_end",
    "event_is_demo",
    "event_is_open",
    "event_caption_id",
    "event_coords",
    "event_num_frames",
    "event_text_tokens",
    "event_text_mask",
    "static_token_event_idx",
)


@dataclasses.dataclass(frozen=True)
class SubgoalInterval:
    """One contiguous run of steps sharing the same (coordinate-stripped) subgoal caption."""

    start: int  # int, first step index (Clock A) included in this interval
    end: int  # int, one past the last step index included -- half-open [start, end)
    caption: str  # str, raw caption text, e.g. "pick up the red cube at <128, 64>"
    template: str  # str, coordinate-stripped caption, e.g. "pick up the red cube at <bbox>"
    coords: tuple[tuple[int, int], ...]  # tuple[tuple[int,int],...], (row,col) bbox(es); empty if none
    event_idx: int  # int, truncated-local ordinal -- renumbered on every truncate_table() call, never stable
    caption_id: int  # int, task-global template-vocabulary id -- stable across truncations
    is_demo: bool  # bool, True only for the synthetic video-demo-prefix interval
    is_open: bool  # bool, True for the most-recent (still-ongoing) interval as of truncation time


@dataclasses.dataclass(frozen=True)
class SubgoalTable:
    """All (or, after truncate_table(), the causally-visible prefix of) one episode's intervals."""

    intervals: list[SubgoalInterval]  # list[SubgoalInterval], sorted, hard-partitioning [0, episode_len)
    view: str  # str, which caption view built this table (CAPTION_VIEW for Task A, "live" for Task B)
    episode_len: int  # int, full episode length in steps (untouched by truncate_table)
    exec_start_idx: int  # int, first non-video-demo step; 0 if the task has no video demo


@dataclasses.dataclass(frozen=True)
class AlignedMemory:
    """The frame<->event map for one (truncated SubgoalTable, sampled frame indices) pair."""

    now: int  # int, the truncation step (Clock A) this alignment was computed at
    events: list[SubgoalInterval]  # list[SubgoalInterval], table.intervals at truncation time
    step_indices: list[int]  # list[int], raw (unpadded) sampled step indices, as given by mem_buffer
    frame_slots: list[int]  # list[int], 0..max_size-1, the padded slot positions
    frames_per_event: dict[int, list[int]]  # dict[int, list[int]], event_idx -> sampled steps inside it
    slot_event_idx: np.ndarray  # int32[max_size], right-padded with PAD_EVENT_IDX
    token_event_idx: np.ndarray  # int32[max_size * num_views * token_per_image], slot_event_idx repeated per-token


def _interval_to_json(iv: SubgoalInterval) -> dict:
    """
    What it does: converts one SubgoalInterval into a JSON-safe dict --
    every field is already a JSON primitive except `coords`, a tuple of
    tuples, which json.dump can't serialize directly and is converted to a
    list of lists here.

    Returns:
        dict -- one JSON object's worth of fields, matching SubgoalInterval's
        own field names exactly.

    Example input:
        _interval_to_json(SubgoalInterval(0, 5, "pick up the cube", "pick up the cube", (), 0, 2, False, False))

    Example output:
        {"start": 0, "end": 5, "caption": "pick up the cube", "template": "pick up the cube", "coords": [], "event_idx": 0, "caption_id": 2, "is_demo": False, "is_open": False}
    """
    return {
        "start": iv.start,
        "end": iv.end,
        "caption": iv.caption,
        "template": iv.template,
        "coords": [list(c) for c in iv.coords],
        "event_idx": iv.event_idx,
        "caption_id": iv.caption_id,
        "is_demo": iv.is_demo,
        "is_open": iv.is_open,
    }


def _interval_from_json(d: dict) -> SubgoalInterval:
    """
    What it does: inverse of _interval_to_json -- rebuilds one
    SubgoalInterval from its JSON dict form, converting `coords` back from a
    list of lists into a tuple of tuples.

    Returns:
        SubgoalInterval.

    Example input:
        _interval_from_json({"start": 0, "end": 5, "caption": "pick up the cube", "template": "pick up the cube", "coords": [], "event_idx": 0, "caption_id": 2, "is_demo": False, "is_open": False})

    Example output:
        SubgoalInterval(start=0, end=5, caption="pick up the cube", template="pick up the cube", coords=(), event_idx=0, caption_id=2, is_demo=False, is_open=False)
    """
    return SubgoalInterval(
        start=d["start"],
        end=d["end"],
        caption=d["caption"],
        template=d["template"],
        coords=tuple(tuple(c) for c in d["coords"]),
        event_idx=d["event_idx"],
        caption_id=d["caption_id"],
        is_demo=d["is_demo"],
        is_open=d["is_open"],
    )


def save_subgoal_table(table: SubgoalTable, path: str) -> None:
    """
    What it does: writes a SubgoalTable to disk as JSON.

    Returns:
        None.

    Example input:
        save_subgoal_table(table, "features/episode_3/subgoal_table.json")

    Example output:
        None (file written)
    """
    payload = {
        "view": table.view,
        "episode_len": table.episode_len,
        "exec_start_idx": table.exec_start_idx,
        "intervals": [_interval_to_json(iv) for iv in table.intervals],
    }  # dict, JSON-safe
    with open(path, "w") as f:
        json.dump(payload, f)


def load_subgoal_table(path: str, max_retries: int = 10, retry_delay_s: float = 2.0) -> SubgoalTable:
    """
    What it does:
        Reads a SubgoalTable previously written by save_subgoal_table.
        Retries with a linear backoff PLUS random jitter on
        FileNotFoundError/OSError/json.JSONDecodeError before giving up.
        Root cause now CONFIRMED (2026-09-21, not just theorized): a
        remote, dual-method ground-truth scan (verify_subgoal_tables_
        remote.py) proved every real file genuinely exists, and directly
        surfaced the real underlying error Modal's volume backend returns
        under sustained concurrent load: `ResourceExhaustedError:
        VolumeListFiles rate limit exceeded. Please wait and retry.` --
        i.e. this is a real, confirmed rate limit, not a mysterious
        transient glitch or a real missing file (originally found via a
        training crash at step ~1020 on 2026-09-20, see RESEARCH_LOG.md's
        entries that day and the next for the full trace). max_retries
        raised 5->10 and retry_delay_s 1.0->2.0 (2026-09-21, matching the
        same strengthening applied to xf_dataset._load_vector_file_with_
        retry the same day, after the FIRST version of this retry --
        5 attempts, 1-4s backoff, no jitter -- still failed to prevent a
        THIRD crash on the exact same file: not enough patience against a
        rate-limit window that can outlast a short backoff, especially
        with num_workers=4 worker processes potentially retrying near-
        simultaneously and re-triggering the same limit). Random jitter
        added for that last reason specifically. Still raises (does not
        silently swallow) once every retry is exhausted, so a genuinely
        missing/corrupt file is not hidden.

    Returns:
        SubgoalTable.

    Example input:
        load_subgoal_table("features/episode_3/subgoal_table.json")

    Example output:
        SubgoalTable(intervals=[...], view="online", episode_len=210, exec_start_idx=42)
    """
    last_error: Exception | None = None  # Exception | None, re-raised only if every retry fails
    for attempt in range(max_retries):  # int
        try:
            with open(path) as f:
                payload = json.load(f)  # dict
            return subgoal_table_from_payload(payload)
        except (FileNotFoundError, OSError, json.JSONDecodeError) as e:
            last_error = e
            if attempt < max_retries - 1:
                jitter = random.uniform(0, retry_delay_s * 0.5)  # float, spreads out concurrent retries
                time.sleep(retry_delay_s * (attempt + 1) + jitter)
    raise last_error


def subgoal_table_from_payload(payload: dict) -> SubgoalTable:
    """
    What it does:
        Parses an already-loaded JSON dict (the same shape save_
        subgoal_table writes) into a SubgoalTable. Factored out of
        load_subgoal_table (2026-09-21) so a caller that obtains the raw
        JSON bytes some OTHER way than open(path) -- e.g. xf_dataset.py's
        direct modal.Volume.read_file() reader, which bypasses this
        container's own FUSE mount entirely after mount-based reads proved
        unable to reliably see episode_33's subgoal_table.json even after
        10 retries with real backoff+jitter (5/5 real training crashes on
        the exact same file, see load_subgoal_table's own docstring) --
        can reuse the exact same parsing logic instead of duplicating it.

    Returns:
        SubgoalTable.

    Example input:
        subgoal_table_from_payload({"intervals": [...], "view": "online", "episode_len": 210, "exec_start_idx": 42})

    Example output:
        SubgoalTable(intervals=[...], view="online", episode_len=210, exec_start_idx=42)
    """
    return SubgoalTable(
        intervals=[_interval_from_json(d) for d in payload["intervals"]],
        view=payload["view"],
        episode_len=payload["episode_len"],
        exec_start_idx=payload["exec_start_idx"],
    )


def load_sentencepiece_tokenizer():
    """
    What it does:
        Loads the same PaliGemma sentencepiece model
        training/config.py's PaligemmaTokenizer wraps for prompt formatting,
        but returns the raw sentencepiece.SentencePieceProcessor directly --
        pack_event_arrays needs plain `sp.encode(template, add_bos=False)`
        tokenization of a bare caption string, not PaligemmaTokenizer's
        "Task: ...\\nAction: " prompt-wrapping behavior.

    Returns:
        sentencepiece.SentencePieceProcessor -- ready to call .encode(text).

    Example input:
        load_sentencepiece_tokenizer()

    Example output:
        <sentencepiece.SentencePieceProcessor object>
    """
    import sentencepiece
    from openpi.shared import download

    path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    with path.open("rb") as f:
        return sentencepiece.SentencePieceProcessor(model_proto=f.read())


def build_subgoal_table(episode_data, exec_start_idx: int) -> SubgoalTable:
    """
    What it does:
        Reads timestep_*/info/grounded_subgoal_online directly from one raw
        H5 episode group -- deliberately bypassing build_robomme_dataset.py's
        cached per-step pickle, whose `if not ts['info']['is_completed'][()]:`
        guard (build_robomme_dataset.py:265) silently reuses the previous
        Python-local caption string on completed steps instead of
        re-reading it, which would corrupt RLE boundaries here. RLE-encodes
        consecutive steps with an equal coordinate-stripped template into one
        SubgoalInterval, and prepends a CAPTION_DEMO sentinel interval
        covering [0, exec_start_idx) when the episode has a video-demo
        prefix. Zero-frame events are a downstream (assign_events_to_frames)
        concern, not filtered here -- every RLE run becomes a kept interval.

    Returns:
        SubgoalTable -- the FULL, untruncated table for this episode
        (episode_len == number of timesteps), every interval is_open=False,
        caption_id left at UNK_CAPTION_ID / DEMO_CAPTION_ID placeholders --
        call apply_caption_vocab() afterward to fill in real template ids.

    Example input:
        build_subgoal_table(h5file["episode_3"], exec_start_idx=42)

    Example output:
        SubgoalTable(
            intervals=[
                SubgoalInterval(0, 42, "watching the demonstration video", ..., is_demo=True),
                SubgoalInterval(42, 80, "pick up the cube at <128, 64>", "pick up the cube at <bbox>", ((128, 64),), ...),
                ...,
            ],
            view="online", episode_len=210, exec_start_idx=42,
        )
    """
    timestep_indices = get_timestep_indices(episode_data)  # list[int], sorted 0..episode_len-1
    episode_len = len(timestep_indices)  # int
    caption_field = f"{SYMBOLIC_TYPE}_{CAPTION_VIEW}"  # str, "grounded_subgoal_online"

    last_valid_raw = None  # str | None, most recent non-"complete" raw caption (for resolve_subgoal)
    raw_captions = []  # list[str], one per timestep, "complete" sentinel already resolved
    for step in timestep_indices:
        ts = episode_data[f"timestep_{step}"]  # h5py.Group
        raw = ts["info"][caption_field][()].decode().lower()  # str
        resolved = resolve_subgoal(raw, last_valid_raw, sentinel="complete")  # str
        if "complete" not in raw:
            last_valid_raw = raw
        raw_captions.append(resolved)

    intervals: list[SubgoalInterval] = []
    if exec_start_idx > 0:
        intervals.append(
            SubgoalInterval(
                start=0,
                end=exec_start_idx,
                caption=CAPTION_DEMO,
                template=CAPTION_DEMO,
                coords=(),
                event_idx=0,
                caption_id=DEMO_CAPTION_ID,
                is_demo=True,
                is_open=False,
            )
        )

    run_start = exec_start_idx  # int, step index where the current RLE run began
    run_template = None  # str | None
    run_caption = None  # str | None
    run_coords = None  # tuple[tuple[int,int],...] | None
    for step in range(exec_start_idx, episode_len):
        raw = raw_captions[step]  # str
        template, coords = preprocess_grounded_subgoal(raw)  # str, list[list[int]]
        if run_template is None:
            run_template, run_caption, run_coords = template, raw, tuple(tuple(c) for c in coords)
        elif template != run_template:
            intervals.append(
                SubgoalInterval(
                    start=run_start,
                    end=step,
                    caption=run_caption,
                    template=run_template,
                    coords=run_coords,
                    event_idx=len(intervals),
                    caption_id=UNK_CAPTION_ID,
                    is_demo=False,
                    is_open=False,
                )
            )
            run_start = step
            run_template, run_caption, run_coords = template, raw, tuple(tuple(c) for c in coords)
    if run_template is not None:
        intervals.append(
            SubgoalInterval(
                start=run_start,
                end=episode_len,
                caption=run_caption,
                template=run_template,
                coords=run_coords,
                event_idx=len(intervals),
                caption_id=UNK_CAPTION_ID,
                is_demo=False,
                is_open=False,
            )
        )

    return SubgoalTable(
        intervals=intervals, view=CAPTION_VIEW, episode_len=episode_len, exec_start_idx=exec_start_idx
    )


def build_caption_vocab(tables) -> dict[str, int]:
    """
    What it does:
        Collects every non-demo template across a set of (usually: all
        training-episode) SubgoalTables into a stable id vocabulary. Ids 0
        and 1 are reserved (UNK_CAPTION_ID, DEMO_CAPTION_ID) so an eval-time
        template never seen in training falls back to UNK rather than
        colliding with a real training template.

    Returns:
        dict[str, int] -- template -> id, ids starting at 2, sorted by
        template text so the mapping is deterministic across runs.

    Example input:
        build_caption_vocab([table_ep0, table_ep1, ...])

    Example output:
        {"pick up the cube at <bbox>": 2, "place the cube at <bbox>": 3, ...}
    """
    templates = set()  # set[str]
    for table in tables:
        for interval in table.intervals:
            if not interval.is_demo:
                templates.add(interval.template)
    return {template: idx + 2 for idx, template in enumerate(sorted(templates))}


def apply_caption_vocab(table: SubgoalTable, vocab: dict[str, int]) -> SubgoalTable:
    """
    What it does:
        Returns a copy of `table` with every interval's caption_id filled in
        from `vocab` (UNK_CAPTION_ID if the template is not in vocab, e.g. an
        eval-time template never seen in training; DEMO_CAPTION_ID for the
        demo sentinel).

    Returns:
        SubgoalTable -- same intervals, caption_id replaced.

    Example input:
        apply_caption_vocab(table, {"pick up the cube at <bbox>": 2, ...})

    Example output:
        SubgoalTable(intervals=[SubgoalInterval(..., caption_id=2), ...], ...)
    """
    new_intervals = [
        dataclasses.replace(
            iv, caption_id=DEMO_CAPTION_ID if iv.is_demo else vocab.get(iv.template, UNK_CAPTION_ID)
        )
        for iv in table.intervals
    ]
    return dataclasses.replace(table, intervals=new_intervals)


def find_interval_at_step(table: SubgoalTable, step: int) -> SubgoalInterval | None:
    """
    What it does:
        Finds the interval containing `step` (the one with
        `start <= step < end`). Used anywhere a single step needs its active
        caption/event looked up -- e.g. subgoal_logger.
        snap_table_to_chunk_grid's replay, and the visual debugging tools
        (inspect_alignment.py / inspect_eval_alignment.py), which previously
        each carried their own copy of this exact lookup.

    Returns:
        SubgoalInterval | None -- the containing interval, or None if `step`
        falls outside every interval in `table` (should not happen for a
        well-formed table, which hard-partitions [0, episode_len)).

    Example input:
        find_interval_at_step(table, 57)

    Example output:
        SubgoalInterval(start=42, end=90, caption="pick up the cube", ...)
    """
    for iv in table.intervals:
        if iv.start <= step < iv.end:
            return iv
    return None


def truncate_table(table: SubgoalTable, now: int) -> SubgoalTable:
    """
    What it does:
        Causal truncation for one training/eval step: keeps only intervals
        with start <= now, clips the last kept interval's end to now+1 and
        marks it is_open=True, and renumbers event_idx fresh (0..k-1, in
        interval order) on every call. event_idx must never leak how many
        subgoals remain -- a stable per-episode ordinal would tell a
        counting-task model "this is subgoal #3 of N" through the index
        alone; caption_id (the task-global template id) is left untouched
        since it carries no positional information.

    Returns:
        SubgoalTable -- the causally-visible prefix of `table` as of `now`,
        with event_idx renumbered and the last interval marked open.

    Example input:
        truncate_table(full_table, now=57)

    Example output:
        SubgoalTable(intervals=[..., SubgoalInterval(42, 58, ..., is_open=True)], ...)
    """
    kept = [iv for iv in table.intervals if iv.start <= now]  # list[SubgoalInterval]
    assert kept, f"no interval starts at or before now={now}"
    kept[-1] = dataclasses.replace(kept[-1], end=now + 1, is_open=True)
    kept = [dataclasses.replace(iv, event_idx=i) for i, iv in enumerate(kept)]
    return dataclasses.replace(table, intervals=kept)


def assign_events_to_frames(
    table: SubgoalTable,
    sampled_step_indices: list[int],
    max_size: int,
    num_views: int = 1,
    token_per_image: int = 16,
) -> AlignedMemory:
    """
    What it does:
        Maps each perceptual-memory frame slot to the subgoal event
        containing it. `sampled_step_indices` is mem_buffer's raw (unpadded)
        indices_to_load; this right-pads slot_event_idx to `max_size` with
        PAD_EVENT_IDX exactly like mem_buffer's right_padding_token_emb
        right-pads the parallel img/pos/state/mask arrays, so the resulting
        token_event_idx stays index-aligned with static_mask once both are
        repeated to token granularity.

    Returns:
        AlignedMemory -- now=table's open interval's (end-1); step_indices=
        sampled_step_indices (raw, unpadded); slot_event_idx=int32[max_size]
        (PAD_EVENT_IDX in unused slots); token_event_idx=int32[max_size *
        num_views * token_per_image].

    Example input:
        assign_events_to_frames(truncated_table, [0, 5, 12, 18], max_size=32)

    Example output:
        AlignedMemory(now=18, ..., slot_event_idx=array([0,0,1,1,-1,...,-1]), ...)
    """
    n_real = min(len(sampled_step_indices), max_size)  # int
    slot_event_idx = np.full((max_size,), PAD_EVENT_IDX, dtype=np.int32)  # int32[max_size]
    # Pre-initialize EVERY truncated event to [] first (temporal-alignment-agent.md's own
    # requirement, Invariant 5) -- this is what structurally guarantees a zero-frame event's
    # event_idx is still a real key with an empty list, not simply absent from the dict.
    # Independent review (2026-09-20) found every current consumer happens to use .get(idx, [])
    # defensively so this had no live consequence yet, but the dict's own documented contract
    # ("event_idx -> sampled steps inside it", implying every truncated event has an entry) was
    # violated -- fixed here rather than left to luck for any future caller that trusts it.
    frames_per_event: dict[int, list[int]] = {iv.event_idx: [] for iv in table.intervals}

    for slot in range(n_real):
        step = sampled_step_indices[slot]  # int
        matched = None  # SubgoalInterval | None
        for iv in table.intervals:
            if iv.start <= step < iv.end:
                matched = iv
                break
        if matched is None:
            raise ValueError(f"step {step} not covered by any interval in truncated table")
        slot_event_idx[slot] = matched.event_idx
        frames_per_event[matched.event_idx].append(step)

    assert sum(len(v) for v in frames_per_event.values()) == n_real

    token_event_idx = np.repeat(slot_event_idx, num_views * token_per_image)  # int32[max_size*num_views*token_per_image]

    return AlignedMemory(
        now=table.intervals[-1].end - 1,
        events=table.intervals,
        step_indices=list(sampled_step_indices),
        frame_slots=list(range(max_size)),
        frames_per_event=frames_per_event,
        slot_event_idx=slot_event_idx,
        token_event_idx=token_event_idx,
    )


def _assert_invariants(table: SubgoalTable, aligned: AlignedMemory, packed: dict) -> None:
    """
    What it does:
        Checks the subset of temporal-alignment-agent.md's 10 invariants that
        are computable from `table`/`aligned`/`packed` alone (the
        `token_event_idx == -1 <=> ~static_mask` invariant needs static_mask,
        which lives outside this module -- Phase 7's CHECK8 checks that one
        at the integration level instead). Extended 2026-09-20 (independent
        review, see RESEARCH_LOG.md) to actually cover invariants 4 and 5,
        which are computable here but were previously silently unchecked
        despite this function's own docstring implying broader coverage
        than it had -- exactly the kind of self-claimed compliance that let
        the frames_per_event pre-init gap (fixed the same review round, see
        assign_events_to_frames) go undetected. Invariants 6/8/9 still
        aren't independently checked here (6/9 are implied by 5 now that
        frames_per_event is pre-initialized per-event; 8 needs the
        pre-truncation table, not available at this call site).

    Returns:
        None -- raises AssertionError on violation.

    Example input:
        _assert_invariants(truncated_table, aligned, packed_arrays)

    Example output:
        None (or raises)
    """
    intervals = table.intervals  # list[SubgoalInterval]
    assert intervals[0].start == 0, "first interval must start at 0"
    assert intervals[-1].is_open, "last (truncated) interval must be open"
    assert intervals[-1].end == aligned.now + 1, "open interval's end must be now+1"
    for i in range(len(intervals) - 1):
        assert intervals[i].end == intervals[i + 1].start, "intervals must hard-partition the episode"
    if not packed["event_overflow"]:
        assert int(packed["event_num_frames"].sum()) == len(aligned.step_indices), (
            "sum(event_num_frames) must equal the number of sampled frames when nothing overflowed"
        )
    # Invariant 4: a demo sentinel, if present, is always the first interval at event_idx 0.
    demo_intervals = [iv for iv in intervals if iv.is_demo]
    assert len(demo_intervals) <= 1, "at most one demo sentinel interval"
    if demo_intervals:
        assert intervals[0].is_demo and intervals[0].event_idx == 0, "demo sentinel must be interval 0"
    # Invariant 5: frames_per_event's key-set must equal every truncated interval's event_idx --
    # true by construction now that assign_events_to_frames pre-initializes every event to [].
    assert set(aligned.frames_per_event.keys()) == {iv.event_idx for iv in intervals}, (
        "frames_per_event must have one entry (possibly empty) per truncated event, no more/fewer"
    )


def pack_event_arrays(
    table: SubgoalTable, aligned: AlignedMemory, sp_tokenizer, E: int, L_c: int
) -> dict[str, np.ndarray]:
    """
    What it does:
        Builds the fixed-shape, E-padded tensors XF's EventEncoder consumes.
        Handles E_max overflow (len(intervals) > E): keeps the demo sentinel
        (if present) plus the most recent E-1 intervals, remaps
        aligned.token_event_idx from old to new (packed) event_idx, and tags
        any frame slot whose true event was dropped with OVERFLOW_EVENT_IDX
        (distinct from PAD_EVENT_IDX, which means "no frame here at all").
        Zero-frame events are kept as real, valid rows (event_mask=True,
        event_num_frames=0) -- never dropped for being empty; per
        temporal-alignment-agent.md this is "the single most important
        behavior", since filtering them out destroys exactly the memory
        content that makes symbolic memory win on counting tasks.

    Returns:
        dict[str, np.ndarray] -- event_mask (E,) bool, event_start/event_end
        (E,) int32, event_is_demo/event_is_open (E,) bool, event_caption_id
        (E,) int32, event_coords (E,2) int32 [(row,col) convention],
        event_num_frames (E,) int32, event_text_tokens (E,L_c) int32,
        event_text_mask (E,L_c) bool, static_token_event_idx
        (max_size*num_views*token_per_image,) int32, plus a plain python bool
        "event_overflow" (diagnostic only, not a fixed-shape model input).

    Example input:
        pack_event_arrays(truncated_table, aligned, sp_tokenizer, E=16, L_c=24)

    Example output:
        {"event_mask": array([True, True, False, ...]), "event_start": array([0, 42, 0, ...]), ...}
    """
    events = list(table.intervals)  # list[SubgoalInterval]
    overflow = len(events) > E  # bool
    dropped_event_indices: set[int] = set()

    if overflow:
        demo = [iv for iv in events if iv.is_demo]  # list[SubgoalInterval], 0 or 1
        rest = [iv for iv in events if not iv.is_demo]  # list[SubgoalInterval]
        keep_n = E - len(demo)  # int, may be <= 0 in pathological tiny-E cases
        kept_rest = rest[-keep_n:] if keep_n > 0 else []  # list[SubgoalInterval]
        dropped = rest[: len(rest) - len(kept_rest)]  # list[SubgoalInterval]
        dropped_event_indices = {iv.event_idx for iv in dropped}
        events = demo + kept_rest

    event_mask = np.zeros((E,), dtype=np.bool_)
    event_start = np.zeros((E,), dtype=np.int32)
    event_end = np.zeros((E,), dtype=np.int32)
    event_is_demo = np.zeros((E,), dtype=np.bool_)
    event_is_open = np.zeros((E,), dtype=np.bool_)
    event_caption_id = np.zeros((E,), dtype=np.int32)
    # -1 (not 0) is the "no bounding box" sentinel -- a real box can legitimately sit at
    # pixel (0, 0), and event_encoder.EventEncoder's has_coords check relies on this exact
    # convention (`jnp.all(event_coords >= 0, axis=-1)`) to tell the two cases apart.
    event_coords = np.full((E, 2), -1, dtype=np.int32)
    event_num_frames = np.zeros((E,), dtype=np.int32)
    event_text_tokens = np.zeros((E, L_c), dtype=np.int32)
    event_text_mask = np.zeros((E, L_c), dtype=np.bool_)

    old_to_new = {iv.event_idx: new_idx for new_idx, iv in enumerate(events)}  # dict[int, int]

    for new_idx, iv in enumerate(events):
        event_mask[new_idx] = True
        event_start[new_idx] = iv.start
        event_end[new_idx] = iv.end
        event_is_demo[new_idx] = iv.is_demo
        event_is_open[new_idx] = iv.is_open
        event_caption_id[new_idx] = iv.caption_id
        if iv.coords:
            # First bounding box only -- event_coords' fixed (E, 2) shape (matching gated-
            # fusion-agent.md section 2's own input table) only ever has room for one box per
            # event. A multi-object caption's later boxes stay available in event_text_tokens
            # (the caption text itself is untruncated), just not in this positional-code path.
            event_coords[new_idx] = iv.coords[0]
        event_num_frames[new_idx] = len(aligned.frames_per_event.get(iv.event_idx, []))
        tokens = sp_tokenizer.encode(iv.template, add_bos=False)[:L_c]  # list[int]
        event_text_tokens[new_idx, : len(tokens)] = tokens
        event_text_mask[new_idx, : len(tokens)] = True

    def _remap(idx: int) -> int:
        if idx == PAD_EVENT_IDX:
            return PAD_EVENT_IDX
        if idx in dropped_event_indices:
            return OVERFLOW_EVENT_IDX
        return old_to_new[idx]

    static_token_event_idx = np.array(
        [_remap(int(i)) for i in aligned.token_event_idx], dtype=np.int32
    )  # int32[max_size*num_views*token_per_image]

    result = {
        "event_mask": event_mask,
        "event_start": event_start,
        "event_end": event_end,
        "event_is_demo": event_is_demo,
        "event_is_open": event_is_open,
        "event_caption_id": event_caption_id,
        "event_coords": event_coords,
        "event_num_frames": event_num_frames,
        "event_text_tokens": event_text_tokens,
        "event_text_mask": event_text_mask,
        "static_token_event_idx": static_token_event_idx,
        "event_overflow": overflow,
    }
    _assert_invariants(table, aligned, result)
    return result
