"""
caption_shift.py

Timing-shift caption augmentation for the hybrid arm (the "verify caption progress against frame
memory" fix, 2026-10-06). For one training sample it decides whether to replace the caption the
sample shows with ANOTHER REAL caption from the same episode -- earlier or later in that
episode's subgoal timeline -- while the action label stays the true one. Samples where the
caption is mistimed can only be predicted correctly by checking progress in the frame memory,
which is what the hybrid fails to do on SwingXtimes (QwenVL says "press the button" 1-6 subgoals
early; RESEARCH_LOG.md 2026-10-06).

Only WHEN a caption is shown changes, never WHAT it says: every shifted caption is a real caption
of the same episode, with its real objects and coordinates. Captions that name a hidden target
("... that hides the ...") are never shifted to or from, so the identity information the unmask
tasks rely on is never contradicted.

Pure Python (no JAX/numpy), so the logic is unit-tested locally (tests/test_caption_shift.py).
The episode timeline comes from XF's subgoal_table.json (verified 400/400 identical to the
training captions, RESEARCH_LOG.md Phase 0).

Role in the system: training/hybrid_dataset.py calls choose_shift() per sample when the history
config has caption_shift.enabled: true.
"""

import dataclasses
import random
import re

# str, marks a caption that names the hidden target in the unmask families -- never shifted to/from.
TARGET_NAMING_PATTERN = re.compile(r"that hides the")
KINDS = ("early_one", "early_k", "terminal", "late_one")  # tuple[str, ...], shift kinds in draw order


@dataclasses.dataclass(frozen=True)
class ShiftConfig:
    """Rates and rules of the augmentation; built from the yaml's caption_shift block."""

    p_counting: float = 0.40  # float, base shift probability for counting tasks
    p_unmask: float = 0.10  # float, for the video/button unmask families
    p_other: float = 0.15  # float, for every other task
    p_overrides: tuple = (("StopCube", 0.30),)  # tuple[tuple[str, float], ...], per-task base probability
    counting_tasks: tuple = ("SwingXtimes", "PickXtimes", "BinFill", "StopCube")  # tuple[str, ...]
    unmask_tasks: tuple = ("VideoUnmask", "VideoUnmaskSwap", "ButtonUnmask", "ButtonUnmaskSwap")  # tuple[str, ...]
    near_end_events: int = 2  # int, "last N events" window where counting tasks' p is multiplied
    near_end_multiplier: float = 2.0  # float
    share_early_one: float = 0.35  # float
    share_early_k: float = 0.10  # float
    share_terminal: float = 0.40  # float, counting tasks only; elsewhere falls back to early_k
    share_late_one: float = 0.15  # float

    def __post_init__(self):
        """Rejects kind shares that do not sum to 1 (a silent mis-weighting otherwise)."""
        total = self.share_early_one + self.share_early_k + self.share_terminal + self.share_late_one  # float
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"caption_shift kind shares must sum to 1, got {total}")


def shift_config_from_yaml(block) -> ShiftConfig | None:
    """
    What it does:
        Builds a ShiftConfig from the history config's caption_shift block
        (an omegaconf node or plain dict). Returns None when the block is
        absent or enabled is false, i.e. the augmentation is OFF by default.

    Returns:
        ShiftConfig | None.

    Example input:
        shift_config_from_yaml({"enabled": True, "p_counting": 0.4, "p_overrides": {"StopCube": 0.3}})

    Example output:
        ShiftConfig(p_counting=0.4, ..., p_overrides=(("StopCube", 0.3),))
    """
    if block is None or not bool(block.get("enabled", False)):
        return None
    fields = {f.name for f in dataclasses.fields(ShiftConfig)}  # set[str]
    kwargs = {}  # dict[str, object]
    for key in block.keys():  # str
        if key == "enabled":
            continue
        if key not in fields:
            raise ValueError(f"unknown caption_shift key {key!r}")
        value = block[key]  # object
        if key == "p_overrides":
            value = tuple((str(k), float(v)) for k, v in dict(value).items())
        elif key in ("counting_tasks", "unmask_tasks"):
            value = tuple(str(v) for v in value)
        kwargs[key] = value
    return ShiftConfig(**kwargs)


def find_true_index(intervals: list[dict], shown_caption: str, step_idx: int) -> int | None:
    """
    What it does:
        Finds which execution interval the sample's shown caption belongs to.
        Matching is on the caption text (exact), because half the samples
        show the recorded caption, which can lag the online-view table by one
        interval. When the caption repeats in the episode (e.g. "put it into
        the bin"), the occurrence nearest to step_idx wins.

    Returns:
        int | None -- index into `intervals`, or None if the caption is not in the timeline.

    Example input:
        find_true_index([{"start": 0, "end": 50, "caption": "a"}, {"start": 50, "end": 90, "caption": "b"}], "b", 40)

    Example output:
        1
    """
    best, best_dist = None, None  # int | None, int | None
    for i, iv in enumerate(intervals):  # int, dict
        if iv["caption"] != shown_caption:
            continue
        dist = 0 if iv["start"] <= step_idx < iv["end"] else min(abs(step_idx - iv["start"]), abs(step_idx - (iv["end"] - 1)))  # int
        if best_dist is None or dist < best_dist:
            best, best_dist = i, dist
    return best


def choose_shift(task: str, intervals: list[dict], true_k: int, rng: random.Random, cfg: ShiftConfig) -> tuple[int, str]:
    """
    What it does:
        Decides the caption to show for a sample whose true caption is
        execution interval `true_k`:
          - base probability by task group (per-task override first), times
            near_end_multiplier for counting tasks within the last
            near_end_events events;
          - kind: early_one (next), early_k (2..remaining ahead), terminal
            (last event; counting tasks with >= 2 remaining, otherwise falls
            back to early_k), late_one (previous);
          - an impossible kind, or a shift to/from a target-naming caption,
            means no shift.
        `intervals` must be the EXECUTION intervals only (no video demo).

    Returns:
        tuple[int, str] -- (index of the caption to show, outcome label). The
        label is the kind applied, "none" (not triggered), or
        "blocked_<kind>" (triggered but impossible / protected).

    Example input:
        choose_shift("SwingXtimes", swing_intervals, 2, random.Random(0), ShiftConfig())

    Example output:
        (6, "terminal")
    """
    n = len(intervals)  # int
    remaining = n - 1 - true_k  # int, events after the true one
    overrides = dict(cfg.p_overrides)  # dict[str, float]
    is_counting = task in cfg.counting_tasks  # bool
    if task in overrides:
        p = overrides[task]  # float
    elif is_counting:
        p = cfg.p_counting
    elif task in cfg.unmask_tasks:
        p = cfg.p_unmask
    else:
        p = cfg.p_other
    if is_counting and true_k >= n - cfg.near_end_events:
        p *= cfg.near_end_multiplier
    if rng.random() >= min(p, 1.0):
        return true_k, "none"

    r = rng.random()  # float
    if r < cfg.share_early_one:
        kind, target = "early_one", (true_k + 1 if remaining >= 1 else None)  # str, int | None
    elif r < cfg.share_early_one + cfg.share_early_k:
        kind, target = "early_k", (true_k + rng.randint(2, remaining) if remaining >= 2 else None)
    elif r < cfg.share_early_one + cfg.share_early_k + cfg.share_terminal:
        if is_counting and remaining >= 2:
            kind, target = "terminal", n - 1
        else:
            kind, target = "early_k", (true_k + rng.randint(2, remaining) if remaining >= 2 else None)
    else:
        kind, target = "late_one", (true_k - 1 if true_k >= 1 else None)

    if target is None:
        return true_k, f"blocked_{kind}"
    if TARGET_NAMING_PATTERN.search(intervals[true_k]["caption"]) or TARGET_NAMING_PATTERN.search(intervals[target]["caption"]):
        return true_k, f"blocked_{kind}"
    return target, kind


class ShiftStats:
    """Running per-task counts of shift outcomes and of fake vs true SHOWN-terminal captions."""

    def __init__(self):
        """Starts empty."""
        self.outcomes = {}  # dict[str, dict[str, int]], task -> outcome label -> count
        self.terminal_shown = {}  # dict[str, list[int]], task -> [fake, total] samples showing the terminal caption
        self.samples = 0  # int

    def record(self, task: str, outcome: str, shown_k: int, true_k: int, n: int) -> None:
        """
        What it does: adds one sample's outcome to the counts.

        Returns:
            None.

        Example input:
            stats.record("SwingXtimes", "terminal", 6, 2, 7)

        Example output:
            None
        """
        self.samples += 1
        per_task = self.outcomes.setdefault(task, {})  # dict[str, int]
        per_task[outcome] = per_task.get(outcome, 0) + 1
        if shown_k == n - 1:
            fake_total = self.terminal_shown.setdefault(task, [0, 0])  # list[int]
            fake_total[1] += 1
            fake_total[0] += int(shown_k != true_k)

    def summary(self) -> str:
        """
        What it does: one line per task: shift rate, outcome counts, fake-terminal share.

        Returns:
            str -- multi-line report.

        Example input:
            stats.summary()

        Example output:
            "SwingXtimes: shifted 31.0% {...} fake-terminal 0.33 (40/121)"
        """
        lines = []  # list[str]
        for task in sorted(self.outcomes):  # str
            counts = self.outcomes[task]  # dict[str, int]
            total = sum(counts.values())  # int
            shifted = sum(v for k, v in counts.items() if k in KINDS)  # int
            fake, shown = self.terminal_shown.get(task, [0, 0])  # int, int
            frac = f"{fake / shown:.2f} ({fake}/{shown})" if shown else "n/a"  # str
            lines.append(f"{task}: shifted {100.0 * shifted / max(total, 1):.1f}% of {total} {dict(sorted(counts.items()))} fake-terminal {frac}")
        return "\n".join(lines)
