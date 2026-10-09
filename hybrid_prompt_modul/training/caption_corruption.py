"""
caption_corruption.py

General, task-agnostic caption-corruption augmentation for the hybrid arm (variant "general",
2026-10-07). Replaces the task-specific timing-shift rules (caption_shift.py, kept unchanged
for reproducibility) with ONE rule applied identically to every task:

  In each training episode, ~`coverage` of the execution steps fall inside corruption WINDOWS
  (random start, length min_len..max_len steps, non-overlapping). Each window is one of:
    - "held_shift": at the window's start, take the caption 1..max_shift subgoals EARLIER or
      LATER in the same episode's timeline and show that one caption, unchanged, for the whole
      window while the robot keeps progressing -- how a VLM caption source's mistakes actually
      behave (it lags or runs ahead and stays wrong for many chunks);
    - "dropout": show NO caption (empty string) for the whole window, so the frame memory must
      carry the task alone.
  The action label is never changed.

Why it is safe for caption-heavy tasks (user requirement): no window ever shows a caption with
the WRONG CONTENT -- held shifts are real captions of the same episode, and captions that name a
hidden target ("... that hides the ...") are never altered (same protection as caption_shift.py),
so the model is never taught that a correctly-timed caption names the wrong object.

Windows are a deterministic function of (seed, epis_idx), so every sample drawn from the same
episode sees the same windows, in any DataLoader worker.

Pure Python (no JAX/numpy); unit-tested locally in tests/test_caption_corruption.py.

Role in the system: training/hybrid_dataset.py calls corrupt_caption() per sample when the
history config has caption_corruption.enabled: true.
"""

import dataclasses
import random

from hybrid_prompt_modul.training.caption_shift import TARGET_NAMING_PATTERN

WINDOW_TYPES = ("held_shift", "dropout")  # tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class CorruptionConfig:
    """Rates of the general corruption; built from the yaml's caption_corruption block."""

    coverage: float = 0.20  # float, target share of execution steps inside corruption windows
    min_len: int = 50  # int, window length bounds in steps (clipped to the episode)
    max_len: int = 300  # int
    p_held_shift: float = 0.5  # float, window-type shares (sum to 1)
    p_dropout: float = 0.5  # float
    max_shift: int = 3  # int, held shifts are 1..max_shift subgoals, earlier or later
    seed: int = 0  # int, combined with epis_idx so windows are fixed per episode
    # Loud in-training guard (added 2026-10-07 after a run silently trained on 36-91% corruption):
    # after guard_after samples per DataLoader worker, every task with >= guard_min_samples samples
    # must have a corrupted share <= guard_max, and >= guard_min unless mostly protected.
    guard_after: int = 2000  # int
    guard_min_samples: int = 150  # int
    guard_min: float = 0.05  # float
    guard_max: float = 0.35  # float

    def __post_init__(self):
        """Rejects inconsistent settings instead of silently mis-weighting."""
        if abs(self.p_held_shift + self.p_dropout - 1.0) > 1e-6:
            raise ValueError("caption_corruption: p_held_shift + p_dropout must be 1")
        if not (0.0 <= self.coverage < 1.0) or self.min_len < 1 or self.max_len < self.min_len or self.max_shift < 1:
            raise ValueError(f"caption_corruption: invalid settings {self}")


def corruption_config_from_yaml(block) -> CorruptionConfig | None:
    """
    What it does:
        Builds a CorruptionConfig from the history config's caption_corruption
        block (omegaconf node or dict). None when absent or enabled is false
        (OFF by default). Unknown keys raise.

    Returns:
        CorruptionConfig | None.

    Example input:
        corruption_config_from_yaml({"enabled": True, "coverage": 0.2})

    Example output:
        CorruptionConfig(coverage=0.2, ...)
    """
    if block is None or not bool(block.get("enabled", False)):
        return None
    fields = {f.name for f in dataclasses.fields(CorruptionConfig)}  # set[str]
    kwargs = {}  # dict[str, object]
    for key in block.keys():  # str
        if key == "enabled":
            continue
        if key not in fields:
            raise ValueError(f"unknown caption_corruption key {key!r}")
        kwargs[key] = block[key]
    return CorruptionConfig(**kwargs)


def interval_at(intervals: list[dict], step: int) -> int | None:
    """
    What it does: index of the execution interval containing `step`.

    Returns:
        int | None -- None if the step is outside the execution timeline.

    Example input:
        interval_at([{"start": 0, "end": 50}, {"start": 50, "end": 90}], 60)

    Example output:
        1
    """
    for i, iv in enumerate(intervals):  # int, dict
        if iv["start"] <= step < iv["end"]:
            return i
    return None


def build_windows(epis_idx: int, intervals: list[dict], cfg: CorruptionConfig) -> list[dict]:
    """
    What it does:
        Deterministically (rng seeded by cfg.seed and epis_idx) places
        non-overlapping windows over the episode's execution steps until about
        cfg.coverage of them are covered, and fixes each window's type and,
        for held shifts, the caption index shown for the whole window
        (interval at the window start + a shift of 1..max_shift, sign random).
        A held shift with no valid target index becomes no window.

    Returns:
        list[dict] -- [{"start", "end", "type", "shown_k" (held_shift only)}], sorted by start.

    Example input:
        build_windows(802, swing_intervals, CorruptionConfig())

    Example output:
        [{"start": 140, "end": 262, "type": "held_shift", "shown_k": 4}, {"start": 400, "end": 455, "type": "dropout"}]
    """
    if not intervals:
        return []
    rng = random.Random(cfg.seed * 1_000_003 + epis_idx)  # random.Random
    lo, hi = intervals[0]["start"], intervals[-1]["end"]  # int, int, execution step range
    target = cfg.coverage * (hi - lo)  # float, steps to cover
    windows, covered = [], 0  # list[dict], int
    for _ in range(50):  # bounded attempts
        remaining = target - covered  # float, steps still to cover
        if remaining <= 0:
            break
        length = min(rng.randint(cfg.min_len, cfg.max_len), hi - lo)  # int
        # A window longer than what is left would overshoot (fixed 2026-10-07: absolute 50-300-step
        # windows covered 1/3 to all of short episodes, giving 36-91% corruption instead of ~20%).
        # Place it only with probability remaining/length and stop, so EXPECTED coverage equals the
        # target at any episode length (short episodes get one window or none).
        last = length > remaining  # bool
        if last and rng.random() >= remaining / length:
            break
        start = rng.randint(lo, hi - length)  # int
        end = start + length  # int
        if any(start < w["end"] and w["start"] < end for w in windows):
            if last:
                break
            continue
        held = rng.random() < cfg.p_held_shift  # bool
        if held:
            k0 = interval_at(intervals, start)  # int | None
            # Valid held targets: 1..max_shift subgoals away, inside the timeline. When the drawn shift
            # would fall outside (common in 1-3-subgoal tasks), pick a valid one instead; with none,
            # the window becomes a dropout window -- skipping it under-covered those tasks (~7-12%).
            valid = [] if k0 is None else [k0 + d * s for d in range(1, cfg.max_shift + 1) for s in (-1, 1)
                                           if 0 <= k0 + d * s < len(intervals)]  # list[int]
            if valid:
                windows.append({"start": start, "end": end, "type": "held_shift", "shown_k": rng.choice(valid)})
            else:
                held = False
        if not held:
            windows.append({"start": start, "end": end, "type": "dropout"})
        covered += length
        if last:
            break
    return sorted(windows, key=lambda w: w["start"])


def corrupt_caption(intervals: list[dict], windows: list[dict], step: int, caption: str) -> tuple[str, str]:
    """
    What it does:
        Returns the caption to SHOW for a sample at `step`: unchanged outside
        windows; inside a held_shift window the window's fixed caption;
        inside a dropout window "". Never alters a sample whose true caption
        or would-be shown caption names a hidden target.

    Returns:
        tuple[str, str] -- (caption to show, outcome: "none" | "held_shift" | "dropout" | "protected").

    Example input:
        corrupt_caption(swing_intervals, [{"start": 100, "end": 200, "type": "dropout"}], 150, "move to ...")

    Example output:
        ("", "dropout")
    """
    window = next((w for w in windows if w["start"] <= step < w["end"]), None)  # dict | None
    if window is None:
        return caption, "none"
    shown = "" if window["type"] == "dropout" else intervals[window["shown_k"]]["caption"]  # str
    if TARGET_NAMING_PATTERN.search(caption or "") or TARGET_NAMING_PATTERN.search(shown):
        return caption, "protected"
    return shown, window["type"]


class CorruptionStats:
    """Running per-task counts of corruption outcomes."""

    def __init__(self):
        """Starts empty."""
        self.outcomes = {}  # dict[str, dict[str, int]]
        self.samples = 0  # int

    def record(self, task: str, outcome: str) -> None:
        """
        What it does: adds one sample's outcome.

        Returns:
            None.

        Example input:
            stats.record("SwingXtimes", "dropout")

        Example output:
            None
        """
        self.samples += 1
        per_task = self.outcomes.setdefault(task, {})  # dict[str, int]
        per_task[outcome] = per_task.get(outcome, 0) + 1

    def check(self, cfg: CorruptionConfig) -> None:
        """
        What it does:
            Raises RuntimeError if any task with at least cfg.guard_min_samples
            samples has a corrupted share above cfg.guard_max, or if its
            windows hit fewer than cfg.guard_min of its samples (corrupted +
            protected: a protected sample is a window hit deliberately blocked
            to keep target-naming captions intact -- on the real data
            VideoUnmask is 1.2% corrupted + 17.8% protected).

        Returns:
            None.

        Example input:
            stats.check(CorruptionConfig())

        Example output:
            None, or RuntimeError("caption_corruption guard: PatternLock corrupted 91.0% > 35%")
        """
        problems = []  # list[str]
        for task, counts in self.outcomes.items():  # str, dict[str, int]
            total = sum(counts.values())  # int
            if total < cfg.guard_min_samples:
                continue
            share = sum(v for k, v in counts.items() if k in WINDOW_TYPES) / total  # float
            protected = counts.get("protected", 0) / total  # float
            if share > cfg.guard_max:
                problems.append(f"{task} corrupted {share:.1%} > {cfg.guard_max:.0%}")
            elif share + protected < cfg.guard_min:
                problems.append(f"{task} corrupted {share:.1%} (+{protected:.1%} protected) < {cfg.guard_min:.0%}")
        if problems:
            raise RuntimeError("caption_corruption guard: " + "; ".join(problems) + "\n" + self.summary())

    def summary(self) -> str:
        """
        What it does: one line per task with the corrupted share and outcome counts.

        Returns:
            str -- multi-line report.

        Example input:
            stats.summary()

        Example output:
            "SwingXtimes: corrupted 19.8% of 216 {'dropout': 21, 'held_shift': 22, 'none': 173}"
        """
        lines = []  # list[str]
        for task in sorted(self.outcomes):  # str
            counts = self.outcomes[task]  # dict[str, int]
            total = sum(counts.values())  # int
            hit = sum(v for k, v in counts.items() if k in WINDOW_TYPES)  # int
            lines.append(f"{task}: corrupted {100.0 * hit / max(total, 1):.1f}% of {total} {dict(sorted(counts.items()))}")
        return "\n".join(lines)
