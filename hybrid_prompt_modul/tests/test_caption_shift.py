"""
test_caption_shift.py

Local unit tests for training/caption_shift.py (pure Python; no JAX, GPU or Modal). Covers the
Phase 1 checklist of the timing-shift plan: neighbour choice per kind, staying inside the
execution timeline, blocking impossible shifts, terminal jumps only in counting tasks,
never shifting target-naming captions, trigger rates within tolerance, the true caption when
not triggered, nearest-occurrence matching of repeated captions, the yaml switch (off by
default), and the fake-terminal accounting.

Run from the repo root:
    python -m pytest hybrid_prompt_modul/tests/test_caption_shift.py -q
"""

import random

import pytest

from hybrid_prompt_modul.training.caption_shift import (
    ShiftConfig,
    ShiftStats,
    choose_shift,
    find_true_index,
    shift_config_from_yaml,
)

# list[str], the real SwingXtimes timeline shape (Phase 0.2): 7 execution events, terminal "press the button".
SWING_CAPTIONS = [
    "pick up the red cube at <64, 89>",
    "move to the top of the right-side target at <95, 76> for the first time",
    "move to the top of the left-side target at <75, 183> for the first time",
    "move to the top of the right-side target at <95, 76> for the second time",
    "move to the top of the left-side target at <75, 183> for the second time",
    "put the red cube on the table",
    "press the button",
]


def make_intervals(captions: list[str], length: int = 50) -> list[dict]:
    """
    What it does: builds consecutive execution intervals of equal length for the given captions.

    Returns:
        list[dict] -- [{"start", "end", "caption"}, ...].

    Example input:
        make_intervals(["a", "b"], 10)

    Example output:
        [{"start": 0, "end": 10, "caption": "a"}, {"start": 10, "end": 20, "caption": "b"}]
    """
    return [{"start": i * length, "end": (i + 1) * length, "caption": c} for i, c in enumerate(captions)]


def forced(kind_share: str) -> ShiftConfig:
    """
    What it does: config that always triggers and always draws one kind (for neighbour tests).

    Returns:
        ShiftConfig.

    Example input:
        forced("late_one")

    Example output:
        ShiftConfig(p_counting=1.0, ..., share_late_one=1.0, other shares 0)
    """
    shares = {"share_early_one": 0.0, "share_early_k": 0.0, "share_terminal": 0.0, "share_late_one": 0.0}  # dict[str, float]
    shares[f"share_{kind_share}"] = 1.0
    return ShiftConfig(p_counting=1.0, p_unmask=1.0, p_other=1.0, p_overrides=(), near_end_multiplier=1.0, **shares)


SWING = make_intervals(SWING_CAPTIONS)  # list[dict]


def test_early_one_picks_next():
    assert choose_shift("SwingXtimes", SWING, 2, random.Random(0), forced("early_one")) == (3, "early_one")


def test_late_one_picks_previous():
    assert choose_shift("SwingXtimes", SWING, 2, random.Random(0), forced("late_one")) == (1, "late_one")


def test_early_k_stays_within_remaining():
    rng = random.Random(1)  # random.Random
    for _ in range(500):
        shown, kind = choose_shift("SwingXtimes", SWING, 1, rng, forced("early_k"))  # int, str
        assert kind == "early_k" and 3 <= shown <= 6


def test_terminal_jump_counting_only():
    assert choose_shift("SwingXtimes", SWING, 1, random.Random(0), forced("terminal")) == (6, "terminal")
    rng = random.Random(2)  # random.Random
    for _ in range(200):  # non-counting task: terminal draw falls back to early_k
        shown, kind = choose_shift("RouteStick", SWING, 1, rng, forced("terminal"))  # int, str
        assert kind == "early_k" and 3 <= shown <= 6


def test_terminal_needs_two_remaining_else_falls_back():
    # true_k = 5 -> only 1 event remains: terminal not allowed, early_k impossible too -> blocked
    assert choose_shift("SwingXtimes", SWING, 5, random.Random(0), forced("terminal")) == (5, "blocked_early_k")


def test_impossible_shifts_are_blocked():
    assert choose_shift("SwingXtimes", SWING, 6, random.Random(0), forced("early_one")) == (6, "blocked_early_one")
    assert choose_shift("SwingXtimes", SWING, 0, random.Random(0), forced("late_one")) == (0, "blocked_late_one")
    one = make_intervals(["pick up the cube"])  # list[dict]
    for kind in ("early_one", "early_k", "terminal", "late_one"):  # str
        assert choose_shift("BinFill", one, 0, random.Random(0), forced(kind))[0] == 0


def test_shown_index_always_inside_execution_timeline():
    rng = random.Random(3)  # random.Random
    cfg = ShiftConfig()  # ShiftConfig
    for _ in range(5000):
        k = rng.randrange(len(SWING))  # int
        shown, _ = choose_shift("SwingXtimes", SWING, k, rng, cfg)  # int, str
        assert 0 <= shown < len(SWING)


def test_target_naming_captions_never_shifted():
    caps = ["press the button", "pick up the container that hides the green cube", "put down the container",
            "pick up the container that hides the blue cube"]  # list[str]
    ivs = make_intervals(caps)  # list[dict]
    rng = random.Random(4)  # random.Random
    for _ in range(3000):
        k = rng.randrange(len(ivs))  # int
        shown, kind = choose_shift("ButtonUnmask", ivs, k, rng, forced(rng.choice(["early_one", "late_one", "early_k"])))  # int, str
        if shown != k:
            assert "that hides the" not in ivs[k]["caption"] and "that hides the" not in ivs[shown]["caption"]


def _trigger_rate(task: str, ivs: list[dict], k: int, n: int = 40000) -> float:
    """Fraction of samples where a shift was triggered (applied or blocked)."""
    rng = random.Random(5)  # random.Random
    cfg = ShiftConfig()  # ShiftConfig
    hits = sum(choose_shift(task, ivs, k, rng, cfg)[1] != "none" for _ in range(n))  # int
    return hits / n


@pytest.mark.parametrize("task,k,expected", [
    ("SwingXtimes", 1, 0.40),   # counting, not near the end
    ("SwingXtimes", 5, 0.80),   # counting, last 2 events -> doubled
    ("StopCube", 0, 0.30),      # per-task override (StopCube has 3 events; k=0 is not near the end)
    ("RouteStick", 1, 0.15),    # other
    ("VideoUnmaskSwap", 1, 0.10),  # unmask family
])
def test_trigger_rates_within_tolerance(task, k, expected):
    ivs = SWING if task != "StopCube" else make_intervals(["move to the top of the button to prepare", "remain static",
                                                             "press the button to stop the cube on the target"])  # list[dict]
    assert abs(_trigger_rate(task, ivs, k) - expected) < 0.01


def test_true_caption_when_not_triggered():
    cfg = ShiftConfig(p_counting=0.0, p_unmask=0.0, p_other=0.0, p_overrides=())  # ShiftConfig
    assert choose_shift("SwingXtimes", SWING, 3, random.Random(0), cfg) == (3, "none")


def test_find_true_index_nearest_repeat():
    caps = ["pick up the first red cube", "put it into the bin", "pick up the second red cube", "put it into the bin"]  # list[str]
    ivs = make_intervals(caps, 10)  # list[dict]; "put it into the bin" at [10,20) and [30,40)
    assert find_true_index(ivs, "put it into the bin", 12) == 1
    assert find_true_index(ivs, "put it into the bin", 33) == 3
    assert find_true_index(ivs, "put it into the bin", 26) == 3  # recorded caption lagging: nearest occurrence
    assert find_true_index(ivs, "not in this episode", 5) is None


def test_yaml_switch_off_by_default_and_strict():
    assert shift_config_from_yaml(None) is None
    assert shift_config_from_yaml({"enabled": False, "p_counting": 0.9}) is None
    cfg = shift_config_from_yaml({"enabled": True, "p_overrides": {"StopCube": 0.3}, "counting_tasks": ["SwingXtimes"]})  # ShiftConfig
    assert cfg.p_overrides == (("StopCube", 0.3),) and cfg.counting_tasks == ("SwingXtimes",)
    with pytest.raises(ValueError):
        shift_config_from_yaml({"enabled": True, "p_countng": 0.4})  # typo must not be silently ignored
    with pytest.raises(ValueError):
        ShiftConfig(share_terminal=0.5)  # shares no longer sum to 1


def test_stats_fake_terminal_accounting():
    stats = ShiftStats()  # ShiftStats
    stats.record("SwingXtimes", "terminal", 6, 2, 7)  # fake terminal
    stats.record("SwingXtimes", "none", 6, 6, 7)  # true terminal
    stats.record("SwingXtimes", "early_one", 3, 2, 7)  # not terminal
    assert stats.terminal_shown["SwingXtimes"] == [1, 2]
    assert "fake-terminal 0.50 (1/2)" in stats.summary()
