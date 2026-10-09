"""
test_caption_corruption.py

Local unit tests for training/caption_corruption.py (pure Python). Covers: windows are
deterministic per episode, non-overlapping and within the length bounds; overall coverage close
to the configured share; a held-shift window shows ONE fixed real caption 1..max_shift subgoals
away for its whole length; dropout shows ""; target-naming captions are never altered; samples
outside windows are untouched; the yaml switch is off by default and strict.

Run from the repo root:
    python -m pytest hybrid_prompt_modul/tests/test_caption_corruption.py -q
"""

import pytest

from hybrid_prompt_modul.training.caption_corruption import (
    CorruptionConfig,
    build_windows,
    corrupt_caption,
    corruption_config_from_yaml,
    interval_at,
)


def make_intervals(captions: list[str], length: int = 60, start: int = 40) -> list[dict]:
    """
    What it does: consecutive execution intervals of equal length starting at `start`.

    Returns:
        list[dict] -- [{"start", "end", "caption"}, ...].

    Example input:
        make_intervals(["a", "b"], 10, 0)

    Example output:
        [{"start": 0, "end": 10, "caption": "a"}, {"start": 10, "end": 20, "caption": "b"}]
    """
    return [{"start": start + i * length, "end": start + (i + 1) * length, "caption": c} for i, c in enumerate(captions)]


SWING = make_intervals([f"swing caption {i}" for i in range(7)])  # list[dict], 420 execution steps


def test_windows_deterministic_nonoverlapping_bounded():
    cfg = CorruptionConfig()  # CorruptionConfig
    for epis in range(200):  # int
        a, b = build_windows(epis, SWING, cfg), build_windows(epis, SWING, cfg)  # list[dict], list[dict]
        assert a == b
        for w in a:  # dict
            assert cfg.min_len <= w["end"] - w["start"] <= cfg.max_len
            assert SWING[0]["start"] <= w["start"] and w["end"] <= SWING[-1]["end"]
        for x, y in zip(a, a[1:]):  # dict, dict
            assert x["end"] <= y["start"]


def _mean_coverage(intervals: list[dict], n_episodes: int, cfg: CorruptionConfig) -> float:
    """Mean share of execution steps inside windows over many episodes (dropout-only, so no window is skipped)."""
    total = intervals[-1]["end"] - intervals[0]["start"]  # int
    shares = [sum(w["end"] - w["start"] for w in build_windows(e, intervals, cfg)) / total for e in range(n_episodes)]  # list[float]
    return sum(shares) / len(shares)


@pytest.mark.parametrize("n_events,length", [
    (20, 100),  # long episode, 2000 steps
    (3, 50),    # short episode, 150 steps -- the case that broke the first "general" run (36-91% corrupted)
    (2, 40),    # very short, 80 steps (shorter than the minimum window)
    (7, 60),    # SwingXtimes-like, 420 steps
])
def test_expected_coverage_matches_target_at_any_length(n_events, length):
    cfg = CorruptionConfig(p_held_shift=0.0, p_dropout=1.0)  # CorruptionConfig
    ivs = make_intervals([f"c{i}" for i in range(n_events)], length=length)  # list[dict]
    assert abs(_mean_coverage(ivs, 4000, cfg) - 0.20) < 0.02


def test_held_shift_fixed_real_caption_within_range():
    cfg = CorruptionConfig(p_held_shift=1.0, p_dropout=0.0)  # CorruptionConfig
    seen = 0  # int
    for epis in range(300):  # int
        for w in build_windows(epis, SWING, cfg):  # dict
            assert w["type"] == "held_shift"
            k0 = interval_at(SWING, w["start"])  # int
            assert 1 <= abs(w["shown_k"] - k0) <= cfg.max_shift
            shown = {corrupt_caption(SWING, [w], s, SWING[interval_at(SWING, s)]["caption"])[0]
                     for s in range(w["start"], w["end"])}  # set[str]
            assert shown == {SWING[w["shown_k"]]["caption"]}  # one caption, held for the whole window
            seen += 1
    assert seen > 100


def test_dropout_and_outside_windows():
    windows = [{"start": 100, "end": 200, "type": "dropout"}]  # list[dict]
    assert corrupt_caption(SWING, windows, 150, "swing caption 1") == ("", "dropout")
    assert corrupt_caption(SWING, windows, 99, "swing caption 0") == ("swing caption 0", "none")
    assert corrupt_caption(SWING, windows, 200, "swing caption 2") == ("swing caption 2", "none")


def test_target_naming_never_altered():
    caps = ["press the button", "pick up the container that hides the green cube", "put down the container"]  # list[str]
    ivs = make_intervals(caps)  # list[dict]
    true_named = corrupt_caption(ivs, [{"start": 0, "end": 1000, "type": "dropout"}], 110, caps[1])  # tuple
    assert true_named == (caps[1], "protected")
    shown_named = corrupt_caption(ivs, [{"start": 0, "end": 1000, "type": "held_shift", "shown_k": 1}], 50, caps[0])  # tuple
    assert shown_named == (caps[0], "protected")


def test_yaml_switch_off_by_default_and_strict():
    assert corruption_config_from_yaml(None) is None
    assert corruption_config_from_yaml({"enabled": False}) is None
    assert corruption_config_from_yaml({"enabled": True, "coverage": 0.1}).coverage == 0.1
    with pytest.raises(ValueError):
        corruption_config_from_yaml({"enabled": True, "coverag": 0.1})
    with pytest.raises(ValueError):
        CorruptionConfig(p_held_shift=0.7, p_dropout=0.5)


def test_guard_raises_on_off_intent_rates():
    from hybrid_prompt_modul.training.caption_corruption import CorruptionStats

    cfg = CorruptionConfig()  # CorruptionConfig
    ok = CorruptionStats()  # CorruptionStats
    for i in range(200):
        ok.record("SwingXtimes", "dropout" if i % 5 == 0 else "none")  # 20% corrupted
        ok.record("VideoUnmask", "protected" if i % 5 == 0 else "none")  # 0% corrupted, 20% protected (real: 1.2% + 17.8%)
        ok.record("ButtonUnmask", "none") if i < 20 else None  # too few samples to judge
    ok.check(cfg)  # must not raise
    bad = CorruptionStats()  # CorruptionStats
    for i in range(200):
        bad.record("PatternLock", "dropout" if i % 10 else "none")  # 90% corrupted -- the 2026-10-07 bug
    with pytest.raises(RuntimeError, match="PatternLock corrupted 90.0%"):
        bad.check(cfg)
    low = CorruptionStats()  # CorruptionStats
    for i in range(200):
        low.record("MoveCube", "none")  # 0%, not protected -> augmentation silently off
    with pytest.raises(RuntimeError, match="MoveCube corrupted 0.0%"):
        low.check(cfg)
