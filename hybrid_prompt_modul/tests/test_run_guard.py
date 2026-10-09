"""
test_run_guard.py

Local unit tests for training/run_guard.py: a preemption restart of the same Modal input
resumes from the LATEST checkpoint (also overriding an older explicit resume step); an explicit
resume needs an existing step; a different launch never silently deletes checkpoints; fresh
start when nothing exists.

Run from the repo root:
    python -m pytest hybrid_prompt_modul/tests/test_run_guard.py -q
"""

import pytest

from hybrid_prompt_modul.training.run_guard import decide_start


def test_preemption_restart_of_fresh_run_resumes_latest():
    # the 2026-10-07 case: fresh launch, checkpoint 2000 saved, preempted, restarted with the same input
    step, why = decide_start([2000], None, False, owner_id="in-1", my_id="in-1")  # int | None, str
    assert step == 2000 and "auto-resume" in why


def test_preemption_restart_of_resume_run_uses_latest_not_requested():
    # resumed from 2000, saved 4000, preempted: must continue from 4000, not redo 2000->4000
    step, _ = decide_start([2000, 4000], 2000, False, owner_id="in-2", my_id="in-2")  # int | None, str
    assert step == 4000


def test_explicit_resume():
    assert decide_start([2000], 2000, False, owner_id="in-1", my_id="in-2")[0] == 2000
    with pytest.raises(RuntimeError, match="not among saved steps"):
        decide_start([2000], 4000, False, owner_id=None, my_id="in-2")


def test_different_launch_never_deletes_checkpoints():
    with pytest.raises(RuntimeError, match="would DELETE"):
        decide_start([2000], None, False, owner_id="in-1", my_id="in-9")
    with pytest.raises(RuntimeError, match="would DELETE"):
        decide_start([2000], None, False, owner_id=None, my_id=None)  # no ids available -> still protected
    assert decide_start([2000], None, True, owner_id="in-1", my_id="in-9")[0] is None  # explicit overwrite only


def test_fresh_start_when_nothing_saved():
    assert decide_start([], None, False, owner_id=None, my_id="in-1") == (None, "fresh start")
    assert decide_start([], None, False, owner_id="in-1", my_id="in-1")[0] is None  # restarted before any save


def test_real_retry_ids_from_2026_10_07_log_match_after_suffix_strip():
    # the two attempts of ONE launch, copied from the real Modal log
    first, retry = "in-01M4BHYPES938HKZ117MCV7JZ6:1791389293039-0", "in-01M4BHYPES938HKZ117MCV7JZ6:1791389459242-0"
    assert first != retry  # full ids differ -> the original comparison could never match
    step, why = decide_start([2000, 4000], 2000, False, owner_id=first.split(":")[0], my_id=retry.split(":")[0])
    assert step == 4000 and "auto-resume" in why
