"""
run_guard.py

Start-up decision for a training run: fresh start, explicit resume, AUTOMATIC resume after a
Modal preemption, or refusal. Added 2026-10-07 after a preemption at step ~2,730 made Modal
restart the general-v2 run "with the same input" -- i.e. as a fresh run, which would have
deleted checkpoint 2000 (the existing overwrite guard refused, so the run stopped instead).

Rules, in order:
  1. this same Modal input owns the run (owner id == my id) and checkpoints exist
     -> resume from the LATEST checkpoint (a preemption restart; never lose saved progress);
  2. an explicit --resum-ckpt-id N -> resume from N, which must exist;
  3. checkpoints exist and this is a different launch asking for a fresh start
     -> refuse unless allow_overwrite;
  4. otherwise -> fresh start.
The owner id is the Modal input id WITHOUT its retry suffix (the part before ":"): verified in a
real log 2026-10-07 to be identical across the retries of one launch, while the full input id
gets a new suffix per retry (so the first version, comparing full ids, never matched). Stored in a
file NEXT TO the run's checkpoint directory -- a fresh start deletes the directory itself.

Pure Python, unit-tested in tests/test_run_guard.py.

Role in the system: launch_hybrid_training._run calls decide_start() before scripts/train.py.
"""


def decide_start(existing_steps: list[int], requested_resume: int | None, allow_overwrite: bool,
                 owner_id: str | None, my_id: str | None) -> tuple[int | None, str]:
    """
    What it does:
        Applies the rules in the module docstring.

    Returns:
        tuple[int | None, str] -- (step to resume from, or None for a fresh start; a one-line
        reason to print). Raises RuntimeError when the start must be refused.

    Example input:
        decide_start([2000, 4000], None, False, owner_id="in-abc", my_id="in-abc")

    Example output:
        (4000, "auto-resume: same Modal input restarted (preemption); resuming from latest checkpoint 4000")
    """
    latest = max(existing_steps) if existing_steps else None  # int | None
    if my_id is not None and owner_id == my_id and latest is not None:
        return latest, f"auto-resume: same Modal input restarted (preemption); resuming from latest checkpoint {latest}"
    if requested_resume is not None:
        if requested_resume not in existing_steps:
            raise RuntimeError(f"--resum-ckpt-id {requested_resume} not among saved steps {existing_steps}")
        return requested_resume, f"explicit resume from checkpoint {requested_resume}"
    if existing_steps and not allow_overwrite:
        raise RuntimeError(
            f"run already has checkpoints {existing_steps}; a fresh run would DELETE them. Resume with "
            f"--resum-ckpt-id {latest}, or pass --allow-overwrite if deleting them is intended."
        )
    return None, "fresh start" + (" (allow_overwrite: existing checkpoints will be deleted)" if existing_steps else "")
