"""
check_eval_load_path.py

Answers, before any checkpoint is pushed to HF Hub or any eval episode is
spent on a second Modal account: can the XF EVAL path actually load the
trained step-18000 checkpoint at all?

Why this exists. `xattn_fusion/mme_vla_suite/policies/xf_policy_config.py`'s
`create_xf_trained_policy` is the factory behind `xf_serve_policy.py`, which
is what serves the policy to `examples/robomme/eval.py`. Every eval row in
the XF eval plan -- including R0, the harness sanity check that makes all the
others interpretable -- goes through it. It has never been run against a real
trained XF checkpoint: `xf_serve_policy.py`'s own module docstring notes the
v1 smoke test deliberately builds an XFPolicy around a randomly-initialized
model instead, skipping this factory entirely.

Two specific, independent reasons to expect it to fail, both found by reading
the code on 2026-09-22 after the same class of bug cost four launch attempts
of `measure_fusion_message.py`:

  1. `xf_policy_config.py:39` does `from openpi.training.weight_loaders
     import _merge_params` -- a direct name binding captured at IMPORT time.
     `launch_xf_training._patch_scripts_train_for_xf`'s "Bug E" fix rebinds
     the module ATTRIBUTE (`_weight_loaders._merge_params = ...`), which does
     not retroactively change a name another module already imported. And
     nothing under `xattn_fusion/eval/` applies that patch at all. So the
     eval path should run the RELEASED `_merge_params`, whose
     `flax.traverse_util.flatten_dict(params, sep="/")` crashes trying to
     `sep.join` a non-str path component -- and `fusion.blocks` /
     `event_encoder.layers` are plain Python lists, so their submodules carry
     INTEGER indices. That is the original Bug E, unfixed on this path.

  2. Even if (1) were survived, `xf_policy_config.py:100` then calls
     `train_config.model.load(merged_params)`. That is the exact call that
     failed repeatedly on 2026-09-22: `BaseModelConfig.load` runs a strict
     `at.check_pytree_equality` BEFORE `state.replace_by_pure_dict` (the
     function that reconciles orbax's string list-indices with nnx's int
     ones), and its `remove_extra_params` branch runs
     `ocp.transform_utils.intersect_trees`, which re-stringifies keys even if
     they were normalized beforehand.

Why it matters operationally: under the eval plan's own reading table, a
broken load presents as "R0 != ~44.5% => the eval path is broken => every
other row is garbage." Debugging that on a second account, after paying to
upload a ~12GB checkpoint, would be considerably more expensive than this
CPU-only check.

Structure. CHECK 1 is nearly free: it builds the model ABSTRACTLY via
`nnx.eval_shape` (shapes only -- the ~3B parameters are never materialized)
and calls the released `_merge_params` on that param tree. The suspected
crash is in `flatten_dict(params, sep="/")`, which depends only on the tree's
key types, not on any values, so an abstract tree exercises it exactly. CHECK
2 is the honest end-to-end test -- the real `create_xf_trained_policy`
against the real checkpoint -- and needs real memory. CHECK 1 runs first so
its answer survives even if CHECK 2 exhausts RAM.

Neither check needs a GPU: this is about tree structure and weight loading,
not compute.

Role in the system: read-only pre-flight for the XF 18k eval. Writes nothing
to any volume and never edits robomme_policy_learning/.

Run with:
    modal run XF_18k_eval/analysis/check_eval_load_path.py
"""

import pathlib

import modal

# str, str -- released policy repo, and this project's XF code.
POLICY_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning")
XF_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "xattn_fusion")

# str -- mount points, MUST match launch_xf_training.py exactly.
CKPT_VOLUME_PATH = "/ckpts"
MAIN_DATA_VOLUME_PATH = "/xf_data"
TRAINING_VOLUME_PATH = "/xf_training"
FEATURE_SHARD_PATHS = [f"/xf_features_shard_{i}" for i in range(4)]

REPO_ID = "xf_full_suite"  # str, must match launch_xf_training.py
EXP_NAME = "full-16task-xattn-fusion"  # str, ditto
CKPT_STEP = 18000  # int

app = modal.App("xf-check-eval-load-path")  # modal.App

ckpt_volume = modal.Volume.from_name("robomme-mme-vla-ckpts")  # modal.Volume
main_data_volume = modal.Volume.from_name("xf-full-suite-data")  # modal.Volume
training_volume = modal.Volume.from_name("xf-full-suite-training")  # modal.Volume
feature_shard_volumes = [modal.Volume.from_name(f"xf-features-shard-{i}") for i in range(4)]  # list[modal.Volume]

volumes = {
    CKPT_VOLUME_PATH: ckpt_volume,
    MAIN_DATA_VOLUME_PATH: main_data_volume,
    TRAINING_VOLUME_PATH: training_volume,
    **{path: vol for path, vol in zip(FEATURE_SHARD_PATHS, feature_shard_volumes)},
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "curl", "libgl1", "libglib2.0-0")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({
        "UV_LINK_MODE": "copy", "UV_PYTHON_DOWNLOADS": "automatic",
        "UV_PROJECT_ENVIRONMENT": "/usr/local",
    })
    .add_local_dir(POLICY_LOCAL_DIR, remote_path="/app", copy=True)
    .run_commands(
        r"""sed -i 's/members = \["packages\/\*", "sandbox2\/flash_attn_jax"\]/members = ["packages\/*"]/' /app/pyproject.toml"""
    )
    .run_commands("cd /app && /root/.local/bin/uv sync --no-dev --python 3.11")
    .run_commands("cd /app && /root/.local/bin/uv pip install --system pytest wandb")
    .add_local_dir(XF_LOCAL_DIR, remote_path="/xf_root/xattn_fusion", copy=True)
)


@app.function(image=image, gpu=None, timeout=3600, memory=98304, cpu=8.0, volumes=volumes)
def check_load_path() -> dict:
    """
    What it does:
        Runs the two checks described in this module's docstring: an
        abstract-tree exercise of the released `_merge_params` (CHECK 1,
        nearly free), then the real `create_xf_trained_policy` against
        checkpoint 18000 (CHECK 2, the end-to-end truth). Each check is
        individually guarded so one failing still reports the other's
        outcome, and each records the exception type and message rather
        than just "failed" -- the distinction between the two predicted
        failure modes is the whole point.

    Returns:
        dict -- {"check1": {...}, "check2": {...}, "notes": list[str]}, each
        check dict having "ok" (bool) and either "detail" (str) or
        "error_type"/"error" (str).

    Example input:
        check_load_path.remote()

    Example output:
        {"check1": {"ok": False, "error_type": "TypeError",
                    "error": "sequence item 1: expected str instance, int found"},
         "check2": {"ok": False, "error_type": "TypeError", "error": "..."},
         "notes": ["..."]}
    """
    import os  # module
    import sys  # module
    import traceback  # module

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/src")
    sys.path.insert(0, "/xf_root")
    os.chdir("/app")  # mme_vla_suite's config loader resolves paths against the process CWD

    import flax.nnx as nnx  # module
    import jax  # module

    for vol in [ckpt_volume, training_volume, main_data_volume, *feature_shard_volumes]:
        vol.reload()

    from xattn_fusion.training.launch_xf_training import _build_train_config

    result = {"check1": {}, "check2": {}, "notes": []}  # dict

    train_config = _build_train_config(num_train_steps=40_000, batch_size=2, resum_ckpt_id=CKPT_STEP)
    ckpt_dir = pathlib.Path(TRAINING_VOLUME_PATH) / "ckpts" / REPO_ID / EXP_NAME / str(CKPT_STEP)  # Path
    print(f"Checkpoint dir: {ckpt_dir}  exists={ckpt_dir.exists()}")
    if ckpt_dir.exists():
        print(f"  contents: {sorted(p.name for p in ckpt_dir.iterdir())}")
    hist_txt = ckpt_dir.parent / "history_config.txt"  # Path
    print(f"  history_config.txt present: {hist_txt.exists()}"
          + (f" -> {hist_txt.read_text()!r}" if hist_txt.exists() else ""))

    # ---------------- CHECK 1: released _merge_params on the abstract param tree ----------------
    # Exercises the suspected `flatten_dict(params, sep="/")` crash without materializing the
    # ~3B parameters: the failure depends only on key TYPES in the tree, not on any values.
    print("\n=== CHECK 1: released _merge_params against the fresh model's param tree ===")
    try:
        from openpi.training.weight_loaders import _merge_params as released_merge_params

        abstract_model = nnx.eval_shape(train_config.model.create, jax.random.key(0))
        fresh_params = nnx.state(abstract_model, nnx.Param).to_pure_dict()  # at.Params
        list_like = [k for k in ("fusion", "event_encoder") if k in fresh_params]  # list[str]
        print(f"  fresh param tree top-level keys include: {sorted(fresh_params)[:8]} ...")
        for k in list_like:
            sub = fresh_params[k]
            inner = "blocks" if k == "fusion" else "layers"
            if isinstance(sub, dict) and inner in sub and isinstance(sub[inner], dict):
                print(f"  {k}/{inner} key types: {[type(kk).__name__ for kk in sub[inner]]}")

        released_merge_params({}, fresh_params, missing_regex=".*")
        result["check1"] = {"ok": True, "detail": "released _merge_params accepted the tree"}
        print("  CHECK1 PASSED (unexpected -- the released _merge_params handled int keys)")
    except Exception as err:  # noqa: BLE001 -- recording the failure IS the result here
        result["check1"] = {"ok": False, "error_type": type(err).__name__, "error": str(err)[:400]}
        print(f"  CHECK1 FAILED: {type(err).__name__}: {err}")
        traceback.print_exc()

    # ---------------- CHECK 2: the real eval factory, end to end ----------------
    print("\n=== CHECK 2: create_xf_trained_policy against the real step-18000 checkpoint ===")
    try:
        from xattn_fusion.mme_vla_suite.policies.xf_policy_config import create_xf_trained_policy

        policy = create_xf_trained_policy(train_config, ckpt_dir, seed=42)
        result["check2"] = {"ok": True, "detail": f"built {type(policy).__name__} successfully"}
        print(f"  CHECK2 PASSED: built {type(policy).__name__}")
    except Exception as err:  # noqa: BLE001
        result["check2"] = {"ok": False, "error_type": type(err).__name__, "error": str(err)[:600]}
        print(f"  CHECK2 FAILED: {type(err).__name__}: {err}")
        traceback.print_exc()

    return result


@app.local_entrypoint()
def main():
    """
    What it does:
        Triggers the remote check and prints a verdict on whether the XF
        eval path can load checkpoint 18000, naming which of the two
        predicted failure modes (if either) actually fired.

    Returns:
        None -- prints to stdout.

    Example input:
        modal run XF_18k_eval/analysis/check_eval_load_path.py

    Example output:
        (stdout) CHECK1/CHECK2 outcomes and a GO / NO-GO on pushing to HF
    """
    r = check_load_path.remote()  # dict

    print("\n============ XF eval load path against checkpoint 18000 ============")
    for name, label in [
        ("check1", "CHECK 1  released _merge_params on the fresh param tree"),
        ("check2", "CHECK 2  create_xf_trained_policy end-to-end"),
    ]:
        c = r[name]  # dict
        status = "PASS" if c.get("ok") else "FAIL"  # str
        print(f"\n{label}\n  -> {status}")
        if c.get("ok"):
            print(f"     {c.get('detail', '')}")
        else:
            print(f"     {c.get('error_type')}: {c.get('error')}")

    print("\n--------------------------------- verdict ---------------------------------")
    if r["check2"].get("ok"):
        print("The eval path CAN load checkpoint 18000. Safe to push to HF and evaluate.")
    else:
        print("The eval path CANNOT load checkpoint 18000 as it stands.")
        print("Fix this BEFORE pushing ~12GB to HF Hub and spending eval episodes on a")
        print("second account -- under the eval plan's reading table a broken load looks")
        print("exactly like 'R0 is wrong, so the whole eval is garbage'.")
        if not r["check1"].get("ok"):
            print("\nCHECK 1 also failed, so the first wall is the released _merge_params")
            print("(the import-time name binding at xf_policy_config.py:39 never receives")
            print("the training patch). Fixing only the .load() call would not be enough.")
    print()
