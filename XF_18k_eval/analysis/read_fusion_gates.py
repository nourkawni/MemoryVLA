"""
read_fusion_gates.py

Reads XF's four learnable fusion gate scalars straight out of every saved
training checkpoint, with no GPU, no data, and no forward pass -- just a
params-array read off the xf-full-suite-training volume.

Why this exists: the XF eval plan's first step is "before you spend a single
episode, check the gate -- pull tanh(a1), tanh(a2) from the logs." There are
no such logs. launch_xf_training.py builds its TrainConfig with
wandb_enabled=False and nothing on the training path ever prints the gate
scalars, so training stdout carries only loss / grad_norm / param_norm. The
checkpoints themselves are the only place these values exist, and reading
them there is exact rather than a logged sample.

What the gates are: fusion_xattn.GatedXAttnBlock has two zero-init scalar
gates, a_x (scaling the cross-attention message residual) and a_d (scaling
the FFW residual), and GatedXAttnFusion stacks 2 such blocks -- so there are
4 gate scalars in total, not 2. tanh(0) == 0 makes the whole fusion block a
bit-exact identity at initialization, which is the point: any nonzero
tanh(a) is training having actively decided to route caption information
into the frame stream. tanh(a) ~= 0 at step 18000 means the mechanism never
engaged and an eval would only reproduce the warm-start baseline.

Why it reads EVERY saved step, not just 18000: a single value cannot
distinguish "the gates opened early and plateaued" from "the gates are still
climbing and 18k is simply early" -- and those imply opposite decisions about
whether to continue to 40k. The full 2000..18000 trajectory costs the same as
one step because each restore pulls only the fusion subtree.

It also reports the out_proj / ffw_out kernel norms per block. The gate alone
does not determine how much the fusion path contributes: a small tanh(a)
multiplying a large message is not the same as a small tanh(a) multiplying a
small one. These norms are the cheap, params-only proxy for message
magnitude; the real ||tanh(a)*msg|| / ||F|| ratio the eval plan asks for
needs a forward pass on real data and is a separate job.

Role in the system: read-only diagnostic for the XF 18k eval. It never
writes to any volume, never touches robomme_policy_learning/, and imports
nothing from xattn_fusion -- restoring raw params needs no model class.

Run with:
    modal run XF_18k_eval/analysis/read_fusion_gates.py
"""

import pathlib

import modal

# str, the released policy repo, mounted at /app so openpi's own restore_params is importable.
POLICY_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning")

# str, str -- MUST match launch_xf_training.py's REPO_ID / EXP_NAME constants, since together
# they form the checkpoint directory path ckpts/{REPO_ID}/{EXP_NAME}/{step} on the volume.
REPO_ID = "xf_full_suite"

# str, the DEFAULT run to read. Left at the ORIGINAL zero-gate run so previously-reported
# numbers reproduce unchanged. The launcher's EXP_NAME moved to
# "full-16task-xattn-fusion-gateinit0.1" on 2026-09-22; pass --exp-name to read that one.
EXP_NAME = "full-16task-xattn-fusion"

TRAINING_VOLUME_PATH = "/xf_training"  # str, mount point for the training volume

app = modal.App("xf-read-fusion-gates")  # modal.App

# modal.Volume, the SAME volume run_training checkpoints into. create_if_missing is deliberately
# omitted so a wrong/missing volume name fails loudly instead of silently mounting an empty one.
training_volume = modal.Volume.from_name("xf-full-suite-training")

# modal.Image. Identical to launch_xf_training.py's image up to (and excluding) its final
# xattn_fusion mount, so every layer here is a cache hit against the already-built training
# image. The xattn_fusion mount is dropped on purpose: a raw params read builds no model and
# imports no XF module, so nothing here needs that code.
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
)


def _key_str(kp):
    """
    What it does:
        Renders one flattened orbax/flax param key path (a tuple whose
        elements may be str OR int, since a plain Python list of submodules
        serializes with integer-like indices) as a single readable string.
        Written as str(k) per element rather than "/".join(kp) because
        "/".join crashes outright on a non-str element -- the exact failure
        mode already hit once in this project's weight loading.

    Returns:
        str -- the path components joined by "/".

    Example input:
        _key_str(("fusion", "blocks", 0, "a_x", "value"))

    Example output:
        "fusion/blocks/0/a_x/value"
    """
    return "/".join(str(k) for k in kp)


def _restore_fusion_subtree(params_path, ocp, jax, np, traverse_util):
    """
    What it does:
        Restores only the fusion/type_emb leaves of an orbax checkpoint. It
        reads the checkpoint's metadata first, filters that structure down
        to the wanted key paths, and hands orbax that reduced tree as the
        restore item -- so the untouched ~3B-parameter backbone is never
        read. Then it strips the trailing "value" component nnx.State adds
        to every key path when a checkpoint is written by training (the same
        normalization openpi's own restore_params performs).

    Returns:
        dict[str, np.ndarray] -- flattened, "/"-joined key path to array,
        containing only fusion and type_emb leaves.

    Example input:
        _restore_fusion_subtree(Path("/xf_training/ckpts/x/y/18000/params"), ocp, jax, np, traverse_util)

    Example output:
        {"fusion/blocks/0/a_x": array(0.031, dtype=float32), ...}
    """
    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)  # orbax metadata tree
        full_item = {"params": metadata["params"]}  # dict
        flat_meta = traverse_util.flatten_dict(full_item)  # dict[tuple, metadata leaf]
        wanted = {kp: leaf for kp, leaf in flat_meta.items() if "fusion" in kp or "type_emb" in kp}  # dict
        if not wanted:
            top_level = sorted({str(kp[1]) for kp in flat_meta if len(kp) > 1})  # list[str]
            raise KeyError(f"No fusion/type_emb keys in checkpoint metadata. Top-level params keys: {top_level}")
        item = traverse_util.unflatten_dict(wanted)  # dict
        restored = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), item),
            ),
        )["params"]

    flat = traverse_util.flatten_dict(restored)  # dict[tuple, np.ndarray]
    if flat and all(kp[-1] == "value" for kp in flat):
        flat = {kp[:-1]: v for kp, v in flat.items()}
    return {_key_str(kp): v for kp, v in flat.items()}


@app.function(image=image, volumes={TRAINING_VOLUME_PATH: training_volume}, timeout=1800, memory=16384)
def read_gates_for_all_steps(exp_name: str = EXP_NAME) -> list[dict]:
    """
    What it does:
        Finds every saved checkpoint step on the training volume, and for
        each one restores ONLY the "fusion" and "type_emb" slices of the
        params tree (a partial orbax restore driven by the checkpoint's own
        metadata, so the ~3B-parameter backbone is never read off disk at
        all), then extracts the raw gate scalars a_x/a_d for every fusion
        block plus the out_proj/ffw_out kernel Frobenius norms.

        If the partial restore is rejected for any step, that step falls
        back to openpi's ordinary full restore_params rather than being
        silently skipped -- a missing step would look identical to a gate
        that never moved, which is the one confusion this diagnostic exists
        to prevent.

    Returns:
        list[dict] -- one dict per saved step, ascending by step, each with
        keys "step" (int), "gates" (dict[str, float] raw a_x/a_d values
        keyed by block), "norms" (dict[str, float] kernel norms), and
        "type_emb_norm" (float). On a per-step failure the dict instead
        carries "error" (str).

    Example input:
        read_gates_for_all_steps.remote()

    Example output:
        [{"step": 2000, "gates": {"blocks/0/a_x": 0.031, "blocks/0/a_d": -0.008,
          "blocks/1/a_x": 0.024, "blocks/1/a_d": 0.011},
          "norms": {"blocks/0/out_proj/kernel": 4.12}, "type_emb_norm": 0.37}]
    """
    import sys  # module

    sys.path.insert(0, "/app/src")

    import flax.traverse_util  # module
    import jax  # module
    import numpy as np  # module
    import orbax.checkpoint as ocp  # module

    import openpi.models.model as _model  # module

    training_volume.reload()  # volumes are not live-synced into an already-running container

    ckpt_root = pathlib.Path(TRAINING_VOLUME_PATH) / "ckpts" / REPO_ID / exp_name  # pathlib.Path
    print(f"Reading run: {exp_name}")
    if not ckpt_root.exists():
        raise FileNotFoundError(f"No checkpoint directory at {ckpt_root}")

    steps = sorted(int(p.name) for p in ckpt_root.iterdir() if p.is_dir() and p.name.isdigit())  # list[int]
    print(f"Saved checkpoint steps found: {steps}")

    results = []  # list[dict]
    for step in steps:
        params_path = ckpt_root / str(step) / "params"  # pathlib.Path
        try:
            flat = _restore_fusion_subtree(params_path, ocp, jax, np, flax.traverse_util)  # dict[str, np.ndarray]
        except Exception as partial_err:  # noqa: BLE001 -- fall back rather than skip; see docstring
            print(f"[step {step}] partial restore failed ({partial_err!r}); falling back to full restore")
            try:
                params = _model.restore_params(params_path, restore_type=np.ndarray)  # at.Params
                flat_all = flax.traverse_util.flatten_dict(params)  # dict[tuple, np.ndarray]
                flat = {_key_str(kp): v for kp, v in flat_all.items() if "fusion" in kp or "type_emb" in kp}
            except Exception as full_err:  # noqa: BLE001
                print(f"[step {step}] FULL restore also failed: {full_err!r}")
                results.append({"step": step, "error": repr(full_err)})
                continue

        if step == steps[0]:
            # Printed once so the real key layout is visible in the logs rather than assumed --
            # list-index components in particular can come back as either "0" or 0.
            print(f"[step {step}] fusion/type_emb keys present: {sorted(flat)}")

        gates = {}  # dict[str, float]
        norms = {}  # dict[str, float]
        type_emb_norm = None  # float or None
        for key, arr in flat.items():
            short = key.replace("fusion/", "")  # str, drop the constant prefix for readability
            if short.endswith("a_x") or short.endswith("a_d"):
                gates[short] = float(np.asarray(arr).reshape(()))
            elif short.endswith("/kernel"):
                # ALL fusion kernels, not just out_proj/ffw_out. Widened 2026-09-23: the eps-floor
                # argument applies to EVERY parameter inside the gated block, not only the two
                # output projections. q_proj/kv_proj sit upstream of the same tanh(a_x) gate, so
                # their gradients are scaled by it too -- meaning the very projections that would
                # learn to bridge the caption and frame embedding spaces were frozen along with
                # everything else. Reporting them makes that checkable instead of assumed.
                norms[short] = float(np.linalg.norm(np.asarray(arr, dtype=np.float64)))
            elif "type_emb" in key:
                type_emb_norm = float(np.linalg.norm(np.asarray(arr, dtype=np.float64)))

        results.append({"step": step, "gates": gates, "norms": norms, "type_emb_norm": type_emb_norm})
        print(f"[step {step}] gates={gates}")

    return results


@app.local_entrypoint()
def main(exp_name: str = EXP_NAME):
    """
    What it does:
        Triggers the remote read and prints one row per saved step: the raw
        gate scalar, its tanh, and the message-path kernel norms -- i.e. the
        table the eval plan's pre-flight gate check needs, plus the
        step-over-step trajectory that tells apart "plateaued" from "still
        climbing".

    Returns:
        None -- prints to stdout.

    Example input:
        modal run XF_18k_eval/analysis/read_fusion_gates.py

    Example output:
        (stdout) a table of step / a_x / tanh(a_x) / a_d / tanh(a_d) per fusion block
    """
    import math  # module

    rows = read_gates_for_all_steps.remote(exp_name)  # list[dict]

    print(f"\n===== XF fusion gates, all saved checkpoints -- {exp_name} =====")
    gate_names = sorted({g for r in rows if "gates" in r for g in r["gates"]})  # list[str]
    header = "step".rjust(7) + "".join(f"  {n:>18}  {'tanh':>9}" for n in gate_names)  # str
    print(header)
    print("-" * len(header))
    for r in rows:
        if "error" in r:
            print(f"{r['step']:>7}  ERROR: {r['error']}")
            continue
        line = f"{r['step']:>7}"  # str
        for n in gate_names:
            v = r["gates"].get(n)  # float or None
            line += f"  {v:>18.6f}  {math.tanh(v):>9.6f}" if v is not None else f"  {'--':>18}  {'--':>9}"
        print(line)

    print("\n---- message-path kernel norms (proxy for message size BEFORE gating) ----")
    norm_names = sorted({n for r in rows if "norms" in r for n in r["norms"]})  # list[str]
    header2 = "step".rjust(7) + "".join(f"  {n:>26}" for n in norm_names) + f"  {'type_emb_norm':>14}"  # str
    print(header2)
    print("-" * len(header2))
    for r in rows:
        if "error" in r:
            continue
        line = f"{r['step']:>7}"  # str
        for n in norm_names:
            v = r["norms"].get(n)  # float or None
            line += f"  {v:>26.6f}" if v is not None else f"  {'--':>26}"
        te = r.get("type_emb_norm")  # float or None
        line += f"  {te:>14.6f}" if te is not None else f"  {'--':>14}"
        print(line)
    print()
