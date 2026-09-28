"""
measure_gate_gradient.py

Phase 1 of the XF fix plan: decide whether raising the fusion output
projection's initialization would actually let the gates learn -- measured on
the checkpoint we already have, BEFORE spending any training time.

The question. `XF_18k_eval/analysis/fusion_message_findings.md` established
that XF's fusion is stuck at initialization: the gates sit at ~3e-4 and
`out_proj`/`ffw_out` are still at their analytic init norms after 18,000
steps. The proposed cause is mutual starvation -- `out_proj`'s gradient is
scaled by `tanh(a_x) ~ 3e-4` so the message never grows, while the gate's own
gradient is proportional to `<dL/dF', msg>` against that tiny message, so the
gate never opens. The proposed fix is to stop double-shrinking that path:
`kernel_init_out_proj` is `normal(stddev=0.002)`, roughly 9x smaller than the
Flamingo reference implementation, which leaves its output projections at
PyTorch's default (~0.018 std for this fan-in) and zero-inits ONLY the gate.

What this measures, and why it is not just a magnitude read. A single
gradient value cannot distinguish "small but consistent" from "noise". The
thing that decides whether a parameter can learn is whether it receives a
CONSISTENT DIRECTION across batches. So this measures `dL/d(alpha)` for all
four gate scalars over several independent real batches and reports:

  - mean and std across batches,
  - **sign consistency**: the fraction of batches whose gradient agrees with
    the mean's sign. ~0.5 means pure noise (the gate random-walks, which is
    exactly the sign-flipping, magnitude-decaying trace observed across
    checkpoints 4000-18000). Consistently above ~0.75 means a real direction.

...in two conditions on the SAME weights and the SAME batches:

  1. as loaded (out_proj at its 0.002-scale init), and
  2. with `out_proj` and `ffw_out` multiplied by 10.

Why condition 2 is a faithful simulation of the proposed fix, not a hack:
`out_proj` is still AT its random initialization (measured 2.046 vs 2.048
predicted analytically from stddev=0.002). Multiplying a random matrix by 10
yields a random matrix with 10x the standard deviation -- distributionally
identical to having initialized it with `kernel_init` (0.02) in the first
place. So this reads the gradient the fixed model would see at step 0.

Reading the result:
  - 10x raises |mean| roughly proportionally AND lifts sign consistency well
    above the 1x condition => the fix should work; proceed to the 2,000-step
    engagement check.
  - 10x changes little, or sign consistency stays ~0.5 in both => the
    starvation story is wrong or incomplete, and retraining on the strength
    of it would waste hours. Stop and rethink.

Note on a bug this file had to fix: `XFModel.compute_loss` returns a TUPLE
`(per_timestep_loss, stats)` despite its docstring stating it returns just
the loss array. An earlier attempt at this measurement (inside
`measure_fusion_message.py`) died on
`TypeError: mean requires ndarray or scalar arguments, got <class 'tuple'>`
for exactly that reason.

Role in the system: read-only diagnostic. Trains nothing, writes to no
volume, and never edits robomme_policy_learning/.

ACCOUNT: this runs on **nour-mkawni** (the default Modal profile), NOT on the
eval account. It mounts `xf-full-suite-training` (the checkpoint),
`xf-full-suite-data`, and the four `xf-features-shard-*` volumes, all of
which live on nour-mkawni. The eval account (noor-koni2002, profile
`arm-d-eval`) has none of them -- it deliberately works from the published HF
checkpoint instead. Every Volume.from_name below omits `create_if_missing`,
so running this on the wrong account fails loudly rather than silently
mounting empty volumes and measuring nothing.

Run with (default profile -- do NOT set MODAL_PROFILE):
    modal run XF_18k_eval/analysis/measure_gate_gradient.py
    modal run XF_18k_eval/analysis/measure_gate_gradient.py --num-batches 12
"""

import pathlib

import modal

POLICY_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning")
XF_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "xattn_fusion")

CKPT_VOLUME_PATH = "/ckpts"  # str
MAIN_DATA_VOLUME_PATH = "/xf_data"  # str
TRAINING_VOLUME_PATH = "/xf_training"  # str
FEATURE_SHARD_PATHS = [f"/xf_features_shard_{i}" for i in range(4)]  # list[str]

REPO_ID = "xf_full_suite"  # str, must match launch_xf_training.py
EXP_NAME = "full-16task-xattn-fusion"  # str, ditto
CKPT_STEP = 18000  # int

DEFAULT_NUM_BATCHES = 8  # int, independent gradient samples per condition
DEFAULT_BATCH_SIZE = 2  # int
SCALE_FACTOR = 10.0  # float, 0.002 -> 0.02, i.e. kernel_init_out_proj -> kernel_init

app = modal.App("xf-measure-gate-gradient")  # modal.App

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

image = (  # modal.Image
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


def _summarize(values, np):
    """
    What it does:
        Summarizes one gate's per-batch gradients: central tendency, spread,
        and the sign-consistency figure that actually decides whether the
        parameter has a direction to move in.

    Returns:
        dict -- {"mean": float, "std": float, "abs_mean": float,
        "sign_consistency": float, "n": int}. sign_consistency is the
        fraction of samples sharing the mean's sign, so ~0.5 is noise and
        1.0 is perfectly consistent.

    Example input:
        _summarize([1e-5, 1.2e-5, -2e-6, 9e-6], np)

    Example output:
        {"mean": 7.5e-06, "std": 5.2e-06, "abs_mean": 8.0e-06, "sign_consistency": 0.75, "n": 4}
    """
    arr = np.asarray(values, dtype=np.float64)  # np.ndarray
    mean = float(arr.mean())  # float
    sign = np.sign(mean) if mean != 0 else 1.0  # float
    return {
        "mean": mean,
        "std": float(arr.std()),
        "abs_mean": float(np.abs(arr).mean()),
        "sign_consistency": float((np.sign(arr) == sign).mean()),
        "n": int(arr.size),
    }


@app.function(image=image, gpu="A10G", timeout=5400, volumes=volumes)
def measure(num_batches: int = DEFAULT_NUM_BATCHES, batch_size: int = DEFAULT_BATCH_SIZE, seed: int = 0) -> dict:
    """
    What it does:
        Loads checkpoint 18000, collects `num_batches` real training batches,
        and measures dL/d(alpha) for all four fusion gates on each batch --
        first with the weights as loaded, then with out_proj/ffw_out scaled
        by SCALE_FACTOR. The SAME batches and the SAME rng are reused across
        both conditions so the comparison isolates the scaling and nothing
        else.

    Returns:
        dict -- {"gates": dict[str,float], "baseline": {gate: summary},
        "scaled": {gate: summary}, "kernel_norms": {...}, "batches": int}.

    Example input:
        measure.remote(num_batches=8, batch_size=2)

    Example output:
        {"baseline": {"blocks/0/a_x": {"mean": 1e-7, "sign_consistency": 0.5, ...}}, ...}
    """
    import os  # module
    import sys  # module

    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/src")
    sys.path.insert(0, "/xf_root")
    os.chdir("/app")  # mme_vla_suite's config loader resolves paths against the process CWD

    import flax.nnx as nnx  # module
    import jax  # module
    import jax.numpy as jnp  # module
    import numpy as np  # module

    for vol in [ckpt_volume, main_data_volume, training_volume, *feature_shard_volumes]:
        vol.reload()

    import mme_vla_suite.training.dataloader as _dataloader  # module -- NOT openpi.training.data_loader;
    # _patch_scripts_train_for_xf rebinds THIS module's create_data_loader.
    import openpi.models.model as _model  # module
    import scripts.train as _train  # module

    from xattn_fusion.training.launch_xf_training import (
        _build_train_config,
        _patch_scripts_train_for_xf,
    )

    _patch_scripts_train_for_xf(_train)

    train_config = _build_train_config(num_train_steps=40_000, batch_size=batch_size, resum_ckpt_id=CKPT_STEP)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)  # DataConfig

    ckpt_params_dir = pathlib.Path(TRAINING_VOLUME_PATH) / "ckpts" / REPO_ID / EXP_NAME / str(CKPT_STEP) / "params"
    if not ckpt_params_dir.exists():
        raise FileNotFoundError(f"No checkpoint params at {ckpt_params_dir}")
    print(f"Loading checkpoint from {ckpt_params_dir} ...")
    raw_params = _model.restore_params(ckpt_params_dir, dtype=jnp.bfloat16)  # at.Params
    # Bypasses .load()'s strict pre-check, which cannot reconcile orbax's string list-indices
    # with nnx's int ones -- replace_by_pure_dict is the function that does reconcile them.
    graphdef, state = nnx.split(nnx.eval_shape(train_config.model.create, jax.random.key(0)))
    state.replace_by_pure_dict(raw_params)
    model = nnx.merge(graphdef, state)  # XFModel
    print("Model loaded.")

    gates = {}  # dict[str, float]
    for i, blk in enumerate(model.fusion.blocks):
        gates[f"blocks/{i}/a_x"] = float(jnp.asarray(blk.a_x.value).reshape(()))
        gates[f"blocks/{i}/a_d"] = float(jnp.asarray(blk.a_d.value).reshape(()))
    print(f"Trained gate scalars: {gates}")
    if all(abs(v) < 1e-9 for v in gates.values()):
        raise ValueError(
            f"All gates are exactly 0.0 -- the checkpoint's fusion weights did not load, so every "
            f"gradient below would describe a randomly-initialized fusion stack. Gates: {gates}"
        )

    num_blocks = len(model.fusion.blocks)  # int
    gate_names = [f"blocks/{i}/{p}" for i in range(num_blocks) for p in ("a_x", "a_d")]  # list[str]
    gate_init = jnp.array([gates[n] for n in gate_names], dtype=jnp.float32)  # float32[2*num_blocks]

    loader = _dataloader.create_data_loader(
        train_config.dataset_path, data_config,
        history_config=train_config.model.history_config,
        action_horizon=train_config.model.action_horizon,
        batch_size=batch_size, shuffle=True, num_batches=num_batches, num_workers=0, seed=seed,
    )
    batches = list(loader)  # list[tuple] -- materialized so BOTH conditions see identical data
    print(f"Collected {len(batches)} batches.")

    def _restore_gates():
        """
        What it does: writes the checkpoint's concrete gate values back onto
        the live module. Necessary after every jax.grad call, because the
        traced function assigns TRACERS into a_x/a_d and those must not leak
        into the next measurement.

        Returns:
            None.

        Example input:
            _restore_gates()

        Example output:
            None
        """
        for i, blk in enumerate(model.fusion.blocks):
            blk.a_x.value = jnp.asarray(gates[f"blocks/{i}/a_x"], dtype=jnp.float32)
            blk.a_d.value = jnp.asarray(gates[f"blocks/{i}/a_d"], dtype=jnp.float32)

    def _grad_for_batch(obs, actions, rng_seed: int):
        """
        What it does: computes dL/d(alpha) for all gate scalars on one batch,
        by writing the gate vector into the live module inside a
        jax.grad-traced function. Unpacks compute_loss's (loss, stats) TUPLE
        -- taking the mean of the tuple itself is what broke the first
        attempt at this measurement.

        Returns:
            np.ndarray -- float64, shape (2*num_blocks,), one gradient per gate.

        Example input:
            _grad_for_batch(obs, actions, 1234)

        Example output:
            array([ 2.7e-07, -1.1e-07,  3.4e-08,  9.0e-08])
        """
        def loss_wrt_gates(gate_vec):
            for i, blk in enumerate(model.fusion.blocks):
                blk.a_x.value = gate_vec[2 * i]
                blk.a_d.value = gate_vec[2 * i + 1]
            loss, _stats = model.compute_loss(jax.random.key(rng_seed), obs, actions, train=True)
            return jnp.mean(loss)

        g = jax.grad(loss_wrt_gates)(gate_init)  # float32[2*num_blocks]
        _restore_gates()
        return np.asarray(g, dtype=np.float64)

    def _run_condition(label: str) -> dict:
        """
        What it does: runs the gradient measurement across every collected
        batch and summarizes each gate. The rng seed is derived from the
        batch index only, so both conditions see identical noise draws.

        Returns:
            dict -- {gate_name: summary dict from _summarize}.

        Example input:
            _run_condition("baseline")

        Example output:
            {"blocks/0/a_x": {"mean": 1e-7, "sign_consistency": 0.5, ...}}
        """
        per_gate = {n: [] for n in gate_names}  # dict[str, list[float]]
        for b_i, (observation, actions) in enumerate(batches):
            g = _grad_for_batch(observation, actions, rng_seed=1234 + b_i)  # np.ndarray
            for j, n in enumerate(gate_names):
                per_gate[n].append(float(g[j]))
            print(f"  [{label}] batch {b_i}: {[f'{v:+.3e}' for v in g]}")
        return {n: _summarize(v, np) for n, v in per_gate.items()}

    def _kernel_norms() -> dict:
        """
        What it does: reports the current Frobenius norms of the message-path
        kernels, so the scaling is visibly applied rather than assumed.

        Returns:
            dict[str, float].

        Example input:
            _kernel_norms()

        Example output:
            {"blocks/0/out_proj": 2.046, "blocks/0/ffw_out": 2.895}
        """
        out = {}  # dict[str, float]
        for i, blk in enumerate(model.fusion.blocks):
            out[f"blocks/{i}/out_proj"] = float(jnp.linalg.norm(blk.out_proj.kernel.value.astype(jnp.float32)))
            out[f"blocks/{i}/ffw_out"] = float(jnp.linalg.norm(blk.ffw_out.kernel.value.astype(jnp.float32)))
        return out

    print("\n=== CONDITION 1: as loaded (out_proj at its stddev=0.002-scale init) ===")
    norms_before = _kernel_norms()  # dict[str, float]
    print(f"  kernel norms: { {k: round(v, 4) for k, v in norms_before.items()} }")
    baseline = _run_condition("baseline")  # dict

    print(f"\n=== CONDITION 2: out_proj / ffw_out scaled x{SCALE_FACTOR} ===")
    # out_proj is still AT its random init, so scaling a random matrix by 10 gives a random
    # matrix with 10x the stddev -- distributionally what kernel_init(0.02) would have produced.
    for blk in model.fusion.blocks:
        blk.out_proj.kernel.value = blk.out_proj.kernel.value * SCALE_FACTOR
        blk.ffw_out.kernel.value = blk.ffw_out.kernel.value * SCALE_FACTOR
    norms_after = _kernel_norms()  # dict[str, float]
    print(f"  kernel norms: { {k: round(v, 4) for k, v in norms_after.items()} }")
    scaled = _run_condition("scaled")  # dict

    return {
        "gates": gates, "baseline": baseline, "scaled": scaled,
        "kernel_norms": {"before": norms_before, "after": norms_after},
        "batches": len(batches), "scale_factor": SCALE_FACTOR,
    }


@app.local_entrypoint()
def main(num_batches: int = DEFAULT_NUM_BATCHES, batch_size: int = DEFAULT_BATCH_SIZE, seed: int = 0):
    """
    What it does:
        Triggers the measurement and prints the two conditions side by side,
        with a GO / NO-GO reading on whether raising the output projection's
        init would give the gates a direction to move in.

    Returns:
        None -- prints to stdout.

    Example input:
        modal run XF_18k_eval/analysis/measure_gate_gradient.py

    Example output:
        (stdout) per-gate mean/std/sign-consistency for both conditions, and a verdict
    """
    r = measure.remote(num_batches=num_batches, batch_size=batch_size, seed=seed)  # dict

    print(f"\n============ dL/d(alpha) at checkpoint {CKPT_STEP} ============")
    print(f"batches per condition: {r['batches']} (batch_size={batch_size})")
    print(f"trained gates: { {k: f'{v:.3e}' for k, v in r['gates'].items()} }")
    print(f"\nkernel norms before: { {k: round(v, 4) for k, v in r['kernel_norms']['before'].items()} }")
    print(f"kernel norms after : { {k: round(v, 4) for k, v in r['kernel_norms']['after'].items()} }")

    hdr = f"\n{'gate':<16} {'condition':<10} {'mean':>12} {'std':>12} {'|mean|':>12} {'sign-consist':>13}"  # str
    print(hdr)
    print("-" * len(hdr))
    ratios = []  # list[float]
    consist_base = []  # list[float]
    consist_scaled = []  # list[float]
    for gate in r["baseline"]:
        b, s = r["baseline"][gate], r["scaled"][gate]  # dict, dict
        print(f"{gate:<16} {'baseline':<10} {b['mean']:>12.4e} {b['std']:>12.4e} "
              f"{abs(b['mean']):>12.4e} {b['sign_consistency']:>13.2f}")
        print(f"{'':<16} {'scaled x' + str(int(r['scale_factor'])):<10} {s['mean']:>12.4e} {s['std']:>12.4e} "
              f"{abs(s['mean']):>12.4e} {s['sign_consistency']:>13.2f}")
        if abs(b["mean"]) > 0:
            ratios.append(abs(s["mean"]) / abs(b["mean"]))
        consist_base.append(b["sign_consistency"])
        consist_scaled.append(s["sign_consistency"])

    import statistics

    mean_ratio = statistics.fmean(ratios) if ratios else float("nan")  # float
    cb = statistics.fmean(consist_base)  # float
    cs = statistics.fmean(consist_scaled)  # float
    print(f"\nmean |grad| ratio (scaled / baseline): {mean_ratio:.2f}x")
    print(f"mean sign consistency: baseline {cb:.2f} -> scaled {cs:.2f}")

    print("\n--------------------------------- reading ---------------------------------")
    print("Sign consistency ~0.5 means the gradient has no consistent direction across batches:")
    print("the gate random-walks, which is exactly the trace seen over checkpoints 4000-18000.")
    if cs >= 0.75 and cs > cb + 0.1:
        print(f"\nGO. Scaling the output projection lifts sign consistency {cb:.2f} -> {cs:.2f} and")
        print(f"|grad| by {mean_ratio:.1f}x. The gate would have a direction to move in.")
        print("Proceed: apply the init fix, then the 2,000-step engagement check.")
    elif mean_ratio > 3.0 and cs > cb:
        print(f"\nWEAK GO. |grad| rises {mean_ratio:.1f}x and consistency improves {cb:.2f} -> {cs:.2f},")
        print("but consistency is still not decisive. The init fix is worth trying, and the")
        print("2,000-step engagement check becomes the real test -- do not skip it.")
    else:
        print(f"\nNO-GO. Scaling changed little (|grad| {mean_ratio:.1f}x, consistency "
              f"{cb:.2f} -> {cs:.2f}).")
        print("The starvation account does not survive this test. Retraining on the strength of")
        print("it would waste hours -- rethink before changing the init.")
    print()
