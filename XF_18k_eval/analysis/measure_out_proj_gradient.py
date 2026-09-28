"""
measure_out_proj_gradient.py

Tests the epsilon-floor hypothesis: is XF's fusion output projection frozen
because its gradients fall BELOW AdamW's epsilon, and would opening the gate
slightly lift them back above it?

The puzzle this exists to resolve. `fusion_message_findings.md` measured that
`out_proj`'s Frobenius norm did not move at all over 18,000 steps (2.046
measured vs 2.048 predicted analytically from its stddev=0.002 init -- 0.1%).
That is hard to explain with scale reasoning alone, for two independent
reasons:

  1. AdamW is approximately scale-invariant. Its update is
     `lr * m / (sqrt(v) + eps)`, which for a gradient of mean mu and spread
     sigma is about `lr * mu / sqrt(mu^2 + sigma^2)`. Scaling a gradient up
     or down leaves that unchanged. So "the gradient is small" is NOT by
     itself a reason a parameter cannot move.
  2. Even PURE NOISE gradients should random-walk the norm upward. At
     lr=5e-5 over 18,000 steps, a per-entry displacement of order
     `lr * sqrt(steps)` accumulated across a [1024,1024] kernel would grow
     the norm far beyond 2.048, not leave it there.

Something is holding `out_proj` still, and plain gradient-magnitude
arguments do not account for it.

The hypothesis. AdamW's scale invariance breaks down when `sqrt(v)` falls
below `eps`. `openpi.training.optimizer.AdamW` uses `eps=1e-8`. `out_proj`'s
gradient is scaled by `tanh(a_x) ~ 3e-4`, so if the unscaled signal is around
1e-6, the actual gradient is around 1e-10 -- two orders of magnitude BELOW
eps. In that regime the update degenerates from `lr * mu/sigma` (order lr) to
roughly `lr * mu/eps` (order lr * 1e-2 or smaller), i.e. the optimizer stops
normalizing and the parameter effectively freezes. That mechanism would
explain the observation that scale reasoning cannot.

What this measures. `dL/d(out_proj)` and `dL/d(ffw_out)` per-element, across
several real batches, in two conditions on the same weights and batches:

  1. gates at their trained values (~3e-4), and
  2. gates forced to 0.1 -- "lever 1" from the fix plan.

For each it reports the per-element gradient RMS and std against `eps=1e-8`,
plus the resulting AdamW update magnitude in each regime. 0.1 / 3e-4 is about
330x, so if the hypothesis holds, condition 1 should sit below eps and
condition 2 above it.

Why this matters for the plan. The previous measurement
(`measure_gate_gradient.py`) showed the gate's gradient scales exactly 10.04x
when `out_proj` is scaled 10x -- mean AND std together, leaving the
signal-to-noise ratio untouched. Because Adam is scale-invariant, that means
simply raising `out_proj`'s init does NOT help the gate open, and the
originally-proposed "swap kernel_init_out_proj for kernel_init" fix is not
justified. The remaining levers change structure rather than scale: a
non-zero gate init, or an auxiliary loss giving `out_proj` a gradient path
not scaled by `tanh(a_x)`. This file tests whether the first of those is
enough, and whether the eps floor is the real reason it is needed.

Reading the result:
  - condition 1 RMS well below 1e-8 and condition 2 well above it => the eps
    floor is real and a non-zero gate init unfreezes `out_proj`. Lever 1 is
    justified on mechanism, not analogy.
  - both well above 1e-8 => the eps floor is NOT the explanation; `out_proj`
    should have been moving and was not, so something else (a freeze filter,
    a detached path) is holding it and must be found before any retrain.
  - both well below => the gate init alone is insufficient; the auxiliary
    loss becomes the necessary lever.

ACCOUNT: runs on **nour-mkawni** (default profile), NOT the eval account --
it mounts the training volumes. Every Volume.from_name omits
`create_if_missing`, so a wrong-account run fails loudly.

Run with (default profile; --detach because a multi-minute GPU job should not
depend on the local client's connection staying up):
    modal run --detach XF_18k_eval/analysis/measure_out_proj_gradient.py
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

DEFAULT_NUM_BATCHES = 4  # int -- fewer than the gate measurement: these gradients are full
# kernels, not scalars, so each sample already averages over ~1M elements.
DEFAULT_BATCH_SIZE = 2  # int
OPEN_GATE_VALUE = 0.1  # float, "lever 1" -- tanh(0.1) ~= 0.0997, ~330x the trained ~3e-4
ADAM_EPS = 1e-8  # float, openpi.training.optimizer.AdamW's default eps
ADAM_LR = 5e-5  # float, this run's peak/decay lr (CosineDecaySchedule peak_lr=decay_lr=5e-5)

app = modal.App("xf-measure-out-proj-gradient")  # modal.App

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


def _adam_update_magnitude(mu: float, sigma: float) -> float:
    """
    What it does:
        Computes the per-step AdamW update magnitude a parameter would
        receive given the mean and spread of its gradient, using the real
        optimizer form `lr * m / (sqrt(v) + eps)` with `m -> mu` and
        `sqrt(v) -> sqrt(mu^2 + sigma^2)`. This is what makes the eps floor
        visible: when sqrt(v) >> eps the result is order `lr`, and when
        sqrt(v) << eps it collapses toward `lr * mu / eps`, which is far
        smaller.

    Returns:
        float -- the expected magnitude of one AdamW step for that parameter.

    Example input:
        _adam_update_magnitude(1e-6, 1e-6)

    Example output:
        3.5355339059327377e-05
    """
    v_sqrt = (mu * mu + sigma * sigma) ** 0.5  # float
    return ADAM_LR * abs(mu) / (v_sqrt + ADAM_EPS)


@app.function(image=image, gpu="A10G", timeout=5400, volumes=volumes)
def measure(num_batches: int = DEFAULT_NUM_BATCHES, batch_size: int = DEFAULT_BATCH_SIZE, seed: int = 0) -> dict:
    """
    What it does:
        Loads checkpoint 18000 and measures dL/d(out_proj) and
        dL/d(ffw_out) across real batches, first with the gates at their
        trained values and then with every gate forced to OPEN_GATE_VALUE.
        The same batches and rng seeds are reused across both conditions so
        the only difference is the gate.

    Returns:
        dict -- {"gates": {...}, "conditions": {"trained_gates": {...},
        "gate_0.1": {...}}, "eps": float, "lr": float, "batches": int},
        where each condition maps a kernel name to
        {"rms": float, "std": float, "mean_abs": float}.

    Example input:
        measure.remote(num_batches=4, batch_size=2)

    Example output:
        {"conditions": {"trained_gates": {"blocks/0/out_proj": {"rms": 1.2e-10, ...}}}}
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

    # mme_vla_suite.training.dataloader, NOT openpi.training.data_loader -- the XF patch rebinds
    # this module's create_data_loader; the openpi one is a different, unpatched function.
    import mme_vla_suite.training.dataloader as _dataloader  # module
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
    print(f"Loading checkpoint from {ckpt_params_dir} ...")
    raw_params = _model.restore_params(ckpt_params_dir, dtype=jnp.bfloat16)  # at.Params
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
        raise ValueError(f"All gates exactly 0.0 -- checkpoint fusion weights did not load. {gates}")

    num_blocks = len(model.fusion.blocks)  # int
    kernel_names = [f"blocks/{i}/{p}" for i in range(num_blocks) for p in ("out_proj", "ffw_out")]  # list[str]

    loader = _dataloader.create_data_loader(
        train_config.dataset_path, data_config,
        history_config=train_config.model.history_config,
        action_horizon=train_config.model.action_horizon,
        batch_size=batch_size, shuffle=True, num_batches=num_batches, num_workers=0, seed=seed,
    )
    batches = list(loader)  # list[tuple] -- materialized so both conditions see identical data
    print(f"Collected {len(batches)} batches.")

    # Snapshot the concrete kernel values once: every jax.grad call writes tracers into the live
    # module, so both the kernels and the gates must be restored afterwards.
    kernel_init_vals = {}  # dict[str, jax.Array]
    for i, blk in enumerate(model.fusion.blocks):
        kernel_init_vals[f"blocks/{i}/out_proj"] = blk.out_proj.kernel.value
        kernel_init_vals[f"blocks/{i}/ffw_out"] = blk.ffw_out.kernel.value

    def _restore(gate_value=None):
        """
        What it does: writes concrete kernel values back onto the live
        module, and sets the gates either to their checkpoint values
        (gate_value=None) or to a fixed override.

        Returns:
            None.

        Example input:
            _restore(gate_value=0.1)

        Example output:
            None
        """
        for i, blk in enumerate(model.fusion.blocks):
            blk.out_proj.kernel.value = kernel_init_vals[f"blocks/{i}/out_proj"]
            blk.ffw_out.kernel.value = kernel_init_vals[f"blocks/{i}/ffw_out"]
            if gate_value is None:
                blk.a_x.value = jnp.asarray(gates[f"blocks/{i}/a_x"], dtype=jnp.float32)
                blk.a_d.value = jnp.asarray(gates[f"blocks/{i}/a_d"], dtype=jnp.float32)
            else:
                blk.a_x.value = jnp.asarray(gate_value, dtype=jnp.float32)
                blk.a_d.value = jnp.asarray(gate_value, dtype=jnp.float32)

    def _run_condition(label: str, gate_value) -> dict:
        """
        What it does: measures dL/d(out_proj) and dL/d(ffw_out) on every
        collected batch with the gates held at `gate_value` (or their
        trained values when None), and summarizes each kernel's per-element
        gradient statistics pooled across batches.

        Returns:
            dict -- {kernel_name: {"rms": float, "std": float,
            "mean_abs": float, "n_elements": int}}.

        Example input:
            _run_condition("trained_gates", None)

        Example output:
            {"blocks/0/out_proj": {"rms": 1.2e-10, "std": 1.2e-10, "mean_abs": 9e-11}}
        """
        acc = {n: [] for n in kernel_names}  # dict[str, list[np.ndarray]]
        for b_i, (observation, actions) in enumerate(batches):
            _restore(gate_value)

            def loss_wrt_kernels(kernels):
                for i, blk in enumerate(model.fusion.blocks):
                    blk.out_proj.kernel.value = kernels[f"blocks/{i}/out_proj"]
                    blk.ffw_out.kernel.value = kernels[f"blocks/{i}/ffw_out"]
                # compute_loss returns (per_timestep_loss, stats) -- a TUPLE. Taking the mean of
                # the tuple itself is what broke the first attempt at a gradient read here.
                loss, _stats = model.compute_loss(
                    jax.random.key(1234 + b_i), observation, actions, train=True
                )
                return jnp.mean(loss)

            grads = jax.grad(loss_wrt_kernels)(dict(kernel_init_vals))  # dict[str, jax.Array]
            for n in kernel_names:
                g = np.asarray(grads[n], dtype=np.float64)  # np.ndarray
                acc[n].append(g)
            print(f"  [{label}] batch {b_i}: "
                  + ", ".join(f"{n.split('/')[-1]}{n.split('/')[1]} rms={np.sqrt((np.asarray(grads[n], dtype=np.float64)**2).mean()):.3e}"
                              for n in kernel_names))
        _restore(None)

        out = {}  # dict[str, dict]
        for n in kernel_names:
            stacked = np.stack(acc[n], axis=0)  # np.ndarray [batches, ...]
            flat = stacked.ravel()  # np.ndarray
            out[n] = {
                "rms": float(np.sqrt((flat ** 2).mean())),
                "std": float(flat.std()),
                "mean_abs": float(np.abs(flat).mean()),
                "n_elements": int(stacked[0].size),
            }
        return out

    print("\n=== CONDITION 1: gates at their TRAINED values (~3e-4) ===")
    trained = _run_condition("trained", None)  # dict
    print(f"\n=== CONDITION 2: gates forced to {OPEN_GATE_VALUE} (lever 1) ===")
    opened = _run_condition(f"gate{OPEN_GATE_VALUE}", OPEN_GATE_VALUE)  # dict

    return {
        "gates": gates,
        "conditions": {"trained_gates": trained, f"gate_{OPEN_GATE_VALUE}": opened},
        "eps": ADAM_EPS, "lr": ADAM_LR, "batches": len(batches),
        "open_gate_value": OPEN_GATE_VALUE,
    }


@app.local_entrypoint()
def main(num_batches: int = DEFAULT_NUM_BATCHES, batch_size: int = DEFAULT_BATCH_SIZE, seed: int = 0):
    """
    What it does:
        Triggers the measurement and prints, per kernel and condition, the
        per-element gradient RMS against AdamW's eps, plus the resulting
        per-step update magnitude -- then reads off whether the eps floor
        explains the frozen output projection and whether a non-zero gate
        init lifts it clear.

    Returns:
        None -- prints to stdout.

    Example input:
        modal run --detach XF_18k_eval/analysis/measure_out_proj_gradient.py

    Example output:
        (stdout) per-kernel RMS vs eps table and a verdict on the eps-floor hypothesis
    """
    r = measure.remote(num_batches=num_batches, batch_size=batch_size, seed=seed)  # dict
    eps, lr = r["eps"], r["lr"]  # float, float
    gv = r["open_gate_value"]  # float

    print(f"\n======== dL/d(out_proj) vs AdamW eps, checkpoint {CKPT_STEP} ========")
    print(f"batches: {r['batches']}   AdamW eps = {eps:.1e}   lr = {lr:.1e}")
    print(f"trained gates: { {k: f'{v:.3e}' for k, v in r['gates'].items()} }")

    hdr = f"\n{'kernel':<20} {'condition':<14} {'grad RMS':>12} {'vs eps':>10} {'adam step':>12}"  # str
    print(hdr)
    print("-" * len(hdr))
    below = {"trained_gates": 0, f"gate_{gv}": 0}  # dict[str, int]
    total = 0  # int
    for kernel in r["conditions"]["trained_gates"]:
        total += 1
        for cond_label, cond in r["conditions"].items():
            s = cond[kernel]  # dict
            rms = s["rms"]  # float
            ratio = rms / eps  # float
            step = lr * s["mean_abs"] / (rms + eps)  # float
            if rms < eps:
                below[cond_label] += 1
            flag = "BELOW" if rms < eps else "above"  # str
            print(f"{kernel if cond_label == 'trained_gates' else '':<20} {cond_label:<14} "
                  f"{rms:>12.3e} {ratio:>9.2f}x {step:>12.3e}  {flag}")

    print(f"\nkernels with gradient RMS BELOW eps: trained gates {below['trained_gates']}/{total}, "
          f"gates at {gv} {below[f'gate_{gv}']}/{total}")

    print("\n--------------------------------- reading ---------------------------------")
    t_below = below["trained_gates"]  # int
    o_below = below[f"gate_{gv}"]  # int
    if t_below >= total - 1 and o_below == 0:
        print(f"EPS FLOOR CONFIRMED. At the trained gates, out_proj's gradients sit BELOW AdamW's")
        print(f"eps={eps:.0e}, so the optimizer stops normalizing and the projection effectively")
        print(f"freezes -- which is what the unchanged 2.046 norm showed. Forcing the gate to {gv}")
        print("lifts them clear. Lever 1 (non-zero gate init) is justified on mechanism.")
    elif t_below == 0 and o_below == 0:
        print("EPS FLOOR NOT THE EXPLANATION. Gradients are above eps in BOTH conditions, so Adam")
        print("should have been normalizing and out_proj should have moved -- it did not. Something")
        print("else is holding it (a freeze filter, a detached path). FIND THAT before any retrain.")
    elif t_below >= total - 1 and o_below >= total - 1:
        print(f"GATE INIT ALONE IS INSUFFICIENT. Gradients remain below eps even at gate={gv},")
        print("so the auxiliary loss on F'/msg -- a path not scaled by tanh(a_x) at all -- becomes")
        print("the necessary lever, not an optional extra.")
    else:
        print("MIXED. The kernels disagree about which side of eps they sit on; read the table")
        print("per-kernel rather than taking a single verdict, and treat this as underpowered.")
    print()
