"""
smoke_test.py

Modal-based verification for Arm B1's architecture (no local JAX/openpi
install exists on this machine, same convention arm_d_dynamic_fusion/
smoke_test.py and modal_reproduction/policy_smoke_test.py already use --
these checks run on a Modal GPU container rather than being guessed correct
from reading the code). Random init only, no checkpoint download: the point
is to confirm every new shape-dependent code path (static_gated_modulator,
history_gemma_static, b1_pi0) actually runs and produces the shapes/behavior
the design calls for, not to produce a trained or even sensible policy.

Three independent checks, each printing its own PASS/FAIL marker so one
failure doesn't hide the others -- same structure as Arm D's smoke test, with
CHECK1 testing the different thing B1 actually needs verified (a FIXED gate,
not a learned one):
  CHECK1 -- StaticGatedModulator in isolation: gate is EXACTLY 0.5/0.5 for
            every call (not just at init -- there is no init-vs-trained
            distinction for a gate with no parameters), the combined
            modulation is a near-identity (RMSNorm(x), small near-zero
            scale/shift) at init, and no balance_loss key exists in stats
            (unlike Arm D, which has one).
  CHECK2 -- StaticMemoryModule (the scanned, multi-layer transformer stack):
            confirms the self.sow(...)-through-nn.scan mechanism used to
            collect per-layer gate diagnostics actually produces
            correctly-shaped, finite, exactly-0.5 values across all layers.
  CHECK3 -- B1Model end-to-end: builds a full (randomly initialized) B1
            policy from config/static-fusion-arm-b1.yaml, runs compute_loss
            and sample_actions on a synthetic batch, and checks output shapes
            and finiteness.

Run with:
    modal run arm_b1_static_fusion/smoke_test.py::shape_and_gate_test
"""

import pathlib

import modal

ROBOMME_LOCAL_DIR = str(  # str
    pathlib.Path(__file__).resolve().parent.parent / "robomme_policy_learning"
)
B1_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent)  # str, this arm_b1_static_fusion/ directory

app = modal.App("robomme-b1-smoke-test")  # modal.App

image = (  # modal.Image
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "curl", "libgl1", "libglib2.0-0")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"UV_LINK_MODE": "copy", "UV_PYTHON_DOWNLOADS": "automatic"})
    .add_local_dir(ROBOMME_LOCAL_DIR, remote_path="/app", copy=True)
    # Same lockfile-workspace fix as arm_d_dynamic_fusion/smoke_test.py --
    # sandbox2/flash_attn_jax is a declared uv workspace member that doesn't
    # exist in this checkout and nothing depends on it.
    .run_commands(
        r"""sed -i 's/members = \["packages\/\*", "sandbox2\/flash_attn_jax"\]/members = ["packages\/*"]/' /app/pyproject.toml"""
    )
    .run_commands("cd /app && /root/.local/bin/uv sync --no-dev --python 3.11")
    # openpi.models_pytorch.gemma_pytorch imports pytest unconditionally at
    # module level; --no-dev excluded it (same fix as policy_smoke_test.py).
    .run_commands("cd /app && /root/.local/bin/uv pip install pytest")
    # arm_b1_static_fusion/ itself, copied so its *contents* land under
    # /b1_root/arm_b1_static_fusion -- i.e. /b1_root on sys.path makes
    # `import arm_b1_static_fusion.models...` resolve, exactly mirroring how
    # /app/src makes `import mme_vla_suite...` resolve above.
    .add_local_dir(B1_LOCAL_DIR, remote_path="/b1_root/arm_b1_static_fusion", copy=True)
)

SMOKE_TEST_SCRIPT = r'''
import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/b1_root")

import jax
import jax.numpy as jnp
import omegaconf

from openpi.models.gemma import Config

from arm_b1_static_fusion.models.static_gated_modulator import StaticGatedModulator
from arm_b1_static_fusion.models.history_gemma_static import StaticMemoryModule
from arm_b1_static_fusion.models.b1_pi0 import B1Config

results = {}

def _record(name, ok, detail=""):
    results[name] = ok
    status = "OK" if ok else "FAIL"
    print(f"CHECK_{name}_{status}: {detail}")

# ---------------------------------------------------------------------------
# CHECK1: StaticGatedModulator in isolation -- exact fixed-gate math.
# ---------------------------------------------------------------------------
try:
    width = 1024
    b, t, s_sym, s_perc = 2, 6, 8, 12
    key = jax.random.key(0)
    k_x, k_sym, k_perc, k_init = jax.random.split(key, 4)

    x = jax.random.normal(k_x, (b, t, width))
    mem_sym = jax.random.normal(k_sym, (b, s_sym, width))
    mem_sym_mask = jnp.ones((b, s_sym), dtype=bool)
    mem_perc = jax.random.normal(k_perc, (b, s_perc, width))
    mem_perc_mask = jnp.ones((b, s_perc), dtype=bool)

    sgm = StaticGatedModulator()
    variables = sgm.init(k_init, x, mem_sym, mem_sym_mask, mem_perc, mem_perc_mask)
    modulated_x, stats = sgm.apply(variables, x, mem_sym, mem_sym_mask, mem_perc, mem_perc_mask)

    # Unlike Arm D's gate (softmax output, only uniform at init), B1's gate
    # has no parameters at all -- it must be EXACTLY 0.5 always, checked with
    # a tight tolerance rather than the "near uniform" check Arm D uses.
    gate_exactly_half_ok = bool(jnp.allclose(stats["gate_sym"], 0.5, atol=1e-7)) and bool(
        jnp.allclose(stats["gate_perc"], 0.5, atol=1e-7)
    )
    no_balance_loss_key_ok = "balance_loss" not in stats

    # MLP_sym/MLP_perc are near-zero-init (kernel_init_out_proj), not exactly
    # zero -- modulated_x is only approximately RMSNorm(x) at init.
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    expected_identity = (x.astype(jnp.float32) * jax.lax.rsqrt(variance + 1e-6)).astype(x.dtype)
    max_abs_diff = float(jnp.max(jnp.abs(modulated_x - expected_identity)))
    near_identity_at_init_ok = max_abs_diff < 0.5

    all_ok = gate_exactly_half_ok and no_balance_loss_key_ok and near_identity_at_init_ok
    _record(
        "1_STATIC_GATED_MODULATOR",
        all_ok,
        f"gate_exactly_half_ok={gate_exactly_half_ok} no_balance_loss_key_ok={no_balance_loss_key_ok} "
        f"near_identity_at_init_ok={near_identity_at_init_ok} max_abs_diff={max_abs_diff:.6f}",
    )
except Exception as e:  # noqa: BLE001
    _record("1_STATIC_GATED_MODULATOR", False, f"{type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# CHECK2: StaticMemoryModule -- the scanned multi-layer stack + sow/scan
# mechanism used to collect per-layer gate diagnostics.
# ---------------------------------------------------------------------------
try:
    import flax.linen as nn

    cfg = Config(width=1024, depth=3, mlp_dim=2048, num_heads=8, num_kv_heads=1, head_dim=256)
    module = StaticMemoryModule(configs=[cfg, cfg], embed_dtype="float32", adarms=True, integration_type="static_fusion")

    key = jax.random.key(1)
    k_init, k_vlm, k_act, k_sym, k_perc = jax.random.split(key, 5)
    # StaticMemoryModule.init(...) SHADOWS flax's built-in Module.init at the
    # Python attribute level, same quirk arm_d_dynamic_fusion/smoke_test.py's
    # CHECK2 documents -- call flax's real Module.init explicitly, pointing it
    # at the convenience method as the trace target.
    variables = nn.Module.init(
        module, k_init, use_adarms=[False, True], mem_mods=[False, True], method=StaticMemoryModule.init
    )

    b, vlm_len, act_len, s_sym, s_perc = 2, 5, 6, 8, 12
    vlm_tokens = jax.random.normal(k_vlm, (b, vlm_len, cfg.width))
    action_tokens = jax.random.normal(k_act, (b, act_len, cfg.width))
    mem_sym = jax.random.normal(k_sym, (b, s_sym, cfg.width))
    mem_sym_mask = jnp.ones((b, s_sym), dtype=bool)
    mem_perc = jax.random.normal(k_perc, (b, s_perc, cfg.width))
    mem_perc_mask = jnp.ones((b, s_perc), dtype=bool)
    adarms_cond = jnp.zeros((b, cfg.width))

    total_len = vlm_len + act_len
    positions = jnp.broadcast_to(jnp.arange(total_len), (b, total_len))
    mask = jnp.ones((b, total_len, total_len), dtype=bool)

    (outputs, kv_cache), mutated = module.apply(
        variables,
        [vlm_tokens, action_tokens],
        positions,
        mask,
        adarms_cond=[None, adarms_cond],
        mem_seq_sym=[None, mem_sym], mem_mask_sym=[None, mem_sym_mask],
        mem_seq_perc=[None, mem_perc], mem_mask_perc=[None, mem_perc_mask],
        mutable=["intermediates"],
    )
    vlm_out, action_out = outputs

    shape_ok = action_out.shape == (b, act_len, cfg.width) and vlm_out.shape == (b, vlm_len, cfg.width)
    finite_ok = bool(jnp.all(jnp.isfinite(action_out))) and bool(jnp.all(jnp.isfinite(vlm_out)))

    intermediates_layers = mutated["intermediates"]["layers"]
    gate_sym_layers = intermediates_layers["gate_sym"][0]
    gate_perc_layers = intermediates_layers["gate_perc"][0]
    depth_ok = gate_sym_layers.shape[0] == cfg.depth
    gate_exactly_half_ok = bool(jnp.allclose(gate_sym_layers, 0.5, atol=1e-7)) and bool(
        jnp.allclose(gate_perc_layers, 0.5, atol=1e-7)
    )

    all_ok = shape_ok and finite_ok and depth_ok and gate_exactly_half_ok
    _record(
        "2_STATIC_MEMORY_MODULE",
        all_ok,
        f"shape_ok={shape_ok} finite_ok={finite_ok} depth_ok={depth_ok} "
        f"gate_exactly_half_ok={gate_exactly_half_ok} gate_sym_layers.shape={gate_sym_layers.shape}",
    )
except Exception as e:  # noqa: BLE001
    _record("2_STATIC_MEMORY_MODULE", False, f"{type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# CHECK3: B1Model end-to-end (compute_loss + sample_actions).
# ---------------------------------------------------------------------------
try:
    history_cfg = omegaconf.OmegaConf.load(
        "/b1_root/arm_b1_static_fusion/config/static-fusion-arm-b1.yaml"
    )
    config = B1Config(
        dtype="float32",
        pi05=True,
        action_dim=8,
        action_horizon=20,
        max_token_len=64,
        use_history=True,
        history_config=history_cfg,
    )
    model = config.create(jax.random.key(2))

    batch_size = 2
    obs_spec, action_spec = config.inputs_spec(batch_size=batch_size)

    def _fill(spec):
        if jnp.issubdtype(spec.dtype, jnp.floating):
            return jnp.full(spec.shape, 0.01, dtype=spec.dtype)
        if spec.dtype == jnp.bool_:
            return jnp.ones(spec.shape, dtype=jnp.bool_)
        return jnp.zeros(spec.shape, dtype=spec.dtype)

    fake_obs = jax.tree_util.tree_map(
        _fill, obs_spec, is_leaf=lambda leaf: isinstance(leaf, jax.ShapeDtypeStruct)
    )
    fake_actions = _fill(action_spec)

    loss, loss_stats = model.compute_loss(jax.random.key(3), fake_obs, fake_actions, train=True)
    loss_shape_ok = loss.shape == (batch_size, config.action_horizon)
    loss_finite_ok = bool(jnp.all(jnp.isfinite(loss)))
    loss_stats_ok = loss_stats is None  # bool, matches Arm A/D's own None contract

    sampled_actions = model.sample_actions(jax.random.key(4), fake_obs, num_steps=2)
    sample_shape_ok = sampled_actions.shape == (batch_size, config.action_horizon, config.action_dim)
    sample_finite_ok = bool(jnp.all(jnp.isfinite(sampled_actions)))

    all_ok = loss_shape_ok and loss_finite_ok and loss_stats_ok and sample_shape_ok and sample_finite_ok
    _record(
        "3_B1_MODEL_END_TO_END",
        all_ok,
        f"loss_shape_ok={loss_shape_ok} loss_finite_ok={loss_finite_ok} "
        f"loss_stats_ok={loss_stats_ok} "
        f"sample_shape_ok={sample_shape_ok} sample_finite_ok={sample_finite_ok} "
        f"loss.shape={loss.shape} sampled_actions.shape={sampled_actions.shape}",
    )
except Exception as e:  # noqa: BLE001
    import traceback
    _record("3_B1_MODEL_END_TO_END", False, f"{type(e).__name__}: {e}\\n{traceback.format_exc()}")

overall_ok = all(results.values())
print("SMOKE_TEST_OVERALL_" + ("OK" if overall_ok else "FAIL"))
'''


@app.function(image=image, gpu="A10G", timeout=1200)
def shape_and_gate_test() -> dict:
    """
    What it does:
        Writes SMOKE_TEST_SCRIPT to a file inside the Modal container and
        runs it with the uv-managed venv's Python, capturing stdout/stderr.

    Returns:
        dict -- {"success": bool, "detail": str}. success is True only if
        every CHECK printed an _OK marker (i.e. SMOKE_TEST_OVERALL_OK is in
        stdout).

    Example input:
        shape_and_gate_test.remote()

    Example output:
        {"success": True, "detail": "CHECK_1_STATIC_GATED_MODULATOR_OK: ...\\n..."}
    """
    import subprocess  # module

    script_path = "/tmp/b1_smoke_test.py"  # str
    with open(script_path, "w") as f:
        f.write(SMOKE_TEST_SCRIPT)

    result = subprocess.run(  # subprocess.CompletedProcess
        ["/app/.venv/bin/python", script_path],
        cwd="/app", capture_output=True, text=True, timeout=1100,
    )
    output = result.stdout + "\n--- STDERR ---\n" + result.stderr  # str
    print(output)
    return {
        "success": result.returncode == 0 and "SMOKE_TEST_OVERALL_OK" in result.stdout,
        "detail": output[-8000:],
    }


@app.local_entrypoint()
def main():
    """
    What it does:
        CLI entrypoint -- runs shape_and_gate_test() and prints the result.

    Returns:
        None -- prints to stdout.

    Example input:
        modal run arm_b1_static_fusion/smoke_test.py

    Example output:
        (stdout) "success: True" followed by the full check-by-check detail.
    """
    result = shape_and_gate_test.remote()  # dict
    print(f"success: {result['success']}")
    print(result["detail"])
