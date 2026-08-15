"""
smoke_test.py

Modal-based verification for Arm D's architecture (no local JAX/openpi
install exists on this machine, and this project's convention -- see
modal_reproduction/policy_smoke_test.py -- is to run these on a Modal GPU
container rather than guess that JAX code is correct from reading it). Random
init only, no checkpoint download: the point is to confirm every new shape-
dependent code path (joint_gated_modulator, history_gemma_dual, arm_d_pi0)
actually runs and produces the shapes/behavior the design calls for, not to
produce a trained or even sensible policy.

Three independent checks, each printing its own PASS/FAIL marker so one
failure doesn't hide the others:
  CHECK1 -- JointGatedModulator in isolation: gate sums to 1 and is in
            [0, 1], the router is exactly uniform (0.5/0.5) at init, the
            combined modulation is an exact identity (RMSNorm(x), no
            scale/shift) at init, and the load-balance loss is ~0 at init.
  CHECK2 -- DualMemoryModule (the scanned, multi-layer transformer stack):
            confirms the self.sow(...)-through-nn.scan mechanism used to
            collect per-layer gate/loss diagnostics actually produces
            correctly-shaped, finite values across all layers.
  CHECK3 -- ArmDModel end-to-end: builds a full (randomly initialized) Arm D
            policy from config/dynamic-fusion-arm-d.yaml, runs compute_loss
            and sample_actions on a synthetic batch, and checks output shapes
            and finiteness.

Run with:
    modal run arm_d_dynamic_fusion/smoke_test.py::shape_and_gate_test
"""

import pathlib

import modal

ROBOMME_LOCAL_DIR = str(  # str
    pathlib.Path(__file__).resolve().parent.parent / "robomme_policy_learning"
)
ARM_D_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent)  # str, this arm_d_dynamic_fusion/ directory

app = modal.App("robomme-arm-d-smoke-test")  # modal.App

image = (  # modal.Image
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "curl", "libgl1", "libglib2.0-0")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"UV_LINK_MODE": "copy", "UV_PYTHON_DOWNLOADS": "automatic"})
    .add_local_dir(ROBOMME_LOCAL_DIR, remote_path="/app", copy=True)
    # Same lockfile-workspace fix as modal_reproduction/policy_smoke_test.py --
    # sandbox2/flash_attn_jax is a declared uv workspace member that doesn't
    # exist in this checkout and nothing depends on it.
    .run_commands(
        r"""sed -i 's/members = \["packages\/\*", "sandbox2\/flash_attn_jax"\]/members = ["packages\/*"]/' /app/pyproject.toml"""
    )
    .run_commands("cd /app && /root/.local/bin/uv sync --no-dev --python 3.11")
    # openpi.models_pytorch.gemma_pytorch imports pytest unconditionally at
    # module level; --no-dev excluded it (same fix as policy_smoke_test.py).
    .run_commands("cd /app && /root/.local/bin/uv pip install pytest")
    # arm_d_dynamic_fusion/ itself, copied so its *contents* land under
    # /arm_d_root/arm_d_dynamic_fusion -- i.e. /arm_d_root on sys.path makes
    # `import arm_d_dynamic_fusion.models...` resolve, exactly mirroring how
    # /app/src makes `import mme_vla_suite...` resolve above.
    .add_local_dir(ARM_D_LOCAL_DIR, remote_path="/arm_d_root/arm_d_dynamic_fusion", copy=True)
)

SMOKE_TEST_SCRIPT = r'''
import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/arm_d_root")

import jax
import jax.numpy as jnp
import omegaconf

from openpi.models.gemma import Config

from arm_d_dynamic_fusion.models.joint_gated_modulator import JointGatedModulator
from arm_d_dynamic_fusion.models.history_gemma_dual import DualMemoryModule
from arm_d_dynamic_fusion.models.arm_d_pi0 import ArmDConfig

results = {}

def _record(name, ok, detail=""):
    results[name] = ok
    status = "OK" if ok else "FAIL"
    print(f"CHECK_{name}_{status}: {detail}")

# ---------------------------------------------------------------------------
# CHECK1: JointGatedModulator in isolation -- exact init-time math.
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

    jgm = JointGatedModulator()
    variables = jgm.init(k_init, x, mem_sym, mem_sym_mask, mem_perc, mem_perc_mask)
    modulated_x, stats = jgm.apply(variables, x, mem_sym, mem_sym_mask, mem_perc, mem_perc_mask)

    gate_sum = stats["gate_sym"] + stats["gate_perc"]
    gate_sum_ok = bool(jnp.allclose(gate_sum, 1.0, atol=1e-4))
    gate_range_ok = bool(jnp.all((stats["gate_sym"] >= 0) & (stats["gate_sym"] <= 1)))
    gate_uniform_ok = bool(jnp.allclose(stats["gate_sym"], 0.5, atol=1e-3))

    # MLP_sym/MLP_perc are near-zero-init (kernel_init_out_proj, stddev
    # 0.002 -- see joint_gated_modulator.py's docstring), not exactly zero,
    # matching the existing MemoryRMSNorm convention. So modulated_x is only
    # approximately RMSNorm(x) at init, not exactly -- check the deviation is
    # small (consistent with that init scale) rather than exactly zero.
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    expected_identity = (x.astype(jnp.float32) * jax.lax.rsqrt(variance + 1e-6)).astype(x.dtype)
    max_abs_diff = float(jnp.max(jnp.abs(modulated_x - expected_identity)))
    near_identity_at_init_ok = max_abs_diff < 0.5

    balance_near_zero_ok = bool(jnp.allclose(stats["balance_loss"], 0.0, atol=1e-3))

    all_ok = gate_sum_ok and gate_range_ok and gate_uniform_ok and near_identity_at_init_ok and balance_near_zero_ok
    _record(
        "1_JOINT_GATED_MODULATOR",
        all_ok,
        f"gate_sum_ok={gate_sum_ok} gate_range_ok={gate_range_ok} "
        f"gate_uniform_ok={gate_uniform_ok} near_identity_at_init_ok={near_identity_at_init_ok} "
        f"max_abs_diff={max_abs_diff:.6f} "
        f"balance_near_zero_ok={balance_near_zero_ok} "
        f"balance_loss={float(stats['balance_loss']):.6f}",
    )
except Exception as e:  # noqa: BLE001
    _record("1_JOINT_GATED_MODULATOR", False, f"{type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# CHECK2: DualMemoryModule -- the scanned multi-layer stack + sow/scan
# mechanism used to collect per-layer gate diagnostics.
# ---------------------------------------------------------------------------
try:
    import flax.linen as nn

    cfg = Config(width=1024, depth=3, mlp_dim=2048, num_heads=8, num_kv_heads=1, head_dim=256)
    module = DualMemoryModule(configs=[cfg, cfg], embed_dtype="float32", adarms=True, integration_type="dynamic_fusion")

    key = jax.random.key(1)
    k_init, k_vlm, k_act, k_sym, k_perc = jax.random.split(key, 5)
    # DualMemoryModule.init(use_adarms, mem_mods) is a convenience method
    # (matching the released history_gemma.Module's own "init" method) that
    # SHADOWS flax's built-in Module.init at the Python attribute level --
    # calling module.init(...) directly reaches that convenience method, not
    # flax's init machinery. Call flax's real Module.init explicitly via the
    # unbound base-class method, pointing it at the convenience method as the
    # trace target (this is exactly what nnx_bridge.ToNNX.lazy_init(...,
    # method="init", ...) does internally for the real ArmDModel).
    variables = nn.Module.init(
        module, k_init, use_adarms=[False, True], mem_mods=[False, True], method=DualMemoryModule.init
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

    # DualMemoryModule.__call__ returns (outputs_list, kv_cache) as its own
    # 2-tuple; with mutable=["intermediates"], .apply() wraps THAT whole
    # thing as element 0 of its own outer 2-tuple, with the sown collections
    # as element 1 -- i.e. ((outputs_list, kv_cache), mutated), not
    # ((vlm_out, action_out), mutated).
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

    # self.sow(...) is called inside DualMemoryHistoryBlock, which is the
    # scanned submodule bound to DualMemoryModule.layers -- sown values are
    # therefore nested under that submodule's attribute name ("layers"), not
    # at the top level of the "intermediates" collection.
    intermediates_layers = mutated["intermediates"]["layers"]
    gate_sym_layers = intermediates_layers["gate_sym"][0]
    gate_perc_layers = intermediates_layers["gate_perc"][0]
    balance_layers = intermediates_layers["balance_loss"][0]
    depth_ok = gate_sym_layers.shape[0] == cfg.depth
    gate_sum_ok = bool(jnp.allclose(gate_sym_layers + gate_perc_layers, 1.0, atol=1e-4))
    balance_finite_ok = bool(jnp.all(jnp.isfinite(balance_layers)))

    all_ok = shape_ok and finite_ok and depth_ok and gate_sum_ok and balance_finite_ok
    _record(
        "2_DUAL_MEMORY_MODULE",
        all_ok,
        f"shape_ok={shape_ok} finite_ok={finite_ok} depth_ok={depth_ok} "
        f"gate_sum_ok={gate_sum_ok} balance_finite_ok={balance_finite_ok} "
        f"gate_sym_layers.shape={gate_sym_layers.shape}",
    )
except Exception as e:  # noqa: BLE001
    _record("2_DUAL_MEMORY_MODULE", False, f"{type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# CHECK3: ArmDModel end-to-end (compute_loss + sample_actions).
# ---------------------------------------------------------------------------
try:
    history_cfg = omegaconf.OmegaConf.load(
        "/arm_d_root/arm_d_dynamic_fusion/config/dynamic-fusion-arm-d.yaml"
    )
    config = ArmDConfig(
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

    loss = model.compute_loss(jax.random.key(3), fake_obs, fake_actions, train=True)
    loss_shape_ok = loss.shape == (batch_size, config.action_horizon)
    loss_finite_ok = bool(jnp.all(jnp.isfinite(loss)))

    sampled_actions = model.sample_actions(jax.random.key(4), fake_obs, num_steps=2)
    sample_shape_ok = sampled_actions.shape == (batch_size, config.action_horizon, config.action_dim)
    sample_finite_ok = bool(jnp.all(jnp.isfinite(sampled_actions)))

    all_ok = loss_shape_ok and loss_finite_ok and sample_shape_ok and sample_finite_ok
    _record(
        "3_ARM_D_MODEL_END_TO_END",
        all_ok,
        f"loss_shape_ok={loss_shape_ok} loss_finite_ok={loss_finite_ok} "
        f"sample_shape_ok={sample_shape_ok} sample_finite_ok={sample_finite_ok} "
        f"loss.shape={loss.shape} sampled_actions.shape={sampled_actions.shape}",
    )
except Exception as e:  # noqa: BLE001
    import traceback
    _record("3_ARM_D_MODEL_END_TO_END", False, f"{type(e).__name__}: {e}\\n{traceback.format_exc()}")

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
        {"success": True, "detail": "CHECK_1_JOINT_GATED_MODULATOR_OK: ...\\n..."}
    """
    import subprocess  # module

    script_path = "/tmp/arm_d_smoke_test.py"  # str
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
        modal run arm_d_dynamic_fusion/smoke_test.py

    Example output:
        (stdout) "success: True" followed by the full check-by-check detail.
    """
    result = shape_and_gate_test.remote()  # dict
    print(f"success: {result['success']}")
    print(result["detail"])
