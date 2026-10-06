"""
smoke_test.py

Modal-based verification for the symbolic-as-modulator probe (no local
JAX/openpi install exists on this machine, same convention every other arm's
smoke_test.py in this project uses -- these checks run on a Modal GPU
container rather than being guessed correct from reading the code). Random
init only, no checkpoint download: the point is to confirm the new code
(symbolic_mem_encoder, symbolic_modulator_pi0) actually runs end-to-end and
produces the shapes the design calls for, not to produce a trained or even
sensible policy. Most of the mechanism this probe exercises
(history_gemma.Module's "modulation" integration path, MemoryAttention,
AdaLN-Zero scale/shift) is UNFORKED code already relied on by the released
perceptual+modulation variants -- so unlike B1/D's smoke tests, there is no
new fused-gate math to check in isolation. Two checks:
  CHECK1 -- SymbolicMemoryEncoder in isolation: projects PaliGemma-width
            (2048) subgoal embeddings to action-expert width (1024) with the
            right output shape, mask passed through unchanged.
  CHECK2 -- SymbolicModulatorModel end-to-end: builds a full (randomly
            initialized) policy from config/symbolic-modulator-only.yaml,
            runs compute_loss and sample_actions on a synthetic batch, and
            checks output shapes and finiteness.

Run with:
    modal run symbolic_as_modulator/smoke_test.py
"""

import pathlib

import modal

ROBOMME_LOCAL_DIR = str(  # str
    pathlib.Path(__file__).resolve().parent.parent / "robomme_policy_learning"
)
PROBE_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent)  # str, this symbolic_as_modulator/ directory

app = modal.App("robomme-symbolic-modulator-smoke-test")  # modal.App

image = (  # modal.Image
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "curl", "libgl1", "libglib2.0-0")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"UV_LINK_MODE": "copy", "UV_PYTHON_DOWNLOADS": "automatic"})
    .add_local_dir(ROBOMME_LOCAL_DIR, remote_path="/app", copy=True)
    # Same lockfile-workspace fix as every other arm's smoke_test.py --
    # sandbox2/flash_attn_jax is a declared uv workspace member that doesn't
    # exist in this checkout and nothing depends on it.
    .run_commands(
        r"""sed -i 's/members = \["packages\/\*", "sandbox2\/flash_attn_jax"\]/members = ["packages\/*"]/' /app/pyproject.toml"""
    )
    .run_commands("cd /app && /root/.local/bin/uv sync --no-dev --python 3.11")
    # openpi.models_pytorch.gemma_pytorch imports pytest unconditionally at
    # module level; --no-dev excluded it (same fix as policy_smoke_test.py).
    .run_commands("cd /app && /root/.local/bin/uv pip install pytest")
    # symbolic_as_modulator/ itself, copied so its *contents* land under
    # /probe_root/symbolic_as_modulator -- i.e. /probe_root on sys.path makes
    # `import symbolic_as_modulator.models...` resolve, exactly mirroring how
    # /app/src makes `import mme_vla_suite...` resolve above.
    .add_local_dir(PROBE_LOCAL_DIR, remote_path="/probe_root/symbolic_as_modulator", copy=True)
)

SMOKE_TEST_SCRIPT = r'''
import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/probe_root")

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import omegaconf

from symbolic_as_modulator.models.symbolic_mem_encoder import SymbolicMemoryEncoder
from symbolic_as_modulator.models.symbolic_modulator_pi0 import SymbolicModulatorConfig

results = {}

def _record(name, ok, detail=""):
    results[name] = ok
    status = "OK" if ok else "FAIL"
    print(f"CHECK_{name}_{status}: {detail}")

# ---------------------------------------------------------------------------
# CHECK1: SymbolicMemoryEncoder in isolation -- the one genuinely new module.
# ---------------------------------------------------------------------------
try:
    b, l, embed_dim, output_dim = 2, 64, 2048, 1024
    key = jax.random.key(0)
    k_embed, k_init = jax.random.split(key, 2)

    subgoal_embeddings = jax.random.normal(k_embed, (b, l, embed_dim))
    subgoal_mask = jnp.ones((b, l), dtype=bool)

    encoder = SymbolicMemoryEncoder(
        rngs=nnx.Rngs(k_init), dtype=jnp.float32, embed_dim=embed_dim, output_dim=output_dim
    )
    m_sym, mask_out = encoder(subgoal_embeddings, subgoal_mask)

    shape_ok = m_sym.shape == (b, l, output_dim)
    finite_ok = bool(jnp.all(jnp.isfinite(m_sym)))
    mask_passthrough_ok = bool(jnp.array_equal(mask_out, subgoal_mask))

    all_ok = shape_ok and finite_ok and mask_passthrough_ok
    _record(
        "1_SYMBOLIC_MEM_ENCODER",
        all_ok,
        f"shape_ok={shape_ok} finite_ok={finite_ok} mask_passthrough_ok={mask_passthrough_ok} "
        f"m_sym.shape={m_sym.shape}",
    )
except Exception as e:  # noqa: BLE001
    _record("1_SYMBOLIC_MEM_ENCODER", False, f"{type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# CHECK2: SymbolicModulatorModel end-to-end (compute_loss + sample_actions).
# ---------------------------------------------------------------------------
try:
    history_cfg = omegaconf.OmegaConf.load(
        "/probe_root/symbolic_as_modulator/config/symbolic-modulator-only.yaml"
    )
    config = SymbolicModulatorConfig(
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
    # Use model.config, NOT the original `config`: SymbolicModulatorConfig.
    # create() doubles max_token_len for representation_type=="symbolic"
    # (matching ModelTransformFactory's own independent doubling of the same
    # yaml field -- see symbolic_modulator_pi0.py's create() docstring), and
    # model.config is the POST-doubling config the model was actually built
    # with. Calling inputs_spec() on the pre-create() `config` here would
    # declare a symbolic_tokenized_prompt shape at the UNDOUBLED length,
    # mismatching what the model's own action_in_proj/BaseModel machinery
    # expects.
    obs_spec, action_spec = model.config.inputs_spec(batch_size=batch_size)

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
    loss_stats_ok = loss_stats is None  # bool, matches this probe's own None contract (see embed_memory)

    sampled_actions = model.sample_actions(jax.random.key(4), fake_obs, num_steps=2)
    sample_shape_ok = sampled_actions.shape == (batch_size, config.action_horizon, config.action_dim)
    sample_finite_ok = bool(jnp.all(jnp.isfinite(sampled_actions)))

    all_ok = loss_shape_ok and loss_finite_ok and loss_stats_ok and sample_shape_ok and sample_finite_ok
    _record(
        "2_SYMBOLIC_MODULATOR_MODEL_END_TO_END",
        all_ok,
        f"loss_shape_ok={loss_shape_ok} loss_finite_ok={loss_finite_ok} "
        f"loss_stats_ok={loss_stats_ok} "
        f"sample_shape_ok={sample_shape_ok} sample_finite_ok={sample_finite_ok} "
        f"loss.shape={loss.shape} sampled_actions.shape={sampled_actions.shape}",
    )
except Exception as e:  # noqa: BLE001
    import traceback
    _record("2_SYMBOLIC_MODULATOR_MODEL_END_TO_END", False, f"{type(e).__name__}: {e}\\n{traceback.format_exc()}")

overall_ok = all(results.values())
print("SMOKE_TEST_OVERALL_" + ("OK" if overall_ok else "FAIL"))
'''


@app.function(image=image, gpu="A10G", timeout=1200)
def shape_test() -> dict:
    """
    What it does:
        Writes SMOKE_TEST_SCRIPT to a file inside the Modal container and
        runs it with the uv-managed venv's Python, capturing stdout/stderr.

    Returns:
        dict -- {"success": bool, "detail": str}. success is True only if
        every CHECK printed an _OK marker (i.e. SMOKE_TEST_OVERALL_OK is in
        stdout).

    Example input:
        shape_test.remote()

    Example output:
        {"success": True, "detail": "CHECK_1_SYMBOLIC_MEM_ENCODER_OK: ...\\n..."}
    """
    import subprocess  # module

    script_path = "/tmp/symbolic_modulator_smoke_test.py"  # str
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
        CLI entrypoint -- runs shape_test() and prints the result.

    Returns:
        None -- prints to stdout.

    Example input:
        modal run symbolic_as_modulator/smoke_test.py

    Example output:
        (stdout) "success: True" followed by the full check-by-check detail.
    """
    result = shape_test.remote()  # dict
    print(f"success: {result['success']}")
    print(result["detail"])
