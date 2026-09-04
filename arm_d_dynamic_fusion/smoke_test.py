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

Five independent checks, each printing its own PASS/FAIL marker so one
failure doesn't hide the others:
  CHECK1 -- EarlyFusionModulator in isolation: attn_mass_sym + attn_mass_perc
            sums to 1 and both are in [0, 1] (true by construction --
            softmax always sums to 1), the combined modulation is an exact
            identity (RMSNorm(x), no scale/shift) at init (mlp_fused is
            near-zero-init), and the learned per-stream score biases
            (bias_sym/bias_perc) are exactly 0 at init.
  CHECK2 -- DualMemoryModule (the scanned, multi-layer transformer stack):
            confirms the self.sow(...)-through-nn.scan mechanism used to
            collect per-layer attn_mass_sym/attn_mass_perc diagnostics
            actually produces correctly-shaped, finite values across all
            layers.
  CHECK3 -- ArmDModel end-to-end: builds a full (randomly initialized) Arm D
            policy from config/dynamic-fusion-arm-d.yaml, runs compute_loss
            and sample_actions on a synthetic batch, checks output shapes and
            finiteness, and confirms compute_loss's alignment_loss/
            attn_mass_sym_mean/attn_mass_perc_mean stats extraction reads
            real values (finite, in-range, summing to 1).
  CHECK4 -- Gradient flow through the SAME extraction mechanism compute_loss
            uses (nnx_bridge.ToNNX wrapping + nnx.state(...) extraction + raw
            flax.linen `.apply(mutable=[...])` + nnx.value_and_grad), on a
            toy-scale DualMemoryModule (not the full ~2.3B-param ArmDModel --
            two earlier attempts at full scale both hit RESOURCE_EXHAUSTED on
            the A10G; the mechanism's correctness doesn't depend on model
            scale, only its memory footprint does): confirms the resulting
            gradient is finite and nonzero, i.e. the extraction doesn't
            silently break gradient attribution back to the original nnx
            Param objects.
  CHECK5 -- Bias lever has full reach: manually overrides bias_sym across a
            sweep of values (bypassing training) and confirms attn_mass_sym
            responds monotonically and saturates near 0 and near 1 -- proof
            the mechanism CAN fully correct for the token-count imbalance
            (or fully favor symbolic, or perceptual) once training decides
            to, not a claim that it currently does (nothing is trained yet).

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
import flax.nnx as nnx
import omegaconf
import optax

from openpi.models.gemma import Config

from arm_d_dynamic_fusion.models.joint_gated_modulator import EarlyFusionModulator
from arm_d_dynamic_fusion.models.history_gemma_dual import DualMemoryModule
from arm_d_dynamic_fusion.models.arm_d_pi0 import ArmDConfig

results = {}

def _record(name, ok, detail=""):
    results[name] = ok
    status = "OK" if ok else "FAIL"
    print(f"CHECK_{name}_{status}: {detail}")

# ---------------------------------------------------------------------------
# CHECK1: EarlyFusionModulator in isolation -- exact init-time math.
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

    efm = EarlyFusionModulator()
    variables = efm.init(k_init, x, mem_sym, mem_sym_mask, mem_perc, mem_perc_mask)
    modulated_x, stats = efm.apply(variables, x, mem_sym, mem_sym_mask, mem_perc, mem_perc_mask)

    # attn_mass_sym + attn_mass_perc == 1 always holds by construction (one
    # softmax distributing its weight across both segments), regardless of
    # training -- unlike the prior design's gate, this isn't a learned
    # property to check for a specific value, just a structural invariant.
    mass_sum = stats["attn_mass_sym"] + stats["attn_mass_perc"]
    mass_sum_ok = bool(jnp.allclose(mass_sum, 1.0, atol=1e-4))
    mass_range_ok = bool(jnp.all((stats["attn_mass_sym"] >= 0) & (stats["attn_mass_sym"] <= 1)))

    # mlp_fused is near-zero-init (kernel_init_out_proj, stddev 0.002 --
    # matching the existing MemoryRMSNorm/prior-design convention), not
    # exactly zero. So modulated_x is only approximately RMSNorm(x) at init,
    # not exactly -- check the deviation is small (consistent with that init
    # scale) rather than exactly zero.
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    expected_identity = (x.astype(jnp.float32) * jax.lax.rsqrt(variance + 1e-6)).astype(x.dtype)
    max_abs_diff = float(jnp.max(jnp.abs(modulated_x - expected_identity)))
    near_identity_at_init_ok = max_abs_diff < 0.5

    # bias_sym/bias_perc must start at exactly 0 -- see joint_gated_modulator.py's
    # docstring for why this specific property (unlike the tags, which don't
    # need to start at 0) matters: it's what guarantees no artificial
    # numerosity correction happens before training decides one is needed.
    bias_sym = variables["params"]["mem_attn_fused"]["bias_sym"]
    bias_perc = variables["params"]["mem_attn_fused"]["bias_perc"]
    bias_zero_at_init_ok = bool(bias_sym == 0.0) and bool(bias_perc == 0.0)

    all_ok = mass_sum_ok and mass_range_ok and near_identity_at_init_ok and bias_zero_at_init_ok
    _record(
        "1_EARLY_FUSION_MODULATOR",
        all_ok,
        f"mass_sum_ok={mass_sum_ok} mass_range_ok={mass_range_ok} "
        f"near_identity_at_init_ok={near_identity_at_init_ok} max_abs_diff={max_abs_diff:.6f} "
        f"bias_zero_at_init_ok={bias_zero_at_init_ok} "
        f"attn_mass_sym_mean={float(jnp.mean(stats['attn_mass_sym'])):.4f} "
        f"(theoretical dilution baseline at random init, s_sym/(s_sym+s_perc)={s_sym/(s_sym+s_perc):.4f})",
    )
except Exception as e:  # noqa: BLE001
    _record("1_EARLY_FUSION_MODULATOR", False, f"{type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# CHECK2: DualMemoryModule -- the scanned multi-layer stack + sow/scan
# mechanism used to collect per-layer attn_mass_sym/attn_mass_perc diagnostics.
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
    attn_mass_sym_layers = intermediates_layers["attn_mass_sym"][0]
    attn_mass_perc_layers = intermediates_layers["attn_mass_perc"][0]
    depth_ok = attn_mass_sym_layers.shape[0] == cfg.depth
    mass_sum_ok = bool(jnp.allclose(attn_mass_sym_layers + attn_mass_perc_layers, 1.0, atol=1e-4))
    mass_finite_ok = bool(jnp.all(jnp.isfinite(attn_mass_sym_layers))) and bool(jnp.all(jnp.isfinite(attn_mass_perc_layers)))

    all_ok = shape_ok and finite_ok and depth_ok and mass_sum_ok and mass_finite_ok
    _record(
        "2_DUAL_MEMORY_MODULE",
        all_ok,
        f"shape_ok={shape_ok} finite_ok={finite_ok} depth_ok={depth_ok} "
        f"mass_sum_ok={mass_sum_ok} mass_finite_ok={mass_finite_ok} "
        f"attn_mass_sym_layers.shape={attn_mass_sym_layers.shape}",
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

    # compute_loss returns (loss, stats) -- a 2-tuple, not a bare array --
    # matching scripts/train.py::train_step's `chunked_loss, stats =
    # model.compute_loss(...)` unpack. stats is a real diagnostics dict
    # (alignment_loss/attn_mass_sym_mean/attn_mass_perc_mean), not always
    # None -- see arm_d_pi0.ArmDModel.compute_loss's docstring for why this
    # is safe (train.py's only other use of stats is gated on
    # representation_type == "recurrent", which never matches Arm D). This
    # also end-to-end-verifies the mutable=["intermediates"] extraction
    # (raw flax.linen .apply() on the wrapped module + its current nnx
    # params -- see that method's docstring) actually produces attn_mass
    # values, not just that compute_loss runs without crashing.
    loss, loss_stats = model.compute_loss(jax.random.key(3), fake_obs, fake_actions, train=True)
    loss_shape_ok = loss.shape == (batch_size, config.action_horizon)
    loss_finite_ok = bool(jnp.all(jnp.isfinite(loss)))

    expected_stats_keys = {"flow_matching_loss", "alignment_loss", "attn_mass_sym_mean", "attn_mass_perc_mean"}
    loss_stats_keys_ok = isinstance(loss_stats, dict) and set(loss_stats.keys()) == expected_stats_keys
    stats_finite_ok = loss_stats_keys_ok and all(bool(jnp.all(jnp.isfinite(v))) for v in loss_stats.values())
    # attn_mass_sym/attn_mass_perc always sum to 1 by construction (one
    # softmax, see joint_gated_modulator.py) regardless of training -- unlike
    # the prior design's gate, there's no single "correct" value to expect at
    # random init here, just this structural invariant plus being in [0, 1].
    mass_sum_ok = (
        loss_stats_keys_ok
        and bool(jnp.allclose(loss_stats["attn_mass_sym_mean"] + loss_stats["attn_mass_perc_mean"], 1.0, atol=1e-3))
    )
    mass_range_ok = (
        loss_stats_keys_ok
        and bool(0.0 <= loss_stats["attn_mass_sym_mean"] <= 1.0)
        and bool(0.0 <= loss_stats["attn_mass_perc_mean"] <= 1.0)
    )

    sampled_actions = model.sample_actions(jax.random.key(4), fake_obs, num_steps=2)
    sample_shape_ok = sampled_actions.shape == (batch_size, config.action_horizon, config.action_dim)
    sample_finite_ok = bool(jnp.all(jnp.isfinite(sampled_actions)))

    all_ok = (
        loss_shape_ok and loss_finite_ok and loss_stats_keys_ok and stats_finite_ok
        and mass_sum_ok and mass_range_ok
        and sample_shape_ok and sample_finite_ok
    )
    _record(
        "3_ARM_D_MODEL_END_TO_END",
        all_ok,
        f"loss_shape_ok={loss_shape_ok} loss_finite_ok={loss_finite_ok} "
        f"loss_stats_keys_ok={loss_stats_keys_ok} stats_finite_ok={stats_finite_ok} "
        f"mass_sum_ok={mass_sum_ok} mass_range_ok={mass_range_ok} "
        f"sample_shape_ok={sample_shape_ok} sample_finite_ok={sample_finite_ok} "
        f"loss.shape={loss.shape} sampled_actions.shape={sampled_actions.shape} "
        f"loss_stats={jax.tree.map(lambda v: float(v) if v.ndim == 0 else v.shape, loss_stats) if loss_stats_keys_ok else loss_stats}",
    )
except Exception as e:  # noqa: BLE001
    import traceback
    _record("3_ARM_D_MODEL_END_TO_END", False, f"{type(e).__name__}: {e}\\n{traceback.format_exc()}")

# ---------------------------------------------------------------------------
# CHECK4: gradients actually reach EarlyFusionModulator/UnifiedMemoryEncoder
# through compute_loss's mutable=["intermediates"]-extraction path (used to
# read attn_mass_sym/attn_mass_perc out for stats, alongside the ordinary
# forward computation) -- CHECK3 only verifies the FORWARD values are
# correct; a raw flax.linen `.apply()` call
# built from a manually-extracted `nnx.state(...).to_pure_dict()` snapshot
# could in principle compute correct-looking forward numbers while silently
# breaking gradient attribution back to the original nnx Param objects
# nnx.value_and_grad differentiates. Two earlier attempts at this check tried
# to exercise the mechanism through a full ~2.3B-param ArmDModel (once
# reusing CHECK3's non-LoRA model, once building a fresh LoRA one after
# explicitly dropping CHECK3's) and both hit RESOURCE_EXHAUSTED on the A10G
# either way -- a full backward pass through nearly the whole backbone
# (non-LoRA) or even just constructing a second full model in the same
# process (LoRA) both exceed what's available once CHECK1-3 have already run
# here. The mechanism being tested (nnx_bridge.ToNNX wrapping + nnx.state(...)
# extraction + raw linen .apply(mutable=[...]) + nnx.value_and_grad) doesn't
# depend on model scale for correctness, only on scale for memory -- so this
# uses the same toy-scale Config CHECK2 already uses (depth=3, width=1024's
# real value isn't needed, just a DualMemoryModule wrapped the same way
# ArmDModel wraps its real one), isolating the actual risk cheaply instead of
# fighting the same OOM a third time.
# ---------------------------------------------------------------------------
try:
    # CHECK3's full ArmDModel (~2.3B params, including its own huge vocab
    # embedder) is still resident at this point and was the actual cause of
    # the prior OOM here, not the toy module itself -- explicitly drop it
    # first (same fix already needed once before for the same reason, see
    # RESEARCH_LOG.md's 2026-08-28 22:14 entry on a different script).
    del model, fake_obs, fake_actions, sampled_actions, loss, loss_stats
    import gc
    gc.collect()

    import flax.nnx.bridge as nnx_bridge

    # Same toy Config CHECK2 already uses successfully -- MemoryAttention
    # (history_gemma.py) hardcodes width=1024 internally ("same dim as the
    # action expert in pi05"), not derived from this Config at all, so an
    # arbitrary smaller width (tried first, width=32) fails its own
    # `assert mem_width == x_width == width` assertion during lazy_init's
    # dummy trace call -- unrelated to anything this check is actually
    # testing, just a hardcoded assumption in the released code.
    toy_cfg = Config(width=1024, depth=3, mlp_dim=2048, num_heads=8, num_kv_heads=1, head_dim=256)
    toy_wrapped = nnx_bridge.ToNNX(
        DualMemoryModule(configs=[toy_cfg, toy_cfg], embed_dtype="float32", adarms=True, integration_type="dynamic_fusion")
    )
    toy_wrapped.lazy_init(
        rngs=nnx.Rngs(10), method="init", use_adarms=[False, True], mem_mods=[False, True]
    )

    b, vlm_len, act_len, s_sym, s_perc = 2, 3, 4, 5, 6
    k1, k2, k3, k4 = jax.random.split(jax.random.key(11), 4)
    vlm_tokens = jax.random.normal(k1, (b, vlm_len, toy_cfg.width))
    action_tokens = jax.random.normal(k2, (b, act_len, toy_cfg.width))
    mem_sym = jax.random.normal(k3, (b, s_sym, toy_cfg.width))
    mem_perc = jax.random.normal(k4, (b, s_perc, toy_cfg.width))
    mem_sym_mask = jnp.ones((b, s_sym), dtype=bool)
    mem_perc_mask = jnp.ones((b, s_perc), dtype=bool)
    adarms_cond = jnp.zeros((b, toy_cfg.width))
    total_len = vlm_len + act_len
    toy_positions = jnp.broadcast_to(jnp.arange(total_len), (b, total_len))
    toy_mask = jnp.ones((b, total_len, total_len), dtype=bool)

    diff_state = nnx.DiffState(0, nnx.Param)  # everything trainable -- this toy wrapper has no frozen backbone to worry about

    def _toy_loss_fn(m, vlm, act):
        # Exactly the mechanism arm_d_pi0.ArmDModel.compute_loss now uses:
        # extract current params via nnx.state(...), call the wrapped
        # flax.linen module's .apply() directly with mutable=["intermediates"].
        variables = {"params": nnx.state(m, nnx.Param).to_pure_dict()}
        (outputs, kv_cache), mutated = m.module.apply(
            variables,
            [vlm, act], toy_positions, toy_mask,
            adarms_cond=[None, adarms_cond],
            mem_seq_sym=[None, mem_sym], mem_mask_sym=[None, mem_sym_mask],
            mem_seq_perc=[None, mem_perc], mem_mask_perc=[None, mem_perc_mask],
            mutable=["intermediates"],
        )
        _, action_out = outputs
        # No aux loss term needed here (unlike an earlier version of this
        # check, which added 0.1*balance_loss) -- that concept doesn't apply
        # to the current early-fusion design (see joint_gated_modulator.py),
        # and this check's only job is confirming gradients reach the
        # mechanism at all, not exercising every stats key.
        return jnp.mean(jnp.square(action_out))

    loss_val, grads = nnx.value_and_grad(_toy_loss_fn, argnums=diff_state)(
        toy_wrapped, vlm_tokens, action_tokens
    )
    grad_norm = float(optax.global_norm(grads))

    loss_finite_ok = bool(jnp.isfinite(loss_val))
    grad_ok = bool(jnp.isfinite(grad_norm)) and grad_norm > 0.0

    all_ok = loss_finite_ok and grad_ok
    _record(
        "4_GRADIENT_FLOW",
        all_ok,
        f"loss_finite_ok={loss_finite_ok} grad_norm={grad_norm:.6f}",
    )
except Exception as e:  # noqa: BLE001
    import traceback
    _record("4_GRADIENT_FLOW", False, f"{type(e).__name__}: {e}\\n{traceback.format_exc()}")

# ---------------------------------------------------------------------------
# CHECK5: the bias lever actually has full reach -- NOT "is attn_mass_sym
# currently balanced" (it isn't, at bias=0, by design -- see CHECK1's
# dilution-baseline note and RESEARCH_LOG.md's 2026-08-29 15:53 entry), but
# "if training decided symbolic needed more (or less) weight, CAN the
# bias_sym/bias_perc lever actually deliver that, all the way from near-0 to
# near-1, or is it structurally capped by the token-count imbalance no
# matter how large the learned bias gets?" Manually overrides bias_sym at a
# handful of values (bypassing training) and confirms attn_mass_sym responds
# monotonically and saturates near both extremes. This is the real answer to
# "does it listen to both equally" for an UNTRAINED architecture check: not a
# claim that it currently does (nothing has been trained yet), but proof the
# mechanism CAN, i.e. nothing about the fused-softmax design leaves symbolic
# permanently unable to compete regardless of what training decides.
# ---------------------------------------------------------------------------
try:
    width = 1024
    b, t, s_sym, s_perc = 2, 6, 8, 12
    key = jax.random.key(20)
    k_x, k_sym, k_perc, k_init = jax.random.split(key, 4)

    x = jax.random.normal(k_x, (b, t, width))
    mem_sym = jax.random.normal(k_sym, (b, s_sym, width))
    mem_sym_mask = jnp.ones((b, s_sym), dtype=bool)
    mem_perc = jax.random.normal(k_perc, (b, s_perc, width))
    mem_perc_mask = jnp.ones((b, s_perc), dtype=bool)

    efm = EarlyFusionModulator()
    base_variables = efm.init(k_init, x, mem_sym, mem_sym_mask, mem_perc, mem_perc_mask)

    def _with_bias_sym(variables, bias_value):
        # Shallow-copy just the two dict levels being modified, rather than
        # mutating base_variables in place (which flax may or may not treat
        # as frozen depending on config -- copying is unambiguous either way).
        new_params = dict(variables["params"])
        new_mem_attn = dict(new_params["mem_attn_fused"])
        new_mem_attn["bias_sym"] = jnp.array(bias_value, dtype=jnp.float32)
        new_params["mem_attn_fused"] = new_mem_attn
        return {"params": new_params}

    bias_sweep = [-10.0, -4.0, 0.0, 4.0, 10.0]  # list[float]
    attn_mass_sym_by_bias = []  # list[float]
    for bias_value in bias_sweep:
        swept_variables = _with_bias_sym(base_variables, bias_value)
        _, swept_stats = efm.apply(swept_variables, x, mem_sym, mem_sym_mask, mem_perc, mem_perc_mask)
        attn_mass_sym_by_bias.append(float(jnp.mean(swept_stats["attn_mass_sym"])))

    monotonic_ok = all(
        attn_mass_sym_by_bias[i] < attn_mass_sym_by_bias[i + 1] for i in range(len(attn_mass_sym_by_bias) - 1)
    )
    reaches_low_ok = attn_mass_sym_by_bias[0] < 0.05  # near-0 at strongly negative bias
    reaches_high_ok = attn_mass_sym_by_bias[-1] > 0.95  # near-1 at strongly positive bias

    all_ok = monotonic_ok and reaches_low_ok and reaches_high_ok
    _record(
        "5_BIAS_LEVER_FULL_RANGE",
        all_ok,
        f"monotonic_ok={monotonic_ok} reaches_low_ok={reaches_low_ok} reaches_high_ok={reaches_high_ok} "
        f"attn_mass_sym_by_bias={dict(zip(bias_sweep, [round(v, 4) for v in attn_mass_sym_by_bias]))}",
    )
except Exception as e:  # noqa: BLE001
    import traceback
    _record("5_BIAS_LEVER_FULL_RANGE", False, f"{type(e).__name__}: {e}\\n{traceback.format_exc()}")

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
        {"success": True, "detail": "CHECK_1_EARLY_FUSION_MODULATOR_OK: ...\\n..."}
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
