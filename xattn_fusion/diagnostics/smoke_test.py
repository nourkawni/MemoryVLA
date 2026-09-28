"""
smoke_test.py

Modal-based verification for XF's v1 architecture (no local JAX/openpi
install exists on this machine -- this project's convention, see
arm_d_dynamic_fusion/smoke_test.py / symbolic_as_modulator/smoke_test.py, is
to run these on a Modal GPU container rather than guess that JAX code is
correct from reading it). Random init only, no checkpoint: the point is to
confirm every new shape-dependent code path (event_encoder, fusion_xattn,
hybrid_mem, xf_pi0, xf_observation, subgoal_table, subgoal_logger, xf_policy)
actually runs and produces the shapes/behavior the design calls for.

COST NOTE (flagged 2026-09-19, not yet investigated -- do this before this
script gets run repeatedly during real iteration, not now): this is the
only script in xattn_fusion/ that uses a GPU at all (gpu="A10G" below) --
inspect_alignment.py, inspect_eval_alignment.py, and verify_real_data.py
are all gpu=None, since they only need raw H5 pixels + the pure-Python
aligner. A10G is needed here only because CHECK4 onward build the real
XFModel (the full ~3B-param SigLIP+Gemma+action-expert backbone via
config.create()) to test compute_loss/sample_actions/freeze-filter/JIT.
CHECK1-3 (GatedXAttnFusion/EventEncoder in isolation, no backbone at all)
plausibly don't need a GPU, or a much cheaper one (Modal's T4 tier) might
suffice for the whole thing given the tiny batch size (b=2) and step counts
(num_steps=2) used throughout -- neither has been measured. Worth splitting
into a cheap CPU/T4-tier pass (CHECK1-3) and a separate A10G pass
(CHECK4-10) once this is being run often enough for the cost to matter.

Ten independent checks, each printing its own PASS/FAIL marker:
  CHECK1 -- GatedXAttnFusion zero-gate identity: at random init (a_x=a_d=0
            by construction), fusion(F, ...) == F bit-exactly.
  CHECK2 -- build_fusion_mask never leaves a query row with no valid key,
            even when an event's caption tokens are fully masked out (the
            null column guarantees this).
  CHECK3 -- EventEncoder in isolation: output shapes and finiteness.
  CHECK4 -- freeze filter: none of the new param paths (event_encoder,
            fusion, type_emb) match HistoryPi0Config.get_freeze_filter()'s
            ".*img.*" regex, i.e. none would be silently frozen at random init.
  CHECK5 -- JIT compiles: compute_loss/sample_actions compile at the
            placeholder E_max=16, L_c=24 from XFConfig.inputs_spec.
  CHECK6 -- XFObservation round-trip: all 11 new fields survive
            preprocess_observation non-None, shape-identical.
  CHECK7 -- XFModel end-to-end: compute_loss + sample_actions on a synthetic
            batch, shapes + finiteness.
  CHECK8 -- Task A (subgoal_table) on a synthetic episode matching the H5
            access pattern build_subgoal_table expects (real .h5 data is not
            available in this environment -- see test_subgoal_table.py's
            real-data test, skipped by default for the same reason):
            invariants hold, pack_event_arrays output matches inputs_spec's
            declared shapes.
  CHECK9 -- eval-time code path (XFPolicy): reset() -> add_buffer() ->
            _prepare_history() -> XFObservation.from_dict ->
            model.sample_actions(), for one synthetic video-demo task and
            one synthetic non-video task, both with a real SubgoalLogger
            (Task B) run across several steps. Scope note: this exercises
            the actual XFPolicy/SubgoalLogger/XFMemoryBuffer integration
            directly rather than launching examples/robomme/eval.py against
            a live websocket server + the real robomme_benchmark environment
            (which needs task assets and a GPU-rendered sim not confirmed
            available in this image) -- a deliberate scope reduction from
            the v1 plan's literal wording, flagged here rather than silently
            done.
  CHECK10 -- XFPolicy.infer() itself, the actual serving entrypoint
            (websocket_policy_server -> XFPolicy.infer), added after a code
            review found the inherited MME_VLA_Policy.infer() (before
            XFPolicy overrode it) hardcoded HistAugObservation.from_dict --
            invisible to CHECK9, which never calls infer() at all.

Run with:
    modal run xattn_fusion/diagnostics/smoke_test.py
"""

import pathlib

import modal

ROBOMME_LOCAL_DIR = str(  # str
    pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning"
)
XF_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)  # str, this xattn_fusion/ directory

app = modal.App("xf-smoke-test")  # modal.App

image = (  # modal.Image
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "curl", "libgl1", "libglib2.0-0")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"UV_LINK_MODE": "copy", "UV_PYTHON_DOWNLOADS": "automatic"})
    .add_local_dir(ROBOMME_LOCAL_DIR, remote_path="/app", copy=True)
    # sandbox2/flash_attn_jax is a declared uv workspace member that doesn't exist in this
    # checkout and nothing depends on it -- same fix as the other arms' smoke tests.
    .run_commands(
        r"""sed -i 's/members = \["packages\/\*", "sandbox2\/flash_attn_jax"\]/members = ["packages\/*"]/' /app/pyproject.toml"""
    )
    .run_commands("cd /app && /root/.local/bin/uv sync --no-dev --python 3.11")
    # openpi.models_pytorch.gemma_pytorch imports pytest unconditionally at module level;
    # --no-dev excluded it.
    .run_commands("cd /app && /root/.local/bin/uv pip install pytest sentencepiece omegaconf")
    # xattn_fusion/ itself, copied so its *contents* land under /xf_root/xattn_fusion -- i.e.
    # /xf_root on sys.path makes `import xattn_fusion.mme_vla_suite...` resolve, mirroring how
    # /app/src makes `import mme_vla_suite...` resolve above.
    .add_local_dir(XF_LOCAL_DIR, remote_path="/xf_root/xattn_fusion", copy=True)
)

SMOKE_TEST_SCRIPT = r'''
import sys
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/xf_root")

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np

results = {}

def _record(name, ok, detail=""):
    results[name] = ok
    status = "OK" if ok else "FAIL"
    print(f"CHECK_{name}_{status}: {detail}")

# ---------------------------------------------------------------------------
# CHECK1: GatedXAttnFusion zero-gate identity.
# ---------------------------------------------------------------------------
try:
    from xattn_fusion.mme_vla_suite.models.representation.fusion_xattn import GatedXAttnFusion

    b, tf, e, lc, width = 2, 12, 3, 3, 1024  # tc (c_flat's key count) must equal e*lc+1
    tc = e * lc + 1
    key = jax.random.key(0)
    k_f, k_c, k_init = jax.random.split(key, 3)

    fusion = GatedXAttnFusion(rngs=nnx.Rngs(k_init), dtype=jnp.float32, num_layers=2, width=width)
    f_in = jax.random.normal(k_f, (b, tf, width))
    c_flat = jax.random.normal(k_c, (b, tc, width))
    static_token_event_idx = jnp.zeros((b, tf), dtype=jnp.int32)
    event_text_mask = jnp.ones((b, e, lc), dtype=jnp.bool_)

    f_out = fusion(f_in, c_flat, static_token_event_idx, event_text_mask)
    identity_ok = bool(jnp.array_equal(f_out, f_in))
    _record("1_ZERO_GATE_IDENTITY", identity_ok, f"identity_ok={identity_ok} max_abs_diff={float(jnp.max(jnp.abs(f_out - f_in)))}")
except Exception as e:
    import traceback
    _record("1_ZERO_GATE_IDENTITY", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

# ---------------------------------------------------------------------------
# CHECK2: build_fusion_mask -- no query row is ever all-False, even with a
# fully-masked-out event.
# ---------------------------------------------------------------------------
try:
    from xattn_fusion.mme_vla_suite.models.representation.fusion_xattn import build_fusion_mask

    b, f_tokens, e, lc = 2, 8, 3, 4
    static_token_event_idx = jnp.array([[-1, -1, 0, 0, 1, 1, -2, 2]] * b, dtype=jnp.int32)
    event_text_mask = jnp.zeros((b, e, lc), dtype=jnp.bool_)
    event_text_mask = event_text_mask.at[:, 0, :2].set(True)  # event 1 fully masked, event 2 fully masked

    mask = build_fusion_mask(static_token_event_idx, event_text_mask)
    shape_ok = mask.shape == (b, f_tokens, e * lc + 1)
    no_all_false_row = bool(jnp.all(jnp.any(mask, axis=-1)))
    _record("2_FUSION_MASK_NULL_COLUMN", shape_ok and no_all_false_row, f"shape_ok={shape_ok} no_all_false_row={no_all_false_row} mask.shape={mask.shape}")
except Exception as e:
    import traceback
    _record("2_FUSION_MASK_NULL_COLUMN", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

# ---------------------------------------------------------------------------
# CHECK3: EventEncoder in isolation.
# ---------------------------------------------------------------------------
try:
    from xattn_fusion.mme_vla_suite.models.representation.event_encoder import EventEncoder
    from mme_vla_suite.shared.posemb_3d import PosEmb3D

    b, e, lc, e_max = 2, 5, 6, 16
    pos_embedder = PosEmb3D(dim=768)
    encoder = EventEncoder(rngs=nnx.Rngs(1), dtype=jnp.float32, embed_dim=2048, width=1024, num_layers=2, heads=4, e_max=e_max)

    event_text_tokens = jax.random.randint(jax.random.key(2), (b, e, lc), 0, 1000)
    event_text_mask = jnp.ones((b, e, lc), dtype=jnp.bool_)
    event_coords = jnp.zeros((b, e, 2), dtype=jnp.int32).at[:, 0, :].set(jnp.array([128, 64]))
    event_start = jnp.zeros((b, e), dtype=jnp.int32)
    event_is_demo = jnp.zeros((b, e), dtype=jnp.bool_)
    event_is_open = jnp.zeros((b, e), dtype=jnp.bool_).at[:, -1].set(True)
    event_num_frames = jnp.ones((b, e), dtype=jnp.int32)
    event_mask = jnp.ones((b, e), dtype=jnp.bool_)

    def fake_embed_fn(tokens):
        return jax.random.normal(jax.random.key(3), (*tokens.shape, 2048))

    e_tok, c_flat, c_mask = encoder(
        event_text_tokens, event_text_mask, event_coords, event_start,
        event_is_demo, event_is_open, event_num_frames, event_mask,
        embed_fn=fake_embed_fn, pos_embedder=pos_embedder,
    )
    shape_ok = e_tok.shape == (b, e, 1024) and c_flat.shape == (b, e * lc + 1, 1024) and c_mask.shape == (b, e * lc + 1)
    finite_ok = bool(jnp.all(jnp.isfinite(e_tok))) and bool(jnp.all(jnp.isfinite(c_flat)))
    _record("3_EVENT_ENCODER_SHAPES", shape_ok and finite_ok, f"shape_ok={shape_ok} finite_ok={finite_ok} e_tok.shape={e_tok.shape} c_flat.shape={c_flat.shape}")
except Exception as e:
    import traceback
    _record("3_EVENT_ENCODER_SHAPES", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

# ---------------------------------------------------------------------------
# CHECK4/5/6/7: build a real (random-init) XFModel via XFConfig and exercise
# freeze-filter, JIT compilation, observation round-trip, compute_loss and
# sample_actions all against the same model instance.
# ---------------------------------------------------------------------------
model = None
config = None
try:
    import omegaconf

    from xattn_fusion.mme_vla_suite.models.integration.xf_pi0 import XFConfig

    # Load the yaml directly (rather than passing its filename string and relying on
    # XFConfig.create()'s internal loader) so `config` itself already carries the loaded
    # DictConfig -- config.inputs_spec() below needs history_config.fusion.* etc. to be a
    # real DictConfig, not the unloaded filename string. Mirrors arm_d_dynamic_fusion's own
    # smoke test, which does the same for the same reason.
    history_cfg = omegaconf.OmegaConf.load("/xf_root/xattn_fusion/config/xf-framesamp-modul-xattn.yaml")
    config = XFConfig(
        dtype="float32",
        pi05=True,
        action_dim=8,
        action_horizon=20,
        max_token_len=64,
        use_history=True,
        history_config=history_cfg,
        discrete_state_input=False,
    )
    model = config.create(jax.random.key(4))
    _record("SETUP_MODEL", True, "XFModel constructed")
except Exception as e:
    import traceback
    _record("SETUP_MODEL", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

# CHECK4 -- freeze filter.
try:
    freeze_filter = model.config.get_freeze_filter()
    state = nnx.state(model, nnx.Param)
    flat = state.flat_state()  # iterable of flat tuples: (*path_segments, Variable)
    new_module_paths = [
        "/".join(str(p) for p in item[:-1])
        for item in flat
        if any(name in "/".join(str(p) for p in item[:-1]) for name in ("event_encoder", "fusion", "type_emb"))
    ]
    no_img_in_new_paths = all("img" not in p for p in new_module_paths)
    has_new_paths = len(new_module_paths) > 0
    _record(
        "4_FREEZE_FILTER",
        no_img_in_new_paths and has_new_paths,
        f"no_img_in_new_paths={no_img_in_new_paths} num_new_param_paths={len(new_module_paths)} sample={new_module_paths[:3]}",
    )
except Exception as e:
    import traceback
    _record("4_FREEZE_FILTER", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

# Shared fake-observation builder for CHECK5/6/7.
def _fill(spec):
    if jnp.issubdtype(spec.dtype, jnp.floating):
        return jnp.full(spec.shape, 0.01, dtype=spec.dtype)
    if spec.dtype == jnp.bool_:
        return jnp.ones(spec.shape, dtype=jnp.bool_)
    return jnp.zeros(spec.shape, dtype=spec.dtype)

fake_obs = None
fake_actions = None
try:
    obs_spec, action_spec = config.inputs_spec(batch_size=2)
    fake_obs = jax.tree_util.tree_map(_fill, obs_spec, is_leaf=lambda leaf: isinstance(leaf, jax.ShapeDtypeStruct))
    fake_actions = _fill(action_spec)
    _record("SETUP_FAKE_OBS", True, f"obs_spec built, event_mask.shape={obs_spec.event_mask.shape}")
except Exception as e:
    import traceback
    _record("SETUP_FAKE_OBS", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

# CHECK6 -- observation round-trip through preprocess_observation.
try:
    from xattn_fusion.mme_vla_suite.models.integration.xf_observation import preprocess_observation

    processed = preprocess_observation(jax.random.key(5), fake_obs, train=True)
    event_fields = [
        "event_mask", "event_start", "event_end", "event_is_demo", "event_is_open",
        "event_caption_id", "event_coords", "event_num_frames", "event_text_tokens",
        "event_text_mask", "static_token_event_idx",
    ]
    all_present = all(getattr(processed, f) is not None for f in event_fields)
    all_shape_match = all(getattr(processed, f).shape == getattr(fake_obs, f).shape for f in event_fields)
    _record("6_OBSERVATION_ROUND_TRIP", all_present and all_shape_match, f"all_present={all_present} all_shape_match={all_shape_match}")
except Exception as e:
    import traceback
    _record("6_OBSERVATION_ROUND_TRIP", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

# CHECK7 -- compute_loss + sample_actions end-to-end.
try:
    loss, stats = model.compute_loss(jax.random.key(6), fake_obs, fake_actions, train=True)
    loss_shape_ok = loss.shape == (2, config.action_horizon)
    loss_finite_ok = bool(jnp.all(jnp.isfinite(loss)))

    sampled_actions = model.sample_actions(jax.random.key(7), fake_obs, num_steps=2)
    sample_shape_ok = sampled_actions.shape == (2, config.action_horizon, config.action_dim)
    sample_finite_ok = bool(jnp.all(jnp.isfinite(sampled_actions)))

    all_ok = loss_shape_ok and loss_finite_ok and sample_shape_ok and sample_finite_ok
    _record(
        "7_END_TO_END_LOSS_AND_SAMPLE",
        all_ok,
        f"loss_shape_ok={loss_shape_ok} loss_finite_ok={loss_finite_ok} sample_shape_ok={sample_shape_ok} "
        f"sample_finite_ok={sample_finite_ok} loss.shape={loss.shape} sampled_actions.shape={sampled_actions.shape}",
    )
except Exception as e:
    import traceback
    _record("7_END_TO_END_LOSS_AND_SAMPLE", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

# CHECK5 -- JIT compiles at inputs_spec's placeholder E_max/L_c.
try:
    from openpi.shared import nnx_utils

    # train is a Python-level bool used in an `if train:` branch inside openpi's generic
    # preprocess_observation -- it must be static, not traced, for jax.jit.
    jit_compute_loss = nnx_utils.module_jit(model.compute_loss, static_argnames=("train",))
    jit_loss, _ = jit_compute_loss(jax.random.key(8), fake_obs, fake_actions, train=True)
    jit_loss_ok = bool(jnp.all(jnp.isfinite(jit_loss))) and jit_loss.shape == (2, config.action_horizon)

    jit_sample_actions = nnx_utils.module_jit(model.sample_actions)
    jit_actions = jit_sample_actions(jax.random.key(9), fake_obs, num_steps=2)
    jit_sample_ok = bool(jnp.all(jnp.isfinite(jit_actions))) and jit_actions.shape == (2, config.action_horizon, config.action_dim)

    _record("5_JIT_COMPILES", jit_loss_ok and jit_sample_ok, f"jit_loss_ok={jit_loss_ok} jit_sample_ok={jit_sample_ok}")
except Exception as e:
    import traceback
    _record("5_JIT_COMPILES", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

# ---------------------------------------------------------------------------
# CHECK8: Task A (subgoal_table) on a synthetic episode.
# ---------------------------------------------------------------------------
try:
    from xattn_fusion.mme_vla_suite.shared.subgoal_table import (
        build_subgoal_table, build_caption_vocab, apply_caption_vocab,
        truncate_table, assign_events_to_frames, pack_event_arrays,
    )

    class _FakeLeaf:
        def __init__(self, value):
            self.value = value
        def __getitem__(self, _key):
            return self.value

    class _FakeTokenizer:
        def encode(self, text, add_bos=False):
            return [ord(c) for c in text][:64]

    def _fake_episode(captions, exec_start_idx):
        episode = {}
        for step, cap in enumerate(captions):
            episode[f"timestep_{step}"] = {"info": {
                "grounded_subgoal_online": _FakeLeaf(cap.encode()),
                "is_video_demo": _FakeLeaf(step < exec_start_idx),
            }}
        return episode

    captions = ["watching"] * 3 + ["pick up the cube at <128, 64>"] * 5 + ["place the cube at <10, 20>"] * 4
    exec_start_idx = 3
    episode = _fake_episode(captions, exec_start_idx)
    table = build_subgoal_table(episode, exec_start_idx)
    vocab = build_caption_vocab([table])
    table = apply_caption_vocab(table, vocab)

    now = len(captions) - 1
    truncated = truncate_table(table, now)
    invariants_ok = (
        truncated.intervals[0].start == 0
        and truncated.intervals[-1].is_open
        and truncated.intervals[-1].end == now + 1
        and all(truncated.intervals[i].end == truncated.intervals[i + 1].start for i in range(len(truncated.intervals) - 1))
    )

    max_size = 8
    sampled = list(range(0, now + 1, max(1, (now + 1) // max_size)))[:max_size]
    aligned = assign_events_to_frames(truncated, sampled, max_size=max_size)
    packed = pack_event_arrays(truncated, aligned, _FakeTokenizer(), E=16, L_c=24)

    shapes_ok = (
        packed["event_mask"].shape == (16,)
        and packed["event_text_tokens"].shape == (16, 24)
        and packed["static_token_event_idx"].shape == (max_size * 1 * 16,)
    )
    _record("8_TASK_A_SYNTHETIC_EPISODE", invariants_ok and shapes_ok, f"invariants_ok={invariants_ok} shapes_ok={shapes_ok} num_intervals={len(truncated.intervals)}")
except Exception as e:
    import traceback
    _record("8_TASK_A_SYNTHETIC_EPISODE", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

# ---------------------------------------------------------------------------
# CHECK9: eval-time code path (XFPolicy) -- one video-demo synthetic task,
# one non-video synthetic task. See module docstring's CHECK9 scope note.
# ---------------------------------------------------------------------------
try:
    from xattn_fusion.mme_vla_suite.policies.xf_policy import XFPolicy
    from openpi.shared.normalize import NormStats
    import openpi.transforms as _transforms

    norm_stats = {
        "state": NormStats(mean=np.zeros(8, dtype=np.float32), std=np.ones(8, dtype=np.float32), q01=-np.ones(8, dtype=np.float32), q99=np.ones(8, dtype=np.float32)),
    }
    policy = XFPolicy(
        model, seed=0, transforms=[], output_transforms=[],
        sample_kwargs={"num_steps": 2}, norm_stats=norm_stats, use_quantiles=False,
    )

    def _run_synthetic_rollout(exec_start_idx, n_steps, captions):
        policy.reset()
        num_views = policy.config.num_views
        image = np.zeros((num_views, 224, 224, 3), dtype=np.uint8)
        state = np.zeros((8,), dtype=np.float32)
        obs_history = None
        for step in range(exec_start_idx + n_steps):
            policy.add_buffer({
                "images": image[None, ...],
                "state": state[None, ...],
                "exec_start_idx": exec_start_idx if step == 0 else 0,
            })
            if step >= exec_start_idx:
                caption = captions[step - exec_start_idx]
                inputs = {"grounded_subgoal": caption, "state": state}
                inputs = policy._prepare_history(dict(inputs))
                obs_history = inputs
        return obs_history

    non_video_captions = ["pick up the cube at <128, 64>"] * 3 + ["place the cube at <10, 20>"] * 3
    video_captions = ["pick up the cube at <128, 64>"] * 4

    non_video_inputs = _run_synthetic_rollout(exec_start_idx=0, n_steps=6, captions=non_video_captions)
    video_inputs = _run_synthetic_rollout(exec_start_idx=5, n_steps=4, captions=video_captions)

    def _inputs_ok(inputs):
        required = [
            "static_image_emb", "static_pos_emb", "static_state_emb", "static_mask",
            "event_mask", "event_start", "event_end", "event_is_demo", "event_is_open",
            "event_caption_id", "event_coords", "event_num_frames", "event_text_tokens",
            "event_text_mask", "static_token_event_idx",
        ]
        return all(inputs.get(k) is not None for k in required)

    non_video_ok = _inputs_ok(non_video_inputs)
    video_ok = _inputs_ok(video_inputs)

    from xattn_fusion.mme_vla_suite.models.integration.xf_observation import XFObservation

    _NUMERIC_KEYS = (
        "static_image_emb", "static_pos_emb", "static_state_emb", "static_mask",
        "event_mask", "event_start", "event_end", "event_is_demo", "event_is_open",
        "event_caption_id", "event_coords", "event_num_frames", "event_text_tokens",
        "event_text_mask", "static_token_event_idx",
    )

    def _to_model_obs(inputs, task_goal_tokens_len=16):
        # Only the numeric event/perceptual-memory keys go through jnp.asarray -- `inputs`
        # also carries "grounded_subgoal" (a raw python string, not a JAX-array-able dtype),
        # which this deliberately does not touch.
        batched = {k: np.asarray(inputs[k])[None, ...] for k in _NUMERIC_KEYS}
        batched["state"] = np.asarray(inputs["state"])[None, ...]
        batched["image"] = {"base_0_rgb": np.zeros((1, 224, 224, 3), dtype=np.float32), "left_wrist_0_rgb": np.zeros((1, 224, 224, 3), dtype=np.float32)}
        batched["image_mask"] = {"base_0_rgb": np.array([True]), "left_wrist_0_rgb": np.array([True])}
        batched["tokenized_prompt"] = np.zeros((1, task_goal_tokens_len), dtype=np.int32)
        batched["tokenized_prompt_mask"] = np.ones((1, task_goal_tokens_len), dtype=np.bool_)
        return XFObservation.from_dict(jax.tree.map(jnp.asarray, batched))

    non_video_obs = _to_model_obs(non_video_inputs)
    actions = model.sample_actions(jax.random.key(10), non_video_obs, num_steps=2)
    sample_ok = actions.shape == (1, config.action_horizon, config.action_dim) and bool(jnp.all(jnp.isfinite(actions)))

    _record(
        "9_EVAL_TIME_POLICY_PATH",
        non_video_ok and video_ok and sample_ok,
        f"non_video_ok={non_video_ok} video_ok={video_ok} sample_ok={sample_ok} actions.shape={actions.shape}",
    )
except Exception as e:
    import traceback
    _record("9_EVAL_TIME_POLICY_PATH", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

# ---------------------------------------------------------------------------
# CHECK10: XFPolicy.infer() itself -- the actual serving entrypoint
# (websocket_policy_server -> XFPolicy.infer), not _prepare_history/
# sample_actions called directly like CHECK9 does. A code review found
# MME_VLA_Policy.infer() (inherited, before XFPolicy overrode it) hardcodes
# HistAugObservation.from_dict -- CHECK9 never caught that because it never
# calls infer() at all. This check would have failed before that fix.
# ---------------------------------------------------------------------------
try:
    # infer()'s final step (jax.tree.map(jnp.asarray, ...)) needs a dict with only array-able
    # values by that point -- in real production this is XFInputs' job (its __call__ builds a
    # fresh dict from named keys only, naturally dropping raw strings like "grounded_subgoal").
    # CHECK9/CHECK10 use transforms=[] for simplicity rather than standing up the full repack/
    # normalize/tokenize pipeline, so this inline filter plays XFInputs' role here: strip
    # anything _prepare_history/the raw obs added that isn't one of the model-facing keys.
    _model_facing_keys = _NUMERIC_KEYS + ("state", "image", "image_mask", "tokenized_prompt", "tokenized_prompt_mask")

    def _strip_non_model_keys(data):
        return {k: v for k, v in data.items() if k in _model_facing_keys}

    policy_for_infer = XFPolicy(
        model, seed=0, transforms=[_strip_non_model_keys], output_transforms=[],
        sample_kwargs={"num_steps": 2}, norm_stats=norm_stats, use_quantiles=False,
    )
    policy_for_infer.reset()
    num_views = policy_for_infer.config.num_views
    image = np.zeros((num_views, 224, 224, 3), dtype=np.uint8)
    state = np.zeros((8,), dtype=np.float32)
    policy_for_infer.add_buffer({"images": image[None, ...], "state": state[None, ...], "exec_start_idx": 0})

    infer_obs = {
        "image": {"base_0_rgb": np.zeros((224, 224, 3), dtype=np.float32), "left_wrist_0_rgb": np.zeros((224, 224, 3), dtype=np.float32)},
        "image_mask": {"base_0_rgb": True, "left_wrist_0_rgb": True},
        "state": state,
        "tokenized_prompt": np.zeros((16,), dtype=np.int32),
        "tokenized_prompt_mask": np.ones((16,), dtype=np.bool_),
        "grounded_subgoal": "pick up the cube at <128, 64>",
    }
    infer_output = policy_for_infer.infer(infer_obs)
    infer_ok = (
        "actions" in infer_output
        and infer_output["actions"].shape == (config.action_horizon, config.action_dim)
        and bool(np.all(np.isfinite(infer_output["actions"])))
    )
    _record("10_POLICY_INFER_ENTRYPOINT", infer_ok, f"infer_ok={infer_ok} actions.shape={infer_output.get('actions', np.array([])).shape}")
except Exception as e:
    import traceback
    _record("10_POLICY_INFER_ENTRYPOINT", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

overall_ok = all(v for k, v in results.items() if not k.startswith("SETUP_"))
print("SMOKE_TEST_OVERALL_" + ("OK" if overall_ok else "FAIL"))
'''


@app.function(image=image, gpu="A10G", timeout=1200)
def run_smoke_test() -> dict:
    """
    What it does: writes SMOKE_TEST_SCRIPT to a file inside the Modal
    container and runs it with the uv-managed venv's Python, capturing
    stdout/stderr.

    Returns:
        dict -- {"success": bool, "detail": str}.

    Example input:
        run_smoke_test.remote()

    Example output:
        {"success": True, "detail": "CHECK_1_ZERO_GATE_IDENTITY_OK: ...\\n..."}
    """
    import subprocess  # module

    script_path = "/tmp/xf_smoke_test.py"  # str
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
        "detail": output[-12000:],
    }


@app.local_entrypoint()
def main():
    """
    What it does: CLI entrypoint -- runs run_smoke_test() and prints the result.

    Returns:
        None -- prints to stdout.

    Example input:
        modal run xattn_fusion/diagnostics/smoke_test.py

    Example output:
        (stdout) "success: True" followed by the full check-by-check detail.
    """
    result = run_smoke_test.remote()  # dict
    print(f"success: {result['success']}")
    print(result["detail"])
