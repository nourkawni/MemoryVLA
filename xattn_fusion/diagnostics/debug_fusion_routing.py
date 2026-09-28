"""
debug_fusion_routing.py

Structural correctness check for the gated cross-attention fusion mechanism
(event_encoder.EventEncoder + fusion_xattn.GatedXAttnFusion) -- WITHOUT any
training. Answers a different question than smoke_test.py's CHECK1
(zero-gate identity, i.e. "does fusion correctly do NOTHING at init"): this
answers "IF the gates were open, would the mechanism route information
correctly" -- same-event masking, differentiability, and residual magnitude
sanity -- independent of whether training ever moves the gates there.

What this can prove (all mechanism/structure, no learned behavior needed):
  1. SAME-EVENT ROUTING: perturbing one event's caption tokens changes ONLY
     the frames assigned to that event's F' -- never any other frame's,
     never a padded/overflow-dropped frame's. This is the load-bearing
     property build_fusion_mask exists to guarantee; the counterfactual
     diff test below checks it empirically on real per-frame outputs,
     not by re-reading the masking code and trusting it's right.
  2. GRADIENT FLOW: a loss on F' produces a finite, nonzero gradient at
     EventEncoder's trainable parameters -- proving the caption pathway is
     genuinely learnable (the raw PaliGemma embedding LOOKUP is stop_
     gradient'd by design, but everything downstream of it, e.g.
     embed_proj/the TinyTransformer/the pooling query, should not be).
  3. MAGNITUDE SANITY: with gates forced open (never happens at real init,
     which is deliberately zero), the resulting residual isn't NaN/Inf and
     isn't wildly larger than F itself.

What this CANNOT prove: whether the gates will ever learn to open, whether
2 xattn_layers is enough, or whether fusion helps task performance -- those
need a real training run.

Deliberately does NOT need a GPU or the full pi0.5 backbone: EventEncoder's
embed_fn is faked (a small, deterministic per-token-id hash -> a fixed
random vector, NOT the real ~2B-param PaliGemma embedder) since the
routing/gradient/magnitude properties above don't depend on what the
caption embeddings actually mean, only on the plumbing being correct. Real
aligner output (event_text_tokens/static_token_event_idx/etc.) still comes
from a real H5 episode via the same aligner code the training/eval paths
use, so this isn't testing against a synthetic toy scenario either.

Run with:
    modal run xattn_fusion/diagnostics/debug_fusion_routing.py
"""

import pathlib

import modal

XF_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
ROBOMME_SRC_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning" / "src")

app = modal.App("xf-debug-fusion-routing")

data_volume = modal.Volume.from_name("robomme-symbolic-modulator-full-suite-data")
DATA_VOLUME_PATH = "/full_suite_data"
RAW_DATA_PATH = f"{DATA_VOLUME_PATH}/raw_h5"

TASK = "BinFill"  # str, 5 distinct events on episode 0 -- enough variety to pick a mid-episode target event to perturb
EPISODE_IDX = 0
E_MAX = 16  # int, matches xf-framesamp-modul-xattn.yaml's fusion.max_events placeholder
L_C = 24  # int, matches fusion.caption_len placeholder
OPEN_GATE_VALUE = 3.0  # float, tanh(3.0) ~= 0.995 -- "fully open" per gated-fusion-agent.md's own gate table

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    # jax/flax/jaxtyping pinned to match robomme_policy_learning/pyproject.toml's own lockfile
    # exactly (jax[cuda12]==0.5.3 there -> jax[cpu]==0.5.3 here, no GPU needed for this debug).
    # Pip's default "latest" for any of these breaks: newer jaxtyping removed a private attribute
    # openpi.shared.array_typing monkey-patches, and newer flax.nnx enforces stricter pytree
    # rules that reject a plain python list of submodules (EventEncoder.layers) without an
    # explicit nnx.List/nnx.data wrapper this project's actual pinned version doesn't require.
    .pip_install(
        "numpy", "h5py", "omegaconf", "flax==0.10.2", "jax[cpu]==0.5.3", "einops", "sentencepiece",
        "jaxtyping==0.2.36", "beartype", "tqdm_loggable", "gcsfs",
    )
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cpu")
    .add_local_dir(ROBOMME_SRC_DIR, remote_path="/app_src", copy=True)
    .add_local_dir(XF_LOCAL_DIR, remote_path="/xf_root/xattn_fusion", copy=True)
)

DEBUG_SCRIPT = r'''
import sys
sys.path.insert(0, "/app_src")
sys.path.insert(0, "/xf_root")

import h5py
import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from mme_vla_suite.dataset_builder.robomme_h5_utils import first_execution_step
from mme_vla_suite.shared.data_utils import even_sampling_indices
from mme_vla_suite.shared.posemb_3d import PosEmb3D

from xattn_fusion.mme_vla_suite.shared.subgoal_table import (
    build_subgoal_table, build_caption_vocab, apply_caption_vocab,
    truncate_table, assign_events_to_frames, pack_event_arrays,
    load_sentencepiece_tokenizer,
)
from xattn_fusion.mme_vla_suite.models.representation.event_encoder import EventEncoder
from xattn_fusion.mme_vla_suite.models.representation.fusion_xattn import GatedXAttnFusion

RAW_DATA_PATH = "/full_suite_data/raw_h5"
TASK = "{TASK}"
EPISODE_IDX = {EPISODE_IDX}
E_MAX = {E_MAX}
L_C = {L_C}
OPEN_GATE_VALUE = {OPEN_GATE_VALUE}
MAX_SIZE = 512 // (16 * 1)
WIDTH = 1024

results = {}
def _record(name, ok, detail=""):
    results[name] = ok
    print(f"CHECK_{name}_{'OK' if ok else 'FAIL'}: {detail}")


def fake_embed_fn(tokens):
    """Deterministic per-token-id embedding via hashing -- no huge PaliGemma vocab table, no GPU,
    but the SAME token id always maps to the SAME vector (so results are reproducible and the
    counterfactual perturbation below is meaningful, not noise)."""
    flat = tokens.reshape(-1).astype(jnp.uint32)
    keys = jax.vmap(jax.random.PRNGKey)(flat)
    embs = jax.vmap(lambda k: jax.random.normal(k, (2048,)))(keys)
    return embs.reshape(tokens.shape + (2048,))


# --- Load one real episode's aligned event data (same aligner code training/eval use) ---
try:
    path = f"{RAW_DATA_PATH}/record_dataset_{TASK}.h5"
    with h5py.File(path, "r") as f:
        episode_data = f[f"episode_{EPISODE_IDX}"]
        exec_start_idx = first_execution_step(episode_data)
        num_timesteps = sum(1 for k in episode_data.keys() if k.startswith("timestep_"))
        table = build_subgoal_table(episode_data, exec_start_idx)
    vocab = build_caption_vocab([table])
    table = apply_caption_vocab(table, vocab)
    now = num_timesteps - 1
    truncated = truncate_table(table, now)
    sampled = even_sampling_indices(now, MAX_SIZE)
    aligned = assign_events_to_frames(truncated, sampled, max_size=MAX_SIZE)
    tokenizer = load_sentencepiece_tokenizer()
    packed = pack_event_arrays(truncated, aligned, tokenizer, E=E_MAX, L_c=L_C)
    print(f"Loaded {TASK} episode {EPISODE_IDX}: {len(truncated.intervals)} real events, "
          f"static_token_event_idx unique values = {sorted(set(packed['static_token_event_idx'].tolist()))}")
    _record("SETUP_REAL_DATA", True, f"{len(truncated.intervals)} events loaded")
except Exception as e:
    import traceback
    _record("SETUP_REAL_DATA", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
    raise SystemExit(1)

# --- Build EventEncoder + GatedXAttnFusion (random init), force gates open ---
try:
    pos_embedder = PosEmb3D(dim=768)
    spatial_dim = pos_embedder.spatial_pe4x4.shape[-1]
    temporal_dim = pos_embedder.temporal_pe.shape[-1]

    event_encoder = EventEncoder(
        rngs=nnx.Rngs(0), dtype=jnp.float32, embed_dim=2048, width=WIDTH,
        num_layers=2, heads=4, e_max=E_MAX, spatial_dim=spatial_dim, temporal_dim=temporal_dim,
    )
    fusion = GatedXAttnFusion(rngs=nnx.Rngs(1), dtype=jnp.float32, num_layers=2, width=WIDTH)

    # Force gates open -- at real init these are exactly 0 (smoke_test.py's CHECK1 already
    # proves that), so routing/magnitude can only be observed here by deliberately overriding
    # them; this does NOT reflect real init behavior, it reflects "if training ever opens them".
    for block in fusion.blocks:
        block.a_x.value = jnp.array(OPEN_GATE_VALUE)
        block.a_d.value = jnp.array(OPEN_GATE_VALUE)
    _record("SETUP_MODULES", True, f"gates forced to tanh({OPEN_GATE_VALUE})={float(jnp.tanh(OPEN_GATE_VALUE)):.4f}")
except Exception as e:
    import traceback
    _record("SETUP_MODULES", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
    raise SystemExit(1)


def to_batch(arr):
    return jnp.asarray(arr)[None, ...]

event_text_tokens = to_batch(packed["event_text_tokens"])
event_text_mask = to_batch(packed["event_text_mask"])
event_coords = to_batch(packed["event_coords"])
event_start = to_batch(packed["event_start"])
event_is_demo = to_batch(packed["event_is_demo"])
event_is_open = to_batch(packed["event_is_open"])
event_num_frames = to_batch(packed["event_num_frames"])
event_mask = to_batch(packed["event_mask"])
static_token_event_idx = to_batch(packed["static_token_event_idx"])

F = jax.random.normal(jax.random.key(2), (1, 512, WIDTH))


def run_fusion(ee, fu, tokens):
    e_tok, c_flat, c_mask = ee(
        tokens, event_text_mask, event_coords, event_start, event_is_demo, event_is_open,
        event_num_frames, event_mask, embed_fn=fake_embed_fn, pos_embedder=pos_embedder,
    )
    f_prime = fu(F, c_flat, static_token_event_idx, event_text_mask)
    return f_prime, e_tok


# --- CHECK 1: counterfactual same-event routing ---
try:
    f_prime_base, _ = run_fusion(event_encoder, fusion, event_text_tokens)

    real_event_ids = sorted(i for i in set(packed["static_token_event_idx"].tolist()) if i >= 0)
    target_event = real_event_ids[len(real_event_ids) // 2]  # a real, non-edge event to perturb

    perturbed_tokens = np.array(packed["event_text_tokens"])
    perturbed_tokens[target_event] = (perturbed_tokens[target_event] * 0 + 999) % 250000  # unrelated fixed tokens
    perturbed_tokens_b = to_batch(perturbed_tokens)

    f_prime_pert, _ = run_fusion(event_encoder, fusion, perturbed_tokens_b)

    diff = jnp.abs(f_prime_base - f_prime_pert)[0]  # [512, WIDTH]
    per_slot_changed = jnp.any(diff > 1e-6, axis=-1)  # bool[512]

    slot_event_ids = packed["static_token_event_idx"]  # int32[512]
    should_change = slot_event_ids == target_event
    should_not_change = slot_event_ids != target_event

    changed_when_expected = bool(jnp.all(per_slot_changed[should_change])) if should_change.any() else None
    unchanged_when_expected = bool(jnp.all(~per_slot_changed[should_not_change])) if should_not_change.any() else None
    n_leaked = int(jnp.sum(per_slot_changed[should_not_change])) if should_not_change.any() else 0

    routing_ok = bool(changed_when_expected) and bool(unchanged_when_expected)
    _record(
        "1_SAME_EVENT_ROUTING",
        routing_ok,
        f"target_event={target_event} frames_in_target={int(should_change.sum())} "
        f"changed_when_expected={changed_when_expected} unchanged_when_expected={unchanged_when_expected} "
        f"leaked_frames={n_leaked}/{int(should_not_change.sum())}",
    )
except Exception as e:
    import traceback
    _record("1_SAME_EVENT_ROUTING", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

# --- CHECK 2: magnitude sanity ---
try:
    residual_norm = float(jnp.linalg.norm(f_prime_base - F))
    f_norm = float(jnp.linalg.norm(F))
    ratio = residual_norm / f_norm
    finite_ok = bool(jnp.all(jnp.isfinite(f_prime_base)))
    sane_ratio_ok = 0.0 < ratio < 5.0  # generous bound -- just ruling out NaN/exploding, not tuning
    _record("2_MAGNITUDE_SANITY", finite_ok and sane_ratio_ok, f"finite_ok={finite_ok} residual/F_norm_ratio={ratio:.4f}")
except Exception as e:
    import traceback
    _record("2_MAGNITUDE_SANITY", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

# --- CHECK 3: gradient flow to EventEncoder's trainable params ---
try:
    def loss_fn(ee, fu, tokens):
        f_prime, _ = run_fusion(ee, fu, tokens)
        return jnp.sum(f_prime.astype(jnp.float32) ** 2)

    diff_state = nnx.DiffState(0, nnx.Param)
    loss_val, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(event_encoder, fusion, event_text_tokens)

    grad_leaves = jax.tree_util.tree_leaves(grads)
    grad_norm = float(jnp.sqrt(sum(jnp.sum(jnp.square(g.astype(jnp.float32))) for g in grad_leaves)))
    loss_finite_ok = bool(jnp.isfinite(loss_val))
    grad_finite_ok = all(bool(jnp.all(jnp.isfinite(g))) for g in grad_leaves)
    grad_nonzero_ok = grad_norm > 0.0

    _record(
        "3_GRADIENT_FLOW",
        loss_finite_ok and grad_finite_ok and grad_nonzero_ok,
        f"loss={float(loss_val):.4f} grad_norm={grad_norm:.6f} loss_finite_ok={loss_finite_ok} "
        f"grad_finite_ok={grad_finite_ok} grad_nonzero_ok={grad_nonzero_ok}",
    )
except Exception as e:
    import traceback
    _record("3_GRADIENT_FLOW", False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

overall_ok = all(v for k, v in results.items() if not k.startswith("SETUP_"))
print("DEBUG_FUSION_ROUTING_OVERALL_" + ("OK" if overall_ok else "FAIL"))
'''.replace("{TASK}", TASK).replace("{EPISODE_IDX}", str(EPISODE_IDX)).replace("{E_MAX}", str(E_MAX)).replace("{L_C}", str(L_C)).replace("{OPEN_GATE_VALUE}", str(OPEN_GATE_VALUE))


@app.function(image=image, gpu=None, timeout=900, volumes={DATA_VOLUME_PATH: data_volume})
def run_debug() -> dict:
    """
    What it does: writes DEBUG_SCRIPT to the container and runs it (CPU
    only -- no GPU requested).

    Returns:
        dict -- {"stdout": str, "returncode": int}.

    Example input:
        run_debug.remote()

    Example output:
        {"stdout": "CHECK_1_SAME_EVENT_ROUTING_OK: ...\\n...", "returncode": 0}
    """
    import subprocess

    script_path = "/tmp/debug_fusion_routing_inner.py"
    with open(script_path, "w") as f:
        f.write(DEBUG_SCRIPT)

    result = subprocess.run(["python", script_path], capture_output=True, text=True, timeout=800)
    output = result.stdout + "\n--- STDERR ---\n" + result.stderr
    print(output)
    return {"stdout": output, "returncode": result.returncode}


@app.local_entrypoint()
def main():
    """
    What it does: runs run_debug() and prints the result.

    Returns:
        None.

    Example input:
        modal run xattn_fusion/diagnostics/debug_fusion_routing.py

    Example output:
        (stdout) the check-by-check detail.
    """
    result = run_debug.remote()
    print(result["stdout"])
