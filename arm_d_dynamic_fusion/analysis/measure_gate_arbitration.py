"""
measure_gate_arbitration.py

Diagnostic (not training, not eval): measures what EarlyFusionModulator's
single fused cross-attention actually does on real data for a trained Arm D
checkpoint -- specifically, how much of its attention mass (attn_mass_sym/
attn_mass_perc, see joint_gated_modulator.py) lands on symbolic vs.
perceptual tokens, and whether that varies meaningfully by task/example
(genuine arbitration) or sits at a fixed lean regardless of content (the
kind of collapse the file's original version found under the PRIOR
two-attention-plus-router design -- see RESEARCH_LOG.md's 2026-08-28 22:14
entry: gate_sym/gate_perc were 0.001%/100.000%, uniform across every layer
and example).

UPDATED 2026-08-29 for the early-fusion redesign (RESEARCH_LOG.md's 15:53
entry): there is no more gate_sym/gate_perc (no 2-way router exists in the
current mechanism) -- this now reads attn_mass_sym/attn_mass_perc, the
realized attention-mass split from the single fused cross-attention. The
published checkpoint on HF Hub (Nkoni/arm-d-counting-suite-pilot, step 9999)
was trained under the OLD mechanism entirely -- its params have router/
mem_attn_sym/mem_attn_perc/mlp_sym/mlp_perc, which don't exist in the new
ArmDModel (now tag_sym/tag_perc/mem_attn_fused/mlp_fused), so
`train_config.model.load(...)` would fail on that checkpoint's params.

UPDATED AGAIN 2026-08-30, same day early-fusion training launched: added
LOCAL_CHECKPOINT_STEP so this can read an in-progress run's checkpoint
directly off the training volume (robomme-arm-d-pilot-training), without
waiting for a finished run to be published to HF Hub -- the whole point of
checking early (per the 2026-08-29 16:20 decision: catch re-collapse toward
perceptual as soon as possible, not just at the end of a 10k-step run).
Set LOCAL_CHECKPOINT_STEP to a step list_checkpoints/check_checkpoints
(launch_pilot_training.py) shows as saved, or to None to fall back to the
HF Hub download path (for a finished, published checkpoint instead).

Harder than measure_representation_alignment.py because attn_mass_sym/
attn_mass_perc are computed once per action-expert layer INSIDE a flax.linen
`nn.scan` (see history_gemma_dual.DualMemoryHistoryBlock.__call__'s
`self.sow(...)` calls and DualMemoryModule.setup's
`variable_axes={"intermediates": 0}`), and are only retrievable via flax's
own `mutable=["intermediates"]` mechanism -- ordinary Python-level
monkeypatching does NOT work here, because values inside a scanned/traced
function body are abstract JAX tracers, not concrete arrays, until the
scan's own machinery stacks and returns them. ArmDModel's actual runtime
path reaches DualMemoryModule through `nnx_bridge.ToNNX` (needed because
ArmDModel itself is a flax.nnx.Module, but the fusion stack underneath is
still flax.linen, unchanged from the released code) -- whether that bridge
transparently forwards a `mutable=` kwarg through to the wrapped module's
`.apply()` is exactly the API-version-dependent question
arm_d_pi0.ArmDModel.compute_loss's docstring already resolved empirically
(it does NOT, for this installed flax version). This script tries the
direct passthrough first (APPROACH A) and prints the full raw return
structure either way, falling back to extracting the wrapped linen module
and its trained params directly and calling
`.apply(..., mutable=["intermediates"])` on it (APPROACH B, the exact call
shape already proven in smoke_test.py's CHECK2, and the same one
compute_loss itself now uses in production) if A's return shape doesn't
match what's expected.

Role in the system: read-only analysis, informs the same design decision
measure_representation_alignment.py's results feed into. robomme_policy_
learning/ is not edited.

Run with:
    modal run arm_d_dynamic_fusion/analysis/measure_gate_arbitration.py
"""

import pathlib

import modal

POLICY_LOCAL_DIR = str(  # str
    pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning"
)
ARM_D_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)  # str, the arm_d_dynamic_fusion/ dir

HF_CKPT_REPO = "Nkoni/arm-d-counting-suite-pilot"  # str
HF_CKPT_STEP = "9999"  # str

# Set to an int (a step list_checkpoints/check_checkpoints shows as saved)
# to read that in-progress run's checkpoint directly off the training
# volume instead of downloading a published HF Hub checkpoint. None falls
# back to the HF Hub path above.
LOCAL_CHECKPOINT_STEP: int | None = 6000  # int | None
TRAIN_CONFIG_NAME = "arm_d_pilot"  # str, must match launch_pilot_training.py's own constant
EXP_NAME = "counting-suite-early-fusion-no-warmstart"  # str, ditto -- updated 2026-08-30 to the attempt-2 (mem_attn_fused trained from scratch) run; the stopped attempt-1 run's checkpoint is still at "counting-suite-early-fusion" if ever needed for comparison

BATCH_SIZE = 32  # int, examples per batch
NUM_BATCHES = 8  # int, 256 examples total -- forward-only, cheap
SEED = 42  # int

app = modal.App("robomme-arm-d-gate-arbitration")  # modal.App

ckpt_volume = modal.Volume.from_name("robomme-arm-d-eval-ckpt-cache", create_if_missing=True)  # modal.Volume, HF-downloaded checkpoints (used when LOCAL_CHECKPOINT_STEP is None)
data_volume = modal.Volume.from_name("robomme-arm-d-pilot-data", create_if_missing=True)  # modal.Volume
train_volume = modal.Volume.from_name("robomme-arm-d-pilot-training", create_if_missing=True)  # modal.Volume, launch_pilot_training.py's own in-progress checkpoints (used when LOCAL_CHECKPOINT_STEP is set)

CKPT_VOLUME_PATH = "/ckpts"  # str
DATA_VOLUME_PATH = "/pilot_data"  # str, must match launch_pilot_training.py's own constant
TRAIN_VOLUME_PATH = "/pilot_training"  # str, must match launch_pilot_training.py's own constant

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
    .run_commands("cd /app && /root/.local/bin/uv pip install --system pytest huggingface_hub")
    .add_local_dir(ARM_D_LOCAL_DIR, remote_path="/arm_d_root/arm_d_dynamic_fusion", copy=True)
)


@app.function(image=image, volumes={CKPT_VOLUME_PATH: ckpt_volume}, timeout=1800)
def download_checkpoint() -> str:
    """
    What it does:
        Downloads/unzips the published Arm D pilot checkpoint (same logic as
        run_pilot_eval.py's and measure_representation_alignment.py's own
        download_checkpoint, duplicated per this project's one-app-per-script
        convention). Idempotent.

    Returns:
        str -- path to the unzipped checkpoint step directory.

    Example input:
        download_checkpoint.remote()

    Example output:
        "/ckpts/arm-d-counting-suite-pilot/9999"
    """
    import subprocess  # module

    import huggingface_hub  # module

    repo_dir = pathlib.Path(CKPT_VOLUME_PATH) / "arm-d-counting-suite-pilot"  # Path
    ckpt_dir = repo_dir / HF_CKPT_STEP  # Path
    zip_path = repo_dir / f"{HF_CKPT_STEP}.zip"  # Path

    if not zip_path.exists():
        repo_dir.mkdir(parents=True, exist_ok=True)
        huggingface_hub.hf_hub_download(
            repo_id=HF_CKPT_REPO, repo_type="model", filename=f"{HF_CKPT_STEP}.zip",
            local_dir=str(repo_dir),
        )
    if not ckpt_dir.exists():
        subprocess.run(["python", "scripts/unzip_ckpt.py", str(repo_dir)], cwd="/app", check=True)

    ckpt_volume.commit()
    return str(ckpt_dir)


ANALYSIS_SCRIPT = r'''
import sys
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/arm_d_root")

import json
import os
import pathlib
import traceback
os.chdir("/app")

import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import mme_vla_suite.training.dataloader as _dataloader

from arm_d_dynamic_fusion.training.arm_d_data import ArmDDataset
from arm_d_dynamic_fusion.training.launch_pilot_training import _build_train_config
from mme_vla_suite.models.integration.history_observation import preprocess_observation
from mme_vla_suite.models.integration.history_pi0 import make_attn_mask

_dataloader.RoboMMEDataset = ArmDDataset

BATCH_SIZE = {BATCH_SIZE}
NUM_BATCHES = {NUM_BATCHES}
SEED = {SEED}


def build_llm_call_inputs(model, observation, actions):
    """
    What it does:
        Replicates the input-assembly portion of ArmDModel.compute_loss
        (noising the action chunk, embed_prefix/embed_suffix, mask/position
        construction) WITHOUT calling model.PaliGemma.llm itself -- so the
        caller can make that final call with an extra `mutable=` kwarg
        compute_loss itself never passes. No model logic is reimplemented
        here, only the small orchestration glue compute_loss already has
        (see arm_d_pi0.ArmDModel.compute_loss for the original).

    Returns:
        dict -- kwargs ready to splat into model.PaliGemma.llm(...): xs,
        mask, positions, adarms_cond, mem_seq_sym, mem_mask_sym,
        mem_seq_perc, mem_mask_perc.

    Example input:
        build_llm_call_inputs(model, observation, actions)

    Example output:
        {"xs": [...], "mask": Array(...), "positions": Array(...), ...}
    """
    rng = jax.random.key(SEED)
    _, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = preprocess_observation(None, observation, train=False)

    batch_shape = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions

    prefix_tokens, prefix_mask, prefix_ar_mask, prefix_na_mask, _ = model.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, suffix_na_mask, adarms_cond = model.embed_suffix(
        observation, x_t, time
    )

    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    na_mask = jnp.concatenate([prefix_na_mask, suffix_na_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask, na_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1

    mem_sym, mem_sym_mask, mem_perc, mem_perc_mask = model.embed_memory(observation)

    return {
        "xs": [prefix_tokens, suffix_tokens],
        "mask": attn_mask,
        "positions": positions,
        "adarms_cond": [None, adarms_cond],
        "mem_seq_sym": [None, mem_sym],
        "mem_mask_sym": [None, mem_sym_mask],
        "mem_seq_perc": [None, mem_perc],
        "mem_mask_perc": [None, mem_perc_mask],
    }


def describe(obj, depth=0, max_depth=3):
    """Prints a short type/shape description of a (possibly nested) result, for diagnosing an unfamiliar return shape without guessing."""
    prefix = "  " * depth
    if depth > max_depth:
        print(f"{prefix}...(truncated)")
        return
    if isinstance(obj, (tuple, list)):
        print(f"{prefix}{type(obj).__name__} of length {len(obj)}:")
        for item in obj:
            describe(item, depth + 1, max_depth)
    elif isinstance(obj, dict):
        print(f"{prefix}dict with keys: {list(obj.keys())}")
        for k, v in obj.items():
            print(f"{prefix}  [{k}]:")
            describe(v, depth + 2, max_depth)
    elif hasattr(obj, "shape"):
        print(f"{prefix}{type(obj).__name__} shape={obj.shape} dtype={obj.dtype}")
    else:
        print(f"{prefix}{type(obj).__name__}: {obj!r:.200}")


train_config = _build_train_config(num_train_steps=1)
data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
model = train_config.model.load(
    _model.restore_params(pathlib.Path("{ckpt_dir}") / "params", dtype=jax.numpy.bfloat16)
)

loader = _dataloader.create_data_loader(
    train_config.dataset_path, data_config,
    history_config=train_config.model.history_config,
    action_horizon=train_config.model.action_horizon,
    batch_size=BATCH_SIZE, shuffle=True, num_batches=1, num_workers=0, seed=SEED,
)
observation, actions = next(iter(loader))
call_inputs = build_llm_call_inputs(model, observation, actions)

print("=== APPROACH A: direct mutable= passthrough on the nnx-wrapped call ===")
approach_a_ok = False
try:
    result = model.PaliGemma.llm(**call_inputs, mutable=["intermediates"])
    print("APPROACH_A_CALL_SUCCEEDED")
    print("Raw result structure:")
    describe(result)
    approach_a_ok = True
except Exception as e:
    print(f"APPROACH_A_FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()

attn_mass_sym = attn_mass_perc = None
if approach_a_ok:
    # Try the shape smoke_test.py's CHECK2 established for a raw linen .apply()
    # call: ((outputs, kv_cache), mutated) where
    # mutated["intermediates"]["layers"]["attn_mass_sym"][0] has shape [depth, b, t, 1].
    try:
        (outputs, kv_cache), mutated = result
        attn_mass_sym = np.asarray(mutated["intermediates"]["layers"]["attn_mass_sym"][0], dtype=np.float32)
        attn_mass_perc = np.asarray(mutated["intermediates"]["layers"]["attn_mass_perc"][0], dtype=np.float32)
        print(f"Extracted via APPROACH A: attn_mass_sym.shape={attn_mass_sym.shape}, attn_mass_perc.shape={attn_mass_perc.shape}")
    except Exception as e:
        print(f"APPROACH_A_EXTRACTION_FAILED (call succeeded but shape did not match expectations): {type(e).__name__}: {e}")

if attn_mass_sym is None:
    print("=== APPROACH B: extract the wrapped linen module + trained params, call .apply() directly ===")
    try:
        import flax.nnx as nnx

        wrapped = model.PaliGemma.llm
        print("type(wrapped):", type(wrapped))
        print("public attrs:", [a for a in dir(wrapped) if not a.startswith("_")])

        linen_module = getattr(wrapped, "module", None)
        print("wrapped.module:", type(linen_module))

        state = nnx.state(wrapped)
        variables = {"params": state.to_pure_dict()}
        print("Extracted variables[params] top-level keys:", list(variables["params"].keys()) if isinstance(variables["params"], dict) else type(variables["params"]))

        (outputs, kv_cache), mutated = linen_module.apply(
            variables,
            call_inputs["xs"], mask=call_inputs["mask"], positions=call_inputs["positions"],
            adarms_cond=call_inputs["adarms_cond"],
            mem_seq_sym=call_inputs["mem_seq_sym"], mem_mask_sym=call_inputs["mem_mask_sym"],
            mem_seq_perc=call_inputs["mem_seq_perc"], mem_mask_perc=call_inputs["mem_mask_perc"],
            mutable=["intermediates"],
        )
        attn_mass_sym = np.asarray(mutated["intermediates"]["layers"]["attn_mass_sym"][0], dtype=np.float32)
        attn_mass_perc = np.asarray(mutated["intermediates"]["layers"]["attn_mass_perc"][0], dtype=np.float32)
        print(f"Extracted via APPROACH B: attn_mass_sym.shape={attn_mass_sym.shape}, attn_mass_perc.shape={attn_mass_perc.shape}")
    except Exception as e:
        print(f"APPROACH_B_FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()

if attn_mass_sym is None:
    print("ATTN_MASS_EXTRACTION_FAILED_BOTH_APPROACHES")
else:
    per_layer_mean_sym = attn_mass_sym.mean(axis=(1, 2, 3)).tolist()  # list[float], one per layer (depth axis is axis 0)
    per_layer_std_sym = attn_mass_sym.std(axis=(1, 2, 3)).tolist()  # list[float]
    overall_mean_sym = float(attn_mass_sym.mean())
    overall_std_sym = float(attn_mass_sym.std())
    overall_mean_perc = float(attn_mass_perc.mean())
    overall_std_perc = float(attn_mass_perc.std())
    result_dict = {
        "n_layers": attn_mass_sym.shape[0],
        "n_examples": attn_mass_sym.shape[1],
        "overall_mean_attn_mass_sym": overall_mean_sym,
        "overall_std_attn_mass_sym": overall_std_sym,
        "overall_mean_attn_mass_perc": overall_mean_perc,
        "overall_std_attn_mass_perc": overall_std_perc,
        "per_layer_mean_attn_mass_sym": per_layer_mean_sym,
        "per_layer_std_attn_mass_sym": per_layer_std_sym,
    }
    print("ATTN_MASS_RESULT_JSON:" + json.dumps(result_dict))
'''


@app.function(
    image=image, gpu="A10G", timeout=1200,
    volumes={CKPT_VOLUME_PATH: ckpt_volume, DATA_VOLUME_PATH: data_volume, TRAIN_VOLUME_PATH: train_volume},
)
def measure_attn_mass() -> dict:
    """
    What it does:
        Resolves the checkpoint directory -- either an in-progress local
        training-volume checkpoint (LOCAL_CHECKPOINT_STEP set, no download
        needed, it's already on a mounted volume) or a published HF Hub
        checkpoint (LOCAL_CHECKPOINT_STEP is None, downloads first) -- then
        writes ANALYSIS_SCRIPT (with ckpt_dir/BATCH_SIZE/NUM_BATCHES/SEED
        substituted in) to a file and runs it. All diagnostic prints (which
        approach worked, raw structure dumps) are printed to stdout
        regardless of success, so a failure is still informative for a
        follow-up fix.

    Returns:
        dict -- the parsed ATTN_MASS_RESULT_JSON line's contents.

    Example input:
        measure_attn_mass.remote()

    Example output:
        {"n_layers": 18, "n_examples": 32, "overall_mean_attn_mass_sym": 0.31, ...}
    """
    import json
    import subprocess  # module

    if LOCAL_CHECKPOINT_STEP is not None:
        ckpt_dir = f"{TRAIN_VOLUME_PATH}/ckpts/{TRAIN_CONFIG_NAME}/{EXP_NAME}/{LOCAL_CHECKPOINT_STEP}"  # str
        train_volume.reload()  # Volumes aren't live-synced into an already-running container
    else:
        ckpt_dir = download_checkpoint.remote()  # str
        ckpt_volume.reload()

    script_text = (
        ANALYSIS_SCRIPT
        .replace("{ckpt_dir}", ckpt_dir)
        .replace("{BATCH_SIZE}", str(BATCH_SIZE))
        .replace("{NUM_BATCHES}", str(NUM_BATCHES))
        .replace("{SEED}", str(SEED))
    )  # str
    script_path = "/tmp/measure_gate_arbitration.py"  # str
    with open(script_path, "w") as f:
        f.write(script_text)

    result = subprocess.run(
        ["python", script_path], cwd="/app", capture_output=True, text=True, timeout=1000,
    )  # subprocess.CompletedProcess
    print(result.stdout)
    print("--- STDERR ---")
    print(result.stderr)

    for line in result.stdout.splitlines():
        if line.startswith("ATTN_MASS_RESULT_JSON:"):
            return json.loads(line[len("ATTN_MASS_RESULT_JSON:"):])
    raise RuntimeError(
        f"Attention-mass extraction did not succeed (returncode={result.returncode}); see full stdout/stderr above for diagnosis."
    )


@app.local_entrypoint()
def main():
    """CLI entrypoint -- runs measure_attn_mass() and prints the summary."""
    result = measure_attn_mass.remote()  # dict
    print(f"\nLayers: {result['n_layers']}, examples: {result['n_examples']}")
    print(f"Overall attn_mass_sym: mean={result['overall_mean_attn_mass_sym']:.4f} std={result['overall_std_attn_mass_sym']:.4f}")
    print(f"Overall attn_mass_perc: mean={result['overall_mean_attn_mass_perc']:.4f} std={result['overall_std_attn_mass_perc']:.4f}")
    print("\nPer-layer attn_mass_sym mean (layer 0 = closest to input):")
    for i, (m, s) in enumerate(zip(result["per_layer_mean_attn_mass_sym"], result["per_layer_std_attn_mass_sym"])):
        print(f"  layer {i:2d}: mean={m:.4f} std={s:.4f}")
