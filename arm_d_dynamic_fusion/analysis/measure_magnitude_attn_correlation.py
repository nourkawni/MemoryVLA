"""
measure_magnitude_attn_correlation.py

Diagnostic (not training, not eval): checks whether attn_mass_sym is being
driven by raw vector MAGNITUDE rather than actual content relevance -- a
follow-up to two prior findings (2026-09-02, see RESEARCH_LOG.md):

1. measure_attn_mass_per_task.py found a small-but-real symbolic-leaning gap
   for BinFill vs. SwingXtimes/StopCube (0.44 vs. 0.41/0.42) -- real, but
   weak, not the large task-dependent split you'd want from confident
   arbitration.
2. measure_representation_alignment.py found the trained checkpoint's
   symbolic-stream tokens are ~6x larger in RMS-norm than perceptual-stream
   tokens (ratio 1.00 at random-init -> 6.03 after training) -- a new
   imbalance that didn't exist before training.

These two facts are suggestive but not conclusively linked: standard scaled
dot-product attention computes scores as (Q . K) / sqrt(d), and Q . K scales
with ||K|| for a roughly-fixed Q direction -- so if symbolic tokens (used as
K/V in the fused attention) are systematically larger in magnitude, they
could win a larger share of attention mass simply by being BIGGER, independent
of whether their CONTENT is more relevant. If that's the dominant effect, the
"weak arbitration" finding in (1) wouldn't reflect the model learning
task-dependent relevance at all -- it would mean the model attends more to
symbolic content whenever a given example's symbolic tokens happen to be
larger, which is a magnitude artifact, not task-dependent reasoning.

This script tests that directly: for a batch of real examples, record BOTH
per-example attn_mass_sym (same forward pass as measure_attn_mass_per_task.py)
AND that same example's symbolic-to-perceptual token RMS-norm ratio (same
per-example computation as measure_representation_alignment.py's
batch_alignment_stats, but not aggregated across the batch -- kept per
example specifically so the two variables can be correlated example-by-
example), then computes the correlation between them across ~2560 pooled
examples (640 per task, reusing the same 4 known-pure task windows
measure_attn_mass_per_task.py established).

Reading the result: if attn_mass_sym tracks the magnitude ratio closely
(high correlation), that confirms raw vector size -- not actual relevance --
is driving a meaningful chunk of the attention split, which is a concrete,
fixable thing (e.g. normalize the K vectors before the dot product) rather
than "just needs more training." A low/near-zero correlation would mean the
magnitude imbalance and the attention split are separate phenomena, and the
weak-arbitration finding needs a different explanation.

Role in the system: read-only analysis, feeding the same unification-
mechanism design decision measure_representation_alignment.py's output
does (user's supervisor's direction, 2026-08-28). robomme_policy_learning/
is not edited.

Run with:
    modal run arm_d_dynamic_fusion/analysis/measure_magnitude_attn_correlation.py
"""

import pathlib

import modal

POLICY_LOCAL_DIR = str(  # str
    pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning"
)
ARM_D_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)  # str, the arm_d_dynamic_fusion/ dir

HF_CKPT_REPO = "Nkoni/arm-d-v1"  # str
HF_CKPT_STEP = "9999"  # str

BATCH_SIZE = 32  # int, examples per batch
NUM_BATCHES = 20  # int, 640 examples per task window -- same size as measure_attn_mass_per_task.py's proven-safe per-subprocess window
SEED = 42  # int

# Same 4 known-pure task windows measure_attn_mass_per_task.py established
# (2026-09-02) -- reused here rather than re-deriving, since the dataset's
# block structure and each block's boundaries are already confirmed.
TASK_WINDOWS = [  # list[tuple[str, int]]
    ("BinFill", 10000),
    ("PickXtimes", 80000),
    ("StopCube", 125000),
    ("SwingXtimes", 160000),
]

app = modal.App("robomme-arm-d-magnitude-attn-correlation")  # modal.App

ckpt_volume = modal.Volume.from_name("robomme-arm-d-eval-ckpt-cache", create_if_missing=True)  # modal.Volume
data_volume = modal.Volume.from_name("robomme-arm-d-pilot-data", create_if_missing=True)  # modal.Volume

CKPT_VOLUME_PATH = "/ckpts"  # str
DATA_VOLUME_PATH = "/pilot_data"  # str, must match launch_pilot_training.py's own constant

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
    """Downloads/unzips the published arm-d-v1 checkpoint (same logic as the other analysis scripts' download_checkpoint). Idempotent."""
    import subprocess  # module

    import huggingface_hub  # module

    repo_dir = pathlib.Path(CKPT_VOLUME_PATH) / "arm-d-v1"  # Path
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
os.chdir("/app")

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx

import openpi.models.model as _model
from openpi.training.data_loader import TorchDataLoader, transform_dataset
from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.models.integration.history_observation import preprocess_observation, HistAugObservation
from mme_vla_suite.models.integration.history_pi0 import make_attn_mask
import mme_vla_suite.training.dataloader as _dataloader

from arm_d_dynamic_fusion.training.arm_d_data import ArmDDataset
from arm_d_dynamic_fusion.training.launch_pilot_training import _build_train_config

_dataloader.RoboMMEDataset = ArmDDataset

BATCH_SIZE = {BATCH_SIZE}
NUM_BATCHES = {NUM_BATCHES}
SEED = {SEED}
TASK_NAME = sys.argv[1]  # str, which of TASK_WINDOWS this subprocess is responsible for
START_IDX = int(sys.argv[2])  # int, index into the raw (untransformed) dataset


def masked_rms_norm_per_example(tokens, mask):
    """
    What it does:
        Per-example RMS-norm of a memory stream's tokens over its valid
        (non-padding) positions -- same math as measure_representation_
        alignment.py's batch_alignment_stats, but kept PER EXAMPLE (not
        summed across the batch) since this script needs to pair each
        example's magnitude with that SAME example's attn_mass_sym.

    Returns:
        np.ndarray -- shape [b], RMS-norm per example.

    Example input:
        masked_rms_norm_per_example(np.ones((2, 4, 8)), np.ones((2, 4)))

    Example output:
        array([2.828, 2.828])  # sqrt(8) for all-ones tokens
    """
    sq_norm = (tokens.astype(np.float32) ** 2).sum(axis=-1)  # np.ndarray [b, t], squared L2 norm per token
    mask_f = mask.astype(np.float32)  # np.ndarray [b, t]
    counts = np.clip(mask_f.sum(axis=1), 1.0, None)  # np.ndarray [b]
    mean_sq = (sq_norm * mask_f).sum(axis=1) / counts  # np.ndarray [b]
    return np.sqrt(mean_sq)  # np.ndarray [b]


train_config = _build_train_config(num_train_steps=1)
data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
model = train_config.model.load(
    _model.restore_params(pathlib.Path("{ckpt_dir}") / "params", dtype=jax.numpy.bfloat16)
)

history_config = get_history_config(train_config.model.history_config)
raw_dataset = ArmDDataset(
    dataset_path=train_config.dataset_path,
    data_config=data_config,
    history_config=history_config,
    action_horizon=train_config.model.action_horizon,
)
print(f"[{TASK_NAME}] window: [{START_IDX}, {START_IDX + NUM_BATCHES * BATCH_SIZE})")

transformed_dataset = transform_dataset(raw_dataset, data_config, skip_norm_stats=False)
end_idx = START_IDX + NUM_BATCHES * BATCH_SIZE
torch_loader = TorchDataLoader(
    transformed_dataset, local_batch_size=BATCH_SIZE, shuffle=False,
    sampler=list(range(START_IDX, end_idx)), num_batches=NUM_BATCHES,
    num_workers=0, seed=SEED, framework="jax",
)

attn_mass_values = []  # list[float]
ratio_values = []  # list[float], sym_rms_norm / perc_rms_norm, same example

for batch_idx, torch_batch in enumerate(torch_loader):
    observation = HistAugObservation.from_dict(torch_batch)
    actions = torch_batch["actions"]

    observation_p = preprocess_observation(None, observation, train=False)
    mem_sym, mem_sym_mask, mem_perc, mem_perc_mask = model.embed_memory(observation_p)

    sym_rms = masked_rms_norm_per_example(np.asarray(mem_sym), np.asarray(mem_sym_mask))  # np.ndarray [b]
    perc_rms = masked_rms_norm_per_example(np.asarray(mem_perc), np.asarray(mem_perc_mask))  # np.ndarray [b]
    ratio = sym_rms / np.clip(perc_rms, 1e-8, None)  # np.ndarray [b]

    llm_variables = {"params": nnx.state(model.PaliGemma.llm, nnx.Param).to_pure_dict()}

    rng = jax.random.key(SEED)
    _, noise_rng, time_rng = jax.random.split(rng, 3)
    batch_shape = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    prefix_tokens, prefix_mask, prefix_ar_mask, prefix_na_mask, _ = model.embed_prefix(observation_p)
    suffix_tokens, suffix_mask, suffix_ar_mask, suffix_na_mask, adarms_cond = model.embed_suffix(observation_p, x_t, time)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    na_mask = jnp.concatenate([prefix_na_mask, suffix_na_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask, na_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1

    (outputs, kv_cache), mutated = model.PaliGemma.llm.module.apply(
        llm_variables,
        [prefix_tokens, suffix_tokens],
        mask=attn_mask, positions=positions,
        adarms_cond=[None, adarms_cond],
        mem_seq_sym=[None, mem_sym], mem_mask_sym=[None, mem_sym_mask],
        mem_seq_perc=[None, mem_perc], mem_mask_perc=[None, mem_perc_mask],
        mutable=["intermediates"],
    )
    attn_mass_sym = np.asarray(mutated["intermediates"]["layers"]["attn_mass_sym"][0], dtype=np.float32)
    per_example = attn_mass_sym.mean(axis=(0, 2, 3))  # np.ndarray [b]

    attn_mass_values.extend(per_example.tolist())
    ratio_values.extend(ratio.tolist())

    print(f"[{TASK_NAME}] batch {batch_idx}: mean attn_mass_sym={float(per_example.mean()):.4f}, mean ratio={float(ratio.mean()):.4f}")

result = {
    "task": TASK_NAME,
    "start_idx": START_IDX,
    "n": len(attn_mass_values),
    "attn_mass_sym": attn_mass_values,
    "sym_perc_rms_ratio": ratio_values,
}
print("CORRELATION_DATA_JSON:" + json.dumps(result))
'''


@app.function(
    image=image, gpu="A10G", timeout=3600,
    volumes={CKPT_VOLUME_PATH: ckpt_volume, DATA_VOLUME_PATH: data_volume},
)
def measure_correlation_data() -> list[dict]:
    """
    What it does:
        Downloads the checkpoint once, then runs ANALYSIS_SCRIPT once per
        entry in TASK_WINDOWS -- each as its own subprocess (fresh model
        load), same OOM-avoidance pattern measure_attn_mass_per_task.py
        established (2026-09-02).

    Returns:
        list[dict] -- one result dict per TASK_WINDOWS entry, each holding
        parallel per-example lists "attn_mass_sym" and "sym_perc_rms_ratio".

    Example input:
        measure_correlation_data.remote()

    Example output:
        [{"task": "BinFill", "n": 640, "attn_mass_sym": [...], "sym_perc_rms_ratio": [...]}, ...]
    """
    import json
    import subprocess  # module

    ckpt_dir = download_checkpoint.remote()  # str
    ckpt_volume.reload()

    script_text = (
        ANALYSIS_SCRIPT
        .replace("{ckpt_dir}", ckpt_dir)
        .replace("{BATCH_SIZE}", str(BATCH_SIZE))
        .replace("{NUM_BATCHES}", str(NUM_BATCHES))
        .replace("{SEED}", str(SEED))
    )  # str
    script_path = "/tmp/measure_magnitude_attn_correlation.py"  # str
    with open(script_path, "w") as f:
        f.write(script_text)

    results = []  # list[dict]
    for task_name, start_idx in TASK_WINDOWS:
        result = subprocess.run(
            ["python", script_path, task_name, str(start_idx)],
            cwd="/app", capture_output=True, text=True, timeout=800,
        )  # subprocess.CompletedProcess
        print(f"=== task={task_name} stdout ===")
        print(result.stdout)
        print(f"=== task={task_name} stderr ===")
        print(result.stderr)

        found = None  # dict | None
        for line in result.stdout.splitlines():
            if line.startswith("CORRELATION_DATA_JSON:"):
                found = json.loads(line[len("CORRELATION_DATA_JSON:"):])
        if found is None:
            raise RuntimeError(
                f"Correlation data extraction did not succeed for task={task_name} "
                f"(returncode={result.returncode}); see stderr above."
            )
        results.append(found)

    return results


@app.local_entrypoint()
def main():
    """CLI entrypoint -- runs measure_correlation_data(), then computes and prints Pearson/Spearman correlation (pooled and per-task) plus a quintile-bucket table."""
    import numpy as np  # numpy.ndarray

    results = measure_correlation_data.remote()  # list[dict]

    all_attn = []  # list[float]
    all_ratio = []  # list[float]
    print(f"\n{'task':<14}{'pearson r':<12}{'n':<6}")
    for r in results:
        attn = np.array(r["attn_mass_sym"])  # np.ndarray
        ratio = np.array(r["sym_perc_rms_ratio"])  # np.ndarray
        task_r = float(np.corrcoef(attn, ratio)[0, 1]) if len(attn) > 1 else float("nan")
        print(f"{r['task']:<14}{task_r:<12.4f}{r['n']:<6}")
        all_attn.extend(r["attn_mass_sym"])
        all_ratio.extend(r["sym_perc_rms_ratio"])

    all_attn = np.array(all_attn)  # np.ndarray
    all_ratio = np.array(all_ratio)  # np.ndarray
    pooled_pearson = float(np.corrcoef(all_attn, all_ratio)[0, 1])  # float

    # Spearman rho = Pearson correlation of the rank-transformed arrays -- no scipy needed.
    def rank(x):
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(x))
        return ranks

    pooled_spearman = float(np.corrcoef(rank(all_attn), rank(all_ratio))[0, 1])  # float

    print(f"\nPOOLED (n={len(all_attn)}): Pearson r = {pooled_pearson:.4f}, Spearman rho = {pooled_spearman:.4f}")

    # Quintile-bucket table: sort by ratio, split into 5 equal-size buckets, report mean attn_mass_sym per bucket.
    order = np.argsort(all_ratio)
    sorted_attn = all_attn[order]
    sorted_ratio = all_ratio[order]
    n = len(sorted_ratio)
    print(f"\n{'ratio quintile':<20}{'ratio range':<24}{'mean attn_mass_sym':<20}{'n':<6}")
    for q in range(5):
        lo = n * q // 5
        hi = n * (q + 1) // 5
        bucket_ratio = sorted_ratio[lo:hi]
        bucket_attn = sorted_attn[lo:hi]
        print(f"Q{q+1:<19}{f'{bucket_ratio.min():.2f}-{bucket_ratio.max():.2f}':<24}{bucket_attn.mean():<20.4f}{len(bucket_attn):<6}")
