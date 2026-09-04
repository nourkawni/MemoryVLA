"""
measure_representation_alignment.py

Diagnostic (not training, not eval): measures how aligned Arm D's two memory
streams actually are in representation space, before any explicit unification
mechanism is built on top of them. M_sym (symbolic subgoal tokens, [b, 64,
1024]) and M_perc (perceptual visual tokens, [b, 512, 1024]) already share the
same last dimension via their own separately-parameterized projectors (see
symbolic_mem_encoder.py / mme_vla_suite's PerceptualMemory) -- but nothing
currently trains them to occupy the same semantic subspace. Each projector
only ever receives gradient through JointGatedModulator's downstream flow-
matching loss, filtered through the gate; same output width is not the same
thing as "unified representation space."

This script quantifies that gap directly on real pilot data, using a
retrieval-style matched-vs-shuffled test (see batch_alignment_stats' docstring
-- the same style of check CLIP-style alignment diagnostics use): for each
training example, does its symbolic-stream summary vector actually sit
closest to its OWN perceptual-stream summary vector, or to some other
example's? Run against two model states so the numbers are interpretable
against a floor, not just reported in isolation:
  - "random_init"        -- a freshly initialized ArmDModel (chance-level
                             floor: no training at all, of any kind).
  - "trained_arm_d_v1"   -- the published early-fusion, no-warm-start
                             checkpoint (Nkoni/arm-d-v1, step 9999), showing
                             how much alignment (if any) ~10k steps of
                             ordinary downstream-loss training already
                             produced with zero explicit alignment objective.

REPOINTED 2026-09-02: was Nkoni/arm-d-counting-suite-pilot (the OLD two-
cross-attention-plus-router design's checkpoint) -- same one-constant change
upload_checkpoint.py already got (2026-08-31). The local cache subdirectory
name was changed too (arm-d-counting-suite-pilot -> arm-d-v1), not just
HF_CKPT_REPO: this diagnostic shares its checkpoint-cache Modal Volume with
run_pilot_eval.py/measure_attn_mass_per_task.py, which may already have the
OLD checkpoint's zip cached under the old subdirectory name from an earlier
run (2026-08-28 21:24 RESEARCH_LOG entry) -- leaving the local dirname
unchanged while only swapping HF_CKPT_REPO would have made download_
checkpoint() see that old zip already present and silently skip downloading
the new one, evaluating the wrong checkpoint under the "arm-d-v1" label. Also
renamed the "trained_pilot_9999" condition label to "trained_arm_d_v1"
throughout, since that RESEARCH_LOG entry used "trained_pilot_9999" for the
OLD checkpoint -- keeping the same label for a different checkpoint would
make the two diagnostics' results ambiguous to compare later.

Role in the system: read-only analysis. Its output is the input to a design
decision (whether/how to build an explicit unification step ahead of
JointGatedModulator -- see the user's supervisor's direction, 2026-08-28), not
part of the trained model or the pilot's training/eval pipeline itself.
robomme_policy_learning/ is not edited; reuses arm_d_data.ArmDDataset directly
and (via launch_pilot_training._build_train_config, unchanged) ArmDConfig/
ArmDDataConfig indirectly -- same "isolated read-only script" shape as
smoke_test.py and run_pilot_eval.py's own smoke_test function.

Run with:
    modal run arm_d_dynamic_fusion/analysis/measure_representation_alignment.py
"""

import pathlib

import modal

POLICY_LOCAL_DIR = str(  # str
    pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning"
)
ARM_D_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)  # str, the arm_d_dynamic_fusion/ dir

# Same public checkpoint measure_attn_mass_per_task.py/upload_checkpoint.py
# evaluate/publish -- see module docstring's "REPOINTED 2026-09-02" note.
HF_CKPT_REPO = "Nkoni/arm-d-v1"  # str
HF_CKPT_STEP = "9999"  # str

BATCH_SIZE = 32  # int, examples per batch -- retrieval chance level is 1/32 (~3.1%)
NUM_BATCHES = 8  # int, 256 examples total per condition -- forward-only, cheap
SEED = 42  # int, shared across both conditions' data loaders so they see the same batches

app = modal.App("robomme-arm-d-repr-alignment")  # modal.App

# Reuses run_pilot_eval.py's checkpoint cache volume by name (Modal Volumes
# are account-level, addressable by name from any app) -- if that eval has
# already run on this account, the checkpoint is already cached here and this
# script needs no fresh download.
ckpt_volume = modal.Volume.from_name("robomme-arm-d-eval-ckpt-cache", create_if_missing=True)  # modal.Volume
# Reuses launch_pilot_training.py's pilot-data volume -- the real
# preprocessed training data this diagnostic draws its batches from.
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
    """
    What it does:
        Downloads and unzips the published Arm D pilot checkpoint from the
        public HF Hub repo, onto this diagnostic's own cache volume. Same
        logic as run_pilot_eval.py's download_checkpoint, duplicated rather
        than cross-app-imported (this project's convention: each script owns
        its own self-contained app/image, see smoke_test.py/run_pilot_eval.py/
        launch_pilot_training.py each defining their own modal.App).
        Idempotent: skips work already done on this account's cache volume.

    Returns:
        str -- path to the unzipped checkpoint step directory.

    Example input:
        download_checkpoint.remote()

    Example output:
        "/ckpts/arm-d-v1/9999"
    """
    import subprocess  # module

    import huggingface_hub  # module

    repo_dir = pathlib.Path(CKPT_VOLUME_PATH) / "arm-d-v1"  # Path
    ckpt_dir = repo_dir / HF_CKPT_STEP  # Path
    zip_path = repo_dir / f"{HF_CKPT_STEP}.zip"  # Path

    if not zip_path.exists():
        repo_dir.mkdir(parents=True, exist_ok=True)
        print(f"[measure_representation_alignment] downloading {HF_CKPT_REPO}/{HF_CKPT_STEP}.zip ...")
        huggingface_hub.hf_hub_download(
            repo_id=HF_CKPT_REPO, repo_type="model", filename=f"{HF_CKPT_STEP}.zip",
            local_dir=str(repo_dir),
        )
    else:
        print(f"[measure_representation_alignment] {zip_path} already downloaded, skipping.")

    if not ckpt_dir.exists():
        print(f"[measure_representation_alignment] unzipping {zip_path} ...")
        subprocess.run(["python", "scripts/unzip_ckpt.py", str(repo_dir)], cwd="/app", check=True)
    else:
        print(f"[measure_representation_alignment] {ckpt_dir} already unzipped, skipping.")

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
import numpy as np

import openpi.models.model as _model
import mme_vla_suite.training.dataloader as _dataloader

from arm_d_dynamic_fusion.training.arm_d_data import ArmDDataset
from arm_d_dynamic_fusion.training.launch_pilot_training import _build_train_config
from mme_vla_suite.models.integration.history_observation import preprocess_observation

_dataloader.RoboMMEDataset = ArmDDataset

BATCH_SIZE = {BATCH_SIZE}  # int, substituted in by measure_alignment() from the outer file's constant
NUM_BATCHES = {NUM_BATCHES}  # int, ditto
SEED = {SEED}  # int, ditto


def masked_mean_pool(tokens, mask):
    """
    What it does:
        Mean-pools a memory stream's tokens over its valid (non-padding)
        positions, per example.

    Returns:
        np.ndarray -- shape [b, d].

    Example input:
        masked_mean_pool(np.zeros((2, 64, 1024)), np.ones((2, 64)))

    Example output:
        np.ndarray of shape (2, 1024), all zeros.
    """
    mask_f = mask.astype(np.float32)[..., None]  # np.ndarray [b, l, 1]
    counts = np.clip(mask_f.sum(axis=1), 1.0, None)  # np.ndarray [b, 1]
    return (tokens * mask_f).sum(axis=1) / counts  # np.ndarray [b, d]


def cosine_sim_matrix(a, b):
    """
    What it does:
        Pairwise cosine similarity between every row of `a` and every row
        of `b`.

    Returns:
        np.ndarray -- shape [a.shape[0], b.shape[0]], values in [-1, 1].

    Example input:
        cosine_sim_matrix(np.eye(2), np.eye(2))

    Example output:
        array([[1., 0.], [0., 1.]])
    """
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)  # np.ndarray [n, d]
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)  # np.ndarray [m, d]
    return a_n @ b_n.T  # np.ndarray [n, m]


def batch_alignment_stats(mem_sym, mem_sym_mask, mem_perc, mem_perc_mask):
    """
    What it does:
        The core retrieval-style alignment check for one batch: mean-pools
        each example's two memory streams to a single vector per stream,
        then asks -- across the batch -- whether an example's symbolic vector
        is closer to its OWN perceptual vector (the matched pair) than to any
        other example's (the shuffled/mismatched pairs). If the two streams
        genuinely shared a representation space, matched pairs should score
        higher than mismatched ones and top-1 retrieval should beat the
        1/batch_size chance level; if the spaces are unrelated, matched vs.
        mismatched similarity should look statistically identical.

    Returns:
        dict -- per-batch raw values (not yet aggregated across batches):
        matched_cos [b], unmatched_cos [b*(b-1)], sym_to_perc_correct [b]
        (bool), perc_to_sym_correct [b] (bool), plus running-sum ingredients
        for the cross-batch centroid/RMS-norm stats (pooled_sum_sym/perc,
        n, tok_sq_norm_sum_sym/perc, tok_count_sym/perc).

    Example input:
        batch_alignment_stats(np.zeros((4,64,1024)), np.ones((4,64)),
                               np.zeros((4,512,1024)), np.ones((4,512)))

    Example output:
        {"matched_cos": array([...]), "unmatched_cos": array([...]), ...}
    """
    sym_pooled = masked_mean_pool(mem_sym, mem_sym_mask)  # np.ndarray [b, d]
    perc_pooled = masked_mean_pool(mem_perc, mem_perc_mask)  # np.ndarray [b, d]
    sim = cosine_sim_matrix(sym_pooled, perc_pooled)  # np.ndarray [b, b]
    b = sim.shape[0]  # int
    eye = np.eye(b, dtype=bool)  # np.ndarray [b, b]

    return {
        "matched_cos": np.diag(sim),  # np.ndarray [b]
        "unmatched_cos": sim[~eye],  # np.ndarray [b*(b-1)]
        "sym_to_perc_correct": np.argmax(sim, axis=1) == np.arange(b),  # np.ndarray [b] bool
        "perc_to_sym_correct": np.argmax(sim, axis=0) == np.arange(b),  # np.ndarray [b] bool
        "pooled_sum_sym": sym_pooled.sum(axis=0),  # np.ndarray [d]
        "pooled_sum_perc": perc_pooled.sum(axis=0),  # np.ndarray [d]
        "n": b,  # int
        "tok_sq_norm_sum_sym": float(((mem_sym ** 2).sum(-1) * mem_sym_mask).sum()),  # float
        "tok_count_sym": float(mem_sym_mask.sum()),  # float
        "tok_sq_norm_sum_perc": float(((mem_perc ** 2).sum(-1) * mem_perc_mask).sum()),  # float
        "tok_count_perc": float(mem_perc_mask.sum()),  # float
    }


def run_condition(model, train_config, data_config, label):
    """
    What it does:
        Builds a real ArmDDataset-backed data loader (same construction
        scripts/train.py's main() uses), draws NUM_BATCHES batches of
        BATCH_SIZE real pilot-training examples, runs each through
        `model.embed_memory`, and aggregates batch_alignment_stats across all
        of them into one summary for this model state.

    Returns:
        dict -- {"label": str, "n_examples": int, "matched_cos_mean": float,
        "matched_cos_std": float, "unmatched_cos_mean": float,
        "unmatched_cos_std": float, "sym_to_perc_acc": float,
        "perc_to_sym_acc": float, "chance_acc": float,
        "centroid_cos": float, "sym_rms_norm": float, "perc_rms_norm": float,
        "rms_norm_ratio": float}.

    Example input:
        run_condition(model, train_config, data_config, "trained_arm_d_v1")

    Example output:
        {"label": "trained_arm_d_v1", "n_examples": 256,
         "matched_cos_mean": 0.041, "unmatched_cos_mean": 0.037, ...}
    """
    loader = _dataloader.create_data_loader(
        train_config.dataset_path,
        data_config,
        history_config=train_config.model.history_config,
        action_horizon=train_config.model.action_horizon,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_batches=NUM_BATCHES,
        num_workers=0,
        seed=SEED,
    )

    matched_all, unmatched_all = [], []  # list[np.ndarray], list[np.ndarray]
    sym_to_perc_hits, perc_to_sym_hits = [], []  # list[np.ndarray], list[np.ndarray]
    pooled_sum_sym = pooled_sum_perc = None  # np.ndarray | None [d]
    n_examples = 0  # int
    tok_sq_norm_sum_sym = tok_count_sym = 0.0  # float
    tok_sq_norm_sum_perc = tok_count_perc = 0.0  # float

    for observation, _actions in loader:
        observation = preprocess_observation(None, observation, train=False)
        mem_sym, mem_sym_mask, mem_perc, mem_perc_mask = model.embed_memory(observation)
        stats = batch_alignment_stats(
            np.asarray(mem_sym, dtype=np.float32), np.asarray(mem_sym_mask),
            np.asarray(mem_perc, dtype=np.float32), np.asarray(mem_perc_mask),
        )
        matched_all.append(stats["matched_cos"])
        unmatched_all.append(stats["unmatched_cos"])
        sym_to_perc_hits.append(stats["sym_to_perc_correct"])
        perc_to_sym_hits.append(stats["perc_to_sym_correct"])
        pooled_sum_sym = stats["pooled_sum_sym"] if pooled_sum_sym is None else pooled_sum_sym + stats["pooled_sum_sym"]
        pooled_sum_perc = stats["pooled_sum_perc"] if pooled_sum_perc is None else pooled_sum_perc + stats["pooled_sum_perc"]
        n_examples += stats["n"]
        tok_sq_norm_sum_sym += stats["tok_sq_norm_sum_sym"]
        tok_count_sym += stats["tok_count_sym"]
        tok_sq_norm_sum_perc += stats["tok_sq_norm_sum_perc"]
        tok_count_perc += stats["tok_count_perc"]

    matched = np.concatenate(matched_all)  # np.ndarray
    unmatched = np.concatenate(unmatched_all)  # np.ndarray
    sym_to_perc_acc = float(np.concatenate(sym_to_perc_hits).mean())  # float
    perc_to_sym_acc = float(np.concatenate(perc_to_sym_hits).mean())  # float
    centroid_sym = pooled_sum_sym / n_examples  # np.ndarray [d]
    centroid_perc = pooled_sum_perc / n_examples  # np.ndarray [d]
    centroid_cos = float(cosine_sim_matrix(centroid_sym[None, :], centroid_perc[None, :])[0, 0])  # float
    sym_rms_norm = float(np.sqrt(tok_sq_norm_sum_sym / tok_count_sym))  # float
    perc_rms_norm = float(np.sqrt(tok_sq_norm_sum_perc / tok_count_perc))  # float

    return {
        "label": label,
        "n_examples": n_examples,
        "matched_cos_mean": float(matched.mean()),
        "matched_cos_std": float(matched.std()),
        "unmatched_cos_mean": float(unmatched.mean()),
        "unmatched_cos_std": float(unmatched.std()),
        "sym_to_perc_acc": sym_to_perc_acc,
        "perc_to_sym_acc": perc_to_sym_acc,
        "chance_acc": 1.0 / BATCH_SIZE,
        "centroid_cos": centroid_cos,
        "sym_rms_norm": sym_rms_norm,
        "perc_rms_norm": perc_rms_norm,
        "rms_norm_ratio": sym_rms_norm / perc_rms_norm,
    }


condition = sys.argv[1]  # str, "random_init" or "trained_arm_d_v1" -- see measure_alignment's docstring
                          # for why this is a separate subprocess per condition rather than both in one.

train_config = _build_train_config(num_train_steps=1)
data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

if condition == "random_init":
    model = train_config.model.create(jax.random.key(SEED))
elif condition == "trained_arm_d_v1":
    model = train_config.model.load(
        _model.restore_params(pathlib.Path("{ckpt_dir}") / "params", dtype=jax.numpy.bfloat16)
    )
else:
    raise ValueError(f"unknown condition: {condition}")

result = run_condition(model, train_config, data_config, condition)
print("ALIGNMENT_RESULT_JSON:" + json.dumps(result))
'''

CONDITIONS = ["random_init", "trained_arm_d_v1"]  # list[str]


@app.function(
    image=image, gpu="A10G", timeout=2400,
    volumes={CKPT_VOLUME_PATH: ckpt_volume, DATA_VOLUME_PATH: data_volume},
)
def measure_alignment() -> list[dict]:
    """
    What it does:
        Downloads the trained pilot checkpoint, writes ANALYSIS_SCRIPT (with
        the checkpoint dir/BATCH_SIZE/NUM_BATCHES/SEED substituted in) to a
        file inside the container, then runs it once per entry in CONDITIONS
        -- each as its OWN subprocess, not both in one process. A first
        version built both the ~2.3B-param random-init model and the loaded
        checkpoint's ~2.3B-param model in the same process (sequentially, but
        without ever releasing the first's device memory) and hit a real
        RESOURCE_EXHAUSTED OOM on the A10G's 24GB (confirmed on an actual run,
        2026-08-28) -- a separate OS process per condition guarantees the GPU
        memory from one condition is fully released (process exit) before the
        next one allocates anything, at the cost of reloading/rebuilding
        common state (train_config, data_config) twice, which is cheap
        relative to the model/checkpoint memory this avoids.

    Returns:
        list[dict] -- one run_condition() result dict per entry in CONDITIONS.

    Example input:
        measure_alignment.remote()

    Example output:
        [{"label": "random_init", "matched_cos_mean": 0.001, ...},
         {"label": "trained_arm_d_v1", "matched_cos_mean": 0.043, ...}]
    """
    import json
    import subprocess  # module

    ckpt_dir = download_checkpoint.remote()  # str
    ckpt_volume.reload()  # Volumes aren't live-synced into an already-running container

    script_text = (
        ANALYSIS_SCRIPT
        .replace("{ckpt_dir}", ckpt_dir)
        .replace("{BATCH_SIZE}", str(BATCH_SIZE))
        .replace("{NUM_BATCHES}", str(NUM_BATCHES))
        .replace("{SEED}", str(SEED))
    )  # str
    script_path = "/tmp/measure_representation_alignment.py"  # str
    with open(script_path, "w") as f:
        f.write(script_text)

    results = []  # list[dict]
    for condition in CONDITIONS:
        result = subprocess.run(
            ["python", script_path, condition],
            cwd="/app", capture_output=True, text=True, timeout=1000,
        )  # subprocess.CompletedProcess
        print(f"=== condition={condition} stdout ===")
        print(result.stdout)
        print(f"=== condition={condition} stderr ===")
        print(result.stderr)

        found = None  # dict | None
        for line in result.stdout.splitlines():
            if line.startswith("ALIGNMENT_RESULT_JSON:"):
                found = json.loads(line[len("ALIGNMENT_RESULT_JSON:"):])
        if found is None:
            raise RuntimeError(
                f"No ALIGNMENT_RESULT_JSON line for condition={condition} "
                f"(returncode={result.returncode}); see stderr above."
            )
        results.append(found)

    return results


@app.local_entrypoint()
def main():
    """
    What it does:
        CLI entrypoint -- runs measure_alignment() and prints a side-by-side
        report comparing "random_init" against "trained_arm_d_v1".

    Returns:
        None -- prints to stdout.

    Example input:
        modal run arm_d_dynamic_fusion/analysis/measure_representation_alignment.py

    Example output:
        (stdout) a table with matched/unmatched cosine similarity, retrieval
        accuracy vs. chance, centroid cosine similarity, and RMS-norm ratio
        for each condition.
    """
    results = measure_alignment.remote()  # list[dict]
    by_label = {r["label"]: r for r in results}  # dict[str, dict]

    print(f"\n{'metric':<28}{'random_init':<18}{'trained_arm_d_v1':<18}")
    rows = [
        ("n_examples", "{:d}"),
        ("matched_cos_mean", "{:.4f}"),
        ("matched_cos_std", "{:.4f}"),
        ("unmatched_cos_mean", "{:.4f}"),
        ("unmatched_cos_std", "{:.4f}"),
        ("sym_to_perc_acc", "{:.4f}"),
        ("perc_to_sym_acc", "{:.4f}"),
        ("chance_acc", "{:.4f}"),
        ("centroid_cos", "{:.4f}"),
        ("sym_rms_norm", "{:.4f}"),
        ("perc_rms_norm", "{:.4f}"),
        ("rms_norm_ratio", "{:.4f}"),
    ]
    for key, fmt in rows:
        r_val = fmt.format(by_label["random_init"][key])
        t_val = fmt.format(by_label["trained_arm_d_v1"][key])
        print(f"{key:<28}{r_val:<18}{t_val:<18}")
