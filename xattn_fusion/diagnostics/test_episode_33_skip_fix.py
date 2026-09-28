"""
test_episode_33_skip_fix.py

Direct, targeted verification of the episode-33/97-exclusion fix in
xf_dataset.py -- rather than waiting for real training to randomly reach a
sample pointing to episode 33 (which took ~1000+ steps in every prior real
run), this script: (1) scans data/*.pkl in parallel to find a real sample
whose epis_idx is 33 (or 97), (2) constructs a real XFDataset (same
construction real training uses), and (3) calls __getitem__ on that EXACT
sample index directly, confirming it no longer crashes and instead returns
a valid, substituted sample.

Run with:
    modal run xattn_fusion/diagnostics/test_episode_33_skip_fix.py
"""

import pathlib

import modal

XF_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
POLICY_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning")

app = modal.App("xf-test-episode-33-skip-fix")

ckpt_volume = modal.Volume.from_name("robomme-mme-vla-ckpts")
main_data_volume = modal.Volume.from_name("xf-full-suite-data")
feature_shard_volumes = [modal.Volume.from_name(f"xf-features-shard-{i}") for i in range(4)]
training_volume = modal.Volume.from_name("xf-full-suite-training", create_if_missing=True)

CKPT_VOLUME_PATH = "/ckpts"
MAIN_DATA_VOLUME_PATH = "/xf_data"
FEATURE_SHARD_PATHS = [f"/xf_features_shard_{i}" for i in range(4)]
TRAINING_VOLUME_PATH = "/xf_training"
DATASET_PATH = f"{MAIN_DATA_VOLUME_PATH}/preprocessed"

REPO_ID = "xf_full_suite"
HISTORY_CONFIG_NAME = "xf-framesamp-modul-xattn.yaml"
TARGET_EPISODES = [33, 97]  # int, the two known-excluded episodes to specifically hunt for and test

volumes = {
    CKPT_VOLUME_PATH: ckpt_volume,
    MAIN_DATA_VOLUME_PATH: main_data_volume,
    TRAINING_VOLUME_PATH: training_volume,
    **{path: vol for path, vol in zip(FEATURE_SHARD_PATHS, feature_shard_volumes)},
}

image = (
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


@app.function(image=image, gpu=None, timeout=1800, volumes=volumes)
def test() -> str:
    """
    What it does:
        Scans data/*.pkl in parallel for a sample pointing at episode 33 or
        97, then builds a real XFDataset and directly calls __getitem__ on
        that exact index, confirming the skip-and-substitute fix works
        without crashing.

    Returns:
        str -- full stdout-style report.

    Example input:
        test.remote()

    Example output:
        "FOUND idx=12345 epis_idx=33 ... SKIP_FIX_VERIFIED_OK"
    """
    import sys
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/src")
    sys.path.insert(0, "/xf_root")
    import os
    import pickle
    import pathlib as _pathlib
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import mme_vla_suite.training.config as _config

    from xattn_fusion.mme_vla_suite.models.integration.xf_pi0 import XFConfig
    from xattn_fusion.mme_vla_suite.models.config.xf_config_utils import get_xf_history_config
    from xattn_fusion.mme_vla_suite.training.xf_config import XFDataConfig
    from xattn_fusion.mme_vla_suite.training.xf_dataset import XFDataset, XFSampleDataset

    out = []

    def log(msg):
        out.append(msg)
        print(msg)

    # --- Step 1: find a real sample idx whose epis_idx is one of TARGET_EPISODES ---
    data_dir = os.path.join(DATASET_PATH, "data")
    real_ids = sorted(int(f.split(".")[0]) for f in os.listdir(data_dir) if f.endswith(".pkl"))
    log(f"Scanning {len(real_ids)} real samples for epis_idx in {TARGET_EPISODES}...")

    found = {}  # dict[int, int], target_episode -> sample_idx (position in real_ids)

    def check_one(pos_and_real_id):
        pos, real_id = pos_and_real_id
        with open(os.path.join(data_dir, f"{real_id}.pkl"), "rb") as f:
            d = pickle.load(f)
        epis_idx = int(d["epis_idx"].item())
        if epis_idx in TARGET_EPISODES:
            return pos, epis_idx
        return None

    # Scan in parallel, stop early once we've found both targets (or scanned everything).
    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = {executor.submit(check_one, (pos, rid)): pos for pos, rid in enumerate(real_ids)}
        checked = 0
        for future in as_completed(futures):
            checked += 1
            result = future.result()
            if result is not None:
                pos, epis_idx = result
                if epis_idx not in found:
                    found[epis_idx] = pos
                    log(f"FOUND: sample idx={pos} (real_id={real_ids[pos]}) has epis_idx={epis_idx}")
            if len(found) == len(TARGET_EPISODES):
                for f in futures:
                    f.cancel()
                break
            if checked % 50000 == 0:
                log(f"...checked {checked}/{len(real_ids)}, found so far: {found}")

    if not found:
        log("NO SAMPLES FOUND pointing at episodes 33/97 -- cannot directly test (unexpected, investigate).")
        return "\n".join(out)

    # --- Step 2: build the REAL XFDataset (same construction real training uses) ---
    model_config = XFConfig(
        pi05=True, action_horizon=20, use_history=True,
        history_config=get_xf_history_config(HISTORY_CONFIG_NAME),
        discrete_state_input=False, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m",
    )
    data_config_spec = XFDataConfig(repo_id=REPO_ID, base_config=_config.DataConfig(prompt_from_task=True))
    assets_dirs = (_pathlib.Path(f"{TRAINING_VOLUME_PATH}/assets") / REPO_ID).resolve()
    data_config = data_config_spec.create(assets_dirs, model_config)

    dataset = XFDataset(
        dataset_path=DATASET_PATH, data_config=data_config,
        history_config=model_config.history_config, action_horizon=model_config.action_horizon,
    )
    log(f"XFDataset built. 33 in valid_episode_ids: {33 in dataset._valid_episode_ids} "
        f"(should be False). 97 in valid_episode_ids: {97 in dataset._valid_episode_ids} (should be False).")

    # --- Step 3: directly call __getitem__ on the found index(es), confirm no crash + real substitution ---
    all_ok = True
    for epis_idx, idx in found.items():
        try:
            data = dataset[idx]
            returned_epis_idx = int(data["epis_idx"].item())
            ok = returned_epis_idx not in TARGET_EPISODES
            all_ok = all_ok and ok
            log(f"dataset[{idx}] (original epis_idx={epis_idx}) -> returned epis_idx={returned_epis_idx}, "
                f"substituted correctly: {ok}, no crash: True")
        except Exception as e:
            all_ok = False
            log(f"dataset[{idx}] (original epis_idx={epis_idx}) CRASHED: {type(e).__name__}: {e}")

    log("SKIP_FIX_VERIFIED_" + ("OK" if all_ok else "FAIL"))
    return "\n".join(out)


@app.local_entrypoint()
def main():
    """
    What it does: runs test() remotely and prints the full report.

    Returns:
        None.

    Example input:
        modal run xattn_fusion/diagnostics/test_episode_33_skip_fix.py

    Example output:
        (stdout) the full scan + verification report.
    """
    print(test.remote())
