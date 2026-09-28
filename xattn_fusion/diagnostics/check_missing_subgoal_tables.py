"""
check_missing_subgoal_tables.py

One-off diagnostic: run_training crashed at step ~1020 (2026-09-20) with
FileNotFoundError on /xf_features_shard_1/features/episode_33/subgoal_table.json
-- that episode's directory has every OTHER expected file (token_emb_*.npy,
kept_indices.json, video files) but not subgoal_table.json, meaning the
earlier full-scale xf_subgoal_table_builder.py run (RESEARCH_LOG 2026-09-19,
"BUILD_OVERALL_OK") silently missed at least this one episode despite
reporting full success. This script checks ALL 1,307 real, mapped episodes
(via episode_mapping.json, the same source XFDataset/training itself uses)
for the same gap, using the exact real feature_episode_dir() path-resolution
code -- not a reimplementation -- so the answer is directly trustworthy.

Run with:
    modal run xattn_fusion/diagnostics/check_missing_subgoal_tables.py
"""

import pathlib

import modal

XF_LOCAL_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
ROBOMME_SRC_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "robomme_policy_learning" / "src")

app = modal.App("xf-check-missing-subgoal-tables")

main_data_volume = modal.Volume.from_name("xf-full-suite-data")
feature_shard_volumes = [modal.Volume.from_name(f"xf-features-shard-{i}") for i in range(4)]

MAIN_DATA_VOLUME_PATH = "/xf_data"
FEATURE_SHARD_PATHS = [f"/xf_features_shard_{i}" for i in range(4)]

volumes = {
    MAIN_DATA_VOLUME_PATH: main_data_volume,
    **{path: vol for path, vol in zip(FEATURE_SHARD_PATHS, feature_shard_volumes)},
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "h5py", "omegaconf", "flax", "jax[cpu]", "einops", "sentencepiece")
    .add_local_dir(ROBOMME_SRC_DIR, remote_path="/app_src", copy=True)
    .add_local_dir(XF_LOCAL_DIR, remote_path="/xf_root/xattn_fusion", copy=True)
)


@app.function(image=image, gpu=None, timeout=600, volumes=volumes)
def check() -> dict:
    """
    What it does: for every episode in episode_mapping.json (the real,
    1,307-entry mapping XFDataset itself trusts), checks whether
    subgoal_table.json exists at feature_episode_dir(epis_idx) -- the exact
    real function XFDataset._load_subgoal_table calls.

    Returns:
        dict -- {"total": int, "missing": list[int], "present": int}.

    Example input:
        check.remote()

    Example output:
        {"total": 1307, "missing": [33], "present": 1306}
    """
    import sys
    sys.path.insert(0, "/app_src")
    sys.path.insert(0, "/xf_root")
    from pathlib import Path
    from xattn_fusion.mme_vla_suite.dataset_builder.xf_subgoal_table_builder import load_episode_mapping
    from xattn_fusion.mme_vla_suite.shared.feature_shard_router import feature_episode_dir

    mapping = load_episode_mapping(f"{MAIN_DATA_VOLUME_PATH}/episode_mapping.json")
    missing = []
    for epis_idx in sorted(mapping.keys()):
        table_path = Path(feature_episode_dir(epis_idx)) / "subgoal_table.json"
        if not table_path.exists():
            missing.append(epis_idx)

    return {"total": len(mapping), "missing": missing, "present": len(mapping) - len(missing), "all_ids": sorted(mapping.keys())}


@app.local_entrypoint()
def main(dump_all_ids: bool = False):
    """
    What it does: runs check(), prints the total/present/missing counts and
    the full list of missing episode indices. With dump_all_ids=True, also
    prints every valid epis_idx on its own line (for driving an external,
    modal-volume-ls-based ground-truth scan instead of trusting this
    script's own Path.exists()-based check -- see RESEARCH_LOG.md's
    2026-09-20 entry on why this script's own "present" answer was found
    to disagree with `modal volume ls` for at least one real file).

    Returns:
        None.

    Example input:
        modal run xattn_fusion/diagnostics/check_missing_subgoal_tables.py --dump-all-ids

    Example output:
        (stdout) "total=1307 present=1306 missing=1: [33]", then one epis_idx per line.
    """
    result = check.remote()
    print(f"total={result['total']} present={result['present']} missing={len(result['missing'])}")
    print(f"missing episode indices: {result['missing']}")
    if dump_all_ids:
        print("ALL_IDS_START")
        for epis_idx in result["all_ids"]:
            print(epis_idx)
        print("ALL_IDS_END")
