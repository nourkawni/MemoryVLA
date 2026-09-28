"""
feature_shard_router.py

Single source of truth for how XF's precomputed perceptual features
(token_emb_*.npy) and cached subgoal_table.json files are routed across
multiple Modal volumes, instead of one preprocessed_data_path/features/
directory. Needed because a single Modal volume on this account hard-caps at
500,000 files (confirmed empirically, 2026-09-19 -- see
RESEARCH_LOG.md's "resharded across 4 volumes" entry), and the full 16-task
dataset's features/ alone needs ~777,000 files -- about 1.5x that ceiling.

Consumed by: xattn_fusion/mme_vla_suite/training/xf_dataset.py (reading
features/subgoal_table.json at training time) and
xattn_fusion/mme_vla_suite/dataset_builder/xf_subgoal_table_builder.py
(writing subgoal_table.json at offline-build time) -- both must agree on the
exact same routing rule, which is why it lives here once instead of being
duplicated in each. xattn_fusion/training/download_xf_full_suite_dataset.py
(the Modal launcher that actually populated the 4 shard volumes) keeps its
own copy of NUM_FEATURE_SHARDS/FEATURE_SHARD_PATHS rather than importing this
module, since that file has a hard Modal dependency and this one deliberately
doesn't -- the two must simply be kept in sync by hand if either changes.
"""

import os

# int, see module docstring for why 4.
NUM_FEATURE_SHARDS = 4
# list[str], the container mount paths for the NUM_FEATURE_SHARDS feature-shard volumes
# (xf-features-shard-0 .. xf-features-shard-{N-1}). Whichever Modal function reads an
# XFDataset for real training must mount all NUM_FEATURE_SHARDS volumes at exactly these
# paths.
FEATURE_SHARD_PATHS = [f"/xf_features_shard_{i}" for i in range(NUM_FEATURE_SHARDS)]


def feature_episode_dir(epis_idx: int) -> str:
    """
    What it does:
        Maps an episode index to the directory holding its precomputed
        features (token_emb_*.npy) and cached subgoal_table.json, routing
        by epis_idx % NUM_FEATURE_SHARDS -- the exact same rule
        download_xf_full_suite_dataset.py's run_download_features_sharded
        used to decide which shard volume each episode's features/
        episode_N.zip was downloaded into.

    Returns:
        str -- absolute path to this episode's feature directory, e.g.
        "/xf_features_shard_2/features/episode_802".

    Example input:
        feature_episode_dir(802)

    Example output:
        "/xf_features_shard_2/features/episode_802"
    """
    shard_id = epis_idx % NUM_FEATURE_SHARDS  # int
    return os.path.join(FEATURE_SHARD_PATHS[shard_id], "features", f"episode_{epis_idx}")
