"""
scratch_scan_ground_truth.py

One-off local script: scans every one of the 1,307 real, mapped XF episodes
across all 4 feature-shard volumes for a missing subgoal_table.json, using
modal.Volume.listdir() directly (the same API the `modal volume ls` CLI
command itself uses) -- NOT a Path.exists() check through a container's
FUSE-style volume mount, which an earlier diagnostic script used and which
an independent review found likely gives false "present" answers (see
RESEARCH_LOG.md's 2026-09-20 entries). Runs locally, no Modal container
spin-up per check, so this is fast even for 1,307 episodes.
"""

import json

import modal

main_vol = modal.Volume.from_name("xf-full-suite-data")
mapping_bytes = b"".join(main_vol.read_file("episode_mapping.json"))
mapping = json.loads(mapping_bytes)
all_ids = sorted(int(k) for k in mapping.keys())

print(f"Checking {len(all_ids)} episodes...")

volumes = {i: modal.Volume.from_name(f"xf-features-shard-{i}") for i in range(4)}

missing = []
for i, epis_idx in enumerate(all_ids):
    shard = epis_idx % 4
    vol = volumes[shard]
    try:
        entries = list(vol.listdir(f"features/episode_{epis_idx}"))
        names = {e.path.split("/")[-1] for e in entries}
        if "subgoal_table.json" not in names:
            missing.append((epis_idx, shard))
            print(f"MISSING: epis_idx={epis_idx} shard={shard}")
    except Exception as e:
        missing.append((epis_idx, shard))
        print(f"ERROR (treated as missing): epis_idx={epis_idx} shard={shard} -- {type(e).__name__}: {e}")
    if (i + 1) % 100 == 0:
        print(f"...checked {i + 1}/{len(all_ids)}, {len(missing)} missing so far")

print(f"\nDONE. total={len(all_ids)} missing={len(missing)}")
print(f"missing list: {missing}")
