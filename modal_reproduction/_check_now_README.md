# full_eval_episodes.csv - column reference

Raw per-episode results for the RoboMME P0 reproduction (arbitrated-memory-proposal.md). One row per completed episode, newest completion first.

| Column | Meaning |
|---|---|
| completed_at_utc | When this episode finished, ISO 8601 UTC. See timestamp_source. |
| seed | Policy sampling seed (0, 42, or 7) - controls the flow-matching model's own action-sampling randomness, NOT the environment. |
| task_id | Which of the 16 RoboMME tasks (e.g. PickXtimes, BinFill). |
| episode_idx | Which of the 50 fixed test scenarios for that task (0-49) - a fixed, benchmark-defined starting condition, not a repeat/retry. |
| success_flag | Outcome: success / fail / timeout (hit the 1300-step cap without resolving) / error (simulator exception). |
| steps | How many simulation steps the episode ran before ending. |
| checkpoint | Which trained model checkpoint was evaluated (perceptual-framesamp-modul, step 79999 - the only publicly released checkpoint for this variant). |
| dataset_split | Which RoboMME data split the episode came from (always 'test' for evaluation). |
| action_space | Action representation used (joint_angle: 7 joint angles + gripper). |
| max_steps_cap | The episode step budget before an automatic timeout (1300, matches the paper). |
| timestamp_source | 'recorded' if completed_at_utc was captured live when the episode finished, or 'file_mtime_backfill' if approximated afterward from the result file's last-modified time (only applies to episodes run before timestamp tracking was added). |
