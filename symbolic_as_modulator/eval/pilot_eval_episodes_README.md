# pilot_eval_episodes.csv - column reference

Raw per-episode results for the symbolic-as-modulator probe's Counting-suite evaluation (checkpoint Nkoni/symbolic-as-modulator-pilot/9999, symbolic_as_modulator/README.md). One row per completed episode, newest completion first.

| Column | Meaning |
|---|---|
| completed_at_utc | When this episode finished, ISO 8601 UTC. See timestamp_source. |
| seed | Policy sampling seed ([0, 42, 7] -- controls the flow-matching model's own action-sampling randomness, NOT the environment). |
| task_id | Which Counting-suite task (BinFill). |
| episode_idx | Which of the fixed test scenarios for that task (0-49) -- a fixed, benchmark-defined starting condition, not a repeat/retry. |
| success_flag | Outcome: success / fail / timeout (hit the 1300-step cap without resolving) / error (simulator exception). |
| steps | How many simulation steps the episode ran before ending. |
| checkpoint | Which checkpoint was evaluated (Nkoni/symbolic-as-modulator-pilot/9999). |
| dataset_split | Which RoboMME data split the episode came from (always 'test' for evaluation). |
| action_space | Action representation used (joint_angle: 7 joint angles + gripper). |
| max_steps_cap | The episode step budget before an automatic timeout (1300, matches the paper). |
| timestamp_source | 'recorded' if completed_at_utc was captured live when the episode finished, or 'file_mtime_backfill' if approximated afterward from the result file's last-modified time. |
