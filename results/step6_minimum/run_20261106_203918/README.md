# Step 6 runner output

| File | One row per | Key columns |
|---|---|---|
| `results.csv` | task (search run) | task_id, search (canonical/shuffled/multistart/random), model_key, sample_position, budget (A/B), explanation_budget, order_seed, control_seed, objective_score, best_probability, still_attack, stopped_reason, explanation_calls, scored_proposals, accounting columns, runtime_seconds |
| `checkpoints.csv` | task x checkpoint | task_id, search, method, strategy, checkpoint, checkpoint_valid, prefix_shortcut, best_score, explanation_calls_at_best, best_candidate_probability, candidate_json, candidate_is_original, constraint_* |
| `best_so_far.csv` | fresh explanation | task_id, search, explanation_calls, score, best_so_far |
| `restarts.csv` | multi-start restart | task_id, restart_id, restart_seed, feature_order, direction_order, fresh_call_cap, explanation_calls_used, stopped_reason, best_score |
| `tasks/<task_id>.json` | task | full payload (spec, summary, checkpoints, curve, restarts) |
| `tasks/<task_id>.history.csv.gz` | proposal | complete candidate history of the task |
| `manifest.json`, `run_config.json`, `run_metadata.json`, `status.json` | run | provenance, selection, manifest hash, completion status |

Comparisons: use `checkpoints.csv` rows with `checkpoint == B` to compare methods at budget B
(explanation calls, baseline included). Accounting: `explanation_evaluations` are LIME explanations
(the budgeted quantity); `lime_internal_prediction_rows` are the model rows LIME evaluated
internally; `top_level_prediction_calls` are the search's own single-row checks.
`constraint_physical_status` is never a claim of physical realism.
Held-out evaluation: `python experiments/unsw_targeted_heldout.py --run-dirs <this directory>`.
