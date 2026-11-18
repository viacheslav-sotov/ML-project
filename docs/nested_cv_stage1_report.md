# Nested CV, Stage 1 (no_ttl39: XGBoost, RF, LR) and official-test isolation check

**Protocol:** `step6-v3-draft-1` (sha `46779f91…`), unchanged. v3 duplicate-disjoint training side (139,520 rows, split digest `1d4c347d…`), grouped category-stratified 5 outer × 3 inner folds, seeds 42/43/44, F1 selection, declared search spaces (XGB 25, RF 15, LR 4 trials).
**Run:** `results/v3_runs/nested_cv_20261117_221501` (local commit `582d8d3`); final model files in `models/` (*.joblib, outside Git).

## Stage 1 results
16:15:01 → 17:54:02 (≈ 99 min): XGBoost 34 min, RF 59 min, LR 6 min. All 5 outer folds and the final refit completed for every model; 0 groups shared between outer train and test in every fold.

| Model | Outer F1 mean ± sd | Recall | FPR | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | 0.9693 ± 0.0008 | 0.9763 | 0.0817 | 0.9969 | 0.9935 |
| Random Forest | 0.9684 ± 0.0006 | 0.9781 | 0.0894 | 0.9967 | 0.9932 |
| Logistic Regression | 0.9478 ± 0.0010 | 0.9769 | 0.1806 | 0.9841 | 0.9690 |

Chosen hyperparameters per outer fold:
- **XGBoost:** folds 0, 1 and 4 chose depth 8, lr 0.03, colsample 0.7, λ 5, subsample 0.9; folds 2 and 3 chose depth 6, lr 0.1, colsample 0.9, λ 1, subsample 1.0. All folds used 500 trees and min_child_weight 1.
- **RF:** folds 0–1 chose 400 trees, no depth limit, leaf 1; folds 2–4 chose depth 20, leaf 2, with 200 trees (fold 2) or 400 trees (folds 3–4). All folds used max_features 0.3 and no class weight.
- **LR:** C = 10 in every fold.

**Final models** (refit on the whole v3 training side):

| Model | Hyperparameters | Inner F1 | SHA-256 |
|---|---|---|---|
| XGBoost | depth 8, lr 0.05, 500 trees, min_child_weight 5, λ 5, subsample 0.9, colsample 1.0 | 0.9689 | `ec41c67035f845541f5c27ba2ed9efc702789b1042045b39b57c7309ea876c0c` |
| RF | 400 trees, no depth limit, max_features 0.3, leaf 1 | 0.9681 | `8f270ee3597289f22320d25908d866cd0d21194ce7a7a06b574b182313b1c7d4` |
| LR | C 10 | 0.9474 | `cc1ae514afa7fe2d9f7b7043d4490fc3054ecd3a2e14feb6e1dcd41db68d642d` |

## Official-test access during Stage 1 (accurate record)
Stage 1 was started before the fix. At process start, `src.unsw.constraints` (imported by `v3_data.feature_order()`) called `load_official_data()`, which **parsed** the official test file and discarded it. No test record was used for any fit, selection or score. The file's last-access time is 16:14:41, the moment Stage 1 started.

## Isolation fix (`e4c9735` on `fix/lock-official-test`, local)
"Keep the official test partition closed in legacy code". It changes `src/unsw/data.py`, the manifest builder, a new guard module, `experiments/run_guarded.py`, `tests/conftest.py`, 20 new tests, and the approved one-line correction in `tests/test_v3_data.py` (the fake-file setup line was removed; the assertion is unchanged). No hash-pinned module, protocol, model, split, seed or result was changed.

| Check | Result |
|---|---|
| 20 isolation tests (5 real-data, none skipped) | 20 passed |
| Full fast suite (guard active) | 352 passed, 0 failed |
| Exact v1 replays (4 tasks) | 4 passed: summaries, checkpoints, best-so-far curves, restarts and histories identical |
| Legacy split (sizes, 20 shared flows), v2 pre-run and G4 manifest datasets/split/samples | identical (tests) |
| v3 data digest `4e2a6644…` and split digest `1d4c347d…` | identical to the benchmark and Stage 1 headers |
| Resume validity (benchmark and Stage 1 run headers vs current code/protocol/data) | all four header fields match |
| Stage 1 units / final model files | 18/18 present; 3 model hashes verified |
| Pinned RF/XGBoost model hashes | `4e49f17d…`, `83a34c61…` unchanged |
| Historical results | `git status` clean; since `704a51c` only the Stage 1 run files were added under `results/`; pilot caches match their recorded SHA-256 (in repo and backups) |
| Official test file during verification | never opened: the guard was active in every run and would have raised; its last-access time is still 16:14:41 (NTFS updates it on access once more than an hour has passed) |
| Stage 2/3 code path | under `run_guarded.py`: the guard and identity lock are active; only `experiments.unsw_nested_cv` and `src.unsw.{constraints,data,evaluation,models,official_test_lock,v3_data,v3_pipelines}` are loaded; no script that hashes or reads the test file is loaded; aliases (case variant, `..` path, Windows short name `UNSW_N~2.CSV`) are recognised as locked (metadata only) |
| Running jobs before Stage 2 | none; on AC power, 119.6 GB disk free, 5.8 GB RAM free |

Stage 2 (full42: XGBoost, RF, LR) started at 18:21 under the guard: run `nested_cv_20261118_002142`.
