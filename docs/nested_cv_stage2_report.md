# Nested CV, Stage 2 (full42: XGBoost, RF, LR)

**Protocol:** `step6-v3-draft-1` (sha `46779f91…`), unchanged from Stage 1: the v3 training side (139,520 rows, split digest `1d4c347d…`, data digest `4e2a6644…`), grouped category-stratified 5 outer × 3 inner folds, seeds 42/43/44, F1 selection, the same search spaces.
**Run:** `results/v3_runs/nested_cv_20261118_002142`. It was launched through `experiments/run_guarded.py`, so the official-test guard with identity lock was active for the whole process (code at `e4c9735`; `unsw_nested_cv.py` module hash `5ee4e1e9…` is the same as Stage 1). Final model files are in `models/` (*.joblib, outside Git).

## Run
18:21:42 → 19:56:59 (≈ 95 min): XGBoost ≈ 34 min, RF ≈ 55 min, LR ≈ 6 min. All 5 outer folds and the final refit completed for every model; 0 groups shared between outer train and test in every fold; no errors.

| Model | Outer F1 mean ± sd | Recall | FPR | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | 0.9699 ± 0.0012 | 0.9765 | 0.0793 | 0.9971 | 0.9938 |
| Random Forest | 0.9686 ± 0.0003 | 0.9781 | 0.0887 | 0.9968 | 0.9934 |
| Logistic Regression | 0.9559 ± 0.0006 | 0.9879 | 0.1689 | 0.9923 | 0.9853 |

For comparison, Stage 1 (no_ttl39, independently tuned) gave outer F1 of 0.9693 (XGBoost), 0.9684 (RF) and 0.9478 (LR). These are descriptive figures on the same grouped outer folds; no test was predeclared for this difference.

Chosen hyperparameters per outer fold:
- **XGBoost:** folds 0, 1 and 3 chose depth 8, lr 0.1, 500 trees, colsample 0.9, λ 1, subsample 1.0; fold 2 chose the same with depth 6; fold 4 chose depth 8, lr 0.1, 200 trees, colsample 0.7, λ 5, subsample 0.8. All folds used min_child_weight 1.
- **RF:** folds 0–1 chose no depth limit, leaf 1; folds 2–4 chose depth 20, leaf 2. All folds used 400 trees, max_features 0.3 and no class weight.
- **LR:** C = 10 in every fold.

**Final models** (refit on the whole v3 training side):

| Model | Hyperparameters | Inner F1 | SHA-256 |
|---|---|---|---|
| XGBoost | depth 8, lr 0.1, 500 trees, min_child_weight 1, λ 1, subsample 1.0, colsample 0.9 | 0.9693 | `670fbcf2b1ac6c7610452ed07bd447f05568e899dd1630da4c9d3e05cf6332a2` |
| RF | 400 trees, no depth limit, max_features 0.3, leaf 2 | 0.9683 | `abda2ca069cc6c9e1d6ce90f9bf5d2ee8e2e8f9374a7840c7c7c6bbf4d4ed757` |
| LR | C 10 | 0.9554 | `ba24a396fcd383002ca5e2889e5aa1956b60dbd04b3abfe14485c61cf5eaac01` |

## Integrity after Stage 2
`verify_nested_cv_state.py`, run under the guard, found no problems on the benchmark, Stage 1 and Stage 2 runs:
- all run headers (protocol, module, data and split digests) match the current state, so resume stays valid;
- 18/18 units are present in each nested-CV run;
- all 6 final model hashes are verified;
- the pinned RF/XGBoost hashes are unchanged;
- no nested-CV process is active.

The official test file was not opened: the guard was active, and this run did not use the pre-fix import path.
