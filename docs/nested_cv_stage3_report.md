# Nested CV, Stage 3 (MLP: no_ttl39 and full42)

**Protocol:** `step6-v3-draft-1` (sha `46779f91…`), unchanged from Stages 1–2:
- data: the v3 training side (139,520 rows, split digest `1d4c347d…`, data digest `4e2a6644…`);
- folds: grouped, category-stratified, 5 outer × 3 inner, seeds 42/43/44;
- selection: F1;
- search: the declared MLP space with 12 trials, standard scaling inside the pipeline.

**Run:** `results/v3_runs/nested_cv_20261118_020018`, launched through `experiments/run_guarded.py`, so the official-test guard (with identity lock) was active for the whole process. Code at `e4c9735`; the `unsw_nested_cv.py` module hash `5ee4e1e9…` is the same as in Stages 1–2. Final model files are in `models/` (*.joblib, outside Git).

## Run
20:00:18 → 00:56:46 (≈ 4 h 56 min):
- no_ttl39: ≈ 159 min;
- full42: ≈ 137 min.

The benchmark projection was ≈ 178–183 min per feature set (range 89–275). All 5 outer folds and the final refit completed for both feature sets, 0 groups were shared between outer train and test in every fold, and there were no errors. The laptop stayed on AC power throughout.

| MLP | Outer F1 mean ± sd | Recall | FPR | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| no_ttl39 (39 features) | 0.9620 ± 0.0010 | 0.9690 | 0.0971 | 0.9955 | 0.9905 |
| full42 (42 features) | 0.9620 ± 0.0011 | 0.9711 | 0.1021 | 0.9955 | 0.9906 |

Chosen hyperparameters per outer fold:
- **no_ttl39:**
  - folds 1–4 chose hidden layers (256, 128), α 1e-4, batch 256; learning rate 3e-4 in folds 1 and 3, 1e-3 in folds 2 and 4;
  - fold 0 chose (128, 64), α 1e-3, batch 512, learning rate 3e-4.
- **full42:**
  - fold 0 chose (256, 128), α 1e-5, batch 512, learning rate 3e-4;
  - folds 1, 2 and 4 chose (128, 64), α 1e-5, batch 256, learning rate 1e-3;
  - fold 3 chose (128, 64), α 1e-3, batch 512, learning rate 3e-4.

**Final models** (refit on the whole v3 training side):

| MLP | Hyperparameters | Inner F1 | SHA-256 |
|---|---|---|---|
| no_ttl39 | (256, 128), α 1e-4, batch 256, lr 3e-4 | 0.9619 | `99c614981ca17306b81d668b8f931aa7a1cb373d438b6ed561ea1144ea9a55ff` |
| full42 | (256, 128), α 1e-4, batch 256, lr 3e-4 | 0.9619 | `6e16abd95d0a240738d2faac7325d0fd2098322cfa4b2c4cafc040523feccba7` |

## All nested-CV results (outer-fold F1, mean ± sd; descriptive only)

| Model | full42 | no_ttl39 |
|---|---|---|
| XGBoost | 0.9699 ± 0.0012 | 0.9693 ± 0.0008 |
| Random Forest | 0.9686 ± 0.0003 | 0.9684 ± 0.0006 |
| MLP | 0.9620 ± 0.0011 | 0.9620 ± 0.0010 |
| Logistic Regression | 0.9559 ± 0.0006 | 0.9478 ± 0.0010 |

These are means over the same 5 grouped outer folds. No test of the with/without-TTL differences was predeclared, so none is claimed. Explanation robustness on these models has not been evaluated yet.

## Integrity after Stage 3
`verify_nested_cv_state.py` ran under the guard on the benchmark and on Stages 1, 2 and 3, and found no problems:
- all 4 run headers (protocol, module, data and split digests) match the current state, so resume stays valid;
- every planned unit is present and every run is complete;
- all 8 final model hashes are verified;
- the pinned RF/XGBoost hashes are unchanged;
- no nested-CV process is active.

The official test file was not opened in Stage 3: the guard was active throughout and the pre-fix import path was not used.

**Nested CV for all four model families and both feature sets is now complete.** No attack experiments have been run on these models.
