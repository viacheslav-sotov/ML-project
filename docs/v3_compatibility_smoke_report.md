# Protocol v3: explainability compatibility and smoke test

**Date:** Nov 18, 2026 · **Branch:** `feature/v3-preparation` (local) · **Run:** `experiments/unsw_v3_compat_smoke.py` under `experiments/run_guarded.py` → `results/v3_prep/compat_smoke/` (`report.json`, `rq5_smoke.csv`, `smoke_cache.jsonl`); 67 s; **no problems reported**.
No attack search was run. The smoke uses one sample flow and its first valid Budget-A lattice candidate only.

## 1. New code

| File | Purpose |
|---|---|
| `src/unsw/v3_final_models.py` | Registry of the 8 final refits (run, path, pinned SHA-256). `load_final_model()` refuses a model unless the file hash = pinned hash = training record, `best_params` = training record, and the fitted preprocessor uses exactly the protocol features in the protocol order (one-hot categoricals; scaling only for MLP/LR; `remainder="drop"`; no TTL feature in no_ttl39) |
| `src/unsw/v3_explain.py` | LIME and TreeSHAP for the new pipelines on **each model's own features** (42 or 39), the v2 ranking metrics (identical definitions), the RQ5 matched metrics, and an identity-keyed resumable explanation cache |
| `experiments/unsw_v3_compat_smoke.py` | The checks below |
| `tests/test_v3_explain.py` (11), `tests/test_v3_attack_sample.py` (10), `tests/test_protocol_v3.py` (4) | Lightweight tests (section 5) |

## 2. Models: hashes, preprocessing, order, predictions

- **Loading:** all 8 final models load with pinned = recorded SHA-256 and the recorded `best_params`. Each fitted preprocessor uses the 42-column protocol order and exactly its feature set. A wrong pin is refused (tested).
- **Deterministic inference:**
  - Random Forest with `n_jobs=-1` sums tree probabilities in thread order, so repeated predictions differed by up to 2.2e-16.
  - Explanations therefore use `n_jobs=1`, which is bit-reproducible (verified). This is an inference setting only: the fitted trees and the file hash are unchanged.
  - XGBoost, MLP and LR predictions are bit-reproducible as loaded. XGBoost is also independent of batch size.
- **Prediction preservation:**
  - Re-scoring the whole pool reproduces every stored probability exactly, for all 8 models.
  - On the 263 sample flows scored alone: RF and XGBoost 0, MLP ≤ 2.2e-16, LR ≤ 3.3e-16 (BLAS summation order depends on batch size), with identical decisions for every flow. The cross-batch tolerance is 1e-12 and stated in the report.

## 3. TTL features (item C3)

The three TTL features were replaced in 763 flows (263 sample + 500 other validation rows) by shuffling them across rows, by 0 and by 255:

| Model | no_ttl39: max change of P(attack) | full42 (sensitivity check) |
|---|---|---|
| RF | **0** | up to 0.97 |
| XGBoost | **0** | up to 1.00 |
| MLP | **0** | up to 1.00 |
| LR | **0** | up to 1.00 |

**The no_ttl39 models cannot be influenced by TTL**, and the test is sensitive. The no_ttl39 explainers do not contain the TTL features at all. Their values are copied from the flow only to satisfy the 42-column input, and the model never reads them. Rankings therefore compare only features each model uses: there are no artificial zero-weight TTL ties.

## 4. LIME and TreeSHAP compatibility

**LIME (all 8 models, flow 57194, seed 11):**
- weights for exactly 42 / 39 features, none of them TTL in the no_ttl39 models;
- the prediction function reproduces the model's probability exactly (gap 0);
- the same seed gives identical weights;
- no exact-zero weights.

The settings are the v2 settings with two documented changes:
- the feature space is the model's own features (kernel width √n · 0.75 for n = 42 or 39);
- the training statistics come from the v3 training side.

Top-5 features for this one flow, for orientation only (not a result):
- full42 RF/XGBoost: led by `sttl`, `ct_state_ttl`, `dttl`;
- no_ttl39: led by other features (e.g. RF `is_sm_ips_ports`, `state`, `proto`).

**TreeSHAP (RF and XGBoost, both sets, 10 sample flows):**
- additivity is checked per flow (tolerance 1e-4): RF on the probability scale, XGBoost on log-odds;
- one-hot columns are summed back into `proto`/`service`/`state`;
- 42 / 39 features, no TTL in no_ttl39;
- up to 3 features have an exact-zero SHAP value (features a tree model never splits on). These are genuine ties, kept and reported (tie-aware Spearman is reported alongside).
- **Documented change:** the background is 100 rows from the v3 training side (`default_rng(42)`). v2 used the old validation split, which now hosts the attack sample.

## 5. Smoke test: one candidate, RQ5 metrics, physical repair, cache and resume

- **Flow and candidate:** flow 57194 (TCP). Its first valid Budget-A `physical_v1` candidate is `spkts` → 21, with the derived features repaired by `physical_repair`. Its status is `unverified_dependencies`, as expected for a packet edit. It is still detected by all four tree models (P rises slightly).
- **RQ5 matched metrics** (LIME seeds 22/33 vs TreeSHAP, each against its own original):

  | Model | LIME top-10 J / ρ | TreeSHAP top-10 J / ρ |
  |---|---|---|
  | RF full42 | 1.00 / 0.9998 | 1.00 / 0.9987 |
  | XGBoost full42 | 1.00 / 0.9997 | 1.00 / 0.9548 |
  | RF no_ttl39 | 1.00 / 1.0000 | 1.00 / 0.9966 |
  | XGBoost no_ttl39 | 1.00 / 1.0000 | 0.82 / 0.9953 |

  These numbers only show that the pipeline works end to end. They are not results.
- **Cache and resume:**
  - The first pass computed 24 explanations (24 cache misses).
  - A second pass with a fresh cache object read from disk had 24 hits, 0 misses and identical results.
  - Cache keys separate explainer, model hash, feature set, settings hash, seed and vector hash (tested).
- **Tests on the laptop, with the guard active:**
  - 21 + 4 new tests pass;
  - full fast suite **373 passed, 0 failed**;
  - exact v1 replays 4/4 identical.

## 6. Not done here

- No attack search and no final explanation runs.
- No change to D8, D11 or D14, or to any earlier result.
- No official-test access.
