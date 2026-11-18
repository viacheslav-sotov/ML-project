# E0 Budget A: pre-flight audit, smoke test and execution summary

**Date:** Nov 18, 2026 · **Branch:** `feature/v3-preparation` · **Protocol:** `step6-v3` (`experiments/step6_protocol_v3.json`), approved for the design and E0 Budget A. **The full run waits for your approval of this summary.**

## 1. Pre-flight audit (`experiments/unsw_v3_preflight.py` → `results/v3_prep/preflight/preflight.json`): 20/20 checks pass

| | Check | Result |
|---|---|---|
| A1 | Excluded pilot ids are the official `id`s of the 10 pilot rows (64810, 73817, 74771, 85162, 92635, 99234, 99900, 123911, 130085, 134318), not their 0-based positions | pass |
| A2–A3 | Positions ≠ ids; no pilot flow in the sample | pass |
| A4–A5 | The sample is not the discarded draw (`93a0f50e…`, on a deny-list in `src/unsw/v3_sample.py`); digest `0ac83281…` and CSV SHA-256 `51c4d872…` equal the protocol pins. Every run reads the sample only through `load_frozen_sample()`, which refuses any other file. The set-aside copies on the laptop were renamed `INVALID_…_DO_NOT_USE` | pass |
| B1–B2 | 263 unique ids, groups and feature vectors; the 42-value vector hashes recomputed from the data are equal | pass |
| B3–B5 | All on the v3 validation side; no vector on the training side; all singletons of a uniform category | pass |
| B6 | Recomputed: 30 per category, Worms 23; 208 TCP / 55 other; 55 short no-reply flows | pass |
| B7–B8 | Predictions of all 8 models recomputed (gap ≤ 3.3e-16, identical decisions); all 263 detected by the three primary full42 models | pass |
| F1 | Isolation guard active with identity lock | pass |
| F2–F3 | 8 model hashes; training, v2 and G4 protocol hashes | pass |
| F4–F5 | `feature/v3-preparation` is a fast-forward of `origin/feature/step6-v2`; clean tree | pass |
| F6–F7 | No model files, raw data, local-only folders or files > 20 MB in the outgoing commits; no AI attribution in outgoing messages. One pre-existing tracked model is reported, not new: `results/models/xgboost_validation_pipeline.joblib` (the v2 XGBoost, in Git since Oct 14) | pass |

**Rule D (frozen in the protocol):** a flow whose original a model variant does not detect is reported as not applicable for that variant. It is never replaced. Affected flows:

| Variant | Not detected |
|---|---|
| RF no_ttl39 | 112351, 132198, 135210 |
| XGBoost no_ttl39 | 135210 |
| MLP no_ttl39 | 99866, 135210 |
| LR full42 | 51776, 80121, 82152, 82691, 92855, 113687 |
| LR no_ttl39 | 70829, 92855 |

All three primary full42 models detect all 263. Paired full42-vs-no_ttl39 comparisons use flows detected by both variants.

## 2. Protocol v3 finalized (check E)

`tests/test_protocol_v3.py` checks that the protocol keeps:
- E0 and G1–G6;
- DEF-1/DEF-2;
- Budgets A/B and explanation budgets 25/50/100;
- seeds 11/22/33/44/55 (+401–404);
- Wilson/Clopper–Pearson intervals, P2 (Wilcoxon + Holm), McNemar and bootstrap, all marked descriptive where approved;
- RQ1–RQ5;
- the limitations (no packet-level validation yet, unverified timing dependencies, G4 diagnostic only, grouped CV not chronological);
- the single predeclared official-test evaluation (D10).

The protocol also records:
- your decisions D-v3-1 to D-v3-7;
- the TreeSHAP background pins (row-id hash `c0725835…` and per-variant matrix hashes);
- Rule D;
- one change made before any E0 result (section 4).

## 3. Exact E0 Budget A plan (`--plan-only`, check C)

All 263 flows; 5,947 unique valid `physical_v1` Budget-A candidates; LIME seeds 11/22/33/44/55 for every detected original and candidate.

| Variant | Flows (Rule D) | Candidates evaluated (not detected) | LIME calls | TreeSHAP | Est. hours |
|---|---|---|---|---|---|
| RF full42 | 263 | 5,939 (8) | 31,010 | 6,202 | 3.31 |
| XGBoost full42 | 263 | 5,939 (8) | 31,010 | 6,202 | 1.17 |
| MLP full42 | 263 | 5,947 (0) | 31,050 | — | 0.57 |
| RF no_ttl39 | 260 (3) | 5,869 (17) | 30,645 | 6,129 | 3.28 |
| XGBoost no_ttl39 | 262 (1) | 5,901 (14) | 30,815 | 6,163 | 1.16 |
| MLP no_ttl39 | 261 (2) | 5,882 (14) | 30,715 | — | 0.56 |
| LR full42 (secondary) | 257 (6) | 5,856 (0) | 30,565 | — | 0.48 |
| LR no_ttl39 (secondary) | 261 (2) | 5,867 (38) | 30,640 | — | 0.48 |
| **Total** | | | **246,450** | **24,696** | **≈ 11.0 h** |

- **Primary only:** 185,245 LIME calls + 24,696 TreeSHAP ≈ **10.05 h**; LR adds ≈ 0.96 h.
- **Confirmation (D14):** computed only for candidates beyond the held-out band. Each needs 4 LIME calls, plus 4 per flow for the original. The worst case (every candidate flagged) is 197,160 calls; the pilot and this smoke flagged none.
- **Earlier "≈ 10 h" estimate:**
  - *Included:* the 6 primary variants (LIME on every candidate and original, 5 seeds) and TreeSHAP for RF/XGBoost, as an upper bound over all 5,947 candidates.
  - *Not included:* LR, confirmation calls and overhead.
  - *Now:* the exact plan is 10.05 h primary + 0.96 h LR. Enumeration takes 6 s. Confirmation is on demand.
- The smoke ran slightly faster than these per-call figures predict, so the estimate is conservative.

## 4. Smoke test (`results/v3_runs_e0/e0_budgetA_smoke_20261118_082509`)

**Flows:**
- 48709: TCP Backdoor, 10 candidates, all consistent.
- 83303: short no-reply Fuzzers, 9 candidates.
- 135210: TCP Generic, 32 candidates; Rule D for RF, XGBoost and MLP no_ttl39.
- 51776: TCP Backdoor; Rule D for LR full42.

All 8 variants were included.

| Step | Result |
|---|---|
| Run with `--max-new-explanations 400` | stopped as designed; status `interrupted`; 400 explanations in the cache |
| `--resume` | 1,754 new + 400 reused = **2,154 = 1,960 LIME + 194 TreeSHAP, exactly the plan**; 0 dropped lines; 193 s for the resumed part |
| `--resume` again | **0 new explanations; every output file byte-identical** (only `status.json` differs: timestamp and counter) |
| `--resume` with other flows | refused: `header differs on ['flows', 'sample_sha256']` |
| Rule D | 135210 listed as not applicable for the 3 no_ttl39 primary variants, 51776 for LR full42; their candidates are counted and not explained |
| Criteria | 61 candidates per full42 tree variant (39 consistent, 22 unverified). **0 exceed the held-out band** for any variant, so 0 LIME successes, 0 confirmation calls and 0 D11 successes. This is a 4-flow pipeline check, not a result |
| TreeSHAP additivity | see below |

**The one problem found and how it was handled.** The first smoke attempt stopped because 2 of 65 RF full42 rows had a TreeSHAP additivity gap of 1.25e-3, above the strict 1e-4 tolerance of the compatibility check.
- **Diagnosis:**
  - Both rows are `sbytes` edits of flow 83303.
  - The gap equals half of one tree's vote out of 400, and in at least one row the edited value lies exactly on an `sbytes` split threshold.
  - Casting to float32 did not change it.
  - This is a traversal tie between SHAP and scikit-learn, not a pipeline error.
- **Change, recorded in the protocol `change_log` before any E0 result:**
  - The abort threshold is now the v2 value of 1e-2 (`unsw_shap_explain.validate_shap_for_model`).
  - Every explanation's gap is stored in the cache and the results, and gaps above 1e-4 are counted per variant.
- **In the smoke:**
  - RF full42: 2 rows above 1e-4, max 1.25e-3.
  - XGBoost full42: 18 rows above 1e-4, max 3.0e-4 on the log-odds scale (float32 margins).
  - All other variants: ≤ 8.8e-6.
- The failed attempt's folder was moved out of `results/`; nothing from it is used.

**Tests after these changes:**
- Full fast suite: 384 passed and 1 failed. The failure was a flaky existing test, `test_v3_pipelines.py::test_no_ttl_model_ignores_ttl_features`: it compares RF `n_jobs=-1` predictions for exact equality, and multi-threaded RF predictions can differ in the last bit between calls.
- That test now uses single-thread RF (`n_jobs=1`), with the exact-equality check kept, and it passed 5 of 5 repeats.
- The 4 exact v1 replays are identical.

## 5. Checkpoint and resume

- Every LIME and TreeSHAP explanation is appended and flushed to `explanations_cache.jsonl`. Its identity key covers explainer, model SHA-256, feature set, settings hash, seed and the vector hash over the model's own features.
- An interruption (crash, sleep, power) loses at most the explanation in progress. A line cut mid-write is dropped and recomputed.
- `--resume <run_dir>` refuses a different header (protocol, code, explainer code, physical policy, sample, split, budget, flows, variants, model hashes, seeds, LIME settings, background pins). It reuses every cached explanation, and the result files are rebuilt from the cache, identical to an uninterrupted run.
- Per-variant result files are written when a variant finishes, in the order RF full42, XGBoost full42, MLP full42, then no_ttl39, then LR. Progress in between is visible as the cache line count.

## 6. Exact command (after your approval)

```
python experiments/run_guarded.py experiments/unsw_v3_e0.py --budget A
```

- Single process, BelowNormal priority, laptop on AC power with the keep-awake helper.
- Output: `results/v3_runs_e0/e0_budgetA_<timestamp>/`.
- Resume if interrupted:
  ```
  python experiments/run_guarded.py experiments/unsw_v3_e0.py --budget A --resume results/v3_runs_e0/e0_budgetA_<timestamp>
  ```
- Expected ≈ 11 h (≈ 10 h without LR), plus any confirmation calls.

## 7. For your approval

1. Start the full E0 Budget A run with the command above.
2. Keep LR (secondary, ≈ 1 h) in this run, or drop it with `--variants` listing only the 6 primary variants.
3. Note the change log entry: TreeSHAP abort threshold 1e-2, with all gaps recorded and reported.
