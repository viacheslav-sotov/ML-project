# Step 6 analysis: Monday meeting summary

Branch `feature/step6-analysis` (commit `5da5a76`, built on `feature/step6-runner` at `d5670e9`). Full report: `results/step6_shared/analysis_run_20261106_203918/STEP6_ANALYSIS_REPORT.md`.

## What I completed

- Added `experiments/unsw_step6_analysis.py`, a read-only analysis of the packaged Step 6 run. It runs 22 completeness and consistency checks (all pass) and produces 18 summary tables, 8 figures and a report.
- Added `tests/test_unsw_step6_analysis.py`, with 9 tests (all pass), including an end-to-end run on the real package.
- Verified the four initial findings against the raw files.
- I didn't touch the runner, protocol, models, data or the original run folder (`unsw_step6_package.py --verify` still reports OK).

## Most important findings

Denominators: **123 search runs** produced **186 candidate records** (best candidate per run at 25 and/or 50 calls), which are **147 unique modified flows**.

| Finding | Number |
|---|---|
| Runs completed | 123 / 123 |
| Targeted - random, held-out top-10 shift, 25 calls | -0.0045 (95% CI -0.015 to +0.007) |
| Targeted - random, held-out top-10 shift, 50 calls | +0.0011 (95% CI -0.010 to +0.012) |
| Median held-out top-10 shift | 0.045 for every method at 50 calls; 0 at 25 calls |
| Combined held-out stability test passed | 0 / 186 (top-10 alone: 26, all on flow 2941; rank correlation alone: 0) |
| Break a checked traffic formula | 75 / 186 records (64 / 147 unique); 0 consistent with all checks |
| Non-compliant with the stricter feature table | 106 / 186 records (85 / 147 unique); this table is not enforced |
| SHAP on modified flows | not computed in Step 6 |

## What worked and what didn't

**Worked**
- The pipeline is solid. Every run finished within budget, every candidate stayed an attack, the saved scores reproduce exactly, and the held-out evaluation matches the earlier Part B LIME-stability data.
- The held-out seed design did its job: it showed that the search-seed gains don't carry over beyond LIME's noise.

**Didn't work**
- Targeted search did not clearly beat random. Random was equal or better at 25 calls, and at 50 calls the gap is small and inside the uncertainty.
- No candidate changed the explanation by more than LIME's own seed-to-seed variation on the combined test.
- The edits aren't physically consistent: every `sbytes` or `dur` edit broke a dependent formula (`smean`, `sload`, `rate`).

## Questions to discuss

1. **Model file:** my local `random_forest_validation_pipeline.joblib` has SHA-256 `2f39fbff...`, but Step 6 used `4e49f17d...`. Because of this, two of the runner tests fail on my machine. Can you share the exact model file, or tell me where the matching copy is?
2. **Canonical greedy:** it's highest on all 3 flows at 50 calls (n=3). Is that worth a targeted follow-up, or is it most likely one ordering that happens to suit these flows?
3. **Is the stability test too strict, or just right?** With only 6 seed pairs, the threshold is the single most extreme pair. Should we keep the minimum (conservative), or predeclare a percentile rule *before* the next run? Either way it must be fixed in advance, not tuned after seeing results.
4. **Flow 2941:** its top 10 never changes across seeds, so one swap counts as a pass. Should we require a minimum noise floor, or report such flows separately?
5. **Constraint policy:** should the stricter feature table (106/186 non-compliant) become the enforced policy, and should the search repair dependent features instead of just reporting violations?
6. **Scope of the next run:** 100 calls, Budget B and XGBoost are all in the protocol. What's the order, and how much compute do we have?

## Recommendations for the next round

- **Physical-consistency constraints: yes, improve them first.** Recompute `smean`, `sload`, `dload` and `rate` whenever `sbytes`, `dbytes`, `spkts` or `dur` change (or block those edits). Decide how to treat `sttl` → `ct_state_ttl` and the inter-packet timing features. Otherwise any future "success" will be dismissed as unrealistic traffic.
- **More samples: yes.** Three flows can't support general claims, and one unusual flow (2941) produced every top-10 pass. Aim for at least 10-20 correctly detected attack flows across attack categories, chosen before running.
- **LIME stability thresholds: review them, and predeclare any change.** Also test whether more LIME samples (more than 5,000) reduce seed noise, and consider optimising the search against several seeds instead of seed 11 alone.
- **Other models: yes, after the constraint fix.** Run XGBoost as the protocol already plans; its LIME explanations are even less stable in Part B, so expect a harder test. Compute SHAP for a subset of candidates so we can finally compare SHAP and LIME on modified flows.
- **Statistics:** with more flows we can predeclare one primary comparison (for example, held-out top-10 shift at 50 calls) and correct the rest, instead of 24 exploratory comparisons.
