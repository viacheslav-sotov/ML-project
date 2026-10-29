# Part B findings: random-perturbation explanation sensitivity (UNSW-NB15 pilot)

**Status: preliminary pilot.** The Part A candidates are *random* constrained perturbations. This
analysis measures how much SHAP and LIME explanations move under those random edits. It is **not** a
targeted attack and does **not** demonstrate explanation spoofing (see "What this does not show").

- Script: `experiments/unsw_spoofing_explanations.py`
- Inputs: `results/spoofing_pilot/run_20261027_164146/candidates.csv` (read-only, unaltered)
- Data: internal 80/20 validation split of the official training set (seed 42) only. The official
  test set was not used for any evaluation.
- Evaluated: **all 600 accepted candidates** (no subset), 12 groups × 50; 594 unique modified vectors
  (6 duplicates in Part A's output, kept as-is).
- Runtime: 407.8 s on a laptop CPU (LIME ≈ 0.57 s per candidate for 5 seeds; RF SHAP 56 s for 300
  candidates; XGBoost SHAP < 1 s).

## Handoff verification

| Check | Result |
|---|---|
| SHA-256: training CSV, testing CSV, XGBoost pipeline, shared manifest | PASS |
| SHA-256: Random Forest pipeline | **FAIL**: expected `4e49f17d…6531`, used `2f39fbff…e173` |
| RF equivalence (file supplied by Part A's author) | 200/200 trees identical (feature, threshold, value); identical preprocessing; max \|ΔP\| 2.2e-16 on all 35,069 validation rows |
| 600 stored P(attack) values, original and modified, both models | reproduced exactly (max gap 1e-16) |
| Original vectors equal their validation rows; 42-feature order | PASS |
| Recomputed original SHAP vs. `results/metrics/shap` | max gap 1e-16 |
| Recomputed original 5-seed LIME vs. `results/lime_stability` | max gap 5e-13 |
| Regression tests | 12/12 pass |

The verifier cannot pass on the RF hash; the hash difference comes from the model bundle's metadata
fields, not the model. Recorded as a deviation in `run_metadata.json`.

## Methods

Each modified candidate is compared with **its own** original sample, separately per model.

- **SHAP**: the setup of `unsw_shap_explain.py` (TreeExplainer, 100 background rows of the
  preprocessed validation split, `default_rng(42)`; RF on the probability scale, XGBoost on the logit
  scale). One-hot columns are summed into the 42 original features. SHAP is deterministic, so one
  comparison per candidate. Raw SHAP magnitudes are never compared between RF and XGBoost.
- **LIME**: the explainer of `unsw_lime.py` (5,000 samples, quartile discretiser, sub-training
  background) under the five stability seeds 11, 22, 33, 44, 55. Each candidate is compared with its
  original **under the same seed**, and metrics are averaged over the 5 seeds.
- **Metrics** (definitions of `unsw_lime_stability.compute_pairwise_metrics`, checked to match it
  exactly): top-1 change; top-5/top-10 Jaccard; Spearman correlation of the 42 rank positions
  (|attribution| descending, ties by name); sign agreement on the shared top-10 features;
  P(attack) original, modified, and ΔP.
- **LIME seed-noise rule (declared in the script before any candidate was explained):** the noise
  band is the 10 seed-to-seed comparisons of the *unmodified* original. A candidate exceeds seed
  noise on a metric if its mean value is strictly below the band's minimum. Primary flag = exceeds on
  top-10 Jaccard **and** Spearman. Top-1 rule: the original's top-1 is stable in all 10 seed pairs and
  the candidate's top-1 differs in ≥ 3 of 5 seeds.

## Verified results

Pooled per model × budget (n = 150 each; per-sample rows in `summary.csv`):

| Explainer | Model | Budget | Top-1 changed | Top-5 J | Top-10 J | Spearman | Sign agr. | Mean ΔP | Min P(mod) | > seed noise |
|---|---|---|---|---|---|---|---|---|---|---|
| SHAP | RF | A | 10 | 0.942 | 0.979 | 0.995 | 1.000 | −0.006 | 0.935 | n/a |
| SHAP | RF | B | 21 | 0.900 | 0.961 | 0.987 | 1.000 | −0.011 | 0.920 | n/a |
| SHAP | XGB | A | 7 | 0.991 | 0.993 | 0.996 | 0.999 | −0.002 | 0.916 | n/a |
| SHAP | XGB | B | 9 | 0.976 | 0.976 | 0.990 | 0.992 | −0.005 | 0.865 | n/a |
| LIME | RF | A | 0 | 0.986 | 0.994 | 0.984 | 1.000 | −0.006 | 0.935 | 0 |
| LIME | RF | B | 0 | 0.979 | 0.992 | 0.980 | 1.000 | −0.011 | 0.920 | 0 |
| LIME | XGB | A | 12 | 0.940 | 0.909 | 0.930 | 0.998 | −0.002 | 0.916 | 0 |
| LIME | XGB | B | 8 | 0.947 | 0.914 | 0.935 | 0.998 | −0.005 | 0.865 | 0 |

(J = mean Jaccard; LIME values are means over 5 same-seed pairs.)

1. **No candidate's LIME shift exceeds same-input seed noise under the primary rule (0/600).** Per
   metric: Spearman 0/600, top-5 Jaccard 0/600, top-10 Jaccard 21/600, top-1 rule 20/600.
2. The 21 top-10 Jaccard exceedances are all **RF, sample 2941** (8 A, 13 B), where LIME's top-10 is
   identical across all 10 seed pairs, so any change counts. Their Spearman stays within noise.
3. The 20 top-1 exceedances are all **XGBoost, sample 4407** (12 A, 8 B), and every one edited `sttl`.
   The original top-1 (`sttl`) moved to `ct_state_ttl` in ≥ 3 of 5 seeds, but the top-10 and Spearman
   shifts stay inside that sample's wide seed-noise band (seed-pair top-10 Jaccard as low as 0.43).
4. **SHAP top-1 changed for 47/600 candidates, every one of them with an `sttl` edit**: RF 3206
   `sttl → ct_state_ttl` (14), RF 4407 `sttl → state` (17), XGB 4407 `sttl → ct_dst_sport_ltm` (16).
   Since SHAP has no seed noise, these are real input-driven rank changes, but they swap the top
   feature with a closely related TTL/state feature and leave the rest of the ranking intact
   (Spearman ≥ 0.94 and top-10 Jaccard ≥ 0.82 for each of these 47 candidates).
5. **Predictions barely move.** Mean ΔP is −0.002 to −0.011 per group; the lowest modified P(attack)
   is 0.865. ΔP is exactly 0 for 60–63% of candidates, and the full 42-feature ranking is unchanged for 16% (RF)
   and 55% (XGB) of candidates under SHAP and 45% / 57% under LIME.
6. **Feature signs are stable**: top-10 sign agreement ≥ 0.986 in every group.
7. The magnitude of RF SHAP shifts tracks the prediction change (correlation of 1 − Spearman with
   |ΔP|: 0.88); for XGBoost it does not (0.15).
8. LIME seed noise is much larger for XGBoost (minimum seed-pair top-10 Jaccard 0.43–0.54) than for
   RF (0.82–1.0), so the same input already gives noticeably different LIME rankings for XGBoost.

**Observation, not verified:** XGBoost LIME shifts take a few discrete values shared by many
candidates (e.g. 20 candidates at exactly 0.520), even when ΔP = 0. This is consistent with LIME's
quartile discretisation, where an edit changes the explanation when it moves a feature into another
quartile bin, but this mechanism was not tested directly.

## Figures

- `figures/rank_change_example.png`: Original vs. modified top-10 ranking for
  `xgboost_A_sample_4407_017`, the candidate with the lowest mean LIME top-10 Jaccard (0.520; 20
  candidates share this value, first by candidate_id shown). This is the largest observed shift, not the typical
  candidate. SHAP (left) is unchanged; LIME seed 11 (right) keeps its top 5 and swaps 4 of ranks
  6–10. Random constrained perturbation, not a targeted attack.
- `figures/lime_shift_vs_seed_noise.png`: Per candidate, the mean LIME top-10 Jaccard between
  original and modified sample (blue = budget A, orange = budget B), against the 10 seed-to-seed
  values of the unmodified original (grey ticks, with counts). Points left of the dashed line are less
  similar than the least similar seed pair; none is, except RF 2941, where seed noise is zero.

## What this does not show, and limitations

Explanation spoofing would require **all** of: the malicious prediction preserved, budget and
validator passed, an explanation shift beyond LIME seed noise, **and** a shift larger than a matched
random control. The first two hold for all 600 candidates; the third holds for none under the
primary rule; the fourth was not tested.

- **No targeted (explanation-guided) search and no matched random control were run.** Part A's
  candidates are themselves random, so `random_control_comparison.csv` was intentionally not created.
- Only **3 malicious samples** (one each of Generic, DoS, Exploits); results may not generalise.
- Feature-level validity is **synthetic**: the validator checks feature-space constraints, not that a
  real network flow with these values exists.
- The perturbations are small in practice: 1.2 (A) and 1.5 (B) features edited on average, against
  allowances of 5 and 10; categorical edits were excluded by Part A.
- Possible **dataset artifacts**: the TTL-related features (`sttl`, `ct_state_ttl`, `dttl`) dominate
  the explanations and drive every top-1 change. These features may encode dataset-collection artifacts rather than attack behaviour.
- **No official test-set results**; validation split only.
- All 600 candidates were evaluated, so no subset was used.
- RF hash mismatch with Part A's metadata (equivalence documented above).

## Report-ready paragraphs

> We evaluated how SHAP and LIME explanations respond to the 600 random, constraint-valid
> perturbations produced in Part A (three correctly classified malicious validation flows × Random
> Forest and XGBoost × two budgets, 50 candidates each). Each modified flow was compared with its own
> original. Predictions were essentially unchanged (mean ΔP(attack) between −0.002 and −0.011; minimum
> modified P(attack) 0.865), and explanation rankings were largely preserved: mean top-10 Jaccard was
> 0.96–0.99 for SHAP and 0.91–0.99 for LIME, mean rank correlations were ≥ 0.93, and the sign of the
> shared top features agreed on average in ≥ 98.6% of cases. The top-ranked SHAP feature changed for 47 of 600 candidates,
> all of which had edited `sttl`; in each case it was replaced by a closely related TTL/state feature.
>
> To separate input-driven change from LIME's own randomness, we compared each candidate's shift with
> the variation of the unmodified flow across five LIME seeds. Under the rule fixed before the run
> (shift larger than the least similar seed pair on both top-10 Jaccard and rank correlation), no
> candidate exceeded seed noise; XGBoost explanations showed the largest shifts, but also the largest
> seed-to-seed variation. These results describe random-perturbation explanation sensitivity only. No
> explanation-guided search or matched random control was run, the study covers three samples, and
> the feature-level validity is synthetic, so this pilot does not demonstrate explanation spoofing.
