# SHAP vs LIME - UNSW-NB15 (Random Forest and XGBoost)

Both methods explain the same 20 correctly classified malicious validation flows (`results/metrics/shap/shared_validation_sample_identifiers.csv`) with the same fitted models. This comparison only reads saved results; nothing was retrained or re-explained.

## How the comparison works

- **Same features:** SHAP's one-hot columns for proto/service/state were summed into one value per original feature (SHAP values are additive), so both methods cover the same 42 original features.
- **Ranks and signs, not raw values:** SHAP output scale is probability for Random Forest and raw_margin (logit) for XGBoost; LIME weights come from a local linear surrogate. Raw magnitudes are therefore not comparable.
- **Measures per flow:** same top-1 feature; top-5 / top-10 overlap (Jaccard = shared / union); Spearman rank correlation of |attribution| over all 42 features; sign agreement on the features both methods put in their top 10.
- **Global:** features ranked by mean |attribution| over the flows, compared as share of total importance.

## Input checks

- PASS: shared file has 20 unique IDs
- PASS: Random Forest: SHAP IDs exactly match the shared file
- PASS: Random Forest: LIME IDs exactly match the shared file
- PASS: Random Forest: no missing values in SHAP or LIME
- PASS: Random Forest: same model in SHAP and LIME (max P(Attack) difference 0.0e+00)
- PASS: XGBoost: SHAP IDs exactly match the shared file
- PASS: XGBoost: LIME IDs exactly match the shared file
- PASS: XGBoost: no missing values in SHAP or LIME
- PASS: XGBoost: same model in SHAP and LIME (max P(Attack) difference 2.9e-08)
- PASS: SHAP values add up to the model output for both models (checked while loading)

## Results per model

| Measure | Random Forest | XGBoost |
|---|---|---|
| Same top-1 feature | 8.0 / 20.0 flows | 20.0 / 20.0 flows |
| Top-5 overlap (mean) | 3.3 of 5 (Jaccard 0.50) | 3.0 of 5 (Jaccard 0.43) |
| Top-10 overlap (mean) | 7.8 of 10 (Jaccard 0.66) | 6.2 of 10 (Jaccard 0.45) |
| Spearman, all 42 features (mean, min-max) | 0.50 (0.25 to 0.72) | 0.36 (0.22 to 0.56) |
| Sign agreement, shared top-10 (mean) | 98% | 88% |
| Global top-10 overlap | 9.0 of 10 (Jaccard 0.82) | 7.0 of 10 (Jaccard 0.54) |
| Global Spearman (mean abs) | 0.51 | 0.49 |

## Global top 10

**Random Forest**

- SHAP: sttl, ct_state_ttl, dttl, state, service, proto, swin, sloss, ct_srv_dst, sbytes
- LIME: sttl, ct_state_ttl, proto, dttl, swin, is_sm_ips_ports, sloss, state, service, ct_srv_dst
- In both: sttl, ct_state_ttl, dttl, state, service, proto, swin, sloss, ct_srv_dst

**XGBoost**

- SHAP: sttl, ct_dst_sport_ltm, sbytes, service, ct_state_ttl, proto, smean, ct_srv_dst, sinpkt, dbytes
- LIME: sttl, ct_state_ttl, ct_srv_dst, service, proto, ct_srv_src, smean, dttl, sbytes, ct_dst_ltm
- In both: sttl, sbytes, service, ct_state_ttl, proto, smean, ct_srv_dst

## Flows where the top-1 feature differs

- **Random Forest** (12): 2941 (SHAP service / LIME sttl), 3040 (SHAP service / LIME sttl), 6978 (SHAP service / LIME sttl), 15137 (SHAP dttl / LIME sttl), 15741 (SHAP dttl / LIME sttl), 18327 (SHAP proto / LIME sttl), 25740 (SHAP dttl / LIME sttl), 26636 (SHAP service / LIME sttl), 27521 (SHAP service / LIME sttl), 29409 (SHAP service / LIME sttl), 30107 (SHAP dttl / LIME sttl), 34186 (SHAP dttl / LIME sttl)

## Caveats

- A LIME weight is about the quartile *range* the value falls in (e.g. `62 < sttl <= 254`); SHAP is about the exact value. Some disagreement is expected even when both are correct.
- The reference data differs: SHAP used 100 validation rows as background, LIME used the sub-training split for its perturbation statistics.
- LIME's local surrogate fit (R²) is only moderate, so LIME is an approximation of the model.
- With 20 flows these numbers describe this sample set; they are a baseline for how much the two explainers naturally disagree, which the spoofing threshold must exceed.
