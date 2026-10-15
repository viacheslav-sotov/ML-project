# LIME results - UNSW-NB15 (Random Forest and XGBoost)

Samples: the 20 malicious validation flows in `results/metrics/shap/shared_validation_sample_identifiers.csv` (SHAP's shared sample set). Models: SHAP's validation pipelines. LIME weights explain P(Attack): **positive = supports the Attack prediction, negative = pushes towards Normal**.

## Settings

- lime: lime 0.2.0.1; seed 42 (per explanation: 42 + validation_position)
- perturbation samples: 5000; weights estimated for all 42 features; top 10 shown
- discretizer: quartile (fitted on the sub-training split); feature_selection: none; kernel width 4.861; surrogate Ridge(alpha=1)
- feature space: original 42 features; proto/service/state as categorical features
- reproducibility check: passed
- full settings: `lime_settings.json`

## Per-sample results (top 3 features)

| Model | Validation position | Official id | Attack category | Prediction | P(Attack) | Local R² | Top 3 LIME features (weight) |
|---|---|---|---|---|---|---|---|
| Random Forest | 2941 | 134318 | Generic | Attack | 1.0000 | 0.668 | sttl (+0.130); ct_state_ttl (+0.053); swin (+0.049) |
| Random Forest | 3040 | 123911 | Generic | Attack | 1.0000 | 0.663 | sttl (+0.129); ct_state_ttl (+0.052); sloss (+0.047) |
| Random Forest | 3206 | 64810 | DoS | Attack | 1.0000 | 0.643 | sttl (+0.128); service (+0.079); ct_state_ttl (+0.054) |
| Random Forest | 4407 | 92635 | Exploits | Attack | 1.0000 | 0.627 | sttl (+0.125); proto (+0.093); ct_state_ttl (+0.057) |
| Random Forest | 6978 | 160610 | Generic | Attack | 1.0000 | 0.654 | sttl (+0.126); ct_state_ttl (+0.052); swin (+0.047) |
| Random Forest | 15137 | 85162 | DoS | Attack | 0.9900 | 0.655 | sttl (+0.126); dttl (+0.075); ct_state_ttl (-0.051) |
| Random Forest | 15338 | 59743 | Exploits | Attack | 1.0000 | 0.641 | sttl (+0.125); ct_state_ttl (+0.057); swin (+0.046) |
| Random Forest | 15741 | 74771 | Fuzzers | Attack | 0.7850 | 0.679 | sttl (+0.128); dttl (+0.072); ct_state_ttl (-0.053) |
| Random Forest | 17889 | 106578 | Exploits | Attack | 1.0000 | 0.644 | proto (+0.133); sttl (-0.125); ct_state_ttl (+0.054) |
| Random Forest | 18327 | 112882 | Exploits | Attack | 1.0000 | 0.640 | sttl (+0.126); proto (+0.069); ct_state_ttl (+0.056) |
| Random Forest | 22855 | 101524 | Exploits | Attack | 1.0000 | 0.644 | sttl (+0.129); proto (+0.120); ct_state_ttl (+0.055) |
| Random Forest | 24358 | 164473 | Generic | Attack | 1.0000 | 0.658 | sttl (+0.127); ct_state_ttl (+0.054); swin (+0.048) |
| Random Forest | 25080 | 142904 | Generic | Attack | 1.0000 | 0.649 | sttl (+0.126); ct_state_ttl (+0.053); swin (+0.047) |
| Random Forest | 25740 | 99900 | Fuzzers | Attack | 0.8100 | 0.671 | sttl (+0.125); dttl (+0.073); ct_state_ttl (-0.054) |
| Random Forest | 26636 | 151877 | Generic | Attack | 1.0000 | 0.661 | sttl (+0.126); ct_state_ttl (+0.053); swin (+0.046) |
| Random Forest | 27094 | 73817 | DoS | Attack | 1.0000 | 0.648 | sttl (+0.130); ct_state_ttl (+0.054); swin (+0.047) |
| Random Forest | 27521 | 124769 | Generic | Attack | 1.0000 | 0.660 | sttl (+0.128); ct_state_ttl (+0.052); swin (+0.048) |
| Random Forest | 29409 | 163470 | Generic | Attack | 1.0000 | 0.660 | sttl (+0.130); is_sm_ips_ports (+0.056); ct_state_ttl (+0.054) |
| Random Forest | 30107 | 130085 | Exploits | Attack | 1.0000 | 0.670 | sttl (+0.128); dttl (+0.074); ct_state_ttl (-0.053) |
| Random Forest | 34186 | 99234 | Exploits | Attack | 1.0000 | 0.693 | sttl (-0.128); dttl (+0.071); ct_state_ttl (-0.051) |
| XGBoost | 2941 | 134318 | Generic | Attack | 1.0000 | 0.796 | sttl (+0.725); ct_state_ttl (+0.213); service (+0.072) |
| XGBoost | 3040 | 123911 | Generic | Attack | 0.9999 | 0.802 | sttl (+0.722); ct_state_ttl (+0.210); ct_srv_dst (-0.081) |
| XGBoost | 3206 | 64810 | DoS | Attack | 0.9633 | 0.791 | sttl (+0.726); ct_state_ttl (+0.206); ct_srv_dst (+0.090) |
| XGBoost | 4407 | 92635 | Exploits | Attack | 0.9996 | 0.780 | sttl (+0.711); ct_state_ttl (+0.219); proto (+0.110) |
| XGBoost | 6978 | 160610 | Generic | Attack | 1.0000 | 0.797 | sttl (+0.714); ct_state_ttl (+0.211); service (+0.079) |
| XGBoost | 15137 | 85162 | DoS | Attack | 0.9968 | 0.795 | sttl (+0.716); ct_state_ttl (-0.210); ct_srv_dst (+0.090) |
| XGBoost | 15338 | 59743 | Exploits | Attack | 0.9999 | 0.795 | sttl (+0.721); ct_state_ttl (+0.213); proto (+0.101) |
| XGBoost | 15741 | 74771 | Fuzzers | Attack | 0.6963 | 0.798 | sttl (+0.716); ct_state_ttl (-0.215); ct_srv_dst (+0.087) |
| XGBoost | 17889 | 106578 | Exploits | Attack | 0.9867 | 0.802 | sttl (-0.722); proto (+0.289); ct_state_ttl (+0.208) |
| XGBoost | 18327 | 112882 | Exploits | Attack | 0.9770 | 0.806 | sttl (+0.725); ct_state_ttl (+0.214); ct_srv_dst (+0.082) |
| XGBoost | 22855 | 101524 | Exploits | Attack | 0.9989 | 0.794 | sttl (+0.720); ct_state_ttl (+0.217); ct_srv_dst (+0.100) |
| XGBoost | 24358 | 164473 | Generic | Attack | 0.9999 | 0.792 | sttl (+0.719); ct_state_ttl (+0.217); service (+0.070) |
| XGBoost | 25080 | 142904 | Generic | Attack | 0.9999 | 0.788 | sttl (+0.714); ct_state_ttl (+0.215); service (+0.076) |
| XGBoost | 25740 | 99900 | Fuzzers | Attack | 0.8637 | 0.776 | sttl (+0.713); ct_state_ttl (-0.217); dttl (+0.036) |
| XGBoost | 26636 | 151877 | Generic | Attack | 0.9999 | 0.809 | sttl (+0.723); ct_state_ttl (+0.207); service (+0.075) |
| XGBoost | 27094 | 73817 | DoS | Attack | 0.9985 | 0.785 | sttl (+0.716); ct_state_ttl (+0.212); proto (+0.101) |
| XGBoost | 27521 | 124769 | Generic | Attack | 0.9999 | 0.795 | sttl (+0.718); ct_state_ttl (+0.211); ct_srv_dst (-0.071) |
| XGBoost | 29409 | 163470 | Generic | Attack | 0.9999 | 0.798 | sttl (+0.715); ct_state_ttl (+0.210); ct_srv_dst (-0.080) |
| XGBoost | 30107 | 130085 | Exploits | Attack | 0.9981 | 0.801 | sttl (+0.720); ct_state_ttl (-0.206); ct_srv_dst (+0.083) |
| XGBoost | 34186 | 99234 | Exploits | Attack | 0.9986 | 0.798 | sttl (-0.716); ct_state_ttl (-0.210); ct_srv_dst (+0.095) |

## Features appearing most often in the top 5

- **Random Forest**: sttl (20), ct_state_ttl (20), swin (19), sloss (13), is_sm_ips_ports (11), state (7), dttl (5), proto (4)
- **XGBoost**: sttl (20), ct_state_ttl (20), ct_srv_dst (18), service (16), ct_srv_src (13), proto (4), is_ftp_login (2), is_sm_ips_ports (2)

## Sample ID checks

- PASS: shared file has 20 unique IDs, all found in the validation split
- PASS: explanations generated = 20 shared IDs x 2 models
- PASS: Random Forest: exactly the shared IDs (no extra, none missing)
- PASS: XGBoost: exactly the shared IDs (no extra, none missing)
- PASS: Random Forest and XGBoost use the same IDs
- PASS: lime_feature_weights_all.csv: IDs exactly match the shared file for both models
- PASS: lime_top_features.csv: IDs exactly match the shared file for both models
- PASS: lime_feature_weights_all.csv: 42 features per explanation
- PASS: figures: exactly 40 plots, one per model and shared ID

## Notes for interpreting and comparing with SHAP

- Join LIME and SHAP results on `validation_position` (and `model`).
- A LIME weight belongs to a *rule* (e.g. `62.00 < sttl <= 254.00`), i.e. the quartile bin the flow falls in, not to the raw value. Compare with SHAP on the `feature` column of `lime_feature_weights_all.csv`.
- LIME works on the 42 original features. SHAP on the pipeline sees one-hot columns for proto/service/state; sum those SHAP values per original feature before comparing.
- TTL values (`sttl`, `ct_state_ttl`) are a known UNSW-NB15 testbed artifact, so a large TTL weight reflects what the models learned, not necessarily real attack behaviour.
- Local surrogate R² is moderate (see table), so LIME is only an approximation of each model near the flow.
- Random Forest probabilities move in steps of 1/200 (200 trees).

## Errors and problems

- None. All validation checks passed for every sample.
