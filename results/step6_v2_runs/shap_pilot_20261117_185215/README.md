# Step 6 v2 TreeSHAP evaluation of the physical_v1 lattice (pilot)

Same 10 flows and the same unique physical_v1 candidates as the exhaustive LIME pilot, explained with
TreeSHAP (Part B configuration) for RF and XGBoost. Pilot only: never pooled with the final experiment.
No candidate is called physically valid.

| File | One row per |
|---|---|
| `shap_summary.csv` | model x unique candidate: SHAP change measures, D11 reference and flags, P(attack) |
| `shap_values.csv` | model x original/candidate: SHAP values summed to the 42 features |
| `d11_references.csv` | model x flow: J_ref and rho_ref (SHAP vs LIME on the original, held-out seeds 22-55) |
| `summary_by_status_group.csv` | model x physical status group |
| `summary_by_flow.csv` | model x flow x physical status group |
| `comparison_with_lime.csv` | model x candidate: SHAP measures next to the LIME pilot's held-out measures |

`shap_success` (D11) = still attack AND consistent_with_checked_rules AND top-10 Jaccard <= J_ref AND
tie-aware Spearman <= rho_ref. Candidates with unverified dependencies are reported but never succeed.
