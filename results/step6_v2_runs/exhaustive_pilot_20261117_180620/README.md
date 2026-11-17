# Step 6 v2 exhaustive evaluation of the physical_v1 lattice (pilot)

Run role: **pilot**. A pilot covers the 10 Step 6 v2 flows and the current RF/XGBoost validation models.
It is not the final experimental design and is never pooled with final results. A smoke test covers a
subset of the flows and only checks the pipeline and runtime.

Scope: definitive only for the enumerated physical_v1 lattice (sbytes/spkts increase-only, 1% grid,
budgets A/B), not for every possible network modification. No candidate is called physically valid.

| File | One row per |
|---|---|
| `identities.csv` | model x sample x budget x lattice candidate (valid or not), with P(attack) |
| `unique_candidates.csv` | unique valid candidate (deduplicated across budgets) with physical status and changes |
| `heldout_summary.csv` | model x unique candidate: held-out LIME metrics, noise flags, search-seed objective, delta P, lime_success |
| `heldout_per_seed.csv` | model x candidate x seed |
| `noise_bands.csv` | model x sample: seed-noise bands of the original |
| `summary_by_status_group.csv` | model x sample x budget x physical status group (consistent vs unverified) |
| `confirmation_noise_bands.csv` | model x sample with a flagged candidate: band of the original on seeds 401-404 |
| `confirmation_per_seed.csv` | model x flagged candidate x confirmation seed |
| `pilot_summary.json` | model x physical status group: candidate and sample counts |
| `explanations_cache.jsonl` | every LIME explanation (resume cache) |

`lime_success` = exceeds the held-out combined test AND physical status consistent_with_checked_rules
(success criterion D8, unchanged). `confirmed_lime_success` = `lime_success` AND the same combined test
passes again on the independent confirmation seeds 401-404. Candidates with unverified packet-timing
dependencies are reported in their own group and never count as a success.
