# Step 6 G4: diagnostic unconstrained upper bound (diagnostic_pilot)

Protocol step6-g4-diagnostic-1 (decision D7). G4 edits up to 10 of 35 numeric features by up to 50%,
derived features included and NOT repaired. It asks only whether LIME/TreeSHAP can be moved beyond seed
noise at all. **A G4 success is never evidence of a physically realizable network attack.** Every
candidate carries physical_report_v2; consistent, unverified and rule-violating candidates are counted
separately, and none is called physically valid.

| File | One row per |
|---|---|
| `tasks/*.json` | search run (method, seed, model, flow): history, accounting, checkpoints |
| `checkpoints.csv` | run x checkpoint: best candidate (or the original if nothing improved) |
| `heldout_summary.csv` | unique checkpoint candidate: physical status, held-out LIME (D8, diagnostic), confirmation, TreeSHAP (D11, diagnostic) |
| `heldout_per_seed.csv`, `noise_bands.csv`, `confirmation_per_seed.csv` | held-out details |
| `summary_by_method.csv` | model x method x checkpoint x physical status group |
| `heldout_cache.jsonl` | held-out and confirmation LIME explanations (resume cache) |
