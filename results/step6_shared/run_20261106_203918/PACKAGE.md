# Step 6 results package: run_20261106_203918

Analysis-ready copy of a finished Step 6 run. Files are byte-for-byte copies of the source run;
checksums in `SHA256SUMS` are computed on LF-normalized content (platform-independent).

## Provenance

| Item | Value |
|---|---|
| Source run | `results/step6_minimum/run_20261106_203918` |
| Held-out run | `results/step6_minimum/run_20261106_203918/heldout/run_20261106_205608` |
| Code commit (from manifest) | `5e3318b29ecebcdf8b7e8549d1a30e830bdd416b` (dirty: False) |
| Protocol | step6-v1 |
| Manifest SHA-256 (recorded) | `0d7b16e85be971af699a675e4f48deb0b194760cd5abbe249d1b034c129ba78c` |
| Manifest hash verification | lf_normalized (runner v1 wrote CRLF on Windows and hashed the LF text; content identical) |
| Model SHA-256 | random_forest: `4e49f17d9b19747af9e4da472866c4385062d8018582bd7b4872983aefd56531`, xgboost: `83a34c617a41b886b628ed01257994360f36e8c6863edbc439ee4b201ac4f4b1` |
| Official test set used | False |
| Tasks / checkpoint candidates | 123 / 186 (all complete, all checkpoints valid) |
| Held-out search-score check | {'checked': 186, 'max_gap': 9.454242944073599e-17, 'mismatches': 0} |
| Selection | `{"budgets": ["A"], "control_seeds": [201, 202, 203, 204, 205, 206, 207, 208, 209, 210], "explanation_budgets": [25, 50], "methods": ["canonical", "shuffled", "multistart", "random"], "models": ["random_forest"], "order_seeds": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110], "samples": [2941, 3206, 4407]}` |
| Packaged at (UTC) | 2026-11-06T21:08:23.344073+00:00 |

## Files

| File | Bytes |
|---|---|
| `results.csv` | 38,690 |
| `checkpoints.csv` | 264,943 |
| `best_so_far.csv` | 894,467 |
| `restarts.csv` | 119,696 |
| `manifest.json` | 8,670 |
| `run_config.json` | 930 |
| `run_metadata.json` | 49,642 |
| `status.json` | 187 |
| `README.md` | 1,789 |
| `heldout/heldout_summary.csv` | 162,396 |
| `heldout/heldout_per_seed.csv` | 119,923 |
| `heldout/noise_bands.csv` | 1,971 |
| `heldout/run_metadata.json` | 3,607 |

`README.md` (from the run) describes the columns. Comparisons at budget B: rows of
`checkpoints.csv` with `checkpoint == B`. Join held-out results on `task_id` + `checkpoint`.

## Not included

`tasks/` (246 files, 4,383,822 bytes): per-task JSON
payloads and complete candidate histories. They remain in the local source run; all packaged CSVs are
consolidated from them. Ask for them if a candidate's full search history is needed.

## Verify

`python experiments/unsw_step6_package.py --verify <this directory>`
