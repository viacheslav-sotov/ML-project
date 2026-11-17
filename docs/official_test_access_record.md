# Official test partition: access record

**File:** `data/raw/UNSW_NB15_training-set.csv`. The file names are swapped: this file holds the **official test** partition (82,332 rows). Its SHA-256 is `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559`, as recorded in existing manifests; it was not recomputed for this note. Last write: Sep 29, 2026.
**Recorded:** Nov 17, 2026, from a code inspection only. The file was not opened for this record.

## What reads the file, and how

Two code paths in the **v1/v2 pipeline** read it. This covers Step 6 v1/v2, the exhaustive and TreeSHAP pilots, G4 and their tests.

1. **`src/unsw/data.py: load_official_data()` / `create_data_split()`**
   - Called through `src/unsw/xai_setup.get_xai_data()`.
   - Parses **both** raw partition files with `pd.read_csv`, assigns their roles by row count, and builds `X_test` / `y_test` / `official_test_df` in memory.
   - Every Step 6 entry point (`unsw_step6_runner`, `unsw_step6_exhaustive`, `unsw_step6_shap`, `unsw_step6_g4`, `unsw_targeted_heldout`) immediately wraps that split in `guarded_split()`. The guard **removes** `X_test`, `y_test` and `official_test_df`, and raises `KeyError` if code asks for them.
   - Outside the guard definitions, no Step 6, G4, TreeSHAP, exhaustive or held-out module references these keys.
2. **`experiments/unsw_step6_manifest.py` (manifest builder)**
   - Streams both raw files byte by byte to count lines (`sum(1 for _ in path.open("rb")) - 1`) and to compute SHA-256.
   - Records `{file, rows, sha256}` per role, with `"official_test_used": false`.
   - No values are parsed.
3. Older v1 scripts (`unsw_targeted_spoofing.py`, `unsw_spoofing_explanations.py`, `verify_spoofing_handoff.py`) compute the file's **SHA-256 only**, for dataset pinning.

The G4 diagnostic pilot, the TreeSHAP and exhaustive pilots, and the fast test suite / v1 replays run after G4 all went through paths 1 and 2. That is the most likely source of the file's last-access time on Nov 17. NTFS last-access times are approximate, so the exact process can't be identified from the timestamp alone.

## What was not done with it

- **Training/fitting:** the current RF/XGBoost pipelines are fitted on `X_subtrain` only (`xai_setup.load_xai_model`). The v3 nested-CV models are fitted only on the v3 training side of the official **training** partition.
- **Tuning / model selection:** v1/v2 tuning used sub-train and validation; v3 uses grouped inner folds of the v3 training side.
- **Attack-sample selection:** flows come from `shared_validation_sample_identifiers.csv` (validation positions of the official training partition), checked by `check_sample_identity` against `X_val`.
- **Performance evaluation:** v1/v2 used the validation split; v3 uses the outer folds of the v3 training side.
- No test records, predictions or metrics appear in any result file. Manifests store only the file name, row count and hash.

## The v3 pipeline (nested CV)

- `src/unsw/v3_data.load_official_train()` opens only `UNSW_NB15_testing-set.csv` (the official **training** partition). It **refuses** the test file's name and any file without 175,341 rows.
- `experiments/unsw_nested_cv.py` and `src/unsw/v3_pipelines.py` do not import `src.unsw.data` loaders. Run headers record `official_test_used: false`.
- The timing benchmark and the nested-CV stages never open the test file.

## Consequence for future work

Any run of the v1/v2 tooling (including the old test suite, the v1 replays and the manifest builder) parses or hashes the official test file. To comply with "do not access the official test file again", **these are not run** until an approved change makes them stop loading it. For example, `create_data_split()` could load only the official training partition, and manifests could pin the test file by its already recorded hash. Changing them touches v1/v2 code (R-1) and needs approval first.
