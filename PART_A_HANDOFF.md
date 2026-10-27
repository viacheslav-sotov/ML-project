# Part A spoofing pilot handoff

This note documents the verified Part A constrained perturbation baseline and the exact handoff steps for a teammate to continue from the same repository state.

## Scope and limits

This is Part A only:
- generate malicious candidates by perturbing numeric DIRECTLY_CONTROLLABLE features only
- keep all other feature categories fixed
- validate every candidate with the canonical validator logic
- do not retrain, overwrite, or replace saved validation pipelines
- do not claim successful explanation spoofing; this is a random perturbation baseline

## Canonical sources of truth

- src/unsw/constraints.py: feature metadata, feature order, and validator logic
- src/unsw/xai_setup.py: shared validation sample manifest loader and model loader contract
- src/unsw/data.py: feature definitions and data contract
- results/metrics/shap/shared_validation_sample_identifiers.csv: canonical shared validation sample manifest

## Verified final run

The verified final run directory is:

results/spoofing_pilot/run_20261027_164146

It contains the full reproducibility bundle for the completed Part A pilot:
- candidates.csv
- rejected_candidates.csv
- attempt_summary.csv
- feature_changes.csv
- run_metadata.json
- feature_order.json
- environment.txt
- README.md
- summary.json

The exact output is the canonical artifact for the 12-group full pilot and should be treated as the reference state when continuing to Part B.

## Exact Git branch and commit instructions

Use the current repo branch and keep the work isolated from unrelated changes.

```bash
git checkout main
git pull --ff-only origin main
git status
```

When ready to save the handoff work:

```bash
git add experiments/unsw_spoofing_candidates.py experiments/verify_spoofing_handoff.py tests/test_unsw_spoofing_candidates.py PART_A_HANDOFF.md results/spoofing_pilot/run_20261027_164146
git commit -m "Add Part A spoofing candidate generator and pilot handoff"
```

If the repo policy rejects the run bundle, commit only the scripts and notes and keep the timestamped run directory untracked locally until a reviewer approves it.

Do not push automatically. Review the diff first.

```bash
git diff --cached
git push origin main
```

## Required repo files and local data model files

The following files must be present in the repository before reproducing or extending Part A:

- experiments/unsw_spoofing_candidates.py
- experiments/verify_spoofing_handoff.py
- tests/test_unsw_spoofing_candidates.py
- src/unsw/constraints.py
- src/unsw/data.py
- src/unsw/xai_setup.py
- results/metrics/shap/shared_validation_sample_identifiers.csv
- results/models/random_forest_validation_pipeline.joblib
- results/models/xgboost_validation_pipeline.joblib

The following datasets are required by the canonical pipeline and must be available locally for a full reproduction:
- data/raw/UNSW_NB15_training-set.csv
- data/raw/UNSW_NB15_testing-set.csv

These files are intentionally not committed because they are large and are already excluded via .gitignore. The repo is designed to keep the code, manifest, and reproducibility metadata in Git while leaving large binary artifacts outside version control.

## SHA-256 verification procedure

From the repo root, validate the run bundle like this:

```bash
python experiments/verify_spoofing_handoff.py results/spoofing_pilot/run_20261027_164146
```

The verifier checks:
- dataset SHA-256 values against the canonical raw files
- shared validation manifest SHA-256
- saved model bundle SHA-256 values
- feature order equality with the project constraints
- JSON reconstruction for original and modified feature vectors
- constraint validity with the canonical validator
- malicious prediction preservation for every accepted candidate

The verification script is the authoritative post-run check. It must PASS before any Part B work begins.

## How to load and inspect the 600 accepted candidates

Use pandas directly from the repo root:

```python
import pandas as pd
import json

run_dir = "results/spoofing_pilot/run_20261027_164146"
candidates = pd.read_csv(f"{run_dir}/candidates.csv")
print(len(candidates))
print(candidates.columns.tolist())
print(candidates.head())
```

Check uniqueness and row count:

```python
assert len(candidates) == 600
assert candidates["candidate_id"].is_unique
```

The accepted candidates are the final Part A output set and should be treated as immutable for Part B unless a new run is intentionally created.

## How to reconstruct original and modified samples

Each row stores the full original and modified feature vectors as JSON strings with the canonical 42-feature order. Reconstruct them as follows:

```python
import json
import pandas as pd

run_dir = "results/spoofing_pilot/run_20261027_164146"
candidates = pd.read_csv(f"{run_dir}/candidates.csv")
row = candidates.iloc[0]
original = json.loads(row["original_feature_values_json"])
modified = json.loads(row["modified_feature_values_json"])

print(type(original), len(original))
print(type(modified), len(modified))
print(original)
print(modified)
```

This preserves the exact 42-feature vector for each original sample and each modified sample, including explicit JSON serialization of every value used during validation.

## How to evaluate candidates with SHAP and LIME

Do not modify Part A artifacts. Save any Part B analysis in a separate output directory.

### SHAP

Use the same validation pipelines and the accepted candidate rows, then write outputs to a separate folder such as:

```bash
mkdir -p results/part_b/shap
python experiments/unsw_shap_model_comparison.py
```

Or adapt a dedicated Part B script to load the candidate CSV and evaluate SHAP explanations against the same validation sample IDs. The critical rule is to never overwrite the Part A run directory or the original model files.

### LIME

Use the same dataset split and the accepted candidate rows as inputs to a separate Part B LIME analysis. Prefer writing to a new directory such as:

```bash
mkdir -p results/part_b/lime
```

Then evaluate the model on the reconstructed original/modified samples without modifying the Part A artifacts. Log every run to a separate metadata file if needed.

## How to save Part B results without modifying Part A

Use a separate directory naming scheme to avoid touching the verified run bundle.

Example:

```bash
mkdir -p results/part_b/run_20261027_164146
```

Part B outputs should be stored under:
- results/part_b/...
- results/metrics/part_b/...
- results/figures/part_b/...

Never edit or overwrite the Part A directory at:

results/spoofing_pilot/run_20261027_164146

## Exact verification command for the full run

From the repository root:

```bash
python experiments/verify_spoofing_handoff.py results/spoofing_pilot/run_20261027_164146
```

This is the minimal command to validate the full Part A run and confirm the run metadata, feature ordering, candidate reconstruction, constraints, and model predictions all remain consistent with the repo state.

## Git ignore policy and safe commit set

The relevant .gitignore entries are intended to exclude large or environment-specific files, not the code or handoff artifacts:
- .venv/
- __pycache__/
- *.pyc
- data/raw/*
- data/processed/*
- *.joblib
- model bundle files under results/models

This means the following are safe to commit:
- experiments/*.py
- tests/*.py
- PART_A_HANDOFF.md
- the small timestamped run bundle which is intended for reproducibility and review

The following are intentionally excluded from Git:
- local virtual environment
- raw UNSW datasets
- processed datasets
- trained model bundles
- any large temporary outputs not explicitly approved for review

No .gitignore change is required for the current handoff because the repository is already excluding the correct large artifacts, and the required small files are not being ignored.

## Summary

Part A is complete and validated:
- 12 experiment groups completed
- 1071 attempts executed
- 600 accepted candidates generated
- 471 rejected attempts recorded
- 12 regression tests passed
- the final handoff bundle is reproducible and verified

This should be treated as a validated baseline for Part B, not as a claim of successful explanation spoofing.
