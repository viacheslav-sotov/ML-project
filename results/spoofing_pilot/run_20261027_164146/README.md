# Part A spoofing pilot run

This directory contains the reproducibility bundle for a small candidate-generation smoke test.

Files:
- candidates.csv: accepted candidates with original and modified feature vectors encoded as JSON
- rejected_candidates.csv: rejected attempts and their rejection reasons
- attempt_summary.csv: one row per model/sample/budget with attempt counts
- feature_changes.csv: per-candidate feature-change summary
- run_metadata.json: full experiment metadata and checksums
- feature_order.json: exact 42-feature order
- environment.txt: Python and package versions
- README.md: operator guidance

How to verify the run:
1. Load `run_metadata.json` and compare the SHA-256 file hashes against the raw datasets and saved model bundles.
2. Load `feature_order.json` and confirm it matches the canonical order used by `src/unsw/constraints.py`.
3. Reconstruct each row in `candidates.csv` by decoding the JSON `original_feature_values_json` and `modified_feature_values_json` columns.
4. Validate each candidate with `validate_candidate()` before using it in Part B.
5. Recompute predictions with the existing validation pipelines and ensure each candidate remains malicious.

Note: large datasets and trained models are intentionally not committed to GitHub; they are referenced by checksum in the run metadata and must be downloaded or restored from the same project branch before full reproduction.
