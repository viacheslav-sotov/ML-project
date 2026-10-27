"""Independent verification for the Part A spoofing pilot outputs.

This script is intentionally conservative and only verifies the reproducibility bundle,
not the model training pipeline itself. It loads the saved run metadata, recomputes
SHA-256 checksums for the canonical artifacts, confirms feature ordering, checks the
validation sample manifest, and re-validates each accepted candidate against the
project-level constraints and prediction logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.unsw.constraints import FEATURE_ORDER, FEATURE_METADATA, NUMERIC_MIN_MAX, validate_candidate
from src.unsw.xai_setup import MODEL_DIR, load_shared_samples, get_xai_data

REQUIRED_FILES = [
    "candidates.csv",
    "rejected_candidates.csv",
    "attempt_summary.csv",
    "feature_changes.csv",
    "run_metadata.json",
    "feature_order.json",
    "environment.txt",
    "README.md",
]

DIRECTLY_CONTROLLABLE_FEATURES = [
    row["feature"] for row in FEATURE_METADATA if row["constraint_category"] == "DIRECTLY_CONTROLLABLE"
]
NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES = [
    feature for feature in DIRECTLY_CONTROLLABLE_FEATURES if feature in NUMERIC_MIN_MAX
]


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _relative_change(original: float, candidate: float) -> float:
    return abs(float(candidate) - float(original)) / max(1.0, abs(float(original)))


def _validate_candidate_budget(row: dict[str, Any], budget: str) -> None:
    original = json.loads(row["original_feature_values_json"])
    modified = json.loads(row["modified_feature_values_json"])
    changed_features = [feature for feature in NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES if original.get(feature) != modified.get(feature)]
    if not changed_features:
        raise AssertionError(f"{row['candidate_id']}: no feature changes detected")
    if budget == "A":
        if len(changed_features) > 5:
            raise AssertionError(f"{row['candidate_id']}: exceeds A budget feature count")
        max_relative = 0.10
    elif budget == "B":
        if len(changed_features) > 10:
            raise AssertionError(f"{row['candidate_id']}: exceeds B budget feature count")
        max_relative = 0.25
    else:
        raise AssertionError(f"Unknown budget {budget!r}")

    per_feature_relative = [
        _relative_change(float(original[feature]), float(modified[feature]))
        for feature in changed_features
    ]
    if any(value > max_relative + 1e-12 for value in per_feature_relative):
        raise AssertionError(f"{row['candidate_id']}: exceeds per-feature relative limit")
    if budget == "B" and sum(per_feature_relative) > 1.5 + 1e-12:
        raise AssertionError(f"{row['candidate_id']}: exceeds aggregate B budget")
    validation = validate_candidate(original, modified)
    if not validation["valid"]:
        raise AssertionError(f"{row['candidate_id']}: validator rejected candidate: {validation['violated_constraints']}")


def verify_run(run_dir: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing run artifacts: {missing}")

    metadata = load_json(run_dir / "run_metadata.json")
    feature_order = json.loads((run_dir / "feature_order.json").read_text(encoding="utf-8"))
    if feature_order != FEATURE_ORDER:
        raise AssertionError("feature_order.json does not match src.unsw.constraints.FEATURE_ORDER")

    dataset_dir = ROOT / "data" / "raw"
    train_dataset = dataset_dir / "UNSW_NB15_training-set.csv"
    test_dataset = dataset_dir / "UNSW_NB15_testing-set.csv"
    if not train_dataset.exists() or not test_dataset.exists():
        raise FileNotFoundError("Canonical raw dataset files are missing from data/raw")
    expected_hashes = {
        "training": sha256(train_dataset),
        "test": sha256(test_dataset),
    }
    if metadata["dataset_sha256"] != expected_hashes:
        raise AssertionError(f"Dataset hashes do not match metadata: {metadata['dataset_sha256']} != {expected_hashes}")

    manifest_path = ROOT / "results" / "metrics" / "shap" / "shared_validation_sample_identifiers.csv"
    expected_manifest_hash = sha256(manifest_path)
    if metadata["shared_validation_manifest_sha256"] != expected_manifest_hash:
        raise AssertionError("Validation manifest SHA-256 mismatches repository state")

    for model_name, model_path in {
        "random_forest": ROOT / "results" / "models" / "random_forest_validation_pipeline.joblib",
        "xgboost": ROOT / "results" / "models" / "xgboost_validation_pipeline.joblib",
    }.items():
        if not model_path.exists():
            raise FileNotFoundError(f"Required model bundle is missing: {model_path}")
        actual_hash = sha256(model_path)
        if metadata["model_sha256"].get(model_name) != actual_hash:
            raise AssertionError(f"Model hash mismatch for {model_name}: expected {metadata['model_sha256'].get(model_name)}, got {actual_hash}")

    df = pd.read_csv(run_dir / "candidates.csv")
    if df.empty:
        raise AssertionError("Accepted candidate CSV is empty")
    if df["candidate_id"].duplicated().any():
        raise AssertionError("Duplicate candidate IDs were detected in the accepted set")

    for _, row in df.iterrows():
        original = json.loads(row["original_feature_values_json"])
        modified = json.loads(row["modified_feature_values_json"])
        if list(original.keys()) != feature_order:
            raise AssertionError(f"{row['candidate_id']}: original feature order does not match FEATURE_ORDER")
        if list(modified.keys()) != feature_order:
            raise AssertionError(f"{row['candidate_id']}: modified feature order does not match FEATURE_ORDER")
        _validate_candidate_budget(row, row["perturbation_budget"])

    rejected_df = read_optional_csv(run_dir / "rejected_candidates.csv")
    attempt_df = read_optional_csv(run_dir / "attempt_summary.csv")
    feature_df = read_optional_csv(run_dir / "feature_changes.csv")

    if len(df) != int(metadata["accepted"]):
        raise AssertionError(f"Accepted candidate count mismatch: metadata {metadata['accepted']} but file has {len(df)}")
    if len(rejected_df) != int(metadata["rejected"]):
        raise AssertionError(f"Rejected candidate count mismatch: metadata {metadata['rejected']} but file has {len(rejected_df)}")
    if sum(int(v) for v in attempt_df["attempts"]) != int(metadata["attempts"]):
        raise AssertionError("Aggregate attempts count mismatch with run metadata")
    if len(feature_df) != len(df):
        raise AssertionError("feature_changes.csv row count does not match the accepted candidate set")

    split = get_xai_data()
    manifest = pd.read_csv(manifest_path)
    manifest_positions = list(map(int, manifest["validation_position"].tolist()))
    expected_positions = {int(value) for value in metadata["validation_positions"].keys()}
    if not expected_positions.issubset(set(manifest_positions)):
        raise AssertionError("Requested validation positions are not in the canonical shared manifest")

    for _, row in df.iterrows():
        original = json.loads(row["original_feature_values_json"])
        modified = json.loads(row["modified_feature_values_json"])
        model_key = row["model_name"]
        model = __import__("src.unsw.xai_setup", fromlist=["load_xai_model"]).load_xai_model(model_key, split)
        original_df = pd.DataFrame([original], columns=FEATURE_ORDER)
        modified_df = pd.DataFrame([modified], columns=FEATURE_ORDER)
        original_prob = float(model.predict_proba(original_df)[0, 1])
        modified_prob = float(model.predict_proba(modified_df)[0, 1])
        if original_prob < 0.5:
            raise AssertionError(f"{row['candidate_id']}: original sample is not classified as malicious")
        if modified_prob < 0.5:
            raise AssertionError(f"{row['candidate_id']}: modified sample is not malicious after perturbation")
        if abs(float(row["original_attack_probability"]) - original_prob) > 1e-9:
            raise AssertionError(f"{row['candidate_id']}: original probability mismatch")
        if abs(float(row["modified_attack_probability"]) - modified_prob) > 1e-9:
            raise AssertionError(f"{row['candidate_id']}: modified probability mismatch")

    print(f"VERIFIED: {len(df)} accepted candidates, {len(rejected_df)} rejected attempts, metadata hash matches repository state.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the reproducibility bundle for the Part A spoofing pilot.")
    parser.add_argument("run_dir", type=str, nargs="?", default=str(ROOT / "results" / "spoofing_pilot"), help="Run directory containing the output bundle")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    verify_run(run_dir)


if __name__ == "__main__":
    main()
