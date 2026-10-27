"""Random-search baseline for explanation-spoofing candidates on UNSW-NB15.

This script is intentionally limited to candidate generation and validation. It does not
retrain the model, modify any saved model bundle, or touch the SHAP/LIME result files.

Important design choice for the first pilot:
- The project's budget is expressed as a relative percentage change on numeric features.
- Categorical changes (`proto`, `service`, `state`) are not assigned a consistent numeric
  percentage budget in the existing validator or pilot design, so the first pilot excludes
  categorical edits completely. These features remain fixed unless the project later adds a
  separate categorical-change policy.
- We therefore generate candidates by perturbing numeric features classified as
  DIRECTLY_CONTROLLABLE in `src/unsw/constraints.py`, while leaving all other categories
  unchanged unless the validator explicitly supports a consistent recomputation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.unsw.constraints import (
    FEATURE_METADATA,
    FEATURE_ORDER,
    NUMERIC_MIN_MAX,
    validate_candidate,
)
from src.unsw.data import CATEGORICAL_FEATURES
from src.unsw.xai_setup import MODEL_DIR, get_xai_data, load_shared_samples, load_xai_model

DEFAULT_MODELS = ["random_forest", "xgboost"]
DEFAULT_SAMPLES = [2941, 3206, 4407]
DEFAULT_BUDGETS = ["A", "B"]
DEFAULT_NUM_CANDIDATES = 20
DEFAULT_SEED = 42

DIRECTLY_CONTROLLABLE_FEATURES = [
    row["feature"]
    for row in FEATURE_METADATA
    if row["constraint_category"] == "DIRECTLY_CONTROLLABLE"
]
NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES = [
    feature
    for feature in DIRECTLY_CONTROLLABLE_FEATURES
    if feature not in CATEGORICAL_FEATURES and feature in NUMERIC_MIN_MAX
]

PACKAGE_NAMES = [
    "numpy",
    "pandas",
    "scikit-learn",
    "xgboost",
    "lime",
    "joblib",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_snapshot() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return {
        "python_version": sys.version.replace("\n", " "),
        "platform": sys.platform,
        "package_versions": versions,
    }


def git_commit_hash() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(ROOT), check=False)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate constrained spoofing candidates for the UNSW-NB15 pilot.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Model keys to evaluate, e.g. random_forest xgboost")
    parser.add_argument("--samples", nargs="+", type=int, default=DEFAULT_SAMPLES, help="Validation positions to evaluate")
    parser.add_argument("--budgets", nargs="+", choices=["A", "B"], default=DEFAULT_BUDGETS, help="Perturbation budgets: A or B")
    parser.add_argument("--num-candidates", type=int, default=DEFAULT_NUM_CANDIDATES, help="Number of random candidates to attempt per model/sample/budget")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "results" / "spoofing_pilot"),
        help="Directory to write candidate CSV outputs. A run-specific timestamped subdirectory is created if the target already exists.",
    )
    return parser.parse_args()


def make_output_directory(base_dir: str) -> Path:
    base = Path(base_dir)
    base = base if base.is_absolute() else ROOT / base
    # Create a time-stamped subdirectory so repeated runs do not overwrite prior pilot outputs.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _relative_change(original: float, candidate: float) -> float:
    return abs(float(candidate) - float(original)) / max(1.0, abs(float(original)))


def _integer_feature_value(value: float, feature: str) -> float:
    if feature in {"dur", "rate", "sload", "dload", "sinpkt", "dinpkt", "sjit", "djit"}:
        return float(value)
    return float(round(value))


def _clamp_numeric_feature(feature: str, value: float) -> float:
    min_value, max_value = NUMERIC_MIN_MAX[feature]
    return max(min_value, min(max_value, float(value)))


def _candidate_for_budget(original_row: pd.Series, budget: str, rng: np.random.Generator) -> pd.Series:
    """Construct one random candidate under the chosen budget.

    The first pilot intentionally excludes categorical edits because the budget is defined as a
    numeric relative-change percentage and the validator does not define a consistent categorical
    semantics for those changes.
    """
    if budget == "A":
        max_features = 5
        max_relative = 0.10
    elif budget == "B":
        max_features = 10
        max_relative = 0.25
    else:
        raise ValueError(f"Unsupported budget {budget!r}")

    candidate = original_row.copy()
    direct_numeric_features = list(NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES)
    if not direct_numeric_features:
        raise ValueError("No numeric direct-controllable features available for perturbation.")

    feature_count = int(rng.integers(1, max_features + 1))
    chosen_features = rng.choice(direct_numeric_features, size=feature_count, replace=False).tolist()

    total_relative = 0.0
    for feature in chosen_features:
        original_value = float(original_row[feature])
        if feature in {"dur", "rate", "sload", "dload", "sinpkt", "dinpkt", "sjit", "djit"}:
            scale = 1.0 if abs(original_value) == 0 else abs(original_value)
        else:
            scale = max(1.0, abs(original_value))
        sign = rng.choice([-1.0, 1.0])
        delta_ratio = rng.uniform(0.0, max_relative)
        candidate_value = original_value * (1.0 + sign * delta_ratio)
        candidate_value = _integer_feature_value(candidate_value, feature)
        candidate_value = _clamp_numeric_feature(feature, candidate_value)
        if feature in {"dur", "rate", "sload", "dload", "sinpkt", "dinpkt", "sjit", "djit"}:
            candidate[feature] = float(candidate_value)
        else:
            candidate[feature] = float(candidate_value)
        total_relative += _relative_change(original_value, candidate_value)

    # Hard cap on the aggregate relative-change budget.
    if budget == "B" and total_relative > 1.5:
        return pd.Series(dtype=float)

    # Reject any candidate that exceeds the per-feature cap after clamping.
    for feature in chosen_features:
        if _relative_change(float(original_row[feature]), float(candidate[feature])) > max_relative + 1e-12:
            return pd.Series(dtype=float)

    return candidate


def _is_malicious_prediction(model: Any, row: pd.DataFrame) -> tuple[bool, float]:
    try:
        prob = float(model.predict_proba(row)[0, 1])
    except Exception as exc:  # pragma: no cover - the script surfaces failures as rejected candidates.
        raise ValueError(f"Model prediction failed: {exc}") from exc
    return prob >= 0.5, prob


def _candidate_is_valid(original_row: pd.Series, candidate_row: pd.Series) -> dict[str, Any]:
    original_dict = original_row.to_dict()
    candidate_dict = candidate_row.to_dict()
    result = validate_candidate(original_dict, candidate_dict)
    return result


def _serialize_values(values: dict[str, Any]) -> str:
    return json.dumps(values, default=str)


def _build_rejected_record(model_key: str, sample_position: int, budget_key: str, seed: int, reason: str, original_row: pd.Series, candidate_row: pd.Series | None = None) -> dict[str, Any]:
    return {
        "candidate_id": f"{model_key}_{budget_key}_sample_{sample_position}_rejected_{seed}",
        "seed": int(seed),
        "model_name": model_key,
        "perturbation_budget": budget_key,
        "validation_position": int(sample_position),
        "original_feature_values_json": _serialize_values(original_row.to_dict()),
        "modified_feature_values_json": _serialize_values(candidate_row.to_dict()) if candidate_row is not None else "{}",
        "reason": reason,
        "candidate_status": "rejected",
    }


def load_strict_model(model_key: str, split: dict[str, Any]):
    """Load the saved validation pipeline without triggering a fallback refit.

    This script is intentionally strict: if the repository model artifact is missing, it raises
    an error instead of retraining. That preserves the existing model-training behavior elsewhere
    in the codebase while making the candidate-generator contract explicit.
    """
    model_path = MODEL_DIR / f"{model_key}_validation_pipeline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Required model bundle not found for {model_key!r}: {model_path}. "
            "Refusing to retrain from the candidate-generator script."
        )
    return load_xai_model(model_key, split)


def _generate_candidates_core(
    model_key: str,
    sample_position: int,
    budget_key: str,
    num_candidates: int,
    seed: int,
    split: dict[str, Any],
    shared_samples: pd.DataFrame | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    X_val = split["X_val"]
    y_val = split["y_val"]
    model = load_strict_model(model_key, split)

    if not 0 <= int(sample_position) < len(X_val):
        raise ValueError(f"Sample position {sample_position} is outside the validation split size {len(X_val)}")

    original_row = X_val.iloc[[int(sample_position)]].iloc[0].copy()
    if int(y_val.iloc[int(sample_position)]) != 1:
        raise ValueError(f"Sample position {sample_position} has label {int(y_val.iloc[int(sample_position)])}, expected 1")

    is_malicious, original_prob = _is_malicious_prediction(model, X_val.iloc[[int(sample_position)]])
    if not is_malicious:
        raise ValueError(f"Sample position {sample_position} is not predicted as Attack by {model_key} (P(Attack)={original_prob})")

    deterministic_offset = sum(ord(ch) for ch in model_key) % 1000
    rng = np.random.default_rng(seed + int(sample_position) + deterministic_offset)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(200, num_candidates * 200)

    while len(accepted) < num_candidates and attempts < max_attempts:
        attempts += 1
        candidate_row = _candidate_for_budget(original_row, budget_key, rng)
        if candidate_row.empty:
            rejected.append(_build_rejected_record(model_key, sample_position, budget_key, seed, "empty candidate", original_row))
            continue

        if candidate_row.equals(original_row):
            rejected.append(_build_rejected_record(model_key, sample_position, budget_key, seed, "candidate equal to original", original_row, candidate_row))
            continue

        changed_features = [feature for feature in NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES if candidate_row[feature] != original_row[feature]]
        if len(changed_features) > (5 if budget_key == "A" else 10):
            rejected.append(_build_rejected_record(model_key, sample_position, budget_key, seed, f"changed feature count exceeds budget {budget_key}", original_row, candidate_row))
            continue
        per_feature_relative = [
            _relative_change(float(original_row[feature]), float(candidate_row[feature]))
            for feature in changed_features
        ]
        if any(value > (0.10 if budget_key == "A" else 0.25) + 1e-12 for value in per_feature_relative):
            rejected.append(_build_rejected_record(model_key, sample_position, budget_key, seed, "relative change exceeds per-feature cap", original_row, candidate_row))
            continue
        if budget_key == "B" and sum(per_feature_relative) > 1.5 + 1e-12:
            rejected.append(_build_rejected_record(model_key, sample_position, budget_key, seed, "aggregate relative change exceeds B budget", original_row, candidate_row))
            continue

        validation_result = _candidate_is_valid(original_row, candidate_row)
        if not validation_result["valid"]:
            rejected.append(_build_rejected_record(model_key, sample_position, budget_key, seed, "; ".join(validation_result["violated_constraints"]), original_row, candidate_row))
            continue

        try:
            candidate_df = candidate_row.to_frame().T
            candidate_prob = float(model.predict_proba(candidate_df)[0, 1])
        except Exception:
            rejected.append(_build_rejected_record(model_key, sample_position, budget_key, seed, "model prediction failed", original_row, candidate_row))
            continue

        if candidate_prob < 0.5:
            rejected.append(_build_rejected_record(model_key, sample_position, budget_key, seed, "modified prediction is benign", original_row, candidate_row))
            continue

        sample_meta = None
        if shared_samples is not None:
            sample_meta = shared_samples[shared_samples["validation_position"] == int(sample_position)]
            if not sample_meta.empty:
                sample_meta = sample_meta.iloc[0]

        candidate_record = {
            "candidate_id": f"{model_key}_{budget_key}_sample_{sample_position}_{len(accepted) + 1:03d}",
            "seed": int(seed),
            "model_name": model_key,
            "perturbation_budget": budget_key,
            "validation_position": int(sample_position),
            "original_training_row_index": int(sample_meta["original_training_row_index"]) if sample_meta is not None and "original_training_row_index" in sample_meta else int(original_row.name) if isinstance(original_row.name, (int, np.integer)) else None,
            "official_id": int(sample_meta["official_id"]) if sample_meta is not None and "official_id" in sample_meta else None,
            "attack_cat": str(sample_meta["attack_cat"]) if sample_meta is not None and "attack_cat" in sample_meta else "unknown",
            "original_feature_values_json": _serialize_values(original_row.to_dict()),
            "modified_feature_values_json": _serialize_values(candidate_row.to_dict()),
            "changed_features_json": json.dumps(changed_features, sort_keys=True),
            "original_attack_probability": float(original_prob),
            "modified_attack_probability": float(candidate_prob),
            "constraint_valid": bool(validation_result["valid"]),
            "violated_constraints_json": json.dumps(validation_result["violated_constraints"], sort_keys=True),
            "candidate_status": "accepted",
            "feature_edit_count": len(changed_features),
            "max_relative_change_per_feature": float(max(per_feature_relative)) if per_feature_relative else 0.0,
            "aggregate_relative_change": float(sum(per_feature_relative)) if per_feature_relative else 0.0,
        }
        accepted.append(candidate_record)

    summary = {
        "model_name": model_key,
        "validation_position": int(sample_position),
        "budget": budget_key,
        "seed": int(seed),
        "attempts": int(attempts),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "original_probability": float(original_prob),
    }
    return accepted, rejected, summary


def generate_candidates(
    model_key: str,
    sample_position: int,
    budget_key: str,
    num_candidates: int,
    seed: int,
    split: dict[str, Any],
    shared_samples: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    accepted, _, _ = _generate_candidates_core(
        model_key=model_key,
        sample_position=sample_position,
        budget_key=budget_key,
        num_candidates=num_candidates,
        seed=seed,
        split=split,
        shared_samples=shared_samples,
    )
    return accepted


def main() -> None:
    args = parse_args()
    start_time = datetime.now(timezone.utc)
    output_root = make_output_directory(args.output_dir)
    split = get_xai_data()
    shared_samples = load_shared_samples(split)
    available_positions = set(int(v) for v in shared_samples["validation_position"])

    selected_samples = [int(v) for v in args.samples if int(v) in available_positions]
    missing = [int(v) for v in args.samples if int(v) not in available_positions]
    if missing:
        raise ValueError(f"Missing shared validation positions: {missing}")

    all_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []

    for model_key in args.models:
        for sample_position in selected_samples:
            for budget_key in args.budgets:
                accepted, rejected, summary = _generate_candidates_core(
                    model_key=model_key,
                    sample_position=sample_position,
                    budget_key=budget_key,
                    num_candidates=args.num_candidates,
                    seed=args.seed,
                    split=split,
                    shared_samples=shared_samples,
                )
                all_records.extend(accepted)
                rejected_records.extend(rejected)
                attempt_rows.append(summary)
                for record in accepted:
                    feature_rows.append(
                        {
                            "candidate_id": record["candidate_id"],
                            "model_name": record["model_name"],
                            "validation_position": record["validation_position"],
                            "perturbation_budget": record["perturbation_budget"],
                            "changed_features_json": record["changed_features_json"],
                            "feature_edit_count": record["feature_edit_count"],
                            "max_relative_change_per_feature": record["max_relative_change_per_feature"],
                            "aggregate_relative_change": record["aggregate_relative_change"],
                            "original_attack_probability": record["original_attack_probability"],
                            "modified_attack_probability": record["modified_attack_probability"],
                        }
                    )

    output_csv = output_root / "candidates.csv"
    rejected_csv = output_root / "rejected_candidates.csv"
    attempts_csv = output_root / "attempt_summary.csv"
    feature_changes_csv = output_root / "feature_changes.csv"
    feature_order_json = output_root / "feature_order.json"
    env_txt = output_root / "environment.txt"
    readme_md = output_root / "README.md"

    pd.DataFrame(all_records).to_csv(output_csv, index=False)
    pd.DataFrame(rejected_records).to_csv(rejected_csv, index=False)
    pd.DataFrame(attempt_rows).to_csv(attempts_csv, index=False)
    pd.DataFrame(feature_rows).to_csv(feature_changes_csv, index=False)
    feature_order_json.write_text(json.dumps(FEATURE_ORDER, indent=2), encoding="utf-8")

    dataset_train = ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"
    dataset_test = ROOT / "data" / "raw" / "UNSW_NB15_testing-set.csv"
    model_paths = {
        "random_forest": ROOT / "results" / "models" / "random_forest_validation_pipeline.joblib",
        "xgboost": ROOT / "results" / "models" / "xgboost_validation_pipeline.joblib",
    }
    manifest_path = ROOT / "results" / "metrics" / "shap" / "shared_validation_sample_identifiers.csv"
    metadata = {
        "run_id": output_root.name,
        "git_commit_hash": git_commit_hash(),
        "python_version": sys.version.replace("\n", " "),
        "package_versions": environment_snapshot()["package_versions"],
        "random_seed": args.seed,
        "dataset_sha256": {
            "training": file_sha256(dataset_train),
            "test": file_sha256(dataset_test),
        },
        "shared_validation_manifest_sha256": file_sha256(manifest_path),
        "model_sha256": {name: file_sha256(path) for name, path in model_paths.items()},
        "validation_split_configuration": {
            "method": "train_test_split",
            "random_state": 42,
            "test_size": 0.20,
            "stratified": True,
            "training_rows": int(len(get_xai_data()["X_train_full"])),
            "validation_rows": int(len(get_xai_data()["X_val"])),
        },
        "validation_positions": {
            str(pos): int(shared_samples[shared_samples["validation_position"] == pos].iloc[0]["official_id"]) for pos in selected_samples
        },
        "experiment_budgets": args.budgets,
        "attempts": int(sum(row["attempts"] for row in attempt_rows)),
        "accepted": int(len(all_records)),
        "rejected": int(len(rejected_records)),
        "runtime_start_utc": start_time.isoformat(),
        "runtime_end_utc": datetime.now(timezone.utc).isoformat(),
        "categorical_change_rule": "Excluded from the first pilot; percentage budgets are applied only to numeric DIRECTLY_CONTROLLABLE features.",
    }
    metadata_path = output_root / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    env_lines = [f"Python: {metadata['python_version']}", "", "Package versions:"]
    for name, version in metadata["package_versions"].items():
        env_lines.append(f"{name}: {version}")
    env_txt.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    readme_md.write_text(
        """# Part A spoofing pilot run

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
""",
        encoding="utf-8",
    )

    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models": args.models,
        "samples": selected_samples,
        "budgets": args.budgets,
        "num_candidates_requested": args.num_candidates,
        "seed": args.seed,
        "accepted_candidates": len(all_records),
        "output_csv": str(output_csv),
        "categorical_change_rule": "Excluded from the first pilot; percentage budgets are applied only to numeric DIRECTLY_CONTROLLABLE features.",
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Saved {len(all_records)} accepted candidates to {output_csv}")
    print(f"Summary: {json.dumps(summary, sort_keys=True)}")


if __name__ == "__main__":
    main()
