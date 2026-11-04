from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.unsw_lime import (
    CategoricalCodec,
    EXPLAINED_LABEL,
    NUM_FEATURES,
    NUM_SAMPLES,
    build_explainer,
    make_predict_fn,
    reseed_explainer,
)
from src.unsw.constraints import FEATURE_METADATA, FEATURE_ORDER, NUMERIC_MIN_MAX, validate_candidate
from src.unsw.data import CATEGORICAL_FEATURES
from src.unsw.xai_setup import MODEL_DIR, get_xai_data, load_shared_samples, load_xai_model

TARGET_SAMPLE_POSITIONS = [2941, 3206, 4407]
DEFAULT_MODELS = ["random_forest", "xgboost"]
DEFAULT_BUDGETS = ["A", "B"]
DEFAULT_SEED = 11
OUTPUT_ROOT = ROOT / "results" / "spoofing_targeted"

DIRECTLY_CONTROLLABLE_FEATURES = [
    row["feature"] for row in FEATURE_METADATA if row["constraint_category"] == "DIRECTLY_CONTROLLABLE"
]
NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES = [
    feature for feature in DIRECTLY_CONTROLLABLE_FEATURES if feature not in CATEGORICAL_FEATURES and feature in NUMERIC_MIN_MAX
]

BUDGET_POLICY = {
    "A": {"max_changed": 5, "max_relative": 0.10, "aggregate_cap": float("inf")},
    "B": {"max_changed": 10, "max_relative": 0.25, "aggregate_cap": 1.5},
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_change(original: float, candidate: float) -> float:
    return abs(float(candidate) - float(original)) / max(1.0, abs(float(original)))


def _clamp_numeric(feature: str, value: float) -> float:
    lo, hi = NUMERIC_MIN_MAX[feature]
    value = float(value)
    return float(max(lo, min(hi, value)))


def _budget_summary(original_row: pd.Series, candidate_row: pd.Series, budget_key: str) -> dict[str, Any]:
    policy = BUDGET_POLICY[budget_key]
    changed = [
        feature
        for feature in NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES
        if float(candidate_row[feature]) != float(original_row[feature])
    ]
    rel_changes = [
        _relative_change(float(original_row[feature]), float(candidate_row[feature]))
        for feature in changed
    ]
    return {
        "changed_features": changed,
        "feature_count": len(changed),
        "max_relative_change": max(rel_changes) if rel_changes else 0.0,
        "aggregate_relative_change": float(sum(rel_changes)) if rel_changes else 0.0,
        "valid": (
            len(changed) <= policy["max_changed"]
            and (max(rel_changes) if rel_changes else 0.0) <= policy["max_relative"] + 1e-12
            and (sum(rel_changes) if rel_changes else 0.0) <= policy["aggregate_cap"] + 1e-12
        ),
    }


def _candidate_matches_budget(original_row: pd.Series, candidate_row: pd.Series, budget_key: str) -> tuple[bool, str | None]:
    policy = BUDGET_POLICY[budget_key]
    changed = [
        feature
        for feature in NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES
        if float(candidate_row[feature]) != float(original_row[feature])
    ]
    if len(changed) > policy["max_changed"]:
        return False, f"changed feature count exceeds {policy['max_changed']}"
    rel_changes = [
        _relative_change(float(original_row[feature]), float(candidate_row[feature]))
        for feature in changed
    ]
    if rel_changes and max(rel_changes) > policy["max_relative"] + 1e-12:
        return False, f"relative change exceeds {policy['max_relative']} per feature"
    if budget_key == "B" and sum(rel_changes) > policy["aggregate_cap"] + 1e-12:
        return False, f"aggregate normalized change exceeds {policy['aggregate_cap']}"
    return True, None


def mutation_proposal(original_row: pd.Series, feature: str, sign: int, step_fraction: float) -> pd.Series:
    proposal = original_row.copy()
    original_value = float(original_row[feature])
    step = float(step_fraction) * max(1.0, abs(original_value))
    candidate_value = original_value + sign * step
    if feature in {"dur", "rate", "sload", "dload", "sinpkt", "dinpkt", "sjit", "djit"}:
        candidate_value = float(candidate_value)
    else:
        candidate_value = float(round(candidate_value))
    proposal[feature] = _clamp_numeric(feature, candidate_value)
    return proposal


def make_random_valid_candidate(original_row: pd.Series, budget_key: str, rng: np.random.Generator) -> pd.Series | None:
    policy = BUDGET_POLICY[budget_key]
    feature_pool = list(NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES)
    if not feature_pool:
        return None
    num_features = int(rng.integers(1, policy["max_changed"] + 1))
    selected = list(rng.choice(feature_pool, size=min(num_features, len(feature_pool)), replace=False))
    proposal = original_row.copy()
    for feature in selected:
        sign = int(rng.choice([-1, 1]))
        rel_step = rng.uniform(0.0, policy["max_relative"])
        proposal = mutation_proposal(proposal, feature, sign, rel_step)

    if proposal.equals(original_row):
        return None
    budget_ok, reason = _candidate_matches_budget(original_row, proposal, budget_key)
    if not budget_ok:
        return None
    validation = validate_candidate(original_row.to_dict(), proposal.to_dict())
    if not validation["valid"]:
        return None
    return proposal


def choose_feature_steps(budget_key: str) -> list[float]:
    if budget_key == "A":
        return [0.02, 0.05, 0.08, 0.10]
    return [0.03, 0.06, 0.10, 0.15, 0.25]


def _safe_model_prob(model: Any, row: pd.DataFrame) -> float:
    return float(model.predict_proba(row)[0, 1])


def _call_explanation(model: Any, row: pd.Series, codec: CategoricalCodec, explainer: Any, predict_fn: Any, seed: int) -> dict[str, Any]:
    encoded = codec.encode(row.to_frame().T)
    reseed_explainer(explainer, seed)
    exp = explainer.explain_instance(
        data_row=np.asarray(encoded[0], dtype=np.float64),
        predict_fn=predict_fn,
        labels=(EXPLAINED_LABEL,),
        num_features=NUM_FEATURES,
        num_samples=NUM_SAMPLES,
        distance_metric="euclidean",
        model_regressor=None,
    )
    raw = exp.as_map()[EXPLAINED_LABEL]
    weights = {FEATURE_ORDER[idx]: float(weight) for idx, weight in raw}
    for feature in FEATURE_ORDER:
        if feature not in weights:
            weights[feature] = 0.0
    return weights


def _top10_set(weights: dict[str, float]) -> set[str]:
    ranked = sorted(weights.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
    return {feature for feature, _ in ranked[:10]}


def _top10_jaccard(original_weights: dict[str, float], candidate_weights: dict[str, float]) -> float:
    top_original = _top10_set(original_weights)
    top_candidate = _top10_set(candidate_weights)
    overlap = top_original & top_candidate
    union = top_original | top_candidate
    return len(overlap) / len(union) if union else 0.0


def _spearman_rho(original_weights: dict[str, float], candidate_weights: dict[str, float]) -> float:
    feature_universe = sorted(set(original_weights) | set(candidate_weights))
    if not feature_universe:
        return 1.0

    original_values = pd.Series(
        [abs(float(original_weights.get(feature, 0.0))) if feature in original_weights else -1.0 for feature in feature_universe],
        index=feature_universe,
    )
    candidate_values = pd.Series(
        [abs(float(candidate_weights.get(feature, 0.0))) if feature in candidate_weights else -1.0 for feature in feature_universe],
        index=feature_universe,
    )
    original_ranks = original_values.rank(method="average", ascending=False)
    candidate_ranks = candidate_values.rank(method="average", ascending=False)

    coef = np.corrcoef(original_ranks.to_numpy(), candidate_ranks.to_numpy())[0, 1]
    if np.isnan(coef):
        return 0.0
    return float(np.clip(coef, -1.0, 1.0))


def objective_score(original_weights: dict[str, float], candidate_weights: dict[str, float]) -> float:
    jaccard = _top10_jaccard(original_weights, candidate_weights)
    rho = _spearman_rho(original_weights, candidate_weights)
    return float(0.7 * (1.0 - jaccard) + 0.3 * ((1.0 - rho) / 2.0))


def explain_and_score(model: Any, row: pd.Series, codec: CategoricalCodec, explainer: Any, predict_fn: Any, seed: int) -> tuple[dict[str, float], float]:
    weights = _call_explanation(model, row, codec, explainer, predict_fn, seed)
    score = float(objective_score({}, weights)) if False else 0.0
    return weights, score


def run_targeted_search(
    model_key: str,
    sample_position: int,
    budget_key: str,
    split: dict[str, Any],
    max_explanation_evaluations: int = 10,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    model = load_xai_model(model_key, split)
    X_val = split["X_val"]
    y_val = split["y_val"]
    if not 0 <= int(sample_position) < len(X_val):
        raise ValueError(f"sample position {sample_position} out of range")
    original_row = X_val.iloc[[int(sample_position)]].iloc[0].copy()
    if int(y_val.iloc[int(sample_position)]) != 1:
        raise ValueError(f"sample position {sample_position} is not a malicious validation row")

    original_prob = _safe_model_prob(model, original_row.to_frame().T)
    if original_prob < 0.5:
        raise ValueError(f"original sample {sample_position} does not remain malicious for {model_key}")

    X_subtrain = split["X_subtrain"]
    codec = CategoricalCodec(X_subtrain)
    explainer = build_explainer(codec.encode(X_subtrain), codec)
    predict_fn = make_predict_fn(model, codec)

    baseline_weights = _call_explanation(model, original_row, codec, explainer, predict_fn, seed)
    best_score = 0.0
    best_candidate = original_row.copy()
    current_candidate = original_row.copy()
    current_weights = baseline_weights
    current_score = 0.0
    evaluation_counts = {
        "explanation_calls": 1,
        "model_predictions": 1,
        "total_proposals": 0,
        "accepted_proposals": 0,
    }
    search_history: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    def log_proposal(status: str, reason: str, candidate: pd.Series, score: float | None = None, proposal_id: str | None = None) -> None:
        row = {
            "method": "targeted",
            "model_key": model_key,
            "validation_position": int(sample_position),
            "budget": budget_key,
            "status": status,
            "reason": reason,
            "candidate_id": proposal_id or f"{model_key}_{budget_key}_{sample_position}_{len(search_history):03d}",
            "score": float(score) if score is not None else None,
            "feature_values_json": json.dumps(candidate.to_dict(), default=str),
        }
        search_history.append(row)
        if status != "accepted":
            rejected.append(row)

    while evaluation_counts["explanation_calls"] < max_explanation_evaluations:
        improved = False
        for feature in NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES:
            for direction in (-1, 1):
                for step_fraction in choose_feature_steps(budget_key):
                    if evaluation_counts["explanation_calls"] >= max_explanation_evaluations:
                        break
                    proposal = mutation_proposal(current_candidate, feature, direction, step_fraction)
                    if proposal.equals(current_candidate):
                        continue
                    if not proposal[feature] == proposal[feature]:
                        continue
                    validation = validate_candidate(original_row.to_dict(), proposal.to_dict())
                    if not validation["valid"]:
                        evaluation_counts["total_proposals"] += 1
                        log_proposal("rejected", "; ".join(validation["violated_constraints"]), proposal)
                        continue
                    budget_ok, budget_reason = _candidate_matches_budget(original_row, proposal, budget_key)
                    if not budget_ok:
                        evaluation_counts["total_proposals"] += 1
                        log_proposal("rejected", budget_reason or "budget violation", proposal)
                        continue
                    candidate_prob = _safe_model_prob(model, proposal.to_frame().T)
                    evaluation_counts["model_predictions"] += 1
                    if candidate_prob < 0.5:
                        evaluation_counts["total_proposals"] += 1
                        log_proposal("rejected", "prediction changed to benign", proposal)
                        continue
                    proposal_weights = _call_explanation(model, proposal, codec, explainer, predict_fn, seed)
                    evaluation_counts["explanation_calls"] += 1
                    evaluation_counts["total_proposals"] += 1
                    score = objective_score(current_weights, proposal_weights)
                    if score > best_score:
                        best_score = score
                        best_candidate = proposal.copy()
                        current_candidate = proposal.copy()
                        current_weights = proposal_weights
                        current_score = score
                        evaluation_counts["accepted_proposals"] += 1
                        log_proposal("accepted", "improved objective", proposal, score)
                        improved = True
                        break
                    log_proposal("rejected", f"score={score:.6f} did not improve best objective", proposal, score)
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    return {
        "model_key": model_key,
        "sample_position": int(sample_position),
        "budget": budget_key,
        "original_probability": float(original_prob),
        "best_candidate_probability": float(_safe_model_prob(model, best_candidate.to_frame().T)),
        "best_score": float(best_score),
        "best_candidate": best_candidate.to_dict(),
        "original_feature_values_json": json.dumps(original_row.to_dict(), default=str),
        "modified_feature_values_json": json.dumps(best_candidate.to_dict(), default=str),
        "search_history": search_history,
        "rejections": rejected,
        "evaluation_counts": evaluation_counts,
        "baseline_weights": baseline_weights,
        "final_weights": current_weights,
        "constraint_valid": bool(validate_candidate(original_row.to_dict(), best_candidate.to_dict())["valid"]),
    }


def random_control_search(
    model_key: str,
    sample_position: int,
    budget_key: str,
    split: dict[str, Any],
    max_explanation_evaluations: int = 10,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    model = load_xai_model(model_key, split)
    X_val = split["X_val"]
    original_row = X_val.iloc[[int(sample_position)]].iloc[0].copy()
    if int(split["y_val"].iloc[int(sample_position)]) != 1:
        raise ValueError(f"sample {sample_position} is not malicious")

    X_subtrain = split["X_subtrain"]
    codec = CategoricalCodec(X_subtrain)
    explainer = build_explainer(codec.encode(X_subtrain), codec)
    predict_fn = make_predict_fn(model, codec)

    rng = np.random.default_rng(seed + int(sample_position) + sum(ord(ch) for ch in model_key))
    baseline_weights = _call_explanation(model, original_row, codec, explainer, predict_fn, seed)
    best_score = 0.0
    best_candidate = original_row.copy()
    history: list[dict[str, Any]] = []
    explanation_calls = 1
    model_predictions = 1
    rejected: list[dict[str, Any]] = []

    while explanation_calls < max_explanation_evaluations:
        proposal = make_random_valid_candidate(original_row, budget_key, rng)
        if proposal is None:
            rejected.append({
                "method": "random_control",
                "model_key": model_key,
                "validation_position": int(sample_position),
                "budget": budget_key,
                "status": "rejected",
                "reason": "invalid random proposal",
            })
            continue
        validation = validate_candidate(original_row.to_dict(), proposal.to_dict())
        if not validation["valid"]:
            history.append({
                "method": "random_control",
                "model_key": model_key,
                "validation_position": int(sample_position),
                "status": "rejected",
                "reason": "; ".join(validation["violated_constraints"]),
            })
            rejected.append(history[-1])
            continue
        prob = _safe_model_prob(model, proposal.to_frame().T)
        model_predictions += 1
        if prob < 0.5:
            history.append({
                "method": "random_control",
                "model_key": model_key,
                "validation_position": int(sample_position),
                "status": "rejected",
                "reason": "prediction changed to benign",
            })
            rejected.append(history[-1])
            continue
        proposal_weights = _call_explanation(model, proposal, codec, explainer, predict_fn, seed)
        explanation_calls += 1
        score = objective_score(baseline_weights, proposal_weights)
        if score > best_score:
            best_score = score
            best_candidate = proposal.copy()
        history.append({
            "method": "random_control",
            "model_key": model_key,
            "validation_position": int(sample_position),
            "budget": budget_key,
            "status": "evaluated",
            "score": float(score),
            "feature_count": sum(float(proposal[feature]) != float(original_row[feature]) for feature in NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES),
            "candidate_probability": float(prob),
            "feature_values_json": json.dumps(proposal.to_dict(), default=str),
        })

    return {
        "model_key": model_key,
        "sample_position": int(sample_position),
        "budget": budget_key,
        "best_score": float(best_score),
        "best_candidate": best_candidate.to_dict(),
        "best_probability": float(_safe_model_prob(model, best_candidate.to_frame().T)),
        "history": history,
        "rejections": rejected,
        "evaluation_counts": {
            "explanation_calls": int(explanation_calls),
            "model_predictions": int(model_predictions),
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _make_run_dir(base_dir: str | Path) -> Path:
    root = Path(base_dir)
    if not root.is_absolute():
        root = ROOT / root
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_experiment(
    models: list[str] | None = None,
    sample_positions: list[int] | None = None,
    budgets: list[str] | None = None,
    max_explanation_evaluations: int = 10,
    seed: int = DEFAULT_SEED,
    output_dir: str = str(OUTPUT_ROOT),
) -> dict[str, Any]:
    model_names = models or DEFAULT_MODELS
    target_positions = sample_positions or TARGET_SAMPLE_POSITIONS
    budget_names = budgets or DEFAULT_BUDGETS

    split = get_xai_data()
    shared_samples = load_shared_samples(split)
    valid_positions = set(int(v) for v in shared_samples["validation_position"])
    missing = [pos for pos in target_positions if pos not in valid_positions]
    if missing:
        raise ValueError(f"Requested validation positions are not in the shared manifest: {missing}")

    run_dir = _make_run_dir(output_dir)
    run_results: list[dict[str, Any]] = []
    control_results: list[dict[str, Any]] = []
    proposal_history: list[dict[str, Any]] = []

    for model_key in model_names:
        for sample_position in target_positions:
            for budget_key in budget_names:
                targeted = run_targeted_search(
                    model_key=model_key,
                    sample_position=sample_position,
                    budget_key=budget_key,
                    split=split,
                    max_explanation_evaluations=max_explanation_evaluations,
                    seed=seed,
                )
                control = random_control_search(
                    model_key=model_key,
                    sample_position=sample_position,
                    budget_key=budget_key,
                    split=split,
                    max_explanation_evaluations=max_explanation_evaluations,
                    seed=seed,
                )
                run_results.append({
                    "model_key": model_key,
                    "sample_position": int(sample_position),
                    "budget": budget_key,
                    "original_probability": float(targeted["original_probability"]),
                    "best_probability": float(targeted["best_candidate_probability"]),
                    "objective_score": float(targeted["best_score"]),
                    "constraint_valid": bool(targeted["constraint_valid"]),
                    "explanation_calls": int(targeted["evaluation_counts"]["explanation_calls"]),
                    "model_predictions": int(targeted["evaluation_counts"]["model_predictions"]),
                    "run_type": "targeted",
                })
                control_results.append({
                    "model_key": model_key,
                    "sample_position": int(sample_position),
                    "budget": budget_key,
                    "objective_score": float(control["best_score"]),
                    "best_probability": float(control["best_probability"]),
                    "explanation_calls": int(control["evaluation_counts"]["explanation_calls"]),
                    "model_predictions": int(control["evaluation_counts"]["model_predictions"]),
                    "run_type": "random_control",
                })
                proposal_history.extend(targeted["search_history"])
                proposal_history.extend([
                    {
                        "method": "random_control",
                        "model_key": model_key,
                        "validation_position": int(sample_position),
                        "budget": budget_key,
                        "status": entry.get("status"),
                        "reason": entry.get("reason"),
                        "score": entry.get("score"),
                    }
                    for entry in control["history"]
                ])

    pd.DataFrame(run_results).to_csv(run_dir / "targeted_results.csv", index=False)
    pd.DataFrame(control_results).to_csv(run_dir / "random_control_results.csv", index=False)
    pd.DataFrame(proposal_history).to_csv(run_dir / "proposal_history.csv", index=False)

    metadata = {
        "run_id": run_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": "bounded greedy coordinate search with matched random controls",
        "models": model_names,
        "sample_positions": [int(pos) for pos in target_positions],
        "budgets": budget_names,
        "max_explanation_evaluations": int(max_explanation_evaluations),
        "search_seed": int(seed),
        "baseline_objective": {
            "formula": "0.7*(1-top10_jaccard) + 0.3*((1-spearman_rho)/2)",
            "spearman_universe": "union of original and candidate top-10 features; missing ranks are assigned worst-rank policy and tie ranks use average rank",
        },
        "budget_policy": {
            key: {
                "max_changed_features": value["max_changed"],
                "max_relative_change_per_feature": value["max_relative"],
                "aggregate_relative_change_cap": value["aggregate_cap"],
            }
            for key, value in BUDGET_POLICY.items()
        },
        "dataset_sha256": {
            "training": file_sha256(ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"),
            "test": file_sha256(ROOT / "data" / "raw" / "UNSW_NB15_testing-set.csv"),
        },
        "shared_manifest_sha256": file_sha256(ROOT / "results" / "metrics" / "shap" / "shared_validation_sample_identifiers.csv"),
        "model_sha256": {
            "random_forest": file_sha256(MODEL_DIR / "random_forest_validation_pipeline.joblib"),
            "xgboost": file_sha256(MODEL_DIR / "xgboost_validation_pipeline.joblib"),
        },
        "feature_order": FEATURE_ORDER,
        "validation_split": {
            "method": "train_test_split",
            "random_state": 42,
            "test_size": 0.20,
            "stratified": True,
            "training_rows": int(len(split["X_train_full"])),
            "validation_rows": int(len(split["X_val"])),
        },
    }
    _write_json(run_dir / "run_metadata.json", metadata)
    summary = {
        "run_id": run_dir.name,
        "targeted_results": run_results,
        "random_control_results": control_results,
        "evaluation_counts_total": {
            "targeted_explanation_calls": sum(item["explanation_calls"] for item in run_results),
            "targeted_model_predictions": sum(item["model_predictions"] for item in run_results),
            "control_explanation_calls": sum(item["explanation_calls"] for item in control_results),
            "control_model_predictions": sum(item["model_predictions"] for item in control_results),
        },
    }
    _write_json(run_dir / "run_summary.json", summary)

    return {
        "run_dir": str(run_dir),
        "metadata": metadata,
        "summary": summary,
        "target_results": run_results,
        "control_results": control_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded targeted explanation-spoofing search for UNSW-NB15.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Saved validation pipelines to attack; defaults to RF and XGBoost.")
    parser.add_argument("--samples", nargs="+", type=int, default=TARGET_SAMPLE_POSITIONS, help="Validation positions to attack.")
    parser.add_argument("--budgets", nargs="+", choices=["A", "B"], default=DEFAULT_BUDGETS, help="Perturbation budgets to enforce.")
    parser.add_argument("--max-explanation-evaluations", type=int, default=10, help="Hard cap on explanation calls per method (targeted and matched control).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="LIME optimization seed for the targeted search.")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_ROOT), help="Directory under which a timestamped run folder is created.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    result = run_experiment(
        models=args.models,
        sample_positions=args.samples,
        budgets=args.budgets,
        max_explanation_evaluations=args.max_explanation_evaluations,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    elapsed = time.perf_counter() - start
    print(f"Run directory: {result['run_dir']}")
    print(f"Elapsed seconds: {elapsed:.3f}")
    print(json.dumps(result["summary"]["evaluation_counts_total"], sort_keys=True))


if __name__ == "__main__":
    main()
