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
from src.unsw.constraints import FEATURE_METADATA, FEATURE_ORDER, INTEGER_FEATURES, NUMERIC_MIN_MAX, validate_candidate
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

# Search strategies.
# - greedy_incremental_v0: the original algorithm, kept unchanged as a baseline. It scores each
#   proposal against the previously accepted explanation (incremental), so its scores are NOT
#   comparable with the random control.
# - greedy_baseline_ref_v1: the same greedy first-improvement coordinate search, but every proposal
#   is scored against the ORIGINAL explanation (the same reference the random control uses). Steps
#   are measured in units of max(1, |original value|) and snap back to the original value within
#   SNAP_TOLERANCE. Integer rounding follows constraints.INTEGER_FEATURES. Candidates that were
#   already evaluated are not explained again.
STRATEGY_V0 = "greedy_incremental_v0"
STRATEGY_V1 = "greedy_baseline_ref_v1"
STRATEGIES = (STRATEGY_V0, STRATEGY_V1)
DEFAULT_STRATEGY = STRATEGY_V1

SNAP_TOLERANCE = 1e-9
# Random control: hard cap on proposal attempts = multiplier * max_explanation_evaluations.
DEFAULT_RANDOM_ATTEMPT_MULTIPLIER = 20

# Evaluation-accounting keys shared by the v1 targeted search and the random control.
# Identities: explanation_calls == 1 + scored_proposals (the 1 is the baseline explanation);
#             model_predictions == 1 + scored_proposals + rejected_benign.
ACCOUNTING_KEYS = (
    "explanation_calls",
    "model_predictions",
    "proposals_generated",
    "scored_proposals",
    "rejected_no_change",
    "rejected_duplicate",
    "rejected_constraint",
    "rejected_budget",
    "rejected_benign",
    "accepted_improvements",
    "ties_not_accepted",
)


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


def mutation_proposal_v1(current_row: pd.Series, original_row: pd.Series, feature: str, sign: int, step_fraction: float) -> pd.Series:
    """Move one feature by step_fraction * max(1, |original|) from its current value (v1).

    Differences from the legacy mutation_proposal():
    - the step unit uses the ORIGINAL value (the same denominator as the budget), not the current one;
    - integer rounding follows constraints.INTEGER_FEATURES, so continuous timing features such as
      synack/ackdat are no longer rounded to integers;
    - a value within SNAP_TOLERANCE of the original snaps back to the exact original value, so an
      up-then-down move does not leave a phantom floating-point change that consumes budget.
    """
    proposal = current_row.copy()
    original_value = float(original_row[feature])
    unit = max(1.0, abs(original_value))
    candidate_value = float(current_row[feature]) + sign * float(step_fraction) * unit
    if feature in INTEGER_FEATURES:
        candidate_value = float(round(candidate_value))
    candidate_value = _clamp_numeric(feature, candidate_value)
    if abs(candidate_value - original_value) <= SNAP_TOLERANCE * unit:
        proposal[feature] = original_row[feature]
    else:
        proposal[feature] = candidate_value
    return proposal


def _mutation_for_strategy(strategy: str):
    """Return a mutate(current_row, original_row, feature, sign, step) callable for the strategy."""
    if strategy == STRATEGY_V0:
        return lambda current, original, feature, sign, step: mutation_proposal(current, feature, sign, step)
    if strategy == STRATEGY_V1:
        return mutation_proposal_v1
    raise ValueError(f"Unknown strategy {strategy!r}; expected one of {STRATEGIES}")


def _row_key(row: pd.Series) -> tuple[float, ...]:
    """Identity of a candidate: the values of the attackable features (all others are frozen)."""
    return tuple(float(row[feature]) for feature in NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES)


def _changed_features(original_row: pd.Series, candidate_row: pd.Series) -> list[str]:
    return [
        feature
        for feature in NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES
        if float(candidate_row[feature]) != float(original_row[feature])
    ]


def _new_counts() -> dict[str, int]:
    counts = {key: 0 for key in ACCOUNTING_KEYS}
    # The baseline explanation of the unmodified sample and its prediction.
    counts["explanation_calls"] = 1
    counts["model_predictions"] = 1
    return counts


def _draw_random_candidate(
    original_row: pd.Series,
    budget_key: str,
    rng: np.random.Generator,
    mutate=None,
    strict_no_change: bool = False,
) -> tuple[pd.Series | None, str | None]:
    """Draw one random budget-sized perturbation; return (proposal, None) or (None, rejection_kind).

    rejection_kind is one of "no_change", "budget", "constraint". The RNG draws are identical to the
    original make_random_valid_candidate(), so seeded runs of the legacy strategy are unchanged.
    strict_no_change (v1 only) also rejects proposals whose attackable values are numerically equal
    to the original even if pandas does not consider the rows equal.
    """
    mutate = mutate or _mutation_for_strategy(STRATEGY_V0)
    policy = BUDGET_POLICY[budget_key]
    feature_pool = list(NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES)
    if not feature_pool:
        return None, "no_change"
    num_features = int(rng.integers(1, policy["max_changed"] + 1))
    selected = list(rng.choice(feature_pool, size=min(num_features, len(feature_pool)), replace=False))
    proposal = original_row.copy()
    for feature in selected:
        sign = int(rng.choice([-1, 1]))
        rel_step = rng.uniform(0.0, policy["max_relative"])
        proposal = mutate(proposal, original_row, feature, sign, rel_step)

    if proposal.equals(original_row):
        return None, "no_change"
    if strict_no_change and not _changed_features(original_row, proposal):
        return None, "no_change"
    budget_ok, _ = _candidate_matches_budget(original_row, proposal, budget_key)
    if not budget_ok:
        return None, "budget"
    validation = validate_candidate(original_row.to_dict(), proposal.to_dict())
    if not validation["valid"]:
        return None, "constraint"
    return proposal, None


def make_random_valid_candidate(original_row: pd.Series, budget_key: str, rng: np.random.Generator) -> pd.Series | None:
    """Legacy interface: a valid random candidate (legacy mutation) or None."""
    proposal, _ = _draw_random_candidate(original_row, budget_key, rng, _mutation_for_strategy(STRATEGY_V0))
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


def _run_targeted_search_v0(
    model_key: str,
    sample_position: int,
    budget_key: str,
    split: dict[str, Any],
    max_explanation_evaluations: int = 10,
    seed: int = DEFAULT_SEED,
    model: Any = None,
) -> dict[str, Any]:
    """Legacy greedy search (greedy_incremental_v0), unchanged apart from the optional model argument
    and the added strategy/stopped_reason/score_reference fields in the result.

    NOTE: proposals are scored against the previously accepted explanation, not the original one.
    """
    model = model if model is not None else load_xai_model(model_key, split)
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
        "strategy": STRATEGY_V0,
        "score_reference": "previous_accepted_explanation",
        "stopped_reason": (
            "explanation_budget_exhausted"
            if evaluation_counts["explanation_calls"] >= max_explanation_evaluations
            else "local_optimum_full_sweep_without_improvement"
        ),
    }


def _run_targeted_search_v1(
    model_key: str,
    sample_position: int,
    budget_key: str,
    split: dict[str, Any],
    max_explanation_evaluations: int = 10,
    seed: int = DEFAULT_SEED,
    model: Any = None,
) -> dict[str, Any]:
    """Greedy first-improvement coordinate search scored against the ORIGINAL explanation (v1).

    Same sweep order, step sizes and stopping rule as v0. Differences: the score reference,
    mutation_proposal_v1 (snap-back, constraint-driven rounding, original-value step unit),
    no re-explanation of already-evaluated candidates, logged ties, and shared accounting.
    """
    model = model if model is not None else load_xai_model(model_key, split)
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
    counts = _new_counts()
    best_score = 0.0
    best_candidate = original_row.copy()
    best_weights = baseline_weights
    current_candidate = original_row.copy()
    seen: set[tuple[float, ...]] = {_row_key(original_row)}
    search_history: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    def log_proposal(status: str, reason: str, candidate: pd.Series, score: float | None = None, probability: float | None = None) -> None:
        row = {
            "method": "targeted",
            "strategy": STRATEGY_V1,
            "model_key": model_key,
            "validation_position": int(sample_position),
            "budget": budget_key,
            "status": status,
            "reason": reason,
            "candidate_id": f"{model_key}_{budget_key}_{sample_position}_{len(search_history):03d}",
            "score": float(score) if score is not None else None,
            "score_reference": "original_explanation",
            "candidate_probability": float(probability) if probability is not None else None,
            "changed_features": ";".join(_changed_features(original_row, candidate)),
            "explanation_calls_so_far": int(counts["explanation_calls"]),
            "feature_values_json": json.dumps(candidate.to_dict(), default=str),
        }
        search_history.append(row)
        if status != "accepted":
            rejected.append(row)

    steps = choose_feature_steps(budget_key)
    stopped_reason = None
    while stopped_reason is None:
        if counts["explanation_calls"] >= max_explanation_evaluations:
            stopped_reason = "explanation_budget_exhausted"
            break
        improved = False
        for feature in NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES:
            for direction in (-1, 1):
                for step_fraction in steps:
                    if counts["explanation_calls"] >= max_explanation_evaluations:
                        break
                    proposal = mutation_proposal_v1(current_candidate, original_row, feature, direction, step_fraction)
                    counts["proposals_generated"] += 1
                    if proposal.equals(current_candidate) or _row_key(proposal) == _row_key(current_candidate):
                        # No representable move (e.g. integer rounding or clamping); nothing to evaluate.
                        counts["rejected_no_change"] += 1
                        continue
                    key = _row_key(proposal)
                    if key in seen:
                        counts["rejected_duplicate"] += 1
                        log_proposal("rejected", "duplicate: candidate already evaluated", proposal)
                        continue
                    seen.add(key)
                    validation = validate_candidate(original_row.to_dict(), proposal.to_dict())
                    if not validation["valid"]:
                        counts["rejected_constraint"] += 1
                        log_proposal("rejected", "; ".join(validation["violated_constraints"]), proposal)
                        continue
                    budget_ok, budget_reason = _candidate_matches_budget(original_row, proposal, budget_key)
                    if not budget_ok:
                        counts["rejected_budget"] += 1
                        log_proposal("rejected", budget_reason or "budget violation", proposal)
                        continue
                    candidate_prob = _safe_model_prob(model, proposal.to_frame().T)
                    counts["model_predictions"] += 1
                    if candidate_prob < 0.5:
                        counts["rejected_benign"] += 1
                        log_proposal("rejected", "prediction changed to benign", proposal, probability=candidate_prob)
                        continue
                    proposal_weights = _call_explanation(model, proposal, codec, explainer, predict_fn, seed)
                    counts["explanation_calls"] += 1
                    counts["scored_proposals"] += 1
                    score = objective_score(baseline_weights, proposal_weights)
                    if score > best_score:
                        best_score = score
                        best_candidate = proposal.copy()
                        best_weights = proposal_weights
                        current_candidate = proposal.copy()
                        counts["accepted_improvements"] += 1
                        log_proposal("accepted", "improved objective vs original explanation", proposal, score, candidate_prob)
                        improved = True
                        break
                    if best_score > 0.0 and score == best_score:
                        counts["ties_not_accepted"] += 1
                        log_proposal("rejected", "tie with best objective (strict improvement required)", proposal, score, candidate_prob)
                    else:
                        log_proposal("rejected", f"score={score:.6f} did not improve best objective", proposal, score, candidate_prob)
                if improved:
                    break
            if improved:
                break
        if not improved:
            stopped_reason = (
                "explanation_budget_exhausted"
                if counts["explanation_calls"] >= max_explanation_evaluations
                else "local_optimum_full_sweep_without_improvement"
            )

    return {
        "model_key": model_key,
        "sample_position": int(sample_position),
        "budget": budget_key,
        "original_probability": float(original_prob),
        "best_candidate_probability": float(_safe_model_prob(model, best_candidate.to_frame().T)),
        "best_score": float(best_score),
        "best_candidate": best_candidate.to_dict(),
        "best_changed_features": _changed_features(original_row, best_candidate),
        "original_feature_values_json": json.dumps(original_row.to_dict(), default=str),
        "modified_feature_values_json": json.dumps(best_candidate.to_dict(), default=str),
        "search_history": search_history,
        "rejections": rejected,
        "evaluation_counts": counts,
        "baseline_weights": baseline_weights,
        "final_weights": best_weights,
        "constraint_valid": bool(validate_candidate(original_row.to_dict(), best_candidate.to_dict())["valid"]),
        "strategy": STRATEGY_V1,
        "score_reference": "original_explanation",
        "stopped_reason": stopped_reason,
    }


def run_targeted_search(
    model_key: str,
    sample_position: int,
    budget_key: str,
    split: dict[str, Any],
    max_explanation_evaluations: int = 10,
    seed: int = DEFAULT_SEED,
    strategy: str = DEFAULT_STRATEGY,
    model: Any = None,
) -> dict[str, Any]:
    """Dispatch to the legacy (v0) or corrected (v1) greedy search."""
    if strategy == STRATEGY_V0:
        return _run_targeted_search_v0(model_key, sample_position, budget_key, split, max_explanation_evaluations, seed, model)
    if strategy == STRATEGY_V1:
        return _run_targeted_search_v1(model_key, sample_position, budget_key, split, max_explanation_evaluations, seed, model)
    raise ValueError(f"Unknown strategy {strategy!r}; expected one of {STRATEGIES}")


def random_control_search(
    model_key: str,
    sample_position: int,
    budget_key: str,
    split: dict[str, Any],
    max_explanation_evaluations: int = 10,
    seed: int = DEFAULT_SEED,
    strategy: str = DEFAULT_STRATEGY,
    max_attempts: int | None = None,
    model: Any = None,
) -> dict[str, Any]:
    """Best-of-N random control with the same objective and accounting as the targeted search.

    Every proposal is scored against the ORIGINAL explanation (as in v0 and v1). The loop stops
    when the explanation-call budget is used up or after max_attempts proposal attempts, whichever
    comes first, so it always terminates. Every attempt is logged. The RNG seed formula is
    unchanged. For STRATEGY_V0 the legacy mutation is used, so seeded legacy draws are reproduced.
    For STRATEGY_V1 the v1 mutation is used and already-evaluated candidates are not re-explained.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; expected one of {STRATEGIES}")
    model = model if model is not None else load_xai_model(model_key, split)
    X_val = split["X_val"]
    if not 0 <= int(sample_position) < len(X_val):
        raise ValueError(f"sample position {sample_position} out of range")
    original_row = X_val.iloc[[int(sample_position)]].iloc[0].copy()
    if int(split["y_val"].iloc[int(sample_position)]) != 1:
        raise ValueError(f"sample {sample_position} is not malicious")
    original_prob = _safe_model_prob(model, original_row.to_frame().T)
    if original_prob < 0.5:
        raise ValueError(f"original sample {sample_position} does not remain malicious for {model_key}")

    max_attempts = int(max_attempts) if max_attempts is not None else DEFAULT_RANDOM_ATTEMPT_MULTIPLIER * int(max_explanation_evaluations)
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    mutate = _mutation_for_strategy(strategy)
    strict = strategy == STRATEGY_V1

    X_subtrain = split["X_subtrain"]
    codec = CategoricalCodec(X_subtrain)
    explainer = build_explainer(codec.encode(X_subtrain), codec)
    predict_fn = make_predict_fn(model, codec)

    rng = np.random.default_rng(seed + int(sample_position) + sum(ord(ch) for ch in model_key))
    baseline_weights = _call_explanation(model, original_row, codec, explainer, predict_fn, seed)
    counts = _new_counts()
    best_score = 0.0
    best_candidate = original_row.copy()
    best_weights = baseline_weights
    seen: set[tuple[float, ...]] = {_row_key(original_row)}
    history: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    attempts = 0

    def log_attempt(status: str, reason: str, candidate: pd.Series | None = None, score: float | None = None, probability: float | None = None) -> None:
        row = {
            "method": "random_control",
            "strategy": strategy,
            "model_key": model_key,
            "validation_position": int(sample_position),
            "budget": budget_key,
            "attempt": int(attempts),
            "status": status,
            "reason": reason,
            "score": float(score) if score is not None else None,
            "score_reference": "original_explanation",
            "candidate_probability": float(probability) if probability is not None else None,
            "feature_count": len(_changed_features(original_row, candidate)) if candidate is not None else 0,
            "changed_features": ";".join(_changed_features(original_row, candidate)) if candidate is not None else "",
            "explanation_calls_so_far": int(counts["explanation_calls"]),
            "feature_values_json": json.dumps(candidate.to_dict(), default=str) if candidate is not None else None,
        }
        history.append(row)
        if status != "evaluated":
            rejected.append(row)

    stopped_reason = "explanation_budget_exhausted"
    while counts["explanation_calls"] < max_explanation_evaluations:
        if attempts >= max_attempts:
            stopped_reason = "attempt_limit_reached"
            break
        attempts += 1
        counts["proposals_generated"] += 1
        proposal, rejection_kind = _draw_random_candidate(original_row, budget_key, rng, mutate, strict_no_change=strict)
        if proposal is None:
            counts[f"rejected_{rejection_kind}"] += 1
            log_attempt("rejected", f"invalid random proposal: {rejection_kind}")
            continue
        if strict:
            key = _row_key(proposal)
            if key in seen:
                counts["rejected_duplicate"] += 1
                log_attempt("rejected", "duplicate: candidate already evaluated", proposal)
                continue
            seen.add(key)
        prob = _safe_model_prob(model, proposal.to_frame().T)
        counts["model_predictions"] += 1
        if prob < 0.5:
            counts["rejected_benign"] += 1
            log_attempt("rejected", "prediction changed to benign", proposal, probability=prob)
            continue
        proposal_weights = _call_explanation(model, proposal, codec, explainer, predict_fn, seed)
        counts["explanation_calls"] += 1
        counts["scored_proposals"] += 1
        score = objective_score(baseline_weights, proposal_weights)
        if score > best_score:
            best_score = score
            best_candidate = proposal.copy()
            best_weights = proposal_weights
            counts["accepted_improvements"] += 1
        log_attempt("evaluated", "scored", proposal, score, prob)

    return {
        "model_key": model_key,
        "sample_position": int(sample_position),
        "budget": budget_key,
        "original_probability": float(original_prob),
        "best_score": float(best_score),
        "best_candidate": best_candidate.to_dict(),
        "best_changed_features": _changed_features(original_row, best_candidate),
        "best_probability": float(_safe_model_prob(model, best_candidate.to_frame().T)),
        "history": history,
        "rejections": rejected,
        "evaluation_counts": counts,
        "attempts": int(attempts),
        "max_attempts": int(max_attempts),
        "baseline_weights": baseline_weights,
        "final_weights": best_weights,
        "constraint_valid": bool(validate_candidate(original_row.to_dict(), best_candidate.to_dict())["valid"]),
        "strategy": strategy,
        "score_reference": "original_explanation",
        "stopped_reason": stopped_reason,
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


def _accounting_row(counts: dict[str, int]) -> dict[str, int]:
    return {key: int(counts.get(key, 0)) for key in ACCOUNTING_KEYS}


def run_experiment(
    models: list[str] | None = None,
    sample_positions: list[int] | None = None,
    budgets: list[str] | None = None,
    max_explanation_evaluations: int = 10,
    seed: int = DEFAULT_SEED,
    output_dir: str = str(OUTPUT_ROOT),
    strategy: str = DEFAULT_STRATEGY,
    max_random_attempts: int | None = None,
) -> dict[str, Any]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; expected one of {STRATEGIES}")
    model_names = models or DEFAULT_MODELS
    target_positions = sample_positions or TARGET_SAMPLE_POSITIONS
    budget_names = budgets or DEFAULT_BUDGETS
    random_attempt_cap = (
        int(max_random_attempts) if max_random_attempts is not None
        else DEFAULT_RANDOM_ATTEMPT_MULTIPLIER * int(max_explanation_evaluations)
    )

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
        model = load_xai_model(model_key, split)  # loaded once, shared by both methods
        for sample_position in target_positions:
            for budget_key in budget_names:
                t0 = time.perf_counter()
                targeted = run_targeted_search(
                    model_key=model_key,
                    sample_position=sample_position,
                    budget_key=budget_key,
                    split=split,
                    max_explanation_evaluations=max_explanation_evaluations,
                    seed=seed,
                    strategy=strategy,
                    model=model,
                )
                t1 = time.perf_counter()
                control = random_control_search(
                    model_key=model_key,
                    sample_position=sample_position,
                    budget_key=budget_key,
                    split=split,
                    max_explanation_evaluations=max_explanation_evaluations,
                    seed=seed,
                    strategy=strategy,
                    max_attempts=random_attempt_cap,
                    model=model,
                )
                t2 = time.perf_counter()
                targeted_counts = targeted["evaluation_counts"]
                run_results.append({
                    "model_key": model_key,
                    "sample_position": int(sample_position),
                    "budget": budget_key,
                    "strategy": targeted["strategy"],
                    "score_reference": targeted["score_reference"],
                    "original_probability": float(targeted["original_probability"]),
                    "best_probability": float(targeted["best_candidate_probability"]),
                    "objective_score": float(targeted["best_score"]),
                    "constraint_valid": bool(targeted["constraint_valid"]),
                    "best_changed_features": ";".join(targeted.get("best_changed_features", [])),
                    "stopped_reason": targeted["stopped_reason"],
                    "explanation_calls": int(targeted_counts["explanation_calls"]),
                    "model_predictions": int(targeted_counts["model_predictions"]),
                    **({} if targeted["strategy"] == STRATEGY_V0 else _accounting_row(targeted_counts)),
                    "runtime_seconds": round(t1 - t0, 3),
                    "run_type": "targeted",
                })
                control_counts = control["evaluation_counts"]
                control_results.append({
                    "model_key": model_key,
                    "sample_position": int(sample_position),
                    "budget": budget_key,
                    "strategy": control["strategy"],
                    "score_reference": control["score_reference"],
                    "original_probability": float(control["original_probability"]),
                    "objective_score": float(control["best_score"]),
                    "best_probability": float(control["best_probability"]),
                    "constraint_valid": bool(control["constraint_valid"]),
                    "best_changed_features": ";".join(control["best_changed_features"]),
                    "stopped_reason": control["stopped_reason"],
                    "attempts": int(control["attempts"]),
                    "max_attempts": int(control["max_attempts"]),
                    **_accounting_row(control_counts),
                    "runtime_seconds": round(t2 - t1, 3),
                    "run_type": "random_control",
                })
                proposal_history.extend(targeted["search_history"])
                proposal_history.extend(control["history"])

    pd.DataFrame(run_results).to_csv(run_dir / "targeted_results.csv", index=False)
    pd.DataFrame(control_results).to_csv(run_dir / "random_control_results.csv", index=False)
    pd.DataFrame(proposal_history).to_csv(run_dir / "proposal_history.csv", index=False)

    metadata = {
        "run_id": run_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "strategy_description": (
            "bounded greedy first-improvement coordinate search with a matched best-of-N random control; "
            + (
                "LEGACY v0: targeted proposals scored against the previously accepted explanation"
                if strategy == STRATEGY_V0
                else "v1: targeted and random proposals both scored against the original explanation"
            )
        ),
        "models": model_names,
        "sample_positions": [int(pos) for pos in target_positions],
        "budgets": budget_names,
        "max_explanation_evaluations": int(max_explanation_evaluations),
        "max_random_attempts": int(random_attempt_cap),
        "search_seed": int(seed),
        "baseline_objective": {
            "formula": "0.7*(1-top10_jaccard) + 0.3*((1-spearman_rho)/2)",
            "reference_explanation": "original (unmodified) sample, same fixed LIME seed",
            "top10": "features ranked by |LIME weight| descending, ties broken by feature name",
            "spearman_universe": (
                "all 42 features: LIME weights are filled with 0.0 for any feature LIME omits, ranks of |weight| "
                "use average ranks for ties, rho = Pearson correlation of the ranks"
            ),
        },
        "evaluation_accounting": {
            "keys": list(ACCOUNTING_KEYS),
            "identities": [
                "explanation_calls == 1 + scored_proposals (1 = baseline explanation)",
                "model_predictions == 1 + scored_proposals + rejected_benign",
            ],
            "explanation_budget": "explanation_calls <= max_explanation_evaluations for both methods (baseline included)",
            "random_attempt_cap": "random control stops after max_random_attempts proposal attempts",
            "duplicates": "v1 only: candidates already evaluated are skipped without a new explanation call",
        },
        "mutation_policy": (
            "legacy mutation_proposal: step = fraction*max(1,|current|); rounding for all features except "
            "dur/rate/sload/dload/sinpkt/dinpkt/sjit/djit"
            if strategy == STRATEGY_V0
            else "mutation_proposal_v1: step = fraction*max(1,|original|); rounding only for constraints.INTEGER_FEATURES; "
            f"values within {SNAP_TOLERANCE}*max(1,|original|) of the original snap to the exact original"
        ),
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
        "dataset_sha256_note": (
            "keys refer to FILE NAMES. In this repository the files are swapped: UNSW_NB15_training-set.csv holds "
            "the 82,332-row official TEST set and UNSW_NB15_testing-set.csv the 175,341-row official TRAIN set. "
            "Loaders assign roles by row count."
        ),
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
        "strategy": strategy,
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
    parser.add_argument("--max-explanation-evaluations", type=int, default=10, help="Hard cap on explanation calls per method (targeted and matched control), baseline included.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="LIME optimization seed for the targeted search.")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_ROOT), help="Directory under which a timestamped run folder is created.")
    parser.add_argument("--strategy", choices=list(STRATEGIES), default=DEFAULT_STRATEGY, help=f"Search strategy; {STRATEGY_V0} reproduces the original incremental-scoring baseline.")
    parser.add_argument("--max-random-attempts", type=int, default=None, help=f"Hard cap on random-control proposal attempts (default {DEFAULT_RANDOM_ATTEMPT_MULTIPLIER} x max explanation evaluations).")
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
        strategy=args.strategy,
        max_random_attempts=args.max_random_attempts,
    )
    elapsed = time.perf_counter() - start
    print(f"Run directory: {result['run_dir']}")
    print(f"Elapsed seconds: {elapsed:.3f}")
    print(json.dumps(result["summary"]["evaluation_counts_total"], sort_keys=True))


if __name__ == "__main__":
    main()
