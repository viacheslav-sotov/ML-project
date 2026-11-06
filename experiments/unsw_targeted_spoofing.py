from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
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
    DISCRETIZE_CONTINUOUS,
    DISCRETIZER,
    EXPLAINED_LABEL,
    FEATURE_SELECTION,
    KERNEL_WIDTH,
    NUM_FEATURES,
    NUM_SAMPLES,
    build_explainer,
    make_predict_fn as _lime_make_predict_fn,
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
# Step 5b (optional strategies; v0 and v1 above are unchanged):
# - greedy_shuffled_v1: one greedy climb from the original sample with a reproducibly shuffled
#   feature order and per-feature direction order (seeded by --order-seed, independent of the LIME seed).
# - multistart_greedy_v1: K independent shuffled climbs, all starting from the original sample, sharing
#   ONE explanation budget (baseline included), fair-share allocation with roll-over, and a candidate
#   cache so no candidate is ever explained or predicted twice.
STRATEGY_SHUFFLED = "greedy_shuffled_v1"
STRATEGY_MULTISTART = "multistart_greedy_v1"
MULTISTART_STRATEGIES = (STRATEGY_SHUFFLED, STRATEGY_MULTISTART)
STRATEGIES = (STRATEGY_V0, STRATEGY_V1, STRATEGY_SHUFFLED, STRATEGY_MULTISTART)
DEFAULT_STRATEGY = STRATEGY_V1
DEFAULT_ORDER_SEED = 42
DEFAULT_RESTARTS = 3

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


class CountingModel:
    """Transparent proxy that counts every predict_proba call on the wrapped model.

    Calls made while LIME's prediction function runs (see make_predict_fn below) are counted as
    LIME-internal batches; all other calls are top-level (the search's single-row checks and the
    final best-candidate report). Cached results never reach the model, so they are never counted.
    """

    def __init__(self, model: Any):
        self._model = model
        self.direct_calls = 0
        self.direct_rows = 0
        self.lime_batches = 0
        self.lime_rows = 0
        self._lime_depth = 0

    def predict_proba(self, X):
        rows = int(len(X))
        if self._lime_depth > 0:
            self.lime_batches += 1
            self.lime_rows += rows
        else:
            self.direct_calls += 1
            self.direct_rows += rows
        return self._model.predict_proba(X)

    def __getattr__(self, name: str):
        return getattr(self._model, name)


def make_predict_fn(model: Any, codec: Any):
    """unsw_lime.make_predict_fn, tagging calls as LIME-internal when the model is a CountingModel."""
    inner = _lime_make_predict_fn(model, codec)
    if not isinstance(model, CountingModel):
        return inner

    def predict_fn(Z):
        model._lime_depth += 1
        try:
            return inner(Z)
        finally:
            model._lime_depth -= 1

    return predict_fn


ACCOUNTING_COLUMNS = (
    "explanation_evaluations",
    "top_level_prediction_calls",
    "top_level_prediction_rows",
    "final_report_prediction_calls",
    "final_report_prediction_rows",
    "lime_internal_prediction_batches",
    "lime_internal_prediction_rows",
    "total_model_prediction_calls",
    "total_model_prediction_rows",
    "accounting_consistent",
)


def prediction_accounting(counts: dict[str, int], meter: CountingModel) -> dict[str, Any]:
    """Measured model usage of one search run.

    - explanation_evaluations: LIME explanations incl. the baseline (the budgeted quantity, unchanged).
    - top_level_prediction_*: the search's own single-row predictions (= counts["model_predictions"]).
    - final_report_prediction_*: predictions made only to report the final best candidate
      (previously uncounted; 1 for v0/v1/random control, 0 for multi-start which reads its cache).
    - lime_internal_prediction_*: batches/rows LIME sent to the model while explaining.
    - total_model_prediction_*: everything the underlying model actually computed.
    accounting_consistent checks one LIME batch per explanation and single-row top-level calls.
    """
    top = int(counts["model_predictions"])
    final_calls = int(meter.direct_calls - top)
    final_rows = int(meter.direct_rows - top)
    explanations = int(counts["explanation_calls"])
    return {
        "explanation_evaluations": explanations,
        "top_level_prediction_calls": top,
        "top_level_prediction_rows": top,
        "final_report_prediction_calls": final_calls,
        "final_report_prediction_rows": final_rows,
        "lime_internal_prediction_batches": int(meter.lime_batches),
        "lime_internal_prediction_rows": int(meter.lime_rows),
        "total_model_prediction_calls": int(meter.direct_calls + meter.lime_batches),
        "total_model_prediction_rows": int(meter.direct_rows + meter.lime_rows),
        "accounting_consistent": bool(
            0 <= final_calls <= 1 and final_rows == final_calls and meter.lime_batches == explanations
        ),
    }


def random_control_rng(seed: int, sample_position: int, model_key: str, budget_key: str, control_seed: int | None = None) -> np.random.Generator:
    """RNG of the random control.

    control_seed=None reproduces the original rule exactly: default_rng(seed + sample + sum(ord(model))),
    which ties the control to the LIME seed. With a control_seed the stream is
    default_rng([control_seed, sample, sum(ord(model)), sum(ord(budget))]), independent of the LIME
    seed and of the search-order seed.
    """
    if control_seed is None:
        return np.random.default_rng(seed + int(sample_position) + sum(ord(ch) for ch in model_key))
    return np.random.default_rng([int(control_seed), int(sample_position), sum(ord(ch) for ch in model_key), sum(ord(ch) for ch in budget_key)])


def row_from_json(text: str, reference_row: pd.Series) -> pd.Series:
    """Rebuild a candidate row from its saved JSON (numeric values were saved as numbers or strings)."""
    values = json.loads(text)
    row = reference_row.copy()
    for feature in FEATURE_ORDER:
        value = values[feature]
        if feature in CATEGORICAL_FEATURES:
            row[feature] = value
        else:
            row[feature] = float(value)
    return row


def best_so_far_checkpoints(
    history: list[dict[str, Any]],
    checkpoints: list[int],
    total_explanation_calls: int,
    original_json: str,
) -> list[dict[str, Any]]:
    """Best candidate found within the first c explanation calls (baseline included), for each c.

    Uses fresh scored rows only (cache hits never add information), in call order, with a strict
    improvement rule (earliest candidate wins ties), the same rule every search uses for its best.
    """
    fresh = [
        row for row in history
        if row.get("score") is not None and not row.get("cache_hit") and row.get("status") in {"accepted", "rejected", "evaluated"}
    ]
    records = []
    for checkpoint in sorted({int(c) for c in checkpoints}):
        best = None
        for row in fresh:
            if int(row["explanation_calls_so_far"]) > checkpoint:
                continue
            if best is None or float(row["score"]) > float(best["score"]):
                best = row
        records.append({
            "checkpoint": checkpoint,
            "budget_reached": bool(total_explanation_calls >= checkpoint),
            "explanation_calls_total": int(total_explanation_calls),
            "best_score": float(best["score"]) if best is not None and float(best["score"]) > 0.0 else 0.0,
            "explanation_calls_at_best": int(best["explanation_calls_so_far"]) if best is not None and float(best["score"]) > 0.0 else 1,
            "best_changed_features": best.get("changed_features", "") if best is not None and float(best["score"]) > 0.0 else "",
            "best_candidate_probability": best.get("candidate_probability") if best is not None and float(best["score"]) > 0.0 else None,
            "candidate_json": best["feature_values_json"] if best is not None and float(best["score"]) > 0.0 else original_json,
            "candidate_is_original": not (best is not None and float(best["score"]) > 0.0),
        })
    return records


def checkpoint_is_valid(strategy: str, checkpoint: int, run_budget: int) -> bool:
    """A checkpoint below the run's own budget stands in for a separate shorter run only when the
    method's decisions do not depend on the total budget (prefix equivalence, tested). Multi-start
    (K restarts) allocates fair-share caps from the total budget, so for it only checkpoint == run
    budget is valid. greedy_shuffled_v1 is the K=1 case (no allocation) and is prefix-equivalent."""
    if int(checkpoint) == int(run_budget):
        return True
    return strategy != STRATEGY_MULTISTART


def manifest_link(manifest_path: str | Path | None) -> dict[str, Any] | None:
    """Validate a written Step 6 manifest and return {path, sha256, protocol_version} for run metadata."""
    if manifest_path is None:
        return None
    from experiments import unsw_step6_manifest as manifest_module

    path = Path(manifest_path)
    if not path.is_absolute():
        path = ROOT / path
    manifest = json.loads(path.read_text(encoding="utf-8"))
    problems = manifest_module.validate_manifest(manifest)
    if problems:
        raise ValueError(f"manifest {path} is invalid: {problems}")
    try:
        shown = str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        shown = str(path)
    return {"path": shown, "sha256": file_sha256(path), "protocol_version": manifest["protocol_version"]}


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


# =============================================================================================
# Step 5b: shuffled and multi-start greedy search
# =============================================================================================
def explanation_config() -> dict[str, Any]:
    """The LIME settings an explanation depends on (besides model, row and seed)."""
    return {
        "num_samples": int(NUM_SAMPLES),
        "num_features": int(NUM_FEATURES),
        "explained_label": int(EXPLAINED_LABEL),
        "discretize_continuous": bool(DISCRETIZE_CONTINUOUS),
        "discretizer": str(DISCRETIZER),
        "feature_selection": str(FEATURE_SELECTION),
        "kernel_width": float(KERNEL_WIDTH),
        "distance_metric": "euclidean",
        "model_regressor": "LIME default (Ridge alpha=1)",
        "background": "X_subtrain via CategoricalCodec",
    }


def weights_digest(weights: dict[str, float]) -> str:
    payload = json.dumps({k: repr(float(v)) for k, v in sorted(weights.items())}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheContext:
    """Everything a cached score/prediction/validity verdict depends on.

    A cached entry is only valid for the same model, sample, budget policy, baseline explanation,
    LIME seed and explanation configuration. The objective is deterministic given these because the
    explainer is reseeded with the same LIME seed before every explanation.
    """

    model_key: str
    model_fingerprint: str
    sample_position: int
    budget_key: str
    lime_seed: int
    explanation_config_digest: str
    baseline_digest: str


def build_cache_context(
    model_key: str,
    model_fingerprint: str,
    sample_position: int,
    budget_key: str,
    lime_seed: int,
    baseline_weights: dict[str, float],
    config: dict[str, Any] | None = None,
) -> CacheContext:
    config = explanation_config() if config is None else config
    config_digest = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return CacheContext(
        model_key=str(model_key),
        model_fingerprint=str(model_fingerprint),
        sample_position=int(sample_position),
        budget_key=str(budget_key),
        lime_seed=int(lime_seed),
        explanation_config_digest=config_digest,
        baseline_digest=weights_digest(baseline_weights),
    )


class CandidateCache:
    """Per-search cache of candidate verdicts, keyed by the attackable feature values.

    Every access must present the CacheContext it was created with; a mismatch raises, so entries
    can never leak across models, samples, budgets, baselines, LIME seeds or LIME configurations.
    Entries: {"status": "scored"|"constraint"|"budget"|"benign", "score", "probability", "weights", "reason"}.
    """

    def __init__(self, context: CacheContext):
        self.context = context
        self._entries: dict[tuple[float, ...], dict[str, Any]] = {}

    def _check(self, context: CacheContext) -> None:
        if context != self.context:
            mismatched = [k for k, v in asdict(context).items() if asdict(self.context)[k] != v]
            raise ValueError(f"candidate cache context mismatch on {mismatched}; cached results are not reusable")

    def get(self, context: CacheContext, key: tuple[float, ...]) -> dict[str, Any] | None:
        self._check(context)
        return self._entries.get(key)

    def put(self, context: CacheContext, key: tuple[float, ...], entry: dict[str, Any]) -> None:
        self._check(context)
        if key in self._entries:
            raise ValueError("candidate already cached; a candidate must be evaluated at most once")
        self._entries[key] = entry

    def __len__(self) -> int:
        return len(self._entries)


def fair_share_cap(total_budget: int, calls_used: int, restarts_remaining: int, fair_share: bool = True) -> int:
    """Fresh explanation calls a restart may use.

    remaining = total_budget - calls_used (the baseline is already in calls_used). With fair share
    the restart gets floor(remaining / restarts_remaining) and any unused calls roll forward to later
    restarts; the last restart gets everything that is left. Without fair share it gets all remaining.
    """
    remaining = max(0, int(total_budget) - int(calls_used))
    if not fair_share or restarts_remaining <= 1:
        return remaining
    return remaining // int(restarts_remaining)


def restart_plans(
    order_seed: int,
    model_key: str,
    sample_position: int,
    budget_key: str,
    restarts: int,
    include_canonical_first: bool = False,
) -> list[dict[str, Any]]:
    """Reproducible, independent restart plans (seed, feature order, direction order per feature).

    Seeds come from numpy SeedSequence(entropy=[order_seed, sample, budget, model]).generate_state(K),
    so the plan of restart r does not depend on K (prefix-stable) and is independent of the LIME seed.
    """
    if restarts < 1:
        raise ValueError("restarts must be >= 1")
    entropy = [int(order_seed), int(sample_position), sum(ord(ch) for ch in budget_key), sum(ord(ch) for ch in model_key)]
    seeds = np.random.SeedSequence(entropy).generate_state(int(restarts), dtype=np.uint32)
    plans = []
    for restart_id, restart_seed in enumerate(seeds):
        restart_seed = int(restart_seed)
        if include_canonical_first and restart_id == 0:
            order = list(NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES)
            directions = {feature: [-1, 1] for feature in order}
            kind = "canonical"
        else:
            rng = np.random.default_rng(restart_seed)
            order = [str(f) for f in rng.permutation(np.array(NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES, dtype=object))]
            directions = {feature: [int(d) for d in rng.permutation([-1, 1])] for feature in order}
            kind = "shuffled"
        plans.append({
            "restart_id": restart_id,
            "restart_seed": restart_seed,
            "order_kind": kind,
            "feature_order": order,
            "direction_order": directions,
        })
    return plans


MULTISTART_EXTRA_KEYS = (
    "cache_hits",
    "cache_hits_scored",
    "cache_hits_rejected",
    "unique_candidates",
    "deferred_at_cap",
    "restarts_started",
    "restarts_completed",
)


def _run_targeted_search_multistart(
    model_key: str,
    sample_position: int,
    budget_key: str,
    split: dict[str, Any],
    max_explanation_evaluations: int = 10,
    seed: int = DEFAULT_SEED,
    model: Any = None,
    strategy: str = STRATEGY_MULTISTART,
    order_seed: int = DEFAULT_ORDER_SEED,
    restarts: int = DEFAULT_RESTARTS,
    fair_share: bool = True,
    include_canonical_first: bool = False,
    model_fingerprint: str | None = None,
) -> dict[str, Any]:
    """greedy_shuffled_v1 (K=1) and multistart_greedy_v1 (K restarts).

    Every restart starts from the ORIGINAL sample with its own seeded feature/direction order and
    climbs by strict first improvement over ITS OWN current score (all scores are measured against
    the original explanation). One global budget covers the baseline and all restarts; counters are
    never reset. A fresh explanation is only requested after the candidate passed the validator, the
    budget check and the malicious-prediction check. A restart stops at a local optimum (no improving
    proposal in a full sweep) or when its fair-share cap or the global budget is reached; unused
    calls roll forward and are never consumed artificially.
    """
    if strategy not in MULTISTART_STRATEGIES:
        raise ValueError(f"{strategy!r} is not a multi-start strategy")
    if strategy == STRATEGY_SHUFFLED:
        restarts = 1
    model = model if model is not None else load_xai_model(model_key, split)
    model_fingerprint = model_fingerprint or f"in-memory-object:{type(model).__name__}:{id(model)}"
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
    counts = _new_counts()  # explanation_calls = 1 (baseline), model_predictions = 1 (original)
    counts.update({key: 0 for key in MULTISTART_EXTRA_KEYS})
    context = build_cache_context(model_key, model_fingerprint, sample_position, budget_key, seed, baseline_weights)
    cache = CandidateCache(context)
    original_key = _row_key(original_row)
    # The original itself is known: objective(baseline, baseline) == 0 and its probability is known.
    cache.put(context, original_key, {"status": "scored", "score": 0.0, "probability": float(original_prob),
                                      "weights": baseline_weights, "reason": "original sample"})

    plans = restart_plans(order_seed, model_key, sample_position, budget_key, restarts, include_canonical_first)
    steps = choose_feature_steps(budget_key)
    history: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    restart_rows: list[dict[str, Any]] = []
    global_best = {"score": 0.0, "key": original_key, "row": original_row.copy(), "restart_id": None}

    def log(plan, sweep, status, reason, candidate, entry=None, cache_hit=False):
        history.append({
            "method": "targeted",
            "strategy": strategy,
            "model_key": model_key,
            "validation_position": int(sample_position),
            "budget": budget_key,
            "restart_id": plan["restart_id"],
            "restart_seed": plan["restart_seed"],
            "sweep": sweep,
            "status": status,
            "reason": reason,
            "cache_hit": bool(cache_hit),
            "candidate_id": f"{model_key}_{budget_key}_{sample_position}_{len(history):04d}",
            "score": (float(entry["score"]) if entry is not None and entry.get("score") is not None else None),
            "score_reference": "original_explanation",
            "candidate_probability": (float(entry["probability"]) if entry is not None and entry.get("probability") is not None else None),
            "changed_features": ";".join(_changed_features(original_row, candidate)),
            "explanation_calls_so_far": int(counts["explanation_calls"]),
            "model_predictions_so_far": int(counts["model_predictions"]),
            "feature_values_json": json.dumps(candidate.to_dict(), default=str),
        })

    for index, plan in enumerate(plans):
        if counts["explanation_calls"] >= max_explanation_evaluations:
            restart_rows.append({**_plan_summary(plan), "status": "not_started", "stopped_reason": "explanation_budget_exhausted",
                                 "fresh_call_cap": 0, "explanation_calls_used": 0, "model_predictions_used": 0,
                                 "proposals": 0, "accepted_steps": 0, "cache_hits": 0, "sweeps": 0,
                                 "best_score": 0.0, "best_changed_features": ""})
            continue
        counts["restarts_started"] += 1
        cap = fair_share_cap(max_explanation_evaluations, counts["explanation_calls"], len(plans) - index, fair_share)
        calls_at_start = counts["explanation_calls"]
        preds_at_start = counts["model_predictions"]
        proposals_at_start = counts["proposals_generated"]
        hits_at_start = counts["cache_hits"]
        current = original_row.copy()
        current_score = 0.0
        accepted_steps = 0
        sweeps = 0
        stopped_reason = None
        while stopped_reason is None:
            sweeps += 1
            improved = False
            for feature in plan["feature_order"]:
                for direction in plan["direction_order"][feature]:
                    for step_fraction in steps:
                        proposal = mutation_proposal_v1(current, original_row, feature, direction, step_fraction)
                        counts["proposals_generated"] += 1
                        if proposal.equals(current) or _row_key(proposal) == _row_key(current):
                            counts["rejected_no_change"] += 1
                            continue
                        key = _row_key(proposal)
                        entry = cache.get(context, key)
                        cache_hit = entry is not None
                        if cache_hit:
                            counts["cache_hits"] += 1
                            counts["cache_hits_scored" if entry["status"] == "scored" else "cache_hits_rejected"] += 1
                        else:
                            validation = validate_candidate(original_row.to_dict(), proposal.to_dict())
                            if not validation["valid"]:
                                entry = {"status": "constraint", "score": None, "probability": None, "weights": None,
                                         "reason": "; ".join(validation["violated_constraints"])}
                                counts["rejected_constraint"] += 1
                                counts["unique_candidates"] += 1
                                cache.put(context, key, entry)
                            else:
                                budget_ok, budget_reason = _candidate_matches_budget(original_row, proposal, budget_key)
                                if not budget_ok:
                                    entry = {"status": "budget", "score": None, "probability": None, "weights": None,
                                             "reason": budget_reason or "budget violation"}
                                    counts["rejected_budget"] += 1
                                    counts["unique_candidates"] += 1
                                    cache.put(context, key, entry)
                                else:
                                    # A fresh evaluation is needed: check the global budget and this restart's cap
                                    # BEFORE spending a model prediction or an explanation call.
                                    if counts["explanation_calls"] >= max_explanation_evaluations:
                                        stopped_reason = "explanation_budget_exhausted"
                                    elif counts["explanation_calls"] - calls_at_start >= cap:
                                        stopped_reason = "fair_share_cap_reached"
                                    if stopped_reason is not None:
                                        counts["deferred_at_cap"] += 1
                                        log(plan, sweeps, "deferred", f"not evaluated: {stopped_reason}", proposal)
                                        break
                                    probability = _safe_model_prob(model, proposal.to_frame().T)
                                    counts["model_predictions"] += 1
                                    counts["unique_candidates"] += 1
                                    if probability < 0.5:
                                        entry = {"status": "benign", "score": None, "probability": float(probability),
                                                 "weights": None, "reason": "prediction changed to benign"}
                                        counts["rejected_benign"] += 1
                                        cache.put(context, key, entry)
                                    else:
                                        weights = _call_explanation(model, proposal, codec, explainer, predict_fn, seed)
                                        counts["explanation_calls"] += 1
                                        counts["scored_proposals"] += 1
                                        score = float(objective_score(baseline_weights, weights))
                                        entry = {"status": "scored", "score": score, "probability": float(probability),
                                                 "weights": weights, "reason": "scored"}
                                        cache.put(context, key, entry)
                                        if score > global_best["score"]:
                                            global_best.update(score=score, key=key, row=proposal.copy(), restart_id=plan["restart_id"])
                                        curve.append({
                                            "method": "targeted", "strategy": strategy, "model_key": model_key,
                                            "validation_position": int(sample_position), "budget": budget_key,
                                            "restart_id": plan["restart_id"],
                                            "explanation_calls": int(counts["explanation_calls"]),
                                            "score": score, "best_so_far": float(global_best["score"]),
                                        })
                        if entry["status"] != "scored":
                            log(plan, sweeps, "rejected", entry["reason"], proposal, entry, cache_hit)
                            continue
                        if entry["score"] > current_score:
                            current = proposal.copy()
                            current_score = float(entry["score"])
                            accepted_steps += 1
                            counts["accepted_improvements"] += 1
                            log(plan, sweeps, "accepted", "improved this restart's objective vs original explanation"
                                + (" (cached score)" if cache_hit else ""), proposal, entry, cache_hit)
                            improved = True
                            break
                        if current_score > 0.0 and entry["score"] == current_score:
                            counts["ties_not_accepted"] += 1
                            log(plan, sweeps, "rejected", "tie with restart's current objective (strict improvement required)", proposal, entry, cache_hit)
                        else:
                            log(plan, sweeps, "rejected", f"score={entry['score']:.6f} did not improve restart's objective", proposal, entry, cache_hit)
                    if improved or stopped_reason is not None:
                        break
                if improved or stopped_reason is not None:
                    break
            if stopped_reason is None and not improved:
                stopped_reason = "local_optimum_full_sweep_without_improvement"
        if stopped_reason == "local_optimum_full_sweep_without_improvement":
            counts["restarts_completed"] += 1
        restart_rows.append({
            **_plan_summary(plan),
            "status": "ran",
            "stopped_reason": stopped_reason,
            "fresh_call_cap": int(cap),
            "explanation_calls_used": int(counts["explanation_calls"] - calls_at_start),
            "model_predictions_used": int(counts["model_predictions"] - preds_at_start),
            "proposals": int(counts["proposals_generated"] - proposals_at_start),
            "accepted_steps": int(accepted_steps),
            "cache_hits": int(counts["cache_hits"] - hits_at_start),
            "sweeps": int(sweeps),
            "best_score": float(current_score),
            "best_changed_features": ";".join(_changed_features(original_row, current)),
        })

    best_entry = cache.get(context, global_best["key"])
    best_row = global_best["row"]
    if counts["explanation_calls"] >= max_explanation_evaluations:
        overall_stop = "explanation_budget_exhausted"
    elif counts["restarts_started"] == len(plans):
        overall_stop = "all_restarts_finished"
    else:
        overall_stop = "explanation_budget_exhausted"
    return {
        "model_key": model_key,
        "sample_position": int(sample_position),
        "budget": budget_key,
        "original_probability": float(original_prob),
        # From the cache: no extra, uncounted model call.
        "best_candidate_probability": float(best_entry["probability"]),
        "best_score": float(global_best["score"]),
        "best_restart_id": global_best["restart_id"],
        "best_candidate": best_row.to_dict(),
        "best_changed_features": _changed_features(original_row, best_row),
        "original_feature_values_json": json.dumps(original_row.to_dict(), default=str),
        "modified_feature_values_json": json.dumps(best_row.to_dict(), default=str),
        "search_history": history,
        "rejections": [row for row in history if row["status"] != "accepted"],
        "evaluation_counts": counts,
        "baseline_weights": baseline_weights,
        "final_weights": best_entry["weights"],
        "constraint_valid": bool(validate_candidate(original_row.to_dict(), best_row.to_dict())["valid"]),
        "strategy": strategy,
        "score_reference": "original_explanation",
        "stopped_reason": overall_stop,
        "order_seed": int(order_seed),
        "restarts_requested": int(len(plans)),
        "fair_share": bool(fair_share),
        "include_canonical_first": bool(include_canonical_first),
        "unused_explanation_calls": int(max(0, max_explanation_evaluations - counts["explanation_calls"])),
        "restart_plans": plans,
        "restarts": restart_rows,
        "best_so_far_curve": curve,
        "cache_context": asdict(context),
        "cache_size": len(cache),
    }


def _plan_summary(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "restart_id": plan["restart_id"],
        "restart_seed": plan["restart_seed"],
        "order_kind": plan["order_kind"],
        "feature_order": ";".join(plan["feature_order"]),
        "direction_order": ";".join(
            f"{feature}:{'+' if plan['direction_order'][feature][0] > 0 else '-'}{'+' if plan['direction_order'][feature][1] > 0 else '-'}"
            for feature in plan["feature_order"]
        ),
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
    order_seed: int = DEFAULT_ORDER_SEED,
    restarts: int = DEFAULT_RESTARTS,
    fair_share: bool = True,
    include_canonical_first: bool = False,
    model_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Dispatch to the legacy (v0), corrected (v1), shuffled or multi-start greedy search.

    order_seed/restarts/fair_share/include_canonical_first/model_fingerprint only apply to the
    Step 5b strategies (greedy_shuffled_v1, multistart_greedy_v1).
    """
    if strategy in MULTISTART_STRATEGIES:
        return _run_targeted_search_multistart(
            model_key, sample_position, budget_key, split, max_explanation_evaluations, seed, model,
            strategy=strategy, order_seed=order_seed, restarts=restarts, fair_share=fair_share,
            include_canonical_first=include_canonical_first, model_fingerprint=model_fingerprint,
        )
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
    control_seed: int | None = None,
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

    rng = random_control_rng(seed, sample_position, model_key, budget_key, control_seed)
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
            "control_seed": control_seed,
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
        "control_seed": control_seed,
        "rng_seed_rule": (
            "legacy: default_rng(lime_seed + sample + sum(ord(model)))" if control_seed is None
            else "default_rng([control_seed, sample, sum(ord(model)), sum(ord(budget))])"
        ),
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
    suffix = 1
    while True:
        try:
            run_dir.mkdir(parents=True, exist_ok=False)  # never reuse an existing directory
            return run_dir
        except FileExistsError:
            run_dir = root / f"run_{timestamp}_{suffix:02d}"
            suffix += 1


def _accounting_row(counts: dict[str, int]) -> dict[str, int]:
    return {key: int(counts.get(key, 0)) for key in ACCOUNTING_KEYS}


def _control_strategy_for(strategies: list[str]) -> str:
    """The random control is unchanged: legacy mutation for v0 runs, v1 mutation otherwise."""
    if all(s == STRATEGY_V0 for s in strategies):
        return STRATEGY_V0
    if any(s == STRATEGY_V0 for s in strategies):
        raise ValueError("greedy_incremental_v0 cannot be combined with other strategies in one run (different random controls)")
    return STRATEGY_V1


def _curve_from_history(history: list[dict[str, Any]], method: str, strategy: str, fresh_only: bool = True) -> list[dict[str, Any]]:
    """Best-so-far vs explanation calls from a history (fresh scored rows only)."""
    rows, best = [], 0.0
    for row in history:
        if row.get("score") is None or (fresh_only and row.get("cache_hit")):
            continue
        if row.get("status") not in {"accepted", "rejected", "evaluated"}:
            continue
        best = max(best, float(row["score"]))
        rows.append({
            "method": method, "strategy": strategy, "model_key": row["model_key"],
            "validation_position": row["validation_position"], "budget": row["budget"],
            "restart_id": row.get("restart_id"),
            "explanation_calls": int(row["explanation_calls_so_far"]),
            "score": float(row["score"]), "best_so_far": best,
        })
    return rows


def run_experiment(
    models: list[str] | None = None,
    sample_positions: list[int] | None = None,
    budgets: list[str] | None = None,
    max_explanation_evaluations: int = 10,
    seed: int = DEFAULT_SEED,
    output_dir: str = str(OUTPUT_ROOT),
    strategy: str = DEFAULT_STRATEGY,
    max_random_attempts: int | None = None,
    strategies: list[str] | None = None,
    order_seed: int = DEFAULT_ORDER_SEED,
    restarts: int = DEFAULT_RESTARTS,
    fair_share: bool = True,
    include_canonical_first: bool = False,
    control_seed: int | None = None,
    checkpoints: list[int] | None = None,
    manifest_path: str | None = None,
) -> dict[str, Any]:
    manifest_info = manifest_link(manifest_path)  # validated before any work starts
    strategy_list = list(strategies) if strategies else [strategy]
    checkpoint_list = sorted({int(c) for c in (checkpoints or [])} | {int(max_explanation_evaluations)})
    if any(c < 1 or c > max_explanation_evaluations for c in checkpoint_list):
        raise ValueError(f"checkpoints must lie in [1, {max_explanation_evaluations}]: {checkpoint_list}")
    for name in strategy_list:
        if name not in STRATEGIES:
            raise ValueError(f"Unknown strategy {name!r}; expected one of {STRATEGIES}")
    if len(set(strategy_list)) != len(strategy_list):
        raise ValueError("duplicate strategies requested")
    control_strategy = _control_strategy_for(strategy_list)
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

    model_sha = {
        "random_forest": file_sha256(MODEL_DIR / "random_forest_validation_pipeline.joblib"),
        "xgboost": file_sha256(MODEL_DIR / "xgboost_validation_pipeline.joblib"),
    }

    run_dir = _make_run_dir(output_dir)
    run_results: list[dict[str, Any]] = []
    control_results: list[dict[str, Any]] = []
    proposal_history: list[dict[str, Any]] = []
    restart_table: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []

    for model_key in model_names:
        model = load_xai_model(model_key, split)  # loaded once, shared by all methods
        for sample_position in target_positions:
            for budget_key in budget_names:
                for strategy_name in strategy_list:
                    meter = CountingModel(model)  # fresh meter per method run
                    t0 = time.perf_counter()
                    targeted = run_targeted_search(
                        model_key=model_key,
                        sample_position=sample_position,
                        budget_key=budget_key,
                        split=split,
                        max_explanation_evaluations=max_explanation_evaluations,
                        seed=seed,
                        strategy=strategy_name,
                        model=meter,
                        order_seed=order_seed,
                        restarts=restarts,
                        fair_share=fair_share,
                        include_canonical_first=include_canonical_first,
                        model_fingerprint=f"sha256:{model_sha[model_key]}",
                    )
                    elapsed = time.perf_counter() - t0
                    targeted_counts = targeted["evaluation_counts"]
                    row = {
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
                    }
                    if targeted["strategy"] != STRATEGY_V0:
                        row.update(_accounting_row(targeted_counts))
                    if targeted["strategy"] in MULTISTART_STRATEGIES:
                        row.update({key: int(targeted_counts[key]) for key in MULTISTART_EXTRA_KEYS})
                        row.update({
                            "order_seed": targeted["order_seed"],
                            "restarts_requested": targeted["restarts_requested"],
                            "fair_share": targeted["fair_share"],
                            "include_canonical_first": targeted["include_canonical_first"],
                            "best_restart_id": targeted["best_restart_id"],
                            "unused_explanation_calls": targeted["unused_explanation_calls"],
                        })
                        for restart_row in targeted["restarts"]:
                            restart_table.append({
                                "model_key": model_key, "sample_position": int(sample_position), "budget": budget_key,
                                "strategy": targeted["strategy"], "order_seed": targeted["order_seed"], **restart_row,
                            })
                        curves.extend(targeted["best_so_far_curve"])
                    else:
                        curves.extend(_curve_from_history(targeted["search_history"], "targeted", targeted["strategy"]))
                    row.update(prediction_accounting(targeted_counts, meter))
                    row.update({"runtime_seconds": round(elapsed, 3), "run_type": "targeted"})
                    run_results.append(row)
                    proposal_history.extend(targeted["search_history"])
                    if targeted["strategy"] != STRATEGY_V0:  # v0 scores are incremental: no checkpoints
                        for record in best_so_far_checkpoints(
                            targeted["search_history"], checkpoint_list, targeted_counts["explanation_calls"],
                            targeted["original_feature_values_json"],
                        ):
                            checkpoint_rows.append({
                                "method": "targeted", "strategy": targeted["strategy"], "model_key": model_key,
                                "sample_position": int(sample_position), "budget": budget_key,
                                "lime_seed": int(seed),
                                "order_seed": targeted.get("order_seed"),
                                "restarts": targeted.get("restarts_requested"),
                                "control_seed": None,
                                "max_explanation_evaluations": int(max_explanation_evaluations),
                                "checkpoint_valid": checkpoint_is_valid(targeted["strategy"], record["checkpoint"], max_explanation_evaluations),
                                "prefix_shortcut": bool(record["checkpoint"] < max_explanation_evaluations),
                                **record,
                            })

                # Random control: once per configuration, unchanged, shared by every targeted strategy.
                control_meter = CountingModel(model)
                t1 = time.perf_counter()
                control = random_control_search(
                    model_key=model_key,
                    sample_position=sample_position,
                    budget_key=budget_key,
                    split=split,
                    max_explanation_evaluations=max_explanation_evaluations,
                    seed=seed,
                    strategy=control_strategy,
                    max_attempts=random_attempt_cap,
                    model=control_meter,
                    control_seed=control_seed,
                )
                t2 = time.perf_counter()
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
                    "control_seed": control["control_seed"],
                    "rng_seed_rule": control["rng_seed_rule"],
                    **_accounting_row(control_counts),
                    **prediction_accounting(control_counts, control_meter),
                    "runtime_seconds": round(t2 - t1, 3),
                    "run_type": "random_control",
                })
                proposal_history.extend(control["history"])
                original_json = json.dumps(split["X_val"].iloc[[int(sample_position)]].iloc[0].to_dict(), default=str)
                for record in best_so_far_checkpoints(control["history"], checkpoint_list, control_counts["explanation_calls"], original_json):
                    checkpoint_rows.append({
                        "method": "random_control", "strategy": control["strategy"], "model_key": model_key,
                        "sample_position": int(sample_position), "budget": budget_key,
                        "lime_seed": int(seed), "order_seed": None, "restarts": None,
                        "control_seed": control["control_seed"],
                        "max_explanation_evaluations": int(max_explanation_evaluations),
                        "checkpoint_valid": checkpoint_is_valid(control["strategy"], record["checkpoint"], max_explanation_evaluations),
                        "prefix_shortcut": bool(record["checkpoint"] < max_explanation_evaluations),
                        **record,
                    })
                curves.extend(_curve_from_history(control["history"], "random_control", control["strategy"]))

    pd.DataFrame(run_results).to_csv(run_dir / "targeted_results.csv", index=False)
    pd.DataFrame(control_results).to_csv(run_dir / "random_control_results.csv", index=False)
    pd.DataFrame(proposal_history).to_csv(run_dir / "proposal_history.csv", index=False)
    pd.DataFrame(curves).to_csv(run_dir / "best_so_far.csv", index=False)
    pd.DataFrame(checkpoint_rows).to_csv(run_dir / "checkpoints.csv", index=False)
    if restart_table:
        pd.DataFrame(restart_table).to_csv(run_dir / "restarts.csv", index=False)

    multistart_used = any(name in MULTISTART_STRATEGIES for name in strategy_list)
    metadata = {
        "run_id": run_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy_list[0] if len(strategy_list) == 1 else strategy_list,
        "strategies": strategy_list,
        "control_strategy": control_strategy,
        "strategy_description": {
            STRATEGY_V0: "LEGACY: targeted proposals scored against the previously accepted explanation",
            STRATEGY_V1: "greedy first-improvement, canonical feature order, scored against the original explanation",
            STRATEGY_SHUFFLED: "one greedy climb with a seeded shuffled feature order and per-feature direction order",
            STRATEGY_MULTISTART: "K seeded shuffled greedy climbs from the original sample, one shared explanation budget, fair-share caps with roll-over, shared candidate cache",
        },
        "random_control_description": "best-of-N random control, unchanged; run once per configuration and shared by all targeted strategies",
        "models": model_names,
        "sample_positions": [int(pos) for pos in target_positions],
        "budgets": budget_names,
        "max_explanation_evaluations": int(max_explanation_evaluations),
        "max_random_attempts": int(random_attempt_cap),
        "control_seed": control_seed,
        "random_control_rng_rule": (
            "legacy: default_rng(lime_seed + sample + sum(ord(model)))" if control_seed is None
            else "default_rng([control_seed, sample, sum(ord(model)), sum(ord(budget))])"
        ),
        "manifest": manifest_info,
        "checkpoints": checkpoint_list,
        "checkpoint_validity": "multi-start checkpoints below the run budget are marked checkpoint_valid=False (fair-share caps depend on the budget); other methods are prefix-equivalent",
        "checkpoint_rule": "best fresh-scored candidate within the first c explanation calls (baseline included), strict improvement, earliest wins ties; v0 excluded (incremental scores)",
        "prediction_accounting": {
            "columns": list(ACCOUNTING_COLUMNS),
            "definitions": prediction_accounting.__doc__.strip(),
            "lime_rows_per_explanation": int(NUM_SAMPLES),
        },
        "search_seed": int(seed),
        "lime_seed": int(seed),
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
            "keys": list(ACCOUNTING_KEYS) + (list(MULTISTART_EXTRA_KEYS) if multistart_used else []),
            "identities": [
                "explanation_calls == 1 + scored_proposals (1 = baseline explanation)",
                "model_predictions == 1 + scored_proposals + rejected_benign",
            ] + ([
                "multi-start: proposals_generated == rejected_no_change + cache_hits + rejected_constraint + rejected_budget + rejected_benign + scored_proposals + deferred_at_cap",
                "multi-start: unique_candidates == rejected_constraint + rejected_budget + rejected_benign + scored_proposals",
            ] if multistart_used else []),
            "explanation_budget": "explanation_calls <= max_explanation_evaluations for every method (baseline included); multi-start shares ONE budget across restarts",
            "random_attempt_cap": "random control stops after max_random_attempts proposal attempts",
            "duplicates": "v1: skipped; multi-start: reused from the candidate cache (no explanation, no model call); random control: v1 skips duplicates",
        },
        "multistart": {
            "order_seed": int(order_seed),
            "restarts": int(restarts),
            "greedy_shuffled_v1_restarts": 1,
            "fair_share": bool(fair_share),
            "fair_share_rule": "restart r may use floor((budget - calls_used) / restarts_left) fresh explanation calls; the last restart gets the rest; unused calls roll forward",
            "include_canonical_first": bool(include_canonical_first),
            "start_point": "every restart starts from the original sample",
            "restart_seed_derivation": "numpy SeedSequence([order_seed, sample_position, sum(ord(budget)), sum(ord(model_key))]).generate_state(K, uint32); prefix-stable in K; independent of the LIME seed",
            "order_and_direction": "default_rng(restart_seed).permutation(features), then one permutation of (-1, +1) per feature in that order; step sizes unchanged",
            "stopping": "a restart stops at a local optimum (full sweep without strict improvement of its own current score), at its fair-share cap, or when the global budget is used; remaining calls are never consumed artificially",
            "cache_validity": "cache entries are valid only for the same model (file SHA-256), sample, budget policy, baseline explanation (weights digest), LIME seed and explanation configuration; any mismatch raises",
            "explanation_config": explanation_config(),
        } if multistart_used else None,
        "mutation_policy": (
            "legacy mutation_proposal: step = fraction*max(1,|current|); rounding for all features except "
            "dur/rate/sload/dload/sinpkt/dinpkt/sjit/djit"
            if control_strategy == STRATEGY_V0
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
        "model_sha256": model_sha,
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
        "strategy": metadata["strategy"],
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
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="LIME seed (fixed for every explanation); also seeds the random control.")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_ROOT), help="Directory under which a timestamped run folder is created.")
    parser.add_argument("--strategy", choices=list(STRATEGIES), default=DEFAULT_STRATEGY, help=f"Search strategy; {STRATEGY_V0} reproduces the original incremental-scoring baseline.")
    parser.add_argument("--strategies", nargs="+", choices=list(STRATEGIES), default=None, help="Several targeted strategies in one run (overrides --strategy); the random control runs once per configuration.")
    parser.add_argument("--max-random-attempts", type=int, default=None, help=f"Hard cap on random-control proposal attempts (default {DEFAULT_RANDOM_ATTEMPT_MULTIPLIER} x max explanation evaluations).")
    parser.add_argument("--order-seed", type=int, default=DEFAULT_ORDER_SEED, help="Seed for shuffled feature/direction orders (independent of the LIME seed).")
    parser.add_argument("--restarts", type=int, default=DEFAULT_RESTARTS, help="Number of restarts K for multistart_greedy_v1 (greedy_shuffled_v1 always uses 1).")
    parser.add_argument("--no-fair-share", action="store_true", help="Disable fair-share allocation across restarts.")
    parser.add_argument("--include-canonical-first", action="store_true", help="Restart 0 uses the canonical feature order and (-1, +1) directions.")
    parser.add_argument("--control-seed", type=int, default=None, help="Seed for the random control, independent of the LIME and order seeds (omit for the original LIME-seed-based rule).")
    parser.add_argument("--checkpoints", nargs="+", type=int, default=None, help="Also record the best candidate at these explanation-call counts (the run budget is always recorded).")
    parser.add_argument("--manifest", type=str, default=None, help="Path of a written Step 6 manifest to validate and link from run_metadata.json.")
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
        strategies=args.strategies,
        order_seed=args.order_seed,
        restarts=args.restarts,
        fair_share=not args.no_fair_share,
        include_canonical_first=args.include_canonical_first,
        control_seed=args.control_seed,
        checkpoints=args.checkpoints,
        manifest_path=args.manifest,
    )
    elapsed = time.perf_counter() - start
    print(f"Run directory: {result['run_dir']}")
    print(f"Elapsed seconds: {elapsed:.3f}")
    print(json.dumps(result["summary"]["evaluation_counts_total"], sort_keys=True))


if __name__ == "__main__":
    main()
