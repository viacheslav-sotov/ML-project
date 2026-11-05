import json

import numpy as np
import pandas as pd
import pytest

import experiments.unsw_targeted_spoofing as ts
from experiments.unsw_targeted_spoofing import (
    BUDGET_POLICY,
    STRATEGY_V0,
    STRATEGY_V1,
    _candidate_matches_budget,
    _relative_change,
    _row_key,
    _spearman_rho,
    mutation_proposal,
    mutation_proposal_v1,
    objective_score,
    random_control_search,
    run_targeted_search,
)
from src.unsw.constraints import FEATURE_ORDER, validate_candidate
from src.unsw.xai_setup import get_xai_data

SAMPLE = 2941


@pytest.fixture(scope="module")
def split():
    return get_xai_data()


@pytest.fixture(scope="module")
def original(split):
    return split["X_val"].iloc[[SAMPLE]].iloc[0].copy()


# ---------------------------------------------------------------------------------------------
# Deterministic fakes: no LIME, no saved models. The fake explanation gives every feature a base
# weight that increases with its position in FEATURE_ORDER, plus a boost proportional to how far an
# attackable feature moved from the original, so larger cumulative edits move the ranking further.
# ---------------------------------------------------------------------------------------------
class FakeModel:
    def __init__(self, original_row: pd.Series, benign_if_changed: bool = False, prob: float = 0.99):
        self.original = original_row
        self.benign_if_changed = benign_if_changed
        self.prob = prob

    def predict_proba(self, X):
        row = X.iloc[0]
        changed = any(
            float(row[f]) != float(self.original[f]) for f in ts.NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES
        )
        p = 0.2 if (self.benign_if_changed and changed) else self.prob
        return np.array([[1.0 - p, p]])


class FakeCodec:
    def __init__(self, X):
        pass

    def encode(self, X):
        return None


def make_fake_explain(original_row: pd.Series):
    base = {feature: float(i + 1) for i, feature in enumerate(FEATURE_ORDER)}

    def fake_explain(model, row, codec, explainer, predict_fn, seed):
        weights = dict(base)
        for feature in ts.NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES:
            weights[feature] += 2000.0 * _relative_change(float(original_row[feature]), float(row[feature]))
        return weights

    return fake_explain


@pytest.fixture
def fakes(monkeypatch, original):
    fake_explain = make_fake_explain(original)
    monkeypatch.setattr(ts, "CategoricalCodec", FakeCodec)
    monkeypatch.setattr(ts, "build_explainer", lambda encoded, codec: None)
    monkeypatch.setattr(ts, "make_predict_fn", lambda model, codec: None)
    monkeypatch.setattr(ts, "_call_explanation", fake_explain)

    def no_model_loading(*args, **kwargs):
        raise AssertionError("tests must pass a fake model explicitly")

    monkeypatch.setattr(ts, "load_xai_model", no_model_loading)
    return fake_explain


def _row_from_json(text: str) -> pd.Series:
    return pd.Series(json.loads(text))


# ---------------------------------------------------------------------------------------------
# Budget, objective and Spearman (existing tests, kept)
# ---------------------------------------------------------------------------------------------
def test_budget_policy_a_and_b_are_enforced(split):
    original = split["X_val"].iloc[[SAMPLE]].iloc[0].copy()
    candidate = original.copy()
    candidate["dur"] = original["dur"] * 1.08
    candidate["spkts"] = original["spkts"] + 1
    candidate["sbytes"] = original["sbytes"] + 10
    candidate["sttl"] = original["sttl"] + 1
    candidate["dttl"] = 10
    candidate["sinpkt"] = original["sinpkt"] * 1.5

    ok_a, reason_a = _candidate_matches_budget(original, candidate, "A")
    ok_b, reason_b = _candidate_matches_budget(original, candidate, "B")
    assert ok_a is False
    assert ok_b is False
    assert reason_a is not None

    candidate = original.copy()
    candidate["dur"] = original["dur"] * 1.05
    candidate["sbytes"] = original["sbytes"] * 1.08
    candidate["sttl"] = original["sttl"] * 1.05
    ok_a, reason_a = _candidate_matches_budget(original, candidate, "A")
    assert ok_a is True


def test_objective_is_in_unit_interval_and_reproducible(split):
    original = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    candidate = {"a": 3.0, "b": 2.0, "c": 1.0, "d": 4.0}
    score = objective_score(original, candidate)
    assert 0.0 <= score <= 1.0

    score2 = objective_score(original, candidate)
    assert score == pytest.approx(score2)


def test_objective_is_zero_for_identical_42_feature_explanations():
    weights = {feature: float(i) for i, feature in enumerate(FEATURE_ORDER)}
    assert objective_score(weights, dict(weights)) == pytest.approx(0.0)


def test_spearman_missing_rank_policy_is_defined(split):
    original = {"a": 10.0, "b": 5.0, "c": 7.0}
    candidate = {"a": 10.0, "c": 3.0}
    rho = _spearman_rho(original, candidate)
    assert -1.0 <= rho <= 1.0
    assert np.isfinite(rho)


def test_large_dur_change_is_valid_for_validator_but_violates_budget_a(split):
    # Formerly named test_targeted_search_rejects_benign_prediction; it tests a budget rejection.
    original = split["X_val"].iloc[[SAMPLE]].iloc[0].copy()
    candidate = original.copy()
    candidate["dur"] = 0.0
    candidate["spkts"] = 1
    validation = validate_candidate(original.to_dict(), candidate.to_dict())
    assert validation["valid"] is True
    ok_a, _ = _candidate_matches_budget(original, candidate, "A")
    assert ok_a is False


# ---------------------------------------------------------------------------------------------
# Mutation: floating-point drift, rounding, synack/ackdat
# ---------------------------------------------------------------------------------------------
def test_v1_up_then_down_returns_exact_original_value(original):
    up = mutation_proposal_v1(original, original, "dur", +1, 0.02)
    assert float(up["dur"]) != float(original["dur"])
    back = mutation_proposal_v1(up, original, "dur", -1, 0.02)
    assert back["dur"] == original["dur"]
    assert _row_key(back) == _row_key(original)
    ok, _ = _candidate_matches_budget(original, back, "A")
    assert ok
    assert ts._changed_features(original, back) == []


def test_legacy_mutation_still_drifts_documenting_v0_behaviour(original):
    up = mutation_proposal(original, "dur", +1, 0.02)
    back = mutation_proposal(up, "dur", -1, 0.02)
    # The legacy step unit is max(1, |current|), and no snap-back is applied.
    assert float(back["dur"]) != float(original["dur"])
    assert abs(float(back["dur"]) - float(original["dur"])) < 1e-12


def test_v1_does_not_round_synack_ackdat_and_validator_enforces_tcprtt(original):
    for feature in ("synack", "ackdat"):
        proposal = mutation_proposal_v1(original, original, feature, +1, 0.05)
        expected = float(original[feature]) + 0.05 * max(1.0, abs(float(original[feature])))
        assert float(proposal[feature]) == pytest.approx(expected)
        assert not float(proposal[feature]).is_integer()
        # tcprtt is derived (not attackable), so a lone synack/ackdat edit breaks tcprtt = synack + ackdat.
        validation = validate_candidate(original.to_dict(), proposal.to_dict())
        assert validation["valid"] is False
        assert any("tcprtt" in v for v in validation["violated_constraints"])
        # The legacy mutation rounded these features, so they could never move from 0.
        legacy = mutation_proposal(original, feature, +1, 0.05)
        assert float(legacy[feature]) == float(round(float(legacy[feature])))


def test_v1_still_rounds_integer_features(original):
    proposal = mutation_proposal_v1(original, original, "sbytes", +1, 0.02)
    assert float(proposal["sbytes"]).is_integer()
    assert float(proposal["sbytes"]) == float(round(float(original["sbytes"]) * 1.02))


# ---------------------------------------------------------------------------------------------
# v1 targeted scoring and shared accounting (fake explainer)
# ---------------------------------------------------------------------------------------------
def _assert_accounting_identities(counts):
    assert counts["explanation_calls"] == 1 + counts["scored_proposals"]
    assert counts["model_predictions"] == 1 + counts["scored_proposals"] + counts["rejected_benign"]


def test_v1_scores_every_candidate_against_original_explanation(split, original, fakes):
    result = run_targeted_search(
        "random_forest", SAMPLE, "A", split, max_explanation_evaluations=20, seed=11,
        strategy=STRATEGY_V1, model=FakeModel(original),
    )
    baseline = fakes(None, original, None, None, None, 11)
    scored = [row for row in result["search_history"] if row["score"] is not None]
    assert scored, "the fake landscape must produce scored proposals"
    for row in scored:
        assert row["score_reference"] == "original_explanation"
        expected = objective_score(baseline, fakes(None, _row_from_json(row["feature_values_json"]), None, None, None, 11))
        assert row["score"] == pytest.approx(expected, abs=1e-12)
    assert result["best_score"] == pytest.approx(
        objective_score(baseline, result["final_weights"]), abs=1e-12
    )
    assert result["best_score"] > 0.0
    assert result["strategy"] == STRATEGY_V1


def test_v1_best_score_is_monotone_over_accepted_steps(split, original, fakes):
    result = run_targeted_search(
        "random_forest", SAMPLE, "B", split, max_explanation_evaluations=25, seed=11,
        strategy=STRATEGY_V1, model=FakeModel(original),
    )
    accepted = [row["score"] for row in result["search_history"] if row["status"] == "accepted"]
    assert accepted == sorted(accepted)
    assert len(set(accepted)) == len(accepted)


def test_targeted_and_random_share_objective_reference_and_accounting(split, original, fakes):
    model = FakeModel(original)
    targeted = run_targeted_search("random_forest", SAMPLE, "A", split, 12, 11, strategy=STRATEGY_V1, model=model)
    control = random_control_search("random_forest", SAMPLE, "A", split, 12, 11, strategy=STRATEGY_V1, model=model)

    assert set(targeted["evaluation_counts"]) == set(control["evaluation_counts"]) == set(ts.ACCOUNTING_KEYS)
    _assert_accounting_identities(targeted["evaluation_counts"])
    _assert_accounting_identities(control["evaluation_counts"])
    assert targeted["score_reference"] == control["score_reference"] == "original_explanation"

    baseline = fakes(None, original, None, None, None, 11)
    for row in control["history"]:
        if row["status"] == "evaluated":
            expected = objective_score(baseline, fakes(None, _row_from_json(row["feature_values_json"]), None, None, None, 11))
            assert row["score"] == pytest.approx(expected, abs=1e-12)
    assert targeted["baseline_weights"] == control["baseline_weights"]


@pytest.mark.parametrize("cap", [1, 2, 7])
def test_explanation_budget_is_a_hard_cap_for_both_methods(split, original, fakes, cap):
    model = FakeModel(original)
    targeted = run_targeted_search("random_forest", SAMPLE, "A", split, cap, 11, strategy=STRATEGY_V1, model=model)
    control = random_control_search("random_forest", SAMPLE, "A", split, cap, 11, strategy=STRATEGY_V1, model=model)
    assert targeted["evaluation_counts"]["explanation_calls"] <= cap
    assert control["evaluation_counts"]["explanation_calls"] <= cap
    if cap == 1:
        assert targeted["evaluation_counts"]["scored_proposals"] == 0
        assert targeted["stopped_reason"] == "explanation_budget_exhausted"
        assert targeted["best_score"] == 0.0


def test_v1_never_explains_the_same_candidate_twice(split, original, fakes):
    result = run_targeted_search(
        "random_forest", SAMPLE, "A", split, max_explanation_evaluations=40, seed=11,
        strategy=STRATEGY_V1, model=FakeModel(original),
    )
    scored_keys = [
        _row_key(_row_from_json(row["feature_values_json"]))
        for row in result["search_history"]
        if row["score"] is not None
    ]
    assert len(scored_keys) == len(set(scored_keys))
    assert _row_key(original) not in scored_keys


def test_targeted_search_rejects_benign_prediction(split, original, fakes):
    result = run_targeted_search(
        "random_forest", SAMPLE, "A", split, max_explanation_evaluations=10, seed=11,
        strategy=STRATEGY_V1, model=FakeModel(original, benign_if_changed=True),
    )
    counts = result["evaluation_counts"]
    assert counts["rejected_benign"] > 0
    assert counts["scored_proposals"] == 0
    assert counts["explanation_calls"] == 1
    assert result["best_score"] == 0.0
    assert ts._changed_features(original, pd.Series(result["best_candidate"])) == []
    assert any(row["reason"] == "prediction changed to benign" for row in result["search_history"])
    assert result["stopped_reason"] == "local_optimum_full_sweep_without_improvement"
    _assert_accounting_identities(counts)


# ---------------------------------------------------------------------------------------------
# Random control: attempt limit and logging
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("strategy", [STRATEGY_V0, STRATEGY_V1])
def test_random_control_terminates_at_attempt_limit_and_logs_every_attempt(split, original, fakes, strategy):
    result = random_control_search(
        "random_forest", SAMPLE, "A", split, max_explanation_evaluations=5, seed=11,
        strategy=strategy, max_attempts=25, model=FakeModel(original, benign_if_changed=True),
    )
    assert result["stopped_reason"] == "attempt_limit_reached"
    assert result["attempts"] == 25
    assert len(result["history"]) == 25
    assert result["evaluation_counts"]["explanation_calls"] == 1
    assert result["evaluation_counts"]["proposals_generated"] == 25
    reasons = {row["reason"] for row in result["history"]}
    assert "prediction changed to benign" in reasons
    _assert_accounting_identities(result["evaluation_counts"])


def test_random_control_logs_every_attempt_in_normal_runs(split, original, fakes):
    result = random_control_search(
        "random_forest", SAMPLE, "A", split, max_explanation_evaluations=8, seed=11,
        strategy=STRATEGY_V1, model=FakeModel(original),
    )
    counts = result["evaluation_counts"]
    assert len(result["history"]) == result["attempts"] == counts["proposals_generated"]
    assert sum(row["status"] == "evaluated" for row in result["history"]) == counts["scored_proposals"]
    assert result["stopped_reason"] == "explanation_budget_exhausted"
    assert counts["explanation_calls"] == 8


def test_legacy_random_draws_match_make_random_valid_candidate(original):
    # The refactored generator must consume the RNG exactly like the original helper.
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    for _ in range(30):
        legacy = ts.make_random_valid_candidate(original, "B", rng_a)
        proposal, _ = ts._draw_random_candidate(original, "B", rng_b, ts._mutation_for_strategy(STRATEGY_V0))
        if legacy is None:
            assert proposal is None
        else:
            assert proposal is not None and proposal.equals(legacy)


# ---------------------------------------------------------------------------------------------
# Legacy v0 preserved as baseline
# ---------------------------------------------------------------------------------------------
def test_legacy_v0_keeps_incremental_scoring(split, original, fakes):
    result = run_targeted_search(
        "random_forest", SAMPLE, "A", split, max_explanation_evaluations=30, seed=11,
        strategy=STRATEGY_V0, model=FakeModel(original),
    )
    assert result["strategy"] == STRATEGY_V0
    assert result["score_reference"] == "previous_accepted_explanation"
    accepted = [row for row in result["search_history"] if row["status"] == "accepted"]
    assert accepted
    previous = fakes(None, original, None, None, None, 11)
    for row in accepted:
        current = fakes(None, _row_from_json(row["feature_values_json"]), None, None, None, 11)
        assert row["score"] == pytest.approx(objective_score(previous, current), abs=1e-12)
        previous = current


def test_unknown_strategy_is_rejected(split, original, fakes):
    with pytest.raises(ValueError):
        run_targeted_search("random_forest", SAMPLE, "A", split, 3, 11, strategy="nope", model=FakeModel(original))
    with pytest.raises(ValueError):
        random_control_search("random_forest", SAMPLE, "A", split, 3, 11, strategy="nope", model=FakeModel(original))


# ---------------------------------------------------------------------------------------------
# Real LIME + saved model. Marked slow (registered in tests/conftest.py):
#   fast unit run:   python -m pytest -m "not slow"
#   only these:      python -m pytest -m slow
# ---------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_random_control_respects_explanation_budget(split):
    result = random_control_search(
        model_key="random_forest",
        sample_position=SAMPLE,
        budget_key="A",
        split=split,
        max_explanation_evaluations=3,
        seed=11,
    )
    assert result["evaluation_counts"]["explanation_calls"] <= 3
    assert result["evaluation_counts"]["model_predictions"] >= 1


@pytest.mark.slow
def test_targeted_search_smoke_run(split):
    result = run_targeted_search(
        model_key="random_forest",
        sample_position=SAMPLE,
        budget_key="A",
        split=split,
        max_explanation_evaluations=10,
        seed=11,
    )
    assert result["constraint_valid"] in {True, False}
    assert result["evaluation_counts"]["explanation_calls"] <= 10
    assert result["evaluation_counts"]["model_predictions"] >= 1
    assert "best_score" in result
