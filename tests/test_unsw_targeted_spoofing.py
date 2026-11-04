import json

import numpy as np
import pandas as pd
import pytest

from experiments.unsw_targeted_spoofing import (
    BUDGET_POLICY,
    _candidate_matches_budget,
    _spearman_rho,
    objective_score,
    random_control_search,
    run_targeted_search,
)
from src.unsw.constraints import validate_candidate
from src.unsw.xai_setup import get_xai_data


@pytest.fixture(scope="module")
def split():
    return get_xai_data()


def test_budget_policy_a_and_b_are_enforced(split):
    original = split["X_val"].iloc[[2941]].iloc[0].copy()
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


def test_spearman_missing_rank_policy_is_defined(split):
    original = {"a": 10.0, "b": 5.0, "c": 7.0}
    candidate = {"a": 10.0, "c": 3.0}
    rho = _spearman_rho(original, candidate)
    assert -1.0 <= rho <= 1.0
    assert np.isfinite(rho)


def test_targeted_search_rejects_benign_prediction(split):
    original = split["X_val"].iloc[[2941]].iloc[0].copy()
    candidate = original.copy()
    candidate["dur"] = 0.0
    candidate["spkts"] = 1
    validation = validate_candidate(original.to_dict(), candidate.to_dict())
    assert validation["valid"] is True
    ok_a, _ = _candidate_matches_budget(original, candidate, "A")
    assert ok_a is False


def test_random_control_respects_explanation_budget(split):
    result = random_control_search(
        model_key="random_forest",
        sample_position=2941,
        budget_key="A",
        split=split,
        max_explanation_evaluations=3,
        seed=11,
    )
    assert result["evaluation_counts"]["explanation_calls"] <= 3
    assert result["evaluation_counts"]["model_predictions"] >= 1


def test_targeted_search_smoke_run(split):
    result = run_targeted_search(
        model_key="random_forest",
        sample_position=2941,
        budget_key="A",
        split=split,
        max_explanation_evaluations=10,
        seed=11,
    )
    assert result["constraint_valid"] in {True, False}
    assert result["evaluation_counts"]["explanation_calls"] <= 10
    assert result["evaluation_counts"]["model_predictions"] >= 1
    assert "best_score" in result
