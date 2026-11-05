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


# =============================================================================================
# Step 5b: greedy_shuffled_v1 and multistart_greedy_v1 (fake explainer; no LIME, no saved model)
# =============================================================================================
from experiments.unsw_targeted_spoofing import (  # noqa: E402
    STRATEGY_MULTISTART,
    STRATEGY_SHUFFLED,
    CandidateCache,
    build_cache_context,
    fair_share_cap,
    restart_plans,
)


class CountingFakeModel(FakeModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def predict_proba(self, X):
        self.calls += 1
        return super().predict_proba(X)


def make_dur_only_explain(original_row: pd.Series):
    """Only `dur` changes the explanation: one improving direction, ties everywhere else."""
    base = {feature: float(i + 1) for i, feature in enumerate(FEATURE_ORDER)}

    def explain(model, row, codec, explainer, predict_fn, seed):
        weights = dict(base)
        weights["dur"] += 2000.0 * _relative_change(float(original_row["dur"]), float(row["dur"]))
        return weights

    return explain


@pytest.fixture
def counting(monkeypatch, original):
    """Install fakes with an explanation-call counter; returns a function to switch the landscape."""
    state = {"explain": make_fake_explain(original), "calls": 0}

    def explain(model, row, codec, explainer, predict_fn, seed):
        state["calls"] += 1
        return state["explain"](model, row, codec, explainer, predict_fn, seed)

    monkeypatch.setattr(ts, "CategoricalCodec", FakeCodec)
    monkeypatch.setattr(ts, "build_explainer", lambda encoded, codec: None)
    monkeypatch.setattr(ts, "make_predict_fn", lambda model, codec: None)
    monkeypatch.setattr(ts, "_call_explanation", explain)

    def no_model_loading(*args, **kwargs):
        raise AssertionError("tests must pass a fake model explicitly")

    monkeypatch.setattr(ts, "load_xai_model", no_model_loading)
    return state


def _ms(split, model, budget=10, strategy=STRATEGY_MULTISTART, **kwargs):
    params = dict(order_seed=42, restarts=3, fair_share=True, include_canonical_first=False, model_fingerprint="fake-model")
    params.update(kwargs)
    lime_seed = params.pop("lime_seed", 11)
    budget_key = params.pop("budget_key", "A")
    return run_targeted_search("random_forest", SAMPLE, budget_key, split, budget, lime_seed, strategy=strategy, model=model, **params)


def _fresh_scored(history):
    return [row for row in history if row["score"] is not None and not row.get("cache_hit") and row["status"] in {"accepted", "rejected"}]


def _assert_multistart_identities(counts):
    _assert_accounting_identities(counts)
    assert counts["proposals_generated"] == (
        counts["rejected_no_change"] + counts["cache_hits"] + counts["rejected_constraint"] + counts["rejected_budget"]
        + counts["rejected_benign"] + counts["scored_proposals"] + counts["deferred_at_cap"]
    )
    assert counts["unique_candidates"] == (
        counts["rejected_constraint"] + counts["rejected_budget"] + counts["rejected_benign"] + counts["scored_proposals"]
    )
    assert counts["cache_hits"] == counts["cache_hits_scored"] + counts["cache_hits_rejected"]


# --- reproducibility -------------------------------------------------------------------------
def test_restart_plans_are_reproducible_prefix_stable_and_permutations():
    a = restart_plans(42, "random_forest", SAMPLE, "A", 3)
    b = restart_plans(42, "random_forest", SAMPLE, "A", 3)
    assert a == b
    longer = restart_plans(42, "random_forest", SAMPLE, "A", 5)
    assert longer[:3] == a  # adding restarts never changes earlier ones
    other = restart_plans(43, "random_forest", SAMPLE, "A", 3)
    assert [p["feature_order"] for p in other] != [p["feature_order"] for p in a]
    for plan in a:
        assert sorted(plan["feature_order"]) == sorted(ts.NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES)
        assert set(plan["direction_order"]) == set(ts.NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES)
        assert all(sorted(d) == [-1, 1] for d in plan["direction_order"].values())
        assert plan["order_kind"] == "shuffled"
    assert len({p["restart_seed"] for p in a}) == 3
    assert len({tuple(p["feature_order"]) for p in a}) == 3
    canonical = restart_plans(42, "random_forest", SAMPLE, "A", 3, include_canonical_first=True)
    assert canonical[0]["feature_order"] == list(ts.NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES)
    assert canonical[1:] == a[1:]


@pytest.mark.parametrize("strategy", [STRATEGY_SHUFFLED, STRATEGY_MULTISTART])
def test_shuffled_runs_are_reproducible(split, original, counting, strategy):
    first = _ms(split, FakeModel(original), budget=12, strategy=strategy)
    second = _ms(split, FakeModel(original), budget=12, strategy=strategy)
    strip = lambda h: [(r["restart_id"], r["status"], r["reason"], r["score"], r["feature_values_json"], r["cache_hit"]) for r in h]
    assert strip(first["search_history"]) == strip(second["search_history"])
    assert first["restarts"] == second["restarts"]
    assert first["evaluation_counts"] == second["evaluation_counts"]
    assert first["best_score"] == second["best_score"]


def test_order_seed_is_independent_of_lime_seed(split, original, counting):
    a = _ms(split, FakeModel(original), budget=8, lime_seed=11)
    b = _ms(split, FakeModel(original), budget=8, lime_seed=22)
    assert a["restart_plans"] == b["restart_plans"]
    assert a["cache_context"]["lime_seed"] == 11 and b["cache_context"]["lime_seed"] == 22
    c = _ms(split, FakeModel(original), budget=8, order_seed=7)
    assert [p["feature_order"] for p in c["restart_plans"]] != [p["feature_order"] for p in a["restart_plans"]]


def test_greedy_shuffled_is_one_restart_with_the_first_multistart_order(split, original, counting):
    shuffled = _ms(split, FakeModel(original), budget=10, strategy=STRATEGY_SHUFFLED, restarts=5)
    multi = _ms(split, FakeModel(original), budget=10, strategy=STRATEGY_MULTISTART)
    assert shuffled["restarts_requested"] == 1
    assert len(shuffled["restarts"]) == 1
    assert shuffled["restart_plans"][0] == multi["restart_plans"][0]
    assert shuffled["restarts"][0]["fresh_call_cap"] == 9  # one restart gets all remaining calls


# --- global evaluation caps and prediction counting -------------------------------------------
@pytest.mark.parametrize("budget", [1, 2, 5, 10, 40])
@pytest.mark.parametrize("restarts", [1, 3])
def test_global_explanation_cap_and_real_call_counts(split, original, counting, budget, restarts):
    model = CountingFakeModel(original)
    counting["calls"] = 0
    result = _ms(split, model, budget=budget, restarts=restarts)
    counts = result["evaluation_counts"]
    assert counts["explanation_calls"] <= budget
    assert counting["calls"] == counts["explanation_calls"]  # actual explainer invocations
    assert model.calls == counts["model_predictions"]  # actual model invocations, none uncounted
    _assert_multistart_identities(counts)
    assert result["unused_explanation_calls"] == budget - counts["explanation_calls"]
    calls_seen = [row["explanation_calls_so_far"] for row in result["search_history"]]
    assert calls_seen == sorted(calls_seen)  # counters never reset between restarts
    preds_seen = [row["model_predictions_so_far"] for row in result["search_history"]]
    assert preds_seen == sorted(preds_seen)
    assert sum(r["explanation_calls_used"] for r in result["restarts"]) + 1 == counts["explanation_calls"]
    assert sum(r["model_predictions_used"] for r in result["restarts"]) + 1 == counts["model_predictions"]


# --- fair share ------------------------------------------------------------------------------
def test_fair_share_cap_rule():
    assert fair_share_cap(10, 1, 3) == 3
    assert fair_share_cap(10, 4, 2) == 3
    assert fair_share_cap(10, 2, 2) == 4  # roll-over: restart 0 used 1 call
    assert fair_share_cap(10, 2, 1) == 8  # last restart gets the rest
    assert fair_share_cap(10, 10, 1) == 0
    assert fair_share_cap(10, 1, 3, fair_share=False) == 9
    assert fair_share_cap(3, 1, 3) == 0


def test_fair_share_caps_are_enforced_per_restart(split, original, counting):
    result = _ms(split, FakeModel(original), budget=10, restarts=3)
    for row in result["restarts"]:
        if row["status"] == "ran":
            assert row["explanation_calls_used"] <= row["fresh_call_cap"]
    assert result["restarts"][0]["fresh_call_cap"] == 3
    no_share = _ms(split, FakeModel(original), budget=10, restarts=3, fair_share=False)
    assert no_share["restarts"][0]["fresh_call_cap"] == 9


def test_unused_calls_roll_forward_and_are_not_consumed_artificially(split, original, counting):
    # Every changed candidate is benign, so no restart can use an explanation call.
    result = _ms(split, CountingFakeModel(original, benign_if_changed=True), budget=10, restarts=3)
    caps = [row["fresh_call_cap"] for row in result["restarts"]]
    assert caps == [3, 4, 9]
    assert all(row["explanation_calls_used"] == 0 for row in result["restarts"])
    assert all(row["stopped_reason"] == "local_optimum_full_sweep_without_improvement" for row in result["restarts"])
    assert result["evaluation_counts"]["explanation_calls"] == 1
    assert result["unused_explanation_calls"] == 9
    assert result["stopped_reason"] == "all_restarts_finished"
    assert result["best_score"] == 0.0


def test_restart_at_local_optimum_leaves_calls_unused(split, original, counting):
    counting["explain"] = make_dur_only_explain(original)
    result = _ms(split, FakeModel(original), budget=500, restarts=3, fair_share=False)
    assert result["stopped_reason"] == "all_restarts_finished"
    assert result["unused_explanation_calls"] > 0
    assert all(r["stopped_reason"] == "local_optimum_full_sweep_without_improvement" for r in result["restarts"])


# --- cache -----------------------------------------------------------------------------------
def test_cache_reuses_score_and_prediction_without_new_calls(split, original, counting):
    model = CountingFakeModel(original)
    counting["calls"] = 0
    result = _ms(split, model, budget=60, restarts=3, fair_share=False)
    history = result["search_history"]
    counts = result["evaluation_counts"]
    assert counts["cache_hits"] > 0
    fresh = {}
    for row in history:
        if row["score"] is not None and not row["cache_hit"] and row["status"] in {"accepted", "rejected"}:
            key = _row_key(_row_from_json(row["feature_values_json"]))
            assert key not in fresh, "a candidate was explained twice"
            fresh[key] = (row["score"], row["candidate_probability"])
    previous_calls = 0
    for row in history:
        if row["cache_hit"]:
            key = _row_key(_row_from_json(row["feature_values_json"]))
            if row["score"] is not None and key != _row_key(original):
                assert (row["score"], row["candidate_probability"]) == fresh[key]
            assert row["explanation_calls_so_far"] == previous_calls  # no explanation spent on a hit
        previous_calls = row["explanation_calls_so_far"]
    assert counting["calls"] == counts["explanation_calls"] == 1 + len(fresh)
    assert model.calls == counts["model_predictions"]
    assert result["cache_size"] == counts["unique_candidates"] + 1  # + the original sample
    _assert_multistart_identities(counts)


def test_later_restart_can_accept_a_cached_candidate(split, original, counting):
    counting["explain"] = make_dur_only_explain(original)
    result = _ms(split, FakeModel(original), budget=300, restarts=2, fair_share=False, include_canonical_first=True)
    accepted_from_cache = [r for r in result["search_history"] if r["status"] == "accepted" and r["cache_hit"]]
    assert accepted_from_cache, "restart 1 should accept the dur move first evaluated by restart 0"
    assert all(r["restart_id"] == 1 for r in accepted_from_cache)
    assert result["restarts"][1]["accepted_steps"] >= 1


def test_cache_context_must_match_exactly(original):
    baseline = {feature: float(i) for i, feature in enumerate(FEATURE_ORDER)}
    ctx = build_cache_context("random_forest", "sha256:abc", SAMPLE, "A", 11, baseline)
    cache = CandidateCache(ctx)
    key = _row_key(original)
    cache.put(ctx, key, {"status": "scored", "score": 0.0})
    assert cache.get(ctx, key)["score"] == 0.0
    with pytest.raises(ValueError):
        cache.put(ctx, key, {"status": "scored", "score": 0.0})  # never evaluate twice
    changed_baseline = dict(baseline)
    changed_baseline["dur"] += 1.0
    other_config = dict(ts.explanation_config())
    other_config["num_samples"] = 1000
    mismatches = [
        build_cache_context("xgboost", "sha256:abc", SAMPLE, "A", 11, baseline),
        build_cache_context("random_forest", "sha256:def", SAMPLE, "A", 11, baseline),
        build_cache_context("random_forest", "sha256:abc", 3206, "A", 11, baseline),
        build_cache_context("random_forest", "sha256:abc", SAMPLE, "B", 11, baseline),
        build_cache_context("random_forest", "sha256:abc", SAMPLE, "A", 22, baseline),
        build_cache_context("random_forest", "sha256:abc", SAMPLE, "A", 11, changed_baseline),
        build_cache_context("random_forest", "sha256:abc", SAMPLE, "A", 11, baseline, config=other_config),
    ]
    for wrong in mismatches:
        assert wrong != ctx
        with pytest.raises(ValueError, match="context mismatch"):
            cache.get(wrong, key)
        with pytest.raises(ValueError, match="context mismatch"):
            cache.put(wrong, (1.0,), {"status": "scored", "score": 0.0})


def test_run_records_cache_context(split, original, counting):
    result = _ms(split, FakeModel(original), budget=5, model_fingerprint="sha256:test")
    ctx = result["cache_context"]
    assert ctx["model_fingerprint"] == "sha256:test"
    assert ctx["lime_seed"] == 11 and ctx["sample_position"] == SAMPLE and ctx["budget_key"] == "A"
    assert ctx["baseline_digest"] == ts.weights_digest(result["baseline_weights"])


# --- candidate validation ----------------------------------------------------------------------
def test_only_valid_malicious_candidates_are_explained(split, original, counting):
    result = _ms(split, FakeModel(original), budget=40, restarts=3, fair_share=False)
    for row in _fresh_scored(result["search_history"]):
        candidate = _row_from_json(row["feature_values_json"])
        assert validate_candidate(original.to_dict(), candidate.to_dict())["valid"]
        assert _candidate_matches_budget(original, candidate, "A")[0]
        assert row["candidate_probability"] >= 0.5
        assert "synack" not in row["changed_features"] and "ackdat" not in row["changed_features"]


def test_benign_candidates_never_reach_the_explainer(split, original, counting):
    model = CountingFakeModel(original, benign_if_changed=True)
    counting["calls"] = 0
    result = _ms(split, model, budget=10, restarts=3)
    counts = result["evaluation_counts"]
    assert counts["explanation_calls"] == 1 and counting["calls"] == 1
    assert counts["rejected_benign"] > 0
    assert model.calls == counts["model_predictions"] == 1 + counts["rejected_benign"]
    assert counts["cache_hits_rejected"] > 0  # later restarts reuse the benign verdicts without new model calls
    _assert_multistart_identities(counts)


# --- restart independence ----------------------------------------------------------------------
def test_every_restart_starts_from_the_original_sample(split, original, counting):
    # Fair share (default) guarantees every restart a share of the 39 non-baseline calls.
    result = _ms(split, FakeModel(original), budget=40, restarts=3)
    assert [r["status"] for r in result["restarts"]] == ["ran", "ran", "ran"]
    for restart_id in range(3):
        rows = [r for r in result["search_history"] if r["restart_id"] == restart_id]
        assert rows
        first = _row_from_json(rows[0]["feature_values_json"])
        assert len(ts._changed_features(original, first)) == 1
        accepted_scores = [r["score"] for r in rows if r["status"] == "accepted"]
        assert accepted_scores == sorted(accepted_scores)  # each restart climbs its own score from 0


# --- K=1 canonical equivalence and v0/v1 regression -------------------------------------------
@pytest.mark.parametrize("budget", [3, 5, 20, 60])
def test_k1_canonical_multistart_matches_v1(split, original, counting, budget):
    v1 = run_targeted_search("random_forest", SAMPLE, "A", split, budget, 11, strategy=STRATEGY_V1, model=FakeModel(original))
    ms = _ms(split, FakeModel(original), budget=budget, restarts=1, include_canonical_first=True)
    key = lambda r: _row_key(_row_from_json(r["feature_values_json"]))
    v1_fresh = [(key(r), r["score"]) for r in v1["search_history"] if r["score"] is not None]
    ms_fresh = [(key(r), r["score"]) for r in _fresh_scored(ms["search_history"])]
    assert ms_fresh == v1_fresh
    v1_acc = [key(r) for r in v1["search_history"] if r["status"] == "accepted"]
    ms_acc = [key(r) for r in ms["search_history"] if r["status"] == "accepted"]
    assert ms_acc == v1_acc
    for name in ("explanation_calls", "model_predictions", "scored_proposals", "rejected_benign", "accepted_improvements"):
        assert ms["evaluation_counts"][name] == v1["evaluation_counts"][name], name
    assert ms["best_score"] == v1["best_score"]
    assert _row_key(pd.Series(ms["best_candidate"])) == _row_key(pd.Series(v1["best_candidate"]))
    expected_stop = "explanation_budget_exhausted" if v1["stopped_reason"] == "explanation_budget_exhausted" else "all_restarts_finished"
    assert ms["stopped_reason"] == expected_stop


def test_random_control_is_unchanged_by_step5b_options(split, original, counting):
    # The control does not take order/restart options; its draws depend only on the LIME seed and config.
    a = random_control_search("random_forest", SAMPLE, "A", split, 8, 11, strategy=STRATEGY_V1, model=FakeModel(original))
    b = random_control_search("random_forest", SAMPLE, "A", split, 8, 11, strategy=STRATEGY_V1, model=FakeModel(original))
    assert [r["feature_values_json"] for r in a["history"]] == [r["feature_values_json"] for r in b["history"]]
    with pytest.raises(ValueError):
        random_control_search("random_forest", SAMPLE, "A", split, 8, 11, strategy=STRATEGY_MULTISTART, model=FakeModel(original))


def test_mixing_v0_with_other_strategies_is_rejected():
    with pytest.raises(ValueError):
        ts._control_strategy_for([STRATEGY_V0, STRATEGY_MULTISTART])
    assert ts._control_strategy_for([STRATEGY_SHUFFLED, STRATEGY_MULTISTART]) == STRATEGY_V1
    assert ts._control_strategy_for([STRATEGY_V0]) == STRATEGY_V0
