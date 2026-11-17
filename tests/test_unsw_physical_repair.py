"""physical_v1 policy (Step 6 v2, stage 2): repair, verification, budget, searches and matched controls.

Real data rows (the 10 v2 samples), fake model and fake explanation (no LIME, no saved model).
"""

import itertools
import json

import numpy as np
import pandas as pd
import pytest

import experiments.unsw_targeted_spoofing as ts
from experiments import unsw_constraint_reporting as cr
from src.unsw import physical_repair as pr
from src.unsw.constraints import FEATURE_ORDER, validate_candidate
from test_unsw_targeted_spoofing import FakeModel, fakes, original, split  # noqa: F401  (fixtures)

V2_SAMPLES = (2941, 3040, 3206, 15137, 27094, 4407, 30107, 34186, 15741, 25740)
TCP_SAMPLE = 15741      # tcp/ftp, spkts 12
SHORT_SAMPLE = 2941     # udp/dns, 2 packets, no reply


def _row(split, position):
    return split["X_val"].iloc[[position]].iloc[0].copy()


def _compose(original_row, assignment):
    """Candidate with the given lattice values, built and repaired as the search does."""
    row = original_row.copy()
    for feature, value in assignment.items():
        row[feature] = value
    return pr.repair(original_row, row)


def _all_lattice_candidates(original_row, budget):
    moves = pr.legal_moves(original_row, budget)
    feats = list(moves)
    for combo in itertools.product(*[[None] + moves[f] for f in feats]):
        assignment = {f: v for f, v in zip(feats, combo) if v is not None}
        if assignment:
            yield assignment, _compose(original_row, assignment)


# ---------------------------------------------------------------------------------------------
# Policy definition and lattice
# ---------------------------------------------------------------------------------------------
def test_budget_caps_match_the_search_policy():
    for key, policy in ts.BUDGET_POLICY.items():
        assert pr.MAX_CHANGED[key] == policy["max_changed"]
        assert pr.MAX_RELATIVE[key] == policy["max_relative"]
        assert pr.AGGREGATE_CAP[key] == policy["aggregate_cap"]


def test_editable_set_is_exactly_the_approved_one():
    assert pr.PHYSICAL_V1_EDITABLE == ("sbytes", "spkts")
    for feature in ("dur", "sttl", "sloss", "swin", "stcpb", "ackdat"):
        assert feature in pr.PHYSICAL_V1_EXCLUDED and feature not in pr.PHYSICAL_V1_EDITABLE
    assert set(pr.PHYSICAL_V1_EDITABLE) <= set(ts.NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES)


@pytest.mark.parametrize("budget", ["A", "B"])
def test_lattice_is_increase_only_within_cap_and_range(split, budget):
    for position in V2_SAMPLES:
        row = _row(split, position)
        for feature, values in pr.legal_moves(row, budget).items():
            original_value = float(row[feature])
            assert values == sorted(set(values))
            for value in values:
                assert value > original_value
                assert (value - original_value) / max(1.0, abs(original_value)) <= pr.MAX_RELATIVE[budget] + 1e-12
                assert float(value).is_integer()  # sbytes and spkts are integer features


def test_two_packet_flows_cannot_add_packets(split):
    for budget in ("A", "B"):
        moves = pr.legal_moves(_row(split, SHORT_SAMPLE), budget)
        assert "spkts" not in moves and "sbytes" in moves


# ---------------------------------------------------------------------------------------------
# Repair and verification
# ---------------------------------------------------------------------------------------------
def test_sbytes_edit_repairs_smean_and_sload(split):
    row = _row(split, SHORT_SAMPLE)
    candidate = _compose(row, {"sbytes": float(row["sbytes"]) + 10})
    assert candidate["smean"] == round(candidate["sbytes"] / candidate["spkts"])
    assert candidate["sload"] == pytest.approx((candidate["sbytes"] - candidate["smean"]) * 8 / candidate["dur"])
    assert candidate["rate"] == row["rate"]  # packet counts unchanged -> rate untouched
    verdict = pr.check_candidate(row, candidate, "A")
    assert verdict["valid"] and verdict["status"] == "consistent_with_checked_rules"
    assert set(verdict["primary_changes"]) == {"sbytes"} and set(verdict["derived_changes"]) <= {"smean", "sload"}


def test_spkts_edit_repairs_rate_and_reports_unverified_timing(split):
    row = _row(split, TCP_SAMPLE)
    candidate = _compose(row, {"spkts": float(row["spkts"]) + 1})
    assert candidate["rate"] == pytest.approx((candidate["spkts"] + candidate["dpkts"] - 1) / candidate["dur"])
    verdict = pr.check_candidate(row, candidate, "A")
    assert verdict["valid"]
    assert verdict["status"] == "unverified_dependencies"
    assert "sinpkt" in verdict["unverified"][0]


@pytest.mark.parametrize("mutation, expected", [
    (lambda r: r.__setitem__("sbytes", float(r["sbytes"]) + 5), "smean is not consistent"),       # stale derived
    (lambda r: r.__setitem__("smean", float(r["smean"]) + 1), "smean is not consistent"),         # derived edited directly
    (lambda r: r.__setitem__("dur", float(r["dur"]) * 1.05), "non-editable feature changed: dur"),
    (lambda r: r.__setitem__("dinpkt", 0.1), "non-editable feature changed: dinpkt"),
    (lambda r: r.__setitem__("sttl", 252), "non-editable feature changed: sttl"),
])
def test_invalid_candidates_are_rejected(split, mutation, expected):
    row = _row(split, SHORT_SAMPLE)
    candidate = row.copy()
    mutation(candidate)
    verdict = pr.check_candidate(row, candidate, "A")
    assert not verdict["valid"] and verdict["status"] == "violates_checked_rule"
    assert any(expected in v for v in verdict["violations"]), verdict["violations"]


def test_decrease_and_over_cap_are_rejected(split):
    row = _row(split, SHORT_SAMPLE)
    lower = _compose(row, {"sbytes": float(row["sbytes"]) - 5})
    assert any("decreased" in v for v in pr.check_candidate(row, lower, "A")["violations"])
    big = _compose(row, {"sbytes": float(round(float(row["sbytes"]) * 1.2))})
    assert any("exceeds" in v for v in pr.check_candidate(row, big, "A")["violations"])
    assert pr.check_candidate(row, big, "B")["valid"]  # 20% is within Budget B


def test_zero_coupling_is_enforced(split):
    row = _row(split, SHORT_SAMPLE)          # dpkts = 0, udp
    assert float(row["dpkts"]) == 0
    assert pr._zero_coupling_violations(row) == []
    bad = row.copy()
    bad["djit"] = 0.5
    assert "djit must be 0 when dpkts = 0" in pr._zero_coupling_violations(bad)
    bad = row.copy()
    bad["swin"] = 255
    assert "swin must be 0 for non-TCP flows" in pr._zero_coupling_violations(bad)


def test_repair_refuses_zero_duration(split):
    row = _row(split, TCP_SAMPLE).copy()
    row["dur"] = 0.0
    with pytest.raises(pr.PhysicalRepairError):
        pr.repair(row, _compose_no_repair(row, {"spkts": float(row["spkts"]) + 1}))


def _compose_no_repair(original_row, assignment):
    row = original_row.copy()
    for feature, value in assignment.items():
        row[feature] = value
    return row


@pytest.mark.parametrize("budget", ["A", "B"])
def test_every_lattice_candidate_on_all_ten_samples_is_valid_or_rejected_for_range_only(split, budget):
    """Every repaired lattice candidate passes the validator, the budget and every checked rule; the only
    allowed reason to reject one is a repaired derived value outside the observed range (e.g. smean)."""
    valid = 0
    for position in V2_SAMPLES:
        row = _row(split, position)
        for assignment, candidate in _all_lattice_candidates(row, budget):
            verdict = pr.check_candidate(row, candidate, budget)
            assert validate_candidate(row.to_dict(), candidate.to_dict())["valid"]
            assert ts._candidate_matches_budget(row, candidate, budget)[0]
            # budget counts edited (primary) features only; repaired derived features are reported separately
            assert set(ts._changed_features(row, candidate)) == set(assignment)
            if verdict["valid"]:
                assert cr.physical_consistency(row.to_dict(), candidate.to_dict())["physical_status"] != "violates_checked_rule"
                valid += 1
            else:
                assert all("outside the observed range" in v for v in verdict["violations"]), (position, assignment, verdict["violations"])
    assert valid > 0


# ---------------------------------------------------------------------------------------------
# Searches and matched random control (fake model and explanation)
# ---------------------------------------------------------------------------------------------
def _evaluated_rows(history, statuses=("accepted", "rejected", "evaluated")):
    for record in history:
        if record.get("score") is not None and record["status"] in statuses and record.get("feature_values_json"):
            yield pd.Series(json.loads(record["feature_values_json"]))


def _assert_all_scored_candidates_valid(history, original_row, budget):
    count = 0
    for candidate in _evaluated_rows(history):
        candidate = candidate.reindex(original_row.index)
        for feature in FEATURE_ORDER:
            if feature not in ("proto", "service", "state"):
                candidate[feature] = float(candidate[feature])
        verdict = pr.check_candidate(original_row, candidate, budget)
        assert verdict["valid"], verdict["violations"]
        count += 1
    return count


@pytest.mark.parametrize("strategy", [ts.STRATEGY_V1, ts.STRATEGY_SHUFFLED, ts.STRATEGY_MULTISTART])
@pytest.mark.parametrize("position, budget", [(TCP_SAMPLE, "A"), (TCP_SAMPLE, "B"), (SHORT_SAMPLE, "B")])
def test_targeted_physical_search_only_scores_valid_candidates(split, fakes, monkeypatch, strategy, position, budget):
    row = _row(split, position)
    monkeypatch.setattr(ts, "_call_explanation", __import__("test_unsw_targeted_spoofing").make_fake_explain(row))
    result = ts.run_targeted_search("random_forest", position, budget, split, 20, 11, strategy=strategy,
                                    model=FakeModel(row), order_seed=101, constraint_policy=ts.POLICY_PHYSICAL)
    history = result["search_history"]
    assert result["constraint_policy"] == ts.POLICY_PHYSICAL
    assert result["evaluation_counts"]["explanation_calls"] <= 20
    assert _assert_all_scored_candidates_valid(history, row, budget) >= result["evaluation_counts"]["scored_proposals"]
    assert set(result["best_changed_features"]) <= set(pr.PHYSICAL_V1_EDITABLE)
    best = pd.Series(result["best_candidate"]).reindex(row.index)
    assert pr.check_candidate(row, best, budget)["valid"]


@pytest.mark.parametrize("position, budget", [(TCP_SAMPLE, "A"), (SHORT_SAMPLE, "A"), (TCP_SAMPLE, "B")])
def test_random_control_uses_the_same_lattice_and_checks(split, fakes, monkeypatch, position, budget):
    row = _row(split, position)
    monkeypatch.setattr(ts, "_call_explanation", __import__("test_unsw_targeted_spoofing").make_fake_explain(row))
    control = ts.random_control_search("random_forest", position, budget, split, 20, 11, strategy=ts.STRATEGY_V1,
                                       max_attempts=500, model=FakeModel(row), control_seed=201,
                                       constraint_policy=ts.POLICY_PHYSICAL)
    targeted = ts.run_targeted_search("random_forest", position, budget, split, 20, 11, strategy=ts.STRATEGY_SHUFFLED,
                                      model=FakeModel(row), order_seed=101, constraint_policy=ts.POLICY_PHYSICAL)
    assert ts.matched_control_problems(targeted, control) == []
    lattice = pr.legal_moves(row, budget)
    assert control["legal_moves"] == lattice == targeted["legal_moves"]
    scored = 0
    for candidate in _evaluated_rows(control["history"], statuses=("evaluated",)):
        for feature in pr.PHYSICAL_V1_EDITABLE:
            value = float(candidate[feature])
            assert value == float(row[feature]) or value in lattice[feature]
        scored += 1
    assert scored == control["evaluation_counts"]["scored_proposals"]
    assert _assert_all_scored_candidates_valid(control["history"], row, budget) == scored
    assert control["evaluation_counts"]["explanation_calls"] <= 20


def test_matched_control_mismatch_is_detected(split, fakes, monkeypatch):
    row = _row(split, TCP_SAMPLE)
    monkeypatch.setattr(ts, "_call_explanation", __import__("test_unsw_targeted_spoofing").make_fake_explain(row))
    physical = ts.run_targeted_search("random_forest", TCP_SAMPLE, "A", split, 8, 11, strategy=ts.STRATEGY_V1,
                                      model=FakeModel(row), constraint_policy=ts.POLICY_PHYSICAL)
    current = ts.random_control_search("random_forest", TCP_SAMPLE, "A", split, 8, 11, strategy=ts.STRATEGY_V1,
                                       max_attempts=200, model=FakeModel(row), control_seed=201)
    assert current["constraint_policy"] == ts.POLICY_CURRENT
    assert any("constraint_policy" in p for p in ts.matched_control_problems(physical, current))
    other_budget = dict(physical, budget="B", legal_moves_digest="x")
    problems = ts.matched_control_problems(physical, other_budget)
    assert any(p.startswith("budget") for p in problems)


def test_physical_prefix_equivalence_for_checkpoints(split, fakes, monkeypatch):
    """A physical run at 20 calls read at 10 equals a separate run at 10 (prefix shortcut stays valid)."""
    row = _row(split, TCP_SAMPLE)
    monkeypatch.setattr(ts, "_call_explanation", __import__("test_unsw_targeted_spoofing").make_fake_explain(row))
    for kind in ("targeted", "random"):
        if kind == "targeted":
            run = lambda n: ts.run_targeted_search("random_forest", TCP_SAMPLE, "B", split, n, 11, strategy=ts.STRATEGY_SHUFFLED,
                                                   model=FakeModel(row), order_seed=103, constraint_policy=ts.POLICY_PHYSICAL)
            hist_key = "search_history"
        else:
            run = lambda n: ts.random_control_search("random_forest", TCP_SAMPLE, "B", split, n, 11, strategy=ts.STRATEGY_V1,
                                                     max_attempts=400, model=FakeModel(row), control_seed=203,
                                                     constraint_policy=ts.POLICY_PHYSICAL)
            hist_key = "history"
        long_run, short_run = run(20), run(10)
        original_json = json.dumps(row.to_dict(), default=str)
        long_cp = ts.best_so_far_checkpoints(long_run[hist_key], [10], long_run["evaluation_counts"]["explanation_calls"], original_json)[0]
        short_cp = ts.best_so_far_checkpoints(short_run[hist_key], [10], short_run["evaluation_counts"]["explanation_calls"], original_json)[0]
        assert long_cp["best_score"] == short_cp["best_score"]
        assert long_cp["candidate_json"] == short_cp["candidate_json"]


def test_unknown_or_unsupported_policies_are_refused(split, fakes):
    row = _row(split, TCP_SAMPLE)
    with pytest.raises(ValueError):
        ts.run_targeted_search("random_forest", TCP_SAMPLE, "A", split, 5, 11, strategy=ts.STRATEGY_V1,
                               model=FakeModel(row), constraint_policy="nonsense")
    with pytest.raises(ValueError):
        ts.run_targeted_search("random_forest", TCP_SAMPLE, "A", split, 5, 11, strategy=ts.STRATEGY_V0,
                               model=FakeModel(row), constraint_policy=ts.POLICY_PHYSICAL)
    with pytest.raises(ValueError):
        ts.random_control_search("random_forest", TCP_SAMPLE, "A", split, 5, 11, strategy=ts.STRATEGY_V0,
                                 model=FakeModel(row), constraint_policy=ts.POLICY_PHYSICAL)


def test_current_policy_is_the_default_and_unchanged(split):
    """Default spec equals the v1 ingredients; restart plans for v1 are unchanged."""
    row = _row(split, SHORT_SAMPLE)
    spec = ts.policy_spec(ts.POLICY_CURRENT, "A", row)
    assert spec["features"] is ts.NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES
    assert spec["directions"] == (-1, 1)
    assert spec["steps_for"]("sbytes") == ts.choose_feature_steps("A")
    assert spec["mutate"] is ts.mutation_proposal_v1 and spec["extra_check"] is None
    assert ts.DEFAULT_CONSTRAINT_POLICY == ts.POLICY_CURRENT
    default = ts.restart_plans(101, "random_forest", 2941, "A", 3)
    explicit = ts.restart_plans(101, "random_forest", 2941, "A", 3, features=ts.NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES, directions=(-1, 1))
    assert default == explicit
    assert all(len(p["direction_order"][f]) == 2 for p in default for f in p["feature_order"])
    physical = ts.restart_plans(101, "random_forest", 15741, "A", 3, features=("sbytes", "spkts"), directions=(1,))
    assert all(p["direction_order"][f] == [1] for p in physical for f in p["feature_order"])
    assert ts._plan_summary(physical[0])["direction_order"].count("+") == 2
