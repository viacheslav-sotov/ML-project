"""G4 diagnostic unconstrained upper bound (experiments/unsw_step6_g4.py, protocol step6-g4-diagnostic-1).

Fake models, LIME and SHAP; the real data split is used. v1/v2 policies must stay untouched.
"""

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

import experiments.unsw_step6_g4 as g4
import experiments.unsw_targeted_spoofing as ts
from experiments import unsw_step6_manifest as mf
from src.unsw import physical_repair as pr
from src.unsw.constraints import FEATURE_ORDER, INTEGER_FEATURES, NUMERIC_MIN_MAX
from test_step6_shap_eval import fake_shap
from test_step6_v2_infrastructure import FakeModel, _fake_lime  # noqa: F401
from test_unsw_step6_runner import split  # noqa: F401  (fixture)


@pytest.fixture(scope="module")
def protocol():
    return g4.load_g4()


@pytest.fixture(scope="module")
def manifest_v2():
    return mf.build_manifest(mf.load_protocol(mf.PROTOCOL_V2_PATH))


def _models(split):
    return {"random_forest": FakeModel(None), "xgboost": FakeModel(None, prob=0.9)}


def _sel(protocol, **kw):
    return g4.selection_from(protocol, smoke=True, **kw)


def _run(tmp_path, split, protocol, manifest, **kw):
    return g4.run_g4(protocol, tmp_path, split=split, manifest=manifest, models=kw.pop("models", _models(split)),
                     explain_fn=kw.pop("explain_fn", _fake_lime(split)[0]), shap_fn=kw.pop("shap_fn", fake_shap()),
                     selection=kw.pop("selection", _sel(protocol)), log=lambda m: None, **kw)


# ---------------------------------------------------------------------------------------------
# Protocol and separation from v1/v2
# ---------------------------------------------------------------------------------------------
def test_g4_protocol_is_valid_and_separate(protocol):
    assert g4.validate_g4(protocol) == []
    pol = protocol["policy"]
    assert pol["name"] == "unconstrained_g4_v1" and pol["name"] not in pr.POLICIES and pol["name"] not in ts.CONSTRAINT_POLICIES
    assert len(pol["editable_features"]) == 35 and sorted(pol["editable_features"] + pol["frozen_features"]) == sorted(FEATURE_ORDER)
    assert pol["max_changed_features"] == 10 and pol["max_relative_change"] == 0.5
    assert "never evidence of a physically realizable network attack" in protocol["interpretation"]
    assert protocol["expected_model_sha256"] == mf.load_protocol(mf.PROTOCOL_V2_PATH)["expected_model_sha256"]
    assert ts.CONSTRAINT_POLICIES == ("current_v1", "physical_v1")      # no G4 policy added to the v1/v2 search code


@pytest.mark.parametrize("change, message", [
    (lambda p: p["policy"].update(max_relative_change=0.6), "D7"),
    (lambda p: p["policy"]["frozen_features"].append("sbytes"), "overlap"),
    (lambda p: p["methods"]["random_control"].update(control_seeds=[22, 202]), "overlap"),
    (lambda p: p.update(confirmation_lime_seeds=[101, 402]), "overlap"),
    (lambda p: p["policy"].update(name="physical_v1"), "separate"),
    (lambda p: p.update(interpretation="anything"), "interpretation"),
    (lambda p: p["smoke_test"].update(explanation_budget=500), "smoke_test"),
])
def test_g4_protocol_validation_catches_errors(protocol, change, message):
    bad = copy.deepcopy(protocol)
    change(bad)
    assert any(message in p for p in g4.validate_g4(bad)), g4.validate_g4(bad)


# ---------------------------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------------------------
def test_policy_mutation_rounding_range_and_budget(protocol, split):
    policy = g4.G4Policy(protocol)
    original = split["X_val"].iloc[[25740]].iloc[0].copy()
    for f in policy.features:
        for sign in (-1, 1):
            row = policy.mutate(original, original, f, sign, 0.5)
            lo, hi = NUMERIC_MIN_MAX[f]
            assert lo <= float(row[f]) <= hi
            if f in INTEGER_FEATURES:
                assert float(row[f]) == round(float(row[f]))
            assert all(row[x] == original[x] for x in policy.frozen)
    row = original.copy()
    for f in policy.features[:11]:
        row = policy.mutate(row, original, f, 1, 0.5)
    changed = policy.changed(original, row)
    ok, reason = policy.check(original, row)
    assert len(changed) > 10 and not ok and "more than 10" in reason
    big = policy.mutate(policy.mutate(original, original, "sbytes", 1, 0.5), original, "sbytes", 1, 0.5)
    assert not policy.check(original, big)[0]                                   # 100% > 50%
    frozen = original.copy()
    frozen["state"] = "CON" if original["state"] != "CON" else "FIN"
    assert "frozen" in policy.check(original, policy.mutate(frozen, original, "sbytes", 1, 0.1))[1]
    assert "rate" in policy.features and "ct_state_ttl" in policy.features   # derived features are editable in G4


def test_searches_respect_budget_seed_and_attack(protocol, split):
    policy = g4.G4Policy(protocol)
    X_val = split["X_val"]
    original = X_val.iloc[[25740]].iloc[0].copy()
    explain, _ = _fake_lime(split)
    prob = lambda m, r: 0.3 if float(r["sbytes"]) < 0.6 * float(original["sbytes"]) else 0.9  # noqa: E731
    a = g4.shuffled_greedy(policy, "random_forest", 25740, original, prob, explain, 101, 25, 11)
    b = g4.shuffled_greedy(policy, "random_forest", 25740, original, prob, explain, 101, 25, 11)
    c = g4.shuffled_greedy(policy, "random_forest", 25740, original, prob, explain, 102, 25, 11)
    assert a == b and a["feature_order"] != c["feature_order"]
    r = g4.random_control(policy, "random_forest", 25740, original, prob, explain, 201, 25, 11, 20)
    for out in (a, c, r):
        assert out["calls"] <= 25 and len(out["history"]) == out["calls"] - 1
        for h in out["history"]:
            cand = g4.apply_changes(original, h["changes"])
            assert policy.check(original, cand)[0] and h["p_attack"] >= 0.5
    cps = g4._checkpoints(a["history"], [10, 25], 25)
    assert cps[10]["best_score"] <= cps[25]["best_score"]


# ---------------------------------------------------------------------------------------------
# Run (fake), outputs, physical labels, resume, safeguards
# ---------------------------------------------------------------------------------------------
def test_smoke_selection_and_plan(protocol):
    sel = _sel(protocol)
    assert sel["run_role"] == "smoke_test" and sel["samples"] == [2941, 25740] and sel["explanation_budget"] == 25
    full = g4.selection_from(protocol, smoke=False)
    assert full["run_role"] == "diagnostic_pilot"
    plan = g4.plan_summary(protocol, full)
    assert plan["tasks"] == 400 and plan["search_lime_calls_max"] == 40_000
    with pytest.raises(g4.G4Error):
        g4.selection_from(protocol, smoke=True, samples=[1])


def test_fake_run_outputs_and_physical_labels(tmp_path, split, protocol, manifest_v2):
    out = _run(tmp_path, split, protocol, manifest_v2)
    run_dir = Path(out["run_dir"])
    assert run_dir.name.startswith("g4_smoke_test_")
    for name in ("checkpoints.csv", "heldout_summary.csv", "noise_bands.csv", "summary_by_method.csv", "README.md", "status.json"):
        assert (run_dir / name).exists(), name
    status = out["status"]
    assert status["tasks"] == 8 and status["search_score_mismatches"] == 0 and status["official_test_used"] is False
    summary = out["summary"]
    assert len(summary) and set(summary["physical_status"]) <= set(g4.PHYSICAL_STATUSES)
    assert (summary["physically_valid_claimed"] == False).all()  # noqa: E712
    assert "lime_diagnostic_success" in summary and "shap_d11_meets_both" in summary
    assert (summary["confirmed_lime_diagnostic_success"] <= summary["lime_diagnostic_success"]).all()
    assert "never evidence of a physically realizable" in (run_dir / "README.md").read_text(encoding="utf-8")


def test_resume_reruns_only_missing_tasks(tmp_path, split, protocol, manifest_v2):
    full = _run(tmp_path / "full", split, protocol, manifest_v2)
    with pytest.raises(g4.Interrupted):
        _run(tmp_path / "crash", split, protocol, manifest_v2, max_tasks=3)
    run_dir = next((tmp_path / "crash").glob("g4_*"))
    resumed = _run(tmp_path / "crash", split, protocol, manifest_v2, resume_dir=run_dir)
    assert resumed["status"]["tasks_reused"] == 3 and resumed["status"]["tasks_run_now"] == 5
    cols = ["candidate_uid", "heldout_top10_jaccard_mean", "heldout_spearman_mean", "physical_status"]
    pd.testing.assert_frame_equal(full["summary"][cols].sort_values("candidate_uid").reset_index(drop=True),
                                  resumed["summary"][cols].sort_values("candidate_uid").reset_index(drop=True))


def test_resume_refuses_changed_selection(tmp_path, split, protocol, manifest_v2):
    with pytest.raises(g4.Interrupted):
        _run(tmp_path, split, protocol, manifest_v2, max_tasks=1)
    run_dir = next(Path(tmp_path).glob("g4_*"))
    other = _sel(protocol, order_seeds=[102])
    with pytest.raises(g4.G4Error, match="selection"):
        _run(tmp_path, split, protocol, manifest_v2, resume_dir=run_dir, selection=other)


def test_wrong_pin_and_identity_are_refused(tmp_path, split, protocol, manifest_v2):
    bad = copy.deepcopy(protocol)
    bad["expected_model_sha256"]["xgboost"] = "3" * 64
    with pytest.raises(Exception, match="pinned hash"):
        _run(tmp_path, split, bad, manifest_v2)
    bad = copy.deepcopy(protocol)
    bad["sample_identity"]["25740"]["official_train_row_index"] += 1
    with pytest.raises(Exception, match="sample identity"):
        _run(tmp_path, split, bad, manifest_v2)
    assert not list(Path(tmp_path).glob("g4_*"))
