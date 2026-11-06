"""Step 6 Stage 0: accounting, independent seeds, prefix equivalence, checkpoints, constraint
reporting and manifest validation. Fake models and explainers only (no LIME, no training)."""

import json

import numpy as np
import pandas as pd
import pytest

import experiments.unsw_targeted_spoofing as ts
from experiments import unsw_constraint_reporting as cr
from experiments import unsw_step6_manifest as mf
from src.unsw.constraints import FEATURE_ORDER, validate_candidate
from src.unsw.xai_setup import get_xai_data

SAMPLE = 2941
LIME_ROWS = 50  # rows the fake LIME sends to the model per explanation (real LIME: 5000)
V1, SHUF, MULTI = ts.STRATEGY_V1, ts.STRATEGY_SHUFFLED, ts.STRATEGY_MULTISTART


@pytest.fixture(scope="module")
def split():
    return get_xai_data()


@pytest.fixture(scope="module")
def original(split):
    return split["X_val"].iloc[[SAMPLE]].iloc[0].copy()


class FakeModel:
    """Counts nothing itself; returns P(attack) for every row. Benign if an attackable feature moved (optional)."""

    def __init__(self, original_row, benign_if_changed=False, prob=0.99):
        self.original = original_row
        self.benign_if_changed = benign_if_changed
        self.prob = prob

    def predict_proba(self, X):
        n = len(X)
        p = self.prob
        if self.benign_if_changed and "dur" in X.columns:
            row = X.iloc[0]
            if any(float(row[f]) != float(self.original[f]) for f in ts.NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES):
                p = 0.2
        return np.tile([[1.0 - p, p]], (n, 1))


class FakeCodec:
    def __init__(self, X):
        pass

    def encode(self, X):
        return None


def make_fake_explain(original_row):
    base = {feature: float(i + 1) for i, feature in enumerate(FEATURE_ORDER)}

    def explain(model, row, codec, explainer, predict_fn, seed):
        predict_fn(np.zeros((LIME_ROWS, 1)))  # one LIME-style batch through the real counting wrapper
        weights = dict(base)
        for feature in ts.NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES:
            weights[feature] += 2000.0 * ts._relative_change(float(original_row[feature]), float(row[feature]))
        return weights

    return explain


@pytest.fixture
def fakes(monkeypatch, original):
    monkeypatch.setattr(ts, "CategoricalCodec", FakeCodec)
    monkeypatch.setattr(ts, "build_explainer", lambda encoded, codec: None)
    # Replace only LIME's inner prediction function; ts.make_predict_fn (the counting wrapper) stays real.
    monkeypatch.setattr(ts, "_lime_make_predict_fn", lambda model, codec: (lambda Z: model.predict_proba(pd.DataFrame(index=range(len(Z))))))
    monkeypatch.setattr(ts, "_call_explanation", make_fake_explain(original))

    def no_loading(*args, **kwargs):
        raise AssertionError("tests pass a fake model explicitly")

    monkeypatch.setattr(ts, "load_xai_model", no_loading)


def run_method(split, model, method, budget, budget_key="B", **kw):
    if method == "random":
        return ts.random_control_search("random_forest", SAMPLE, budget_key, split, budget, 11, strategy=V1,
                                        max_attempts=kw.pop("max_attempts", 2000), model=model, **kw)
    params = dict(order_seed=101, restarts=3, fair_share=True, include_canonical_first=False, model_fingerprint="fake")
    params.update(kw)
    return ts.run_targeted_search("random_forest", SAMPLE, budget_key, split, budget, 11, strategy=method, model=model, **params)


def history_of(result):
    return result["search_history"] if "search_history" in result else result["history"]


# =============================================================================================
# Prediction-call accounting
# =============================================================================================
@pytest.mark.parametrize("method", [ts.STRATEGY_V0, V1, SHUF, MULTI, "random"])
def test_accounting_separates_top_level_lime_and_final_predictions(split, original, fakes, method):
    meter = ts.CountingModel(FakeModel(original))
    result = run_method(split, meter, method, 20)
    counts = result["evaluation_counts"]
    acc = ts.prediction_accounting(counts, meter)
    assert acc["accounting_consistent"]
    assert acc["explanation_evaluations"] == counts["explanation_calls"] <= 20
    assert acc["lime_internal_prediction_batches"] == counts["explanation_calls"]  # one batch per explanation, baseline included
    assert acc["lime_internal_prediction_rows"] == LIME_ROWS * counts["explanation_calls"]
    assert acc["top_level_prediction_calls"] == counts["model_predictions"] == acc["top_level_prediction_rows"]
    expected_final = 0 if method in (SHUF, MULTI) else 1  # multi-start reports from its cache
    assert acc["final_report_prediction_calls"] == expected_final == acc["final_report_prediction_rows"]
    assert acc["total_model_prediction_calls"] == meter.direct_calls + meter.lime_batches
    assert acc["total_model_prediction_rows"] == meter.direct_rows + meter.lime_rows
    assert acc["total_model_prediction_calls"] == counts["model_predictions"] + expected_final + counts["explanation_calls"]


def test_cache_hits_never_reach_the_model(split, original, fakes):
    meter = ts.CountingModel(FakeModel(original))
    result = run_method(split, meter, MULTI, 60, fair_share=False)
    counts = result["evaluation_counts"]
    assert counts["cache_hits"] > 0
    assert meter.direct_calls == counts["model_predictions"] == 1 + counts["scored_proposals"] + counts["rejected_benign"]
    assert meter.lime_batches == counts["explanation_calls"] == 1 + counts["scored_proposals"]


def test_benign_candidates_cost_one_prediction_and_no_explanation(split, original, fakes):
    meter = ts.CountingModel(FakeModel(original, benign_if_changed=True))
    result = run_method(split, meter, MULTI, 10)
    counts = result["evaluation_counts"]
    acc = ts.prediction_accounting(counts, meter)
    assert counts["explanation_calls"] == 1 and meter.lime_batches == 1
    assert acc["top_level_prediction_calls"] == 1 + counts["rejected_benign"]
    assert counts["cache_hits_rejected"] > 0  # benign verdicts reused across restarts without new calls
    assert acc["accounting_consistent"]


# =============================================================================================
# Independent random-control seeds
# =============================================================================================
def _draws(rng, n=8):
    return rng.integers(0, 2**31, size=n).tolist()


def test_omitting_control_seed_reproduces_the_original_rule():
    legacy = np.random.default_rng(11 + SAMPLE + sum(ord(c) for c in "random_forest"))
    assert _draws(ts.random_control_rng(11, SAMPLE, "random_forest", "A", None)) == _draws(legacy)


def test_control_seed_is_independent_of_lime_seed_and_order_seed(split, original, fakes):
    seqs = {}
    for lime_seed in (11, 22):
        r = ts.random_control_search("random_forest", SAMPLE, "B", split, 12, lime_seed, strategy=V1,
                                     max_attempts=2000, model=FakeModel(original), control_seed=201)
        seqs[lime_seed] = [row["feature_values_json"] for row in r["history"]]
    assert seqs[11] == seqs[22]  # the LIME seed no longer drives the control's draws
    legacy = {}
    for lime_seed in (11, 22):
        r = ts.random_control_search("random_forest", SAMPLE, "B", split, 12, lime_seed, strategy=V1,
                                     max_attempts=2000, model=FakeModel(original))
        legacy[lime_seed] = [row["feature_values_json"] for row in r["history"]]
    assert legacy[11] != legacy[22]  # documents the old coupling when control_seed is omitted
    other = ts.random_control_search("random_forest", SAMPLE, "B", split, 12, 11, strategy=V1,
                                     max_attempts=2000, model=FakeModel(original), control_seed=202)
    assert [row["feature_values_json"] for row in other["history"]] != seqs[11]
    assert other["control_seed"] == 202 and "control_seed" in other["rng_seed_rule"]
    # The order seed is not an input of the random control at all.
    import inspect
    assert "order_seed" not in inspect.signature(ts.random_control_search).parameters


def test_control_seed_streams_differ_by_budget_and_model():
    a = _draws(ts.random_control_rng(11, SAMPLE, "random_forest", "A", 201))
    assert a != _draws(ts.random_control_rng(11, SAMPLE, "random_forest", "B", 201))
    assert a != _draws(ts.random_control_rng(11, SAMPLE, "xgboost", "A", 201))
    assert a == _draws(ts.random_control_rng(99, SAMPLE, "random_forest", "A", 201))


# =============================================================================================
# Budget-prefix equivalence (justifies the 100-call shortcut for 25/50)
# =============================================================================================
def _trim_deferred(history):
    return history[:-1] if history and history[-1].get("status") == "deferred" else history


@pytest.mark.parametrize("method", [V1, SHUF, "random"])
@pytest.mark.parametrize("short_budget", [25, 50])
def test_100_call_run_prefix_equals_separate_short_run(split, original, fakes, method, short_budget):
    kw = {"control_seed": 201} if method == "random" else {}
    long = run_method(split, FakeModel(original), method, 100, **kw)
    short = run_method(split, FakeModel(original), method, short_budget, **kw)
    long_hist, short_hist = history_of(long), _trim_deferred(history_of(short))
    assert short_hist == long_hist[: len(short_hist)], "candidate history / random draws diverge before the short budget ends"
    # The checkpoint read from the long run equals the separate short run's best.
    original_json = json.dumps(original.to_dict(), default=str)
    from_long = ts.best_so_far_checkpoints(long_hist, [short_budget], long["evaluation_counts"]["explanation_calls"], original_json)[0]
    from_short = ts.best_so_far_checkpoints(history_of(short), [short_budget], short["evaluation_counts"]["explanation_calls"], original_json)[0]
    for key in ("best_score", "explanation_calls_at_best", "candidate_json", "best_changed_features", "candidate_is_original"):
        assert from_long[key] == from_short[key], key
    assert from_short["best_score"] == pytest.approx(short["best_score"])
    if method == "random":
        n = short["attempts"]
        assert [r["feature_values_json"] for r in long["history"][:n]] == [r["feature_values_json"] for r in short["history"]]


def test_multistart_is_not_prefix_equivalent_so_it_runs_per_budget(split, original, fakes):
    long = run_method(split, FakeModel(original), MULTI, 100)
    short = run_method(split, FakeModel(original), MULTI, 25)
    assert [r["fresh_call_cap"] for r in short["restarts"]][0] == 8
    assert [r["fresh_call_cap"] for r in long["restarts"]][0] == 33
    protocol = mf.load_protocol()
    assert protocol["strategies"]["multistart_greedy_v1"]["prefix_shortcut"] is False
    for name in ("greedy_baseline_ref_v1", "greedy_shuffled_v1", "random_control"):
        assert protocol["strategies"][name]["prefix_shortcut"] is True


# =============================================================================================
# Checkpoint candidate reconstruction
# =============================================================================================
@pytest.mark.parametrize("method", [V1, SHUF, MULTI, "random"])
def test_checkpoints_reconstruct_valid_candidates(split, original, fakes, method):
    kw = {"control_seed": 205} if method == "random" else {}
    result = run_method(split, FakeModel(original), method, 40, **kw)
    total = result["evaluation_counts"]["explanation_calls"]
    original_json = json.dumps(original.to_dict(), default=str)
    records = ts.best_so_far_checkpoints(history_of(result), [10, 20, 40], total, original_json)
    assert [r["checkpoint"] for r in records] == [10, 20, 40]
    scores = [r["best_score"] for r in records]
    assert scores == sorted(scores)
    final = records[-1]
    assert final["best_score"] == pytest.approx(result["best_score"])
    rebuilt = ts.row_from_json(final["candidate_json"], original)
    assert ts._row_key(rebuilt) == ts._row_key(pd.Series(result["best_candidate"]))
    for record in records:
        assert record["explanation_calls_at_best"] <= record["checkpoint"]
        row = ts.row_from_json(record["candidate_json"], original)
        assert validate_candidate(original.to_dict(), row.to_dict())["valid"]
        assert ts._candidate_matches_budget(original, row, "B")[0]
        assert record["candidate_is_original"] == (record["best_score"] == 0.0)
        if record["candidate_is_original"]:
            assert ts._row_key(row) == ts._row_key(original)


def test_row_from_json_round_trips_the_original(original):
    rebuilt = ts.row_from_json(json.dumps(original.to_dict(), default=str), original)
    for feature in FEATURE_ORDER:
        if feature in ("proto", "service", "state"):
            assert rebuilt[feature] == original[feature]
        else:
            assert float(rebuilt[feature]) == float(original[feature])


def test_run_directories_are_unique_and_never_reused(tmp_path):
    first = ts._make_run_dir(tmp_path)
    (first / "keep.txt").write_text("existing result", encoding="utf-8")
    second = ts._make_run_dir(tmp_path)
    third = ts._make_run_dir(tmp_path)
    assert len({first, second, third}) == 3
    assert (first / "keep.txt").read_text(encoding="utf-8") == "existing result"


def test_run_experiment_writes_accounting_checkpoints_and_control_seed(split, original, fakes, monkeypatch, tmp_path):
    monkeypatch.setattr(ts, "load_xai_model", lambda key, s: FakeModel(original))
    out = ts.run_experiment(models=["random_forest"], sample_positions=[SAMPLE], budgets=["A"],
                            max_explanation_evaluations=20, seed=11, output_dir=str(tmp_path),
                            strategies=[V1, MULTI], order_seed=101, restarts=3, control_seed=201,
                            max_random_attempts=2000, checkpoints=[10])
    run_dir = out["run_dir"]
    targeted = pd.read_csv(f"{run_dir}/targeted_results.csv")
    control = pd.read_csv(f"{run_dir}/random_control_results.csv")
    checkpoints = pd.read_csv(f"{run_dir}/checkpoints.csv")
    for frame in (targeted, control):
        for column in ts.ACCOUNTING_COLUMNS:
            assert column in frame.columns
        assert frame["accounting_consistent"].all()
        assert (frame["lime_internal_prediction_rows"] == LIME_ROWS * frame["explanation_evaluations"]).all()
    assert list(targeted.loc[targeted.strategy == MULTI, "final_report_prediction_calls"]) == [0]
    assert list(targeted.loc[targeted.strategy == V1, "final_report_prediction_calls"]) == [1]
    assert int(control["control_seed"].iloc[0]) == 201
    assert set(checkpoints["checkpoint"]) == {10, 20}
    assert set(checkpoints["method"]) == {"targeted", "random_control"}
    meta = json.loads(open(f"{run_dir}/run_metadata.json", encoding="utf-8").read())
    assert meta["control_seed"] == 201 and meta["checkpoints"] == [10, 20]
    assert "prediction_accounting" in meta


# =============================================================================================
# Constraint reporting
# =============================================================================================
@pytest.fixture(scope="module")
def table():
    return cr.load_teammate_table()


def _edit(original, **changes):
    row = original.copy()
    for feature, value in changes.items():
        row[feature] = value
    return row


def _report(original, candidate, budget_key="B", table=None):
    ok, reason = ts._candidate_matches_budget(original, candidate, budget_key)
    return cr.constraint_report(original.to_dict(), candidate.to_dict(), ok, reason, table)


def test_unchanged_candidate_is_never_called_physically_valid(original, table):
    r = _report(original, original.copy(), table=table)
    assert r["experimental_valid"] and r["teammate_table_status"] == "compliant"
    assert r["physical_status"] == "consistent_with_checked_rules"
    assert r["physically_valid_claimed"] is False
    assert r["physical_rules_checked"] == len(cr.CHECKED_RULES)


def test_dur_edit_is_experimentally_valid_and_compliant_but_breaks_rate(original, table):
    cand = _edit(original, dur=float(original["dur"]) + 0.02)
    r = _report(original, cand, table=table)
    assert r["experimental_valid"]
    assert r["teammate_table_status"] == "compliant"
    if cr._rate_rule(original.to_dict()):
        assert r["physical_status"] == "violates_checked_rule" and "rate" in r["physical_violations"]
    else:
        assert "rate" in r["physical_rules_failing_on_original"]


def test_sbytes_increase_needs_review_and_decrease_is_non_compliant(original, table):
    up = _report(original, _edit(original, sbytes=float(original["sbytes"]) + 2), table=table)
    down = _report(original, _edit(original, sbytes=float(original["sbytes"]) - 2), table=table)
    assert up["teammate_table_status"] == "needs_review"
    assert down["teammate_table_status"] == "non_compliant"
    if cr._smean_rule(original.to_dict()):
        assert "smean" in up["physical_violations"]


def test_destination_side_edit_is_non_compliant_and_unverified(original, table):
    r = _report(original, _edit(original, djit=0.1), table=table)
    assert r["experimental_valid"]
    assert r["teammate_table_status"] == "non_compliant"
    assert r["physical_status"] == "unverified_dependencies"


def test_tcp_only_field_on_non_tcp_flow_violates_physical_rule(original, table):
    assert str(original["proto"]) != "tcp"
    r = _report(original, _edit(original, swin=1.0), table=table)
    assert r["physical_status"] == "violates_checked_rule"
    assert "TCP-only" in r["physical_violations"]
    assert r["teammate_table_status"] == "needs_review"


def test_synack_edit_fails_experimental_policy_and_physics(original, table):
    r = _report(original, _edit(original, synack=0.05), table=table)
    assert not r["experimental_valid"] and "tcprtt" in r["experimental_reasons"]
    assert r["teammate_table_status"] == "non_compliant"
    assert r["physical_status"] == "violates_checked_rule"


def test_loss_above_packets_violates(original, table):
    r = _report(original, _edit(original, sloss=float(original["spkts"]) + 1), table=table)
    assert "sloss <= spkts" in r["physical_violations"]


def test_teammate_status_mapping():
    assert cr.teammate_feature_status("Yes", +1)[0] == "compliant"
    assert cr.teammate_feature_status("Yes (via packet timing)", -1)[0] == "compliant"
    assert cr.teammate_feature_status("Yes (increase only) - Needs review", +1)[0] == "needs_review"
    assert cr.teammate_feature_status("Yes (increase only) - Needs review", -1)[0] == "non_compliant"
    assert cr.teammate_feature_status("Needs review", +1)[0] == "needs_review"
    for rating in ("No", "Limited", "Indirectly (derived)", "Indirectly (other flows)"):
        assert cr.teammate_feature_status(rating, +1)[0] == "non_compliant"


def test_reporting_requires_the_teammate_table_and_records_versions(tmp_path):
    with pytest.raises(FileNotFoundError):
        cr.load_teammate_table(tmp_path / "missing.csv")
    versions = cr.policy_versions()
    assert versions["constraints_py_sha256"] == cr.file_sha256(cr.CONSTRAINTS_PY_PATH)
    assert versions["teammate_table_enforced"].startswith("no")


# =============================================================================================
# Manifest
# =============================================================================================
def test_predeclared_protocol_is_valid_and_matches_the_approved_design():
    protocol = mf.load_protocol()
    assert mf.validate_protocol(protocol) == []
    assert protocol["lime_search_seed"] == 11
    assert protocol["heldout_lime_seeds"] == [22, 33, 44, 55]
    assert protocol["order_seeds"] == list(range(101, 111))
    assert protocol["control_seeds"] == list(range(201, 211))
    assert protocol["explanation_budgets"] == [25, 50, 100]
    multi = protocol["strategies"]["multistart_greedy_v1"]
    assert multi["restarts"] == 3 and multi["fair_share"] is True and multi["include_canonical_first"] is False
    assert protocol["models"] == ["random_forest", "xgboost"] and protocol["sample_positions"] == [2941, 3206, 4407]
    assert protocol["perturbation_budgets"] == ["A", "B"]


@pytest.mark.parametrize("mutate, message", [
    (lambda p: p.update(control_seeds=[101, 202]), "overlap"),
    (lambda p: p.update(heldout_lime_seeds=[11, 22, 33]), "search_seed"),
    (lambda p: p.update(explanation_budgets=[50, 25, 100]), "explanation_budgets"),
    (lambda p: p["strategies"]["multistart_greedy_v1"].update(prefix_shortcut=True), "separate runs"),
    (lambda p: p["strategies"]["multistart_greedy_v1"].update(restarts=0), "restarts"),
    (lambda p: p["strategies"]["multistart_greedy_v1"].update(fair_share=False), "fair_share"),
    (lambda p: p["strategies"]["multistart_greedy_v1"].update(include_canonical_first=True), "include_canonical_first"),
    (lambda p: p["strategies"].pop("random_control"), "random_control"),
    (lambda p: p.update(max_random_attempts=100), "max_random_attempts"),
    (lambda p: p.update(models=["mlp"]), "unknown model"),
    (lambda p: p.update(order_seeds=[11, 102]), "LIME seeds"),
])
def test_protocol_validation_catches_errors(mutate, message):
    protocol = json.loads(json.dumps(mf.load_protocol()))
    mutate(protocol)
    problems = mf.validate_protocol(protocol)
    assert any(message in p for p in problems), problems


@pytest.fixture(scope="module")
def manifest():
    return mf.build_manifest()


def test_resolved_manifest_is_valid_and_complete(manifest):
    assert mf.validate_manifest(manifest) == []
    roles = manifest["datasets"]["by_role"]
    assert roles["official_train"]["rows"] == 175_341 and roles["official_train"]["file"].endswith("testing-set.csv")
    assert roles["official_test"]["rows"] == 82_332 and roles["official_test"]["file"].endswith("training-set.csv")
    assert manifest["datasets"]["official_test_used"] is False
    assert [s["official_id"] for s in manifest["samples"]] == [134318, 64810, 92635]
    assert manifest["split"]["validation_rows"] == 35_069 and manifest["split"]["subtrain_rows"] == 140_272
    assert manifest["models"]["random_forest"]["sha256"].startswith("4e49f17d")
    assert manifest["models"]["xgboost"]["sha256"].startswith("83a34c61")
    assert manifest["software"]["lime"] == "0.2.0.1"
    assert manifest["seeds"]["control_seeds"] == list(range(201, 211))
    for key in ("objective_score", "pair_metrics_part_b", "targeted_search_script", "heldout_script"):
        assert len(manifest["implementation_hashes"][key]) == 64
    assert manifest["constraint_policy"]["teammate_table_enforced"].startswith("no")


def test_manifest_tampering_is_detected(manifest):
    tampered = json.loads(json.dumps(manifest))
    tampered["models"]["random_forest"]["sha256"] = "0" * 64
    assert any("random_forest hash changed" in p for p in mf.validate_manifest(tampered))
    tampered = json.loads(json.dumps(manifest))
    tampered["protocol"]["order_seeds"] = list(range(301, 311))
    assert any("protocol_sha256" in p for p in mf.validate_manifest(tampered, check_files=False))


def test_manifest_files_are_never_overwritten(manifest, tmp_path):
    first = mf.write_manifest(manifest, tmp_path)
    second = mf.write_manifest(manifest, tmp_path)
    assert first != second
    assert json.loads(first.read_text(encoding="utf-8"))["protocol_version"] == "step6-v1"
