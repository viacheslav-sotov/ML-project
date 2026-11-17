"""Resumable nested CV (experiments/unsw_nested_cv.py) with fake estimators on a small real sample. No real training."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import experiments.unsw_nested_cv as ncv
from src.unsw import v3_data as vd
from src.unsw import v3_pipelines as vp


@pytest.fixture(scope="module")
def protocol():
    return ncv.load_protocol()


@pytest.fixture(scope="module")
def official():
    return vd.load_official_train()


@pytest.fixture(scope="module")
def small_data(official):
    df = official.sample(2500, random_state=3).reset_index(drop=True)
    info = vd.build_groups(df)
    keep = ~info["label_conflict"].to_numpy()
    return {"X": df[keep][vd.feature_order()].reset_index(drop=True), "y": df["label"].astype(int).to_numpy()[keep],
            "strata": info["stratum"].to_numpy()[keep], "groups": info["group_id"].to_numpy()[keep],
            "categories": info["attack_cat"].to_numpy()[keep], "ids": info["id"].to_numpy()[keep], "split_digest": "test"}


def fake_pipeline(model, fs):
    return Pipeline([("preprocessor", vp.build_preprocessor(fs, False)), ("model", DecisionTreeClassifier(random_state=0))])


def fake_space(model):
    return {"model__max_depth": [2, 3, 4, 5]}


def _run(tmp_path, protocol, data, **kw):
    return ncv.run(protocol, tmp_path, data=data, make_pipeline=fake_pipeline, space_fn=fake_space, n_iter=kw.pop("n_iter", 2),
                   models=kw.pop("models", ["random_forest"]), feature_sets=kw.pop("feature_sets", ["full42"]), log=lambda m: None, **kw)


def test_protocol_draft_is_valid_and_states_not_time_based(protocol):
    assert ncv.validate_protocol(protocol) == []
    assert "NOT time-based" in protocol["nested_cv"]["applies_to"]
    assert "CHRONOLOGICAL" in protocol["time_split_robustness_S2"]["validation"]
    assert protocol["time_split_robustness_S2"]["status"].startswith("postponed")
    assert protocol["statistics"]["intervals"].startswith("Wilson 95% primary")


@pytest.mark.parametrize("change, message", [
    (lambda p: p["nested_cv"]["inner"].update(random_state=42), "seeds must differ"),
    (lambda p: p["nested_cv"]["outer"].update(splitter="StratifiedKFold"), "grouped"),
    (lambda p: p["nested_cv"].update(applies_to="random folds"), "not time-based"),
    (lambda p: p["nested_cv"]["search_spaces"].pop("mlp"), "mlp"),
    (lambda p: p["data"].update(label_conflict_rule="maybe"), "label_conflict_rule"),
])
def test_protocol_validation_catches_errors(protocol, change, message):
    bad = copy.deepcopy(protocol)
    change(bad)
    assert any(message in p for p in ncv.validate_protocol(bad))


def test_plan_and_fit_counts(protocol):
    assert len(ncv.plan_units(protocol, vp.MODELS, vp.FEATURE_SETS)) == 4 * 2 * 6
    assert len(ncv.plan_units(protocol, vp.MODELS, vp.FEATURE_SETS, benchmark=True)) == 8
    assert ncv.fit_count(protocol, "xgboost") == 5 * (3 * 25 + 1) + (3 * 25 + 1) == 456
    assert ncv.fit_count(protocol, "random_forest") == 276 and ncv.fit_count(protocol, "mlp") == 222
    assert ncv.fit_count(protocol, "mlp", benchmark=True, n_iter=2) == 7


def test_folds_are_grouped_and_stratified(protocol, small_data):
    folds = ncv.make_folds(small_data["strata"], small_data["groups"], protocol["nested_cv"]["outer"])
    ncv.check_disjoint(small_data["groups"], folds)
    assert len(folds) == 5
    assert sorted(np.concatenate([b for _, b in folds]).tolist()) == list(range(len(small_data["y"])))
    with pytest.raises(ncv.NestedCVError):
        ncv.check_disjoint(np.zeros(len(small_data["y"]), int), [(np.array([0]), np.array([1]))])


def test_full_unit_set_runs_and_saves_final_model(tmp_path, protocol, small_data):
    out = _run(tmp_path, protocol, small_data)
    recs = {r["unit"]: r for r in out["records"]}
    assert sorted(recs) == ["final", "outer0", "outer1", "outer2", "outer3", "outer4"]
    for k in range(5):
        r = recs[f"outer{k}"]
        assert r["outer_groups_shared"] == 0 and r["fits"] == 3 * 2 + 1 and 0 <= r["outer_metrics"]["f1"] <= 1
        assert "recall_by_category" in r["outer_metrics"]
    final = recs["final"]
    assert Path(final["model_file"]).exists() and final["model_sha256"] == ncv.file_sha256(Path(final["model_file"]))
    summary = out["summary"]["random_forest__full42"]
    assert len(summary["outer"]) == 5 and summary["final"]["model_sha256"] == final["model_sha256"]
    assert json.loads((Path(out["run_dir"]) / "run_config.json").read_text(encoding="utf-8"))["official_test_used"] is False


def test_resume_after_interruption_gives_identical_results(tmp_path, protocol, small_data):
    full = _run(tmp_path / "full", protocol, small_data)
    with pytest.raises(ncv.Interrupted):
        _run(tmp_path / "crash", protocol, small_data, max_units=2)
    run_dir = next((tmp_path / "crash").iterdir())
    resumed = _run(tmp_path / "crash", protocol, small_data, resume_dir=run_dir)
    assert resumed["status"]["units_reused"] == 2 and resumed["status"]["units_run_now"] == 4
    strip = lambda r: {k: v for k, v in r.items() if k not in ("search_seconds", "seconds_per_fit", "candidates", "model_file", "model_sha256")}  # noqa: E731
    assert [strip(r) for r in full["records"]] == [strip(r) for r in resumed["records"]]


def test_resume_refuses_changed_data(tmp_path, protocol, small_data):
    with pytest.raises(ncv.Interrupted):
        _run(tmp_path, protocol, small_data, max_units=1)
    run_dir = next(Path(tmp_path).iterdir())
    other = dict(small_data, y=1 - small_data["y"])
    with pytest.raises(ncv.NestedCVError, match="data_digest"):
        _run(tmp_path, protocol, other, resume_dir=run_dir)


def test_v3_train_side_from_real_data_is_disjoint(protocol, official):
    data = ncv.load_v3_train(protocol, official)
    assert data["audit"]["shared_feature_vectors"] == 0
    assert len(data["y"]) == data["audit"]["rows_train"] and len(set(data["ids"])) == len(data["ids"])


def test_leaking_split_is_refused(protocol, official, monkeypatch):
    real = vd.audit
    monkeypatch.setattr(vd, "audit", lambda split, df: {**real(split, df), "shared_feature_vectors": 5})
    with pytest.raises(ncv.NestedCVError, match="not duplicate-disjoint"):
        ncv.load_v3_train(protocol, official)
