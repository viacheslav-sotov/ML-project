"""Step 6 runner and integration fixes (fake models/explainers; no LIME, no training)."""

import gzip
import json
import os

import numpy as np
import pandas as pd
import pytest

import experiments.unsw_step6_runner as runner
import experiments.unsw_targeted_spoofing as ts
from experiments import unsw_step6_manifest as mf
from experiments import unsw_targeted_heldout as ho
from src.unsw.constraints import FEATURE_ORDER
from src.unsw.xai_setup import get_xai_data

SAMPLE = 2941
LIME_ROWS = 50


@pytest.fixture(scope="module")
def split():
    return get_xai_data()


@pytest.fixture(scope="module")
def original(split):
    return split["X_val"].iloc[[SAMPLE]].iloc[0].copy()


@pytest.fixture(scope="module")
def manifest():
    return mf.build_manifest()


@pytest.fixture(scope="module")
def protocol():
    return mf.load_protocol()


class FakeModel:
    def __init__(self, original_row, prob=0.99):
        self.original, self.prob = original_row, prob

    def predict_proba(self, X):
        return np.tile([[1.0 - self.prob, self.prob]], (len(X), 1))


class FakeCodec:
    def __init__(self, X):
        pass

    def encode(self, X):
        return None


def make_fake_explain(original_row, counter):
    base = {feature: float(i + 1) for i, feature in enumerate(FEATURE_ORDER)}

    def explain(model, row, codec, explainer, predict_fn, seed):
        counter["calls"] += 1
        predict_fn(np.zeros((LIME_ROWS, 1)))
        weights = dict(base)
        for feature in ts.NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES:
            weights[feature] += 2000.0 * ts._relative_change(float(original_row[feature]), float(row[feature]))
        return weights

    return explain


@pytest.fixture
def fakes(monkeypatch, original):
    counter = {"calls": 0}
    monkeypatch.setattr(ts, "CategoricalCodec", FakeCodec)
    monkeypatch.setattr(ts, "build_explainer", lambda encoded, codec: None)
    monkeypatch.setattr(ts, "_lime_make_predict_fn", lambda model, codec: (lambda Z: model.predict_proba(pd.DataFrame(index=range(len(Z))))))
    monkeypatch.setattr(ts, "_call_explanation", make_fake_explain(original, counter))

    def no_loading(*args, **kwargs):
        raise AssertionError("the runner must use the injected model loader")

    monkeypatch.setattr(ts, "load_xai_model", no_loading)
    return counter


PILOT = dict(models=["random_forest"], samples=[SAMPLE], budgets=["A"], explanation_budgets=[25, 50],
             order_seeds=[101, 102], control_seeds=[201, 202])


def run(tmp_path, split, manifest, original, resume_dir=None, selection=None, **kw):
    selection = selection if selection is not None or resume_dir is not None else runner.make_selection(mf.load_protocol(), **PILOT)
    return runner.run_step6(selection, output_root=tmp_path, resume_dir=resume_dir, split=split, manifest=manifest,
                            model_loader=lambda key, s: FakeModel(original), log=lambda msg: None, **kw)


def read(run_dir, name):
    return pd.read_csv(f"{run_dir}/{name}")


# =============================================================================================
# Selection, seeds and planning
# =============================================================================================
def test_plan_has_one_canonical_task_and_independent_seed_families(protocol):
    tasks = runner.plan_tasks(runner.make_selection(protocol, **PILOT))
    by_method = {}
    for task in tasks:
        by_method.setdefault(task.method, []).append(task)
    assert len(by_method["canonical"]) == 1  # deterministic: never repeated per seed
    assert [t.order_seed for t in by_method["shuffled"]] == [101, 102]
    assert sorted((t.order_seed, t.explanation_budget) for t in by_method["multistart"]) == [(101, 25), (101, 50), (102, 25), (102, 50)]
    assert [t.control_seed for t in by_method["random"]] == [201, 202]
    for task in tasks:
        if task.method in ("shuffled", "multistart"):
            assert task.order_seed is not None and task.control_seed is None
        elif task.method == "random":
            assert task.control_seed is not None and task.order_seed is None
        else:
            assert task.order_seed is None and task.control_seed is None
        if task.method == "multistart":
            assert task.checkpoints == (task.explanation_budget,)
        else:
            assert task.explanation_budget == 50 and task.checkpoints == (25, 50)
    assert len({t.task_id for t in tasks}) == len(tasks) == 9


@pytest.mark.parametrize("kwargs, message", [
    (dict(order_seeds=[201]), "order_seeds"),
    (dict(control_seeds=[101]), "control_seeds"),
    (dict(explanation_budgets=[30]), "explanation_budgets"),
    (dict(models=["mlp"]), "models"),
    (dict(samples=[1]), "samples"),
    (dict(budgets=["C"]), "budgets"),
    (dict(order_seeds=[101, 101]), "duplicate"),
    (dict(methods=["genetic"]), "methods"),
])
def test_selection_must_come_from_the_protocol(protocol, kwargs, message):
    with pytest.raises(ValueError, match=message):
        runner.make_selection(protocol, **kwargs)


def test_default_selection_is_the_full_protocol(protocol):
    selection = runner.make_selection(protocol)
    assert selection.order_seeds == tuple(range(101, 111)) and selection.control_seeds == tuple(range(201, 211))
    tasks = runner.plan_tasks(selection)
    # 2 models x 3 samples x 2 budgets x (1 canonical + 10 shuffled + 30 multistart + 10 random)
    assert len(tasks) == 12 * 51


# =============================================================================================
# End-to-end with fakes: budgets, checkpoints, accounting, schema
# =============================================================================================
@pytest.fixture
def pilot(tmp_path, split, manifest, original, fakes):
    out = run(tmp_path, split, manifest, original)
    return out, fakes


def test_every_task_respects_its_budget_and_stays_an_attack(pilot):
    out, _ = pilot
    results = read(out["run_dir"], "results.csv")
    assert len(results) == 9 and out["status"]["failed"] == []
    assert (results["explanation_calls"] <= results["explanation_budget"]).all()
    assert (results["explanation_calls"] + results["unused_explanation_calls"] == results["explanation_budget"]).all()
    random_rows = results[results.search == "random"]
    assert (random_rows["explanation_calls"] == random_rows["explanation_budget"]).all()
    assert results["still_attack"].all() and (results["best_probability"] >= 0.5).all()
    assert (results["lime_seed"] == 11).all()


def test_checkpoints_are_valid_and_multistart_only_at_its_own_budget(pilot):
    out, _ = pilot
    cp = read(out["run_dir"], "checkpoints.csv")
    assert cp["checkpoint_valid"].all()
    multi = cp[cp.search == "multistart"]
    assert (multi["checkpoint"] == multi["explanation_budget"]).all() and not multi["prefix_shortcut"].any()
    for search in ("canonical", "shuffled", "random"):
        rows = cp[cp.search == search]
        assert set(rows["checkpoint"]) == {25, 50}
        assert (rows["prefix_shortcut"] == (rows["checkpoint"] < rows["explanation_budget"])).all()
    assert (cp["explanation_calls_at_best"] <= cp["checkpoint"]).all()
    # every budget is covered for every method: comparison rows exist at 25 and 50
    for budget in (25, 50):
        assert set(cp[cp.checkpoint == budget]["search"]) == {"canonical", "shuffled", "multistart", "random"}


def test_accounting_keeps_explanations_and_lime_rows_separate(pilot):
    out, counter = pilot
    results = read(out["run_dir"], "results.csv")
    assert results["accounting_consistent"].all()
    assert (results["explanation_evaluations"] == results["explanation_calls"]).all()
    assert (results["lime_internal_prediction_batches"] == results["explanation_evaluations"]).all()
    assert (results["lime_internal_prediction_rows"] == LIME_ROWS * results["explanation_evaluations"]).all()
    assert (results["top_level_prediction_calls"] == results["model_predictions"]).all()
    assert counter["calls"] == results["explanation_calls"].sum()  # real explainer invocations
    cached = results.search.isin(["multistart", "shuffled"])  # both run on the multi-start path and report from the cache
    assert (results[cached]["final_report_prediction_calls"] == 0).all()
    assert (results[~cached]["final_report_prediction_calls"] == 1).all()


def test_output_schema_and_heldout_compatibility(pilot, split):
    out, _ = pilot
    run_dir = out["run_dir"]
    for name in ("results.csv", "checkpoints.csv", "best_so_far.csv", "restarts.csv", "manifest.json",
                 "run_config.json", "run_metadata.json", "status.json", "README.md"):
        assert os.path.exists(f"{run_dir}/{name}"), name
    cp = read(run_dir, "checkpoints.csv")
    for column in ("task_id", "search", "method", "strategy", "model_key", "sample_position", "budget", "checkpoint",
                   "checkpoint_valid", "best_score", "candidate_json", "candidate_is_original", "order_seed",
                   "control_seed", "max_explanation_evaluations", "constraint_experimental_valid",
                   "constraint_teammate_table_status", "constraint_physical_status", "constraint_physically_valid_claimed"):
        assert column in cp.columns, column
    assert not cp["constraint_physically_valid_claimed"].any()
    curve = read(run_dir, "best_so_far.csv")
    assert {"task_id", "search", "explanation_calls", "score", "best_so_far"} <= set(curve.columns)
    restarts = read(run_dir, "restarts.csv")
    assert set(restarts["search"]) == {"multistart", "shuffled"} and {"restart_seed", "feature_order", "fresh_call_cap"} <= set(restarts.columns)
    assert (restarts[restarts.search == "shuffled"].groupby("task_id").size() == 1).all()  # K=1
    assert (restarts[restarts.search == "multistart"].groupby("task_id").size() == 3).all()  # K=3
    for task_id in read(run_dir, "results.csv")["task_id"]:
        with gzip.open(f"{run_dir}/tasks/{task_id}.history.csv.gz", "rt", encoding="utf-8") as fh:
            assert len(pd.read_csv(fh)) > 0
    candidates, sources = ho.load_checkpoint_candidates([run_dir], split["X_val"])
    assert len(candidates) == len(cp) and sources[0]["skipped_invalid_checkpoints"] == 0
    assert {c["search"] for c in candidates} == {"canonical", "shuffled", "multistart", "random"}


def test_physical_reporting_is_separate_from_validator_acceptance(pilot):
    out, _ = pilot
    cp = read(out["run_dir"], "checkpoints.csv")
    assert cp["constraint_experimental_valid"].all()  # the searches only save validator-accepted candidates
    assert set(cp["constraint_physical_status"]) <= set(runner.cr.PHYSICAL_STATUS)
    edited = cp[~cp["candidate_is_original"]]
    if len(edited):  # validator acceptance does not imply physical consistency
        assert (edited["constraint_physical_status"] != "consistent_with_checked_rules").any() or edited.empty


# =============================================================================================
# Determinism and manifest
# =============================================================================================
def _strip(frame):
    return frame.drop(columns=[c for c in frame.columns if c in ("runtime_seconds",)]).reset_index(drop=True)


def test_execution_is_deterministic(tmp_path, split, manifest, original, fakes):
    a = run(tmp_path / "a", split, manifest, original)
    b = run(tmp_path / "b", split, manifest, original)
    for name in ("results.csv", "checkpoints.csv", "best_so_far.csv", "restarts.csv"):
        pd.testing.assert_frame_equal(_strip(read(a["run_dir"], name)), _strip(read(b["run_dir"], name)))


def test_manifest_is_written_hashed_and_linked(pilot):
    out, _ = pilot
    run_dir = out["run_dir"]
    text = open(f"{run_dir}/manifest.json", encoding="utf-8").read()
    config = json.loads(open(f"{run_dir}/run_config.json", encoding="utf-8").read())
    meta = json.loads(open(f"{run_dir}/run_metadata.json", encoding="utf-8").read())
    assert runner.sha256_text(text) == config["manifest_sha256"] == meta["manifest"]["sha256"]
    assert meta["manifest"]["path"] == "manifest.json"
    assert mf.validate_manifest(json.loads(text)) == []
    assert meta["official_test_used"] is False and json.loads(text)["datasets"]["official_test_used"] is False
    assert meta["model_sha256"]["random_forest"].startswith("4e49f17d")


# =============================================================================================
# Resume and failure reporting
# =============================================================================================
def test_failures_are_reported_and_resume_runs_only_missing_work(tmp_path, split, manifest, original, fakes, monkeypatch):
    real_execute = runner.execute_task
    state = {"fail": True, "executed": []}
    target = "random_forest_s2941_A_shuffled_b50_o102"

    def flaky(task, ctx):
        state["executed"].append(task.task_id)
        if state["fail"] and task.task_id == target:
            raise RuntimeError("simulated crash")
        return real_execute(task, ctx)

    monkeypatch.setattr(runner, "execute_task", flaky)
    with pytest.raises(runner.RunnerError, match="1 task\\(s\\) failed"):
        run(tmp_path / "x", split, manifest, original)
    run_dir = next((tmp_path / "x").iterdir())
    failed = json.loads((run_dir / "tasks" / f"{target}.failed.json").read_text(encoding="utf-8"))
    assert "simulated crash" in failed["error"] and "Traceback" in failed["traceback"]
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["failed"] == [target] and status["completed"] == 8
    assert json.loads((run_dir / "failures.json").read_text(encoding="utf-8"))["failed_tasks"] == [target]

    state.update(fail=False, executed=[])
    out = run(tmp_path / "x", split, manifest, original, resume_dir=run_dir)
    assert state["executed"] == [target]  # nothing completed is re-run
    assert out["status"]["skipped_already_done"] == 8 and out["status"]["ran_now"] == 1 and out["status"]["failed"] == []
    assert not (run_dir / "tasks" / f"{target}.failed.json").exists() and not (run_dir / "failures.json").exists()

    clean = run(tmp_path / "clean", split, manifest, original)
    for name in ("results.csv", "checkpoints.csv", "best_so_far.csv"):
        pd.testing.assert_frame_equal(_strip(read(run_dir, name)), _strip(read(clean["run_dir"], name)))

    state["executed"] = []
    again = run(tmp_path / "x", split, manifest, original, resume_dir=run_dir)
    assert state["executed"] == [] and again["status"]["ran_now"] == 0


def test_resume_refuses_changed_inputs(tmp_path, split, manifest, original, fakes, protocol):
    out = run(tmp_path, split, manifest, original)
    run_dir = out["run_dir"]
    other = runner.make_selection(protocol, **{**PILOT, "order_seeds": [101]})
    with pytest.raises(runner.RunnerError, match="selection differs"):
        run(tmp_path, split, manifest, original, resume_dir=run_dir, selection=other)
    changed = json.loads(json.dumps(manifest))
    changed["models"]["random_forest"]["sha256"] = "0" * 64
    with pytest.raises(runner.RunnerError, match="changed since the run started"):
        run(tmp_path, split, changed, original, resume_dir=run_dir)
    path = f"{run_dir}/manifest.json"
    text = open(path, encoding="utf-8").read()
    open(path, "w", encoding="utf-8").write(text.replace('"manifest_version": 1', '"manifest_version": 2'))
    with pytest.raises(runner.RunnerError, match="modified after the run started"):
        run(tmp_path, split, manifest, original, resume_dir=run_dir)


def test_missing_model_file_fails_without_retraining(tmp_path, split, manifest, original, fakes):
    broken = json.loads(json.dumps(manifest))
    broken["models"]["random_forest"]["file"] = "results/models/does_not_exist.joblib"
    with pytest.raises(runner.RunnerError, match="never retrains"):
        run(tmp_path, split, broken, original)


# =============================================================================================
# No official test-set usage
# =============================================================================================
class Boom:
    def __getattr__(self, name):
        raise AssertionError("official test data was touched")

    def __len__(self):
        raise AssertionError("official test data was touched")


def test_official_test_set_is_unreachable(tmp_path, split, manifest, original, fakes):
    guarded = runner.guarded_split(split)
    for key in runner.FORBIDDEN_SPLIT_KEYS:
        assert key not in guarded
        with pytest.raises(KeyError, match="official test set"):
            guarded[key]
        with pytest.raises(KeyError):
            guarded.get(key)
    poisoned = {**split, **{key: Boom() for key in runner.FORBIDDEN_SPLIT_KEYS}}
    out = run(tmp_path, poisoned, manifest, original)
    assert out["status"]["failed"] == []


# =============================================================================================
# Integration fixes in unsw_targeted_spoofing.py and unsw_targeted_heldout.py
# =============================================================================================
def test_checkpoint_validity_rule():
    assert ts.checkpoint_is_valid(ts.STRATEGY_MULTISTART, 50, 50)
    assert not ts.checkpoint_is_valid(ts.STRATEGY_MULTISTART, 25, 50)
    for strategy in (ts.STRATEGY_V1, ts.STRATEGY_SHUFFLED):
        assert ts.checkpoint_is_valid(strategy, 25, 50)


def test_run_experiment_flags_invalid_multistart_checkpoints_and_links_manifest(tmp_path, split, manifest, original, fakes, monkeypatch):
    monkeypatch.setattr(ts, "load_xai_model", lambda key, s: FakeModel(original))
    manifest_path = mf.write_manifest(manifest, tmp_path / "manifests")
    out = ts.run_experiment(models=["random_forest"], sample_positions=[SAMPLE], budgets=["A"], max_explanation_evaluations=20,
                            seed=11, output_dir=str(tmp_path / "runs"), strategies=[ts.STRATEGY_V1, ts.STRATEGY_MULTISTART],
                            order_seed=101, control_seed=201, max_random_attempts=2000, checkpoints=[10],
                            manifest_path=str(manifest_path))
    cp = pd.read_csv(f"{out['run_dir']}/checkpoints.csv")
    multi = cp[cp.strategy == ts.STRATEGY_MULTISTART]
    assert list(multi.sort_values("checkpoint")["checkpoint_valid"]) == [False, True]
    assert cp[cp.strategy != ts.STRATEGY_MULTISTART]["checkpoint_valid"].all()
    meta = json.loads(open(f"{out['run_dir']}/run_metadata.json", encoding="utf-8").read())
    assert meta["manifest"]["sha256"] == mf.file_sha256(manifest_path)
    assert meta["manifest"]["protocol_version"] == "step6-v1"
    candidates, sources = ho.load_checkpoint_candidates([out["run_dir"]], split["X_val"])
    assert sources[0]["skipped_invalid_checkpoints"] == 1 and len(candidates) == len(cp) - 1


def test_invalid_manifest_is_rejected_before_any_work(tmp_path, manifest):
    bad = json.loads(json.dumps(manifest))
    bad["models"]["random_forest"]["sha256"] = "0" * 64
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        ts.manifest_link(path)


def test_heldout_skips_invalid_rows_in_legacy_files_without_the_flag(tmp_path, split, original):
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    (run_dir / "run_metadata.json").write_text(json.dumps({"run_id": "legacy", "model_sha256": {"random_forest": "x"}, "lime_seed": 11}), encoding="utf-8")
    original_json = json.dumps(original.to_dict(), default=str)
    rows = [
        dict(method="targeted", strategy=ts.STRATEGY_MULTISTART, model_key="random_forest", sample_position=SAMPLE, budget="A",
             checkpoint=25, max_explanation_evaluations=50, best_score=0.0, candidate_json=original_json, candidate_is_original=True),
        dict(method="targeted", strategy=ts.STRATEGY_MULTISTART, model_key="random_forest", sample_position=SAMPLE, budget="A",
             checkpoint=50, max_explanation_evaluations=50, best_score=0.0, candidate_json=original_json, candidate_is_original=True),
    ]
    pd.DataFrame(rows).to_csv(run_dir / "checkpoints.csv", index=False)
    candidates, sources = ho.load_checkpoint_candidates([run_dir], split["X_val"])
    assert len(candidates) == 1 and candidates[0]["checkpoint"] == 50
    assert sources[0]["skipped_invalid_checkpoints"] == 1


def _fake_heldout_explain(model_key, sample, row, seed):
    rng = np.random.default_rng(1000 + seed)
    weights = {f: 100.0 - i for i, f in enumerate(FEATURE_ORDER)}
    for f, eps in zip(FEATURE_ORDER, rng.normal(0.0, 0.6, len(FEATURE_ORDER))):
        weights[f] += eps
    for f in ts.NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES:
        weights[f] += 60.0 * float(row[f])
    return weights


def test_heldout_checks_that_the_search_score_is_reproduced():
    def row(**values):
        base = {feature: 0.0 for feature in FEATURE_ORDER}
        base.update({"proto": "udp", "service": "dns", "state": "INT"}, **values)
        return pd.Series(base)

    original, candidate = row(), row(sjit=1.0, djit=1.0)
    true_score = ts.objective_score(_fake_heldout_explain("m", 0, original, 11), _fake_heldout_explain("m", 0, candidate, 11))
    members = [
        {"candidate_uid": "right", "model_key": "m", "sample_position": 0, "original_row": original, "candidate_row": candidate,
         "search_seed_objective": true_score},
        {"candidate_uid": "wrong", "model_key": "m", "sample_position": 0, "original_row": original, "candidate_row": candidate,
         "search_seed_objective": true_score + 1e-3},
    ]
    summary = {r["candidate_uid"]: r for r in ho.evaluate_candidates(members, _fake_heldout_explain)["summary"]}
    assert summary["right"]["search_score_reproduced"] is True and summary["right"]["search_score_gap"] == 0.0
    assert summary["wrong"]["search_score_reproduced"] is False
    assert summary["wrong"]["search_score_gap"] == pytest.approx(1e-3)


# =============================================================================================
# Manifest hash and line endings (regression: runner v1 wrote CRLF on Windows but hashed LF text)
# =============================================================================================
def test_manifest_hash_equals_sha256_of_the_file_bytes(pilot):
    out, _ = pilot
    run_dir = out["run_dir"]
    config = json.loads(open(f"{run_dir}/run_config.json", encoding="utf-8").read())
    raw = open(f"{run_dir}/manifest.json", "rb").read()
    assert b"\r\n" not in raw
    assert runner.hashlib.sha256(raw).hexdigest() == config["manifest_sha256"]
    assert runner.manifest_hash_status(f"{run_dir}/manifest.json", config["manifest_sha256"]) == "raw_bytes"
    for name in ("run_config.json", "run_metadata.json", "status.json", "README.md"):
        assert b"\r\n" not in open(f"{run_dir}/{name}", "rb").read(), name


def test_legacy_crlf_manifest_still_verifies_and_tampering_is_detected(tmp_path):
    text = '{\n  "a": 1,\n  "b": [\n    2\n  ]\n}'
    recorded = runner.sha256_text(text)  # what runner v1 recorded (hash of the LF text)
    legacy = tmp_path / "manifest.json"
    legacy.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))  # what runner v1 wrote on Windows
    assert runner.manifest_hash_status(legacy, recorded) == "lf_normalized"
    modern = tmp_path / "manifest_lf.json"
    modern.write_bytes(text.encode("utf-8"))
    assert runner.manifest_hash_status(modern, recorded) == "raw_bytes"
    tampered = tmp_path / "manifest_bad.json"
    tampered.write_bytes(text.replace("1", "3").replace("\n", "\r\n").encode("utf-8"))
    assert runner.manifest_hash_status(tampered, recorded) is None


def test_resume_accepts_a_legacy_crlf_manifest(tmp_path, split, manifest, original, fakes):
    out = run(tmp_path, split, manifest, original)
    path = f"{out['run_dir']}/manifest.json"
    raw = open(path, "rb").read()
    open(path, "wb").write(raw.replace(b"\n", b"\r\n"))  # simulate a v1 run written on Windows
    again = run(tmp_path, split, manifest, original, resume_dir=out["run_dir"])
    assert again["status"]["ran_now"] == 0 and again["status"]["failed"] == []
