"""Step 6 v2 infrastructure (stage 3): protocol v2, manifest, runner identity/policy/resume, exhaustive evaluator.

Fake models and explanations only: no LIME, no training, no real search.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import experiments.unsw_step6_exhaustive as ex
import experiments.unsw_step6_runner as runner
import experiments.unsw_targeted_spoofing as ts
from experiments import unsw_step6_manifest as mf
from src.unsw import physical_repair as pr
from src.unsw.constraints import FEATURE_ORDER
from test_unsw_step6_runner import FakeModel, fakes, original, split  # noqa: F401  (fixtures)

ROOT = Path(__file__).resolve().parents[1]
V1_RUN = ROOT / "results" / "step6_minimum" / "run_20261106_203918"


@pytest.fixture(scope="module")
def protocol_v1():
    return mf.load_protocol()


@pytest.fixture(scope="module")
def protocol_v2():
    return mf.load_protocol(mf.PROTOCOL_V2_PATH)


@pytest.fixture(scope="module")
def manifest_v2(protocol_v2):
    return mf.build_manifest(protocol_v2)


# ---------------------------------------------------------------------------------------------
# Protocol v2
# ---------------------------------------------------------------------------------------------
def test_v2_protocol_is_valid_and_separate_from_v1(protocol_v1, protocol_v2):
    assert mf.validate_protocol(protocol_v2) == []
    assert mf.is_v2(protocol_v2) and not mf.is_v2(protocol_v1)
    assert mf.validate_protocol(protocol_v1) == []
    assert protocol_v2["sample_positions"] == [2941, 3040, 3206, 27094, 15137, 4407, 30107, 34186, 15741, 25740]
    assert protocol_v2["search_objectives"]["multi_seed_k3"]["lime_seeds"] == [11, 66, 77]
    assert protocol_v2["primary_analysis"]["fallback"] == "none adopted"
    assert "definitive only for the enumerated" in protocol_v2["exhaustive_evaluation"]["scope"]


@pytest.mark.parametrize("change, message", [
    (lambda p: p["search_objectives"]["multi_seed_k3"].update(lime_seeds=[11, 22, 77]), "overlap"),
    (lambda p: p["search_objectives"]["multi_seed_k3"].update(lime_seeds=[66, 77]), "include the search seed"),
    (lambda p: p.update(defense_lime_seeds=[33, 302]), "defense_lime_seeds"),
    (lambda p: p.update(constraint_policies=["physical_v1", "magic"]), "constraint_policies"),
    (lambda p: p["exhaustive_evaluation"].update(lime_seeds=[11, 22]), "exhaustive_evaluation lime_seeds"),
    (lambda p: p["grids"]["G1"].update(search_objectives=["nope"]), "unknown search objective"),
])
def test_v2_protocol_validation_catches_errors(protocol_v2, change, message):
    bad = copy.deepcopy(protocol_v2)
    change(bad)
    assert any(message in p for p in mf.validate_protocol(bad)), mf.validate_protocol(bad)


# ---------------------------------------------------------------------------------------------
# Planner: task identity and counts
# ---------------------------------------------------------------------------------------------
def _grid(protocol, name, **extra):
    grid = protocol["grids"][name]
    return runner.make_selection(protocol, methods=grid["methods"], constraint_policies=grid["constraint_policies"],
                                 search_objectives=grid["search_objectives"], **extra)


@pytest.mark.parametrize("grid, tasks, lime, checkpoints", [
    ("G1", 2040, 154_000, 3720),
    ("G2", 800, 80_000, 2400),
    ("G3", 840, 84_000, 2520),
])
def test_v2_planned_task_counts(protocol_v2, grid, tasks, lime, checkpoints):
    summary = runner.plan_summary(runner.plan_tasks(_grid(protocol_v2, grid)))
    assert (summary["tasks"], summary["max_lime_calls"], summary["checkpoint_candidates"]) == (tasks, lime, checkpoints)


def test_v2_task_identity_includes_policy_and_objective(protocol_v2):
    tasks = runner.plan_tasks(runner.make_selection(protocol_v2))
    ids = [t.task_id for t in tasks]
    assert len(ids) == len(set(ids)) == 2 * 2 * 2040   # 2 policies x 2 objectives x full grid
    t = next(t for t in tasks if t.constraint_policy == "physical_v1" and t.search_objective == "multi_seed_k3")
    assert "_physical_v1_multi_seed_k3_" in t.task_id
    assert t.spec()["constraint_policy"] == "physical_v1" and t.spec()["search_objective"] == "multi_seed_k3"


def test_v1_selection_and_specs_are_unchanged(protocol_v1):
    selection = runner.make_selection(protocol_v1)
    assert selection.constraint_policies is None and selection.search_objectives is None
    assert set(selection.to_json()) == {"models", "samples", "budgets", "explanation_budgets", "order_seeds", "control_seeds", "methods"}
    with pytest.raises(ValueError, match="v1"):
        runner.make_selection(protocol_v1, constraint_policies=["physical_v1"])
    spec = runner.plan_tasks(selection)[0].spec()
    assert "constraint_policy" not in spec and "search_objective" not in spec


# ---------------------------------------------------------------------------------------------
# Manifest v2
# ---------------------------------------------------------------------------------------------
def test_v2_manifest_records_the_new_hashes(manifest_v2, protocol_v2):
    assert mf.validate_manifest(manifest_v2) == []
    hashes = manifest_v2["implementation_hashes"]
    assert hashes["physical_repair_module"] == mf.file_sha256(ROOT / "src" / "unsw" / "physical_repair.py")
    assert hashes["physical_report_v2"] == mf.source_sha256(pr.physical_report)
    assert hashes["exhaustive_script"] == mf.file_sha256(ROOT / "experiments" / "unsw_step6_exhaustive.py")
    assert manifest_v2["seeds"]["search_objectives"]["multi_seed_k3"] == [11, 66, 77]
    assert manifest_v2["datasets"]["official_test_used"] is False
    assert manifest_v2["constraint_policy"]["physical_v1"]["editable"] == ["sbytes", "spkts"]


def test_v1_manifest_structure_is_unchanged():
    saved = json.loads((V1_RUN / "manifest.json").read_text(encoding="utf-8"))
    fresh = mf.build_manifest(mf.load_protocol())
    assert set(fresh) == set(saved)
    assert set(fresh["implementation_hashes"]) == set(saved["implementation_hashes"])
    assert set(fresh["seeds"]) == set(saved["seeds"])
    assert fresh["protocol_sha256"] == saved["protocol_sha256"]


# ---------------------------------------------------------------------------------------------
# Runner: v2 execution, policy enforcement, physical_report_v2, matched groups, resume
# ---------------------------------------------------------------------------------------------
V2_SMALL = dict(models=["random_forest"], samples=[15741], budgets=["A"], explanation_budgets=[25, 50],
                order_seeds=[101, 102], control_seeds=[201, 202], search_objectives=["single_seed"])


def _run_v2(tmp_path, split, original, protocol_v2, manifest_v2, policies, resume_dir=None):
    selection = None if resume_dir else runner.make_selection(protocol_v2, constraint_policies=policies, **V2_SMALL)
    return runner.run_step6(selection, output_root=tmp_path, resume_dir=resume_dir, split=split, manifest=manifest_v2,
                            protocol=protocol_v2, model_loader=lambda key, s: FakeModel(original), log=lambda m: None)


def test_v2_run_enforces_policy_and_uses_physical_report_v2(tmp_path, split, original, fakes, protocol_v2, manifest_v2):
    out = _run_v2(tmp_path, split, original, protocol_v2, manifest_v2, ["physical_v1", "current_v1"])
    assert out["status"]["completed"] == out["status"]["planned"] == 18 and not out["status"]["failed"]
    run_dir = Path(out["run_dir"])
    results = pd.read_csv(run_dir / "results.csv")
    checkpoints = pd.read_csv(run_dir / "checkpoints.csv")
    assert set(results["constraint_policy"]) == {"physical_v1", "current_v1"}
    assert all("_physical_v1_single_seed_" in t or "_current_v1_single_seed_" in t for t in results["task_id"])

    physical = results[results["constraint_policy"] == "physical_v1"]
    original_15741 = split["X_val"].iloc[[15741]].iloc[0]
    lattice = len(pr.valid_lattice_candidates(original_15741, "A"))
    assert (physical["lattice_valid_candidates"] == lattice).all()
    assert ((physical["lattice_coverage"] > 0) & (physical["lattice_coverage"] <= 1)).all()
    assert physical["legal_moves_digest"].nunique() == 1
    assert set(results["stopped_early"]) <= {True, False}
    assert (results[results["constraint_policy"] == "current_v1"]["lattice_coverage"].isna()).all()

    assert (checkpoints["constraint_physical_report_version"] == "physical_report_v2").all()
    assert "constraint_legacy_v1_physical_status" in checkpoints.columns
    assert (checkpoints["constraint_physically_valid_claimed"] == False).all()  # noqa: E712
    assert set(checkpoints["search_objective"]) == {"single_seed"}
    phys_cp = checkpoints[checkpoints["constraint_policy"] == "physical_v1"]
    assert phys_cp["constraint_policy_valid"].astype(bool).all()
    assert (phys_cp["constraint_physical_status"] != "violates_checked_rule").all()
    for text in phys_cp["constraint_primary_changes"]:
        assert set(json.loads(text)) <= {"sbytes", "spkts"}
    # The new report labels spkts edits as unverified (the v1 classification did not).
    for _, row in phys_cp.iterrows():
        if "spkts" in json.loads(row["constraint_primary_changes"]):
            assert row["constraint_physical_status"] == "unverified_dependencies"


def test_multi_seed_objective_is_refused_until_implemented(tmp_path, split, original, fakes, protocol_v2, manifest_v2):
    selection = runner.make_selection(protocol_v2, **{**V2_SMALL, "search_objectives": ["multi_seed_k3"]},
                                      constraint_policies=["physical_v1"], methods=["random"])
    with pytest.raises(runner.RunnerError, match="failed"):
        runner.run_step6(selection, output_root=tmp_path, split=split, manifest=manifest_v2, protocol=protocol_v2,
                         model_loader=lambda key, s: FakeModel(original), log=lambda m: None)
    failed = list(Path(tmp_path).rglob("*.failed.json"))
    assert failed and "not implemented" in failed[0].read_text(encoding="utf-8")


def test_matched_group_problems_detects_a_different_lattice():
    rows = [dict(model_key="random_forest", sample_position=1, budget="A", constraint_policy="physical_v1",
                 search_objective="single_seed", legal_moves_digest=d) for d in ("x", "x", "y")]
    assert runner.matched_group_problems(rows[:2]) == []
    assert runner.matched_group_problems(rows) and "unmatched" in runner.matched_group_problems(rows)[0]
    assert runner.matched_group_problems([dict(rows[0], constraint_policy=None)]) == []


def test_v2_resume_reruns_only_missing_tasks(tmp_path, split, original, fakes, protocol_v2, manifest_v2):
    out = _run_v2(tmp_path, split, original, protocol_v2, manifest_v2, ["physical_v1"])
    run_dir = Path(out["run_dir"])
    before = pd.read_csv(run_dir / "results.csv").drop(columns=["runtime_seconds"])
    victim = sorted((run_dir / "tasks").glob("*_random_*.json"))[0]
    victim.unlink()
    calls_before = fakes["calls"]
    again = _run_v2(tmp_path, split, original, protocol_v2, manifest_v2, ["physical_v1"], resume_dir=run_dir)
    assert again["status"]["ran_now"] == 1 and again["status"]["skipped_already_done"] == 8
    assert fakes["calls"] - calls_before <= 50  # one task's explanations, not the whole run
    after = pd.read_csv(run_dir / "results.csv").drop(columns=["runtime_seconds"])
    pd.testing.assert_frame_equal(before, after)


# ---------------------------------------------------------------------------------------------
# Exhaustive evaluator
# ---------------------------------------------------------------------------------------------
EX_SAMPLES = [2941, 15741]


def _fake_lime(split):
    X_val = split["X_val"]
    base = {f: float(i + 1) for i, f in enumerate(FEATURE_ORDER)}
    counter = {"calls": 0}

    def explain(model_key, sample, row, seed):
        counter["calls"] += 1
        original = X_val.iloc[[sample]].iloc[0]
        weights = {f: v + 0.01 * ((seed * (i + 3)) % 7) for i, (f, v) in enumerate(base.items())}
        for f in pr.PHYSICAL_V1_EDITABLE + tuple(pr.REPAIRED_FEATURES):
            weights[f] += 300.0 * ts._relative_change(float(original[f]), float(row[f])) * (2 if model_key == "xgboost" else 1)
        return weights

    return explain, counter


def _models():
    return {"random_forest": FakeModel(None), "xgboost": FakeModel(None, prob=0.9)}


def _expected_unique(split):
    keys = set()
    for s in EX_SAMPLES:
        row = split["X_val"].iloc[[s]].iloc[0]
        for b in ("A", "B"):
            keys |= {(s, ex.candidate_key(c)) for _, c, _ in pr.valid_lattice_candidates(row, b)}
    return keys


class TrapSplit(dict):
    def __getitem__(self, key):
        if key in runner.FORBIDDEN_SPLIT_KEYS:
            raise AssertionError(f"official test set accessed: {key}")
        return super().__getitem__(key)


def test_exhaustive_deduplicates_and_keeps_identities(tmp_path, split, protocol_v2, manifest_v2):
    explain, counter = _fake_lime(split)
    out = ex.run_exhaustive(protocol_v2, tmp_path, split=TrapSplit(split), manifest=manifest_v2, models=_models(),
                            explain_fn=explain, samples=EX_SAMPLES, log=lambda m: None)
    run_dir = Path(out["run_dir"])
    expected = _expected_unique(split)
    plan = out["plan"]
    assert plan["unique_valid_candidates"] == len(expected)
    assert plan["lime_calls_total"] == 2 * (len(expected) + len(EX_SAMPLES)) * 5
    assert counter["calls"] == plan["lime_calls_total"]                      # each unique explanation exactly once
    identities = pd.read_csv(run_dir / "identities.csv")
    valid = identities[identities["valid"]]
    assert set(valid["budget"]) == {"A", "B"} and set(valid["policy"]) == {"physical_v1"}
    assert set(valid["model_key"]) == {"random_forest", "xgboost"}
    assert valid["candidate_uid"].nunique() == len(expected)
    a_uids = set(valid[valid["budget"] == "A"]["candidate_uid"])
    assert a_uids <= set(valid[valid["budget"] == "B"]["candidate_uid"])     # A lattice is part of B: shared uids
    summary = pd.read_csv(run_dir / "heldout_summary.csv")
    assert len(summary) == 2 * len(expected)
    assert (summary["physically_valid_claimed"] == False).all()  # noqa: E712
    assert not summary.loc[summary["physical_status"] != "consistent_with_checked_rules", "lime_success"].any()
    groups = pd.read_csv(run_dir / "summary_by_status_group.csv")
    assert set(groups["physical_status_group"]) == set(ex.STATUS_GROUPS)
    s2941 = groups[(groups["sample_position"] == 2941) & (groups["physical_status_group"] == "unverified_dependencies")]
    assert (s2941["valid_lattice_candidates"] == 0).all()                    # 2-packet flow: no spkts edits
    assert json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))["official_test_used"] is False


def test_exhaustive_resume_after_crash_and_truncated_line(tmp_path, split, protocol_v2, manifest_v2):
    explain_full, _ = _fake_lime(split)
    full = ex.run_exhaustive(protocol_v2, tmp_path / "full", split=split, manifest=manifest_v2, models=_models(),
                             explain_fn=explain_full, samples=EX_SAMPLES, log=lambda m: None)

    explain, counter = _fake_lime(split)
    with pytest.raises(ex.Interrupted):
        ex.run_exhaustive(protocol_v2, tmp_path / "crash", split=split, manifest=manifest_v2, models=_models(),
                          explain_fn=explain, samples=EX_SAMPLES, max_new_explanations=300, log=lambda m: None)
    run_dir = next((tmp_path / "crash").glob("exhaustive_*"))
    assert json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"] == "interrupted"
    with (run_dir / ex.CACHE_NAME).open("a", encoding="utf-8") as fh:
        fh.write('{"type": "explanation", "model_key": "random_fo')               # crash in the middle of a write
    first = counter["calls"]
    assert first == 300
    resumed = ex.run_exhaustive(protocol_v2, tmp_path / "crash", resume_dir=run_dir, split=split, manifest=manifest_v2,
                                models=_models(), explain_fn=explain, samples=EX_SAMPLES, log=lambda m: None)
    assert counter["calls"] == full["plan"]["lime_calls_total"]             # nothing computed twice
    assert resumed["status"]["reused"] >= 300
    cols = [c for c in full["summary"].columns]
    pd.testing.assert_frame_equal(full["summary"][cols].reset_index(drop=True), resumed["summary"][cols].reset_index(drop=True))


def test_exhaustive_resume_refuses_changed_inputs(tmp_path, split, protocol_v2, manifest_v2):
    explain, _ = _fake_lime(split)
    with pytest.raises(ex.Interrupted):
        ex.run_exhaustive(protocol_v2, tmp_path, split=split, manifest=manifest_v2, models=_models(), explain_fn=explain,
                          samples=EX_SAMPLES, max_new_explanations=5, log=lambda m: None)
    run_dir = next(Path(tmp_path).glob("exhaustive_*"))
    changed = copy.deepcopy(manifest_v2)
    changed["models"]["random_forest"]["sha256"] = "0" * 64
    with pytest.raises(ex.ExhaustiveError, match="model_sha256"):
        ex.run_exhaustive(protocol_v2, tmp_path, resume_dir=run_dir, split=split, manifest=changed, models=_models(),
                          explain_fn=explain, samples=EX_SAMPLES, log=lambda m: None)


def test_exhaustive_refuses_v1_protocol_and_foreign_samples(tmp_path, split, protocol_v1, protocol_v2, manifest_v2):
    with pytest.raises(ex.ExhaustiveError):
        ex.run_exhaustive(protocol_v1, tmp_path, split=split, manifest=manifest_v2, models=_models(),
                          explain_fn=_fake_lime(split)[0], log=lambda m: None)
    with pytest.raises(ex.ExhaustiveError, match="protocol"):
        ex.run_exhaustive(protocol_v2, tmp_path, split=split, manifest=manifest_v2, models=_models(),
                          explain_fn=_fake_lime(split)[0], samples=[1], log=lambda m: None)
