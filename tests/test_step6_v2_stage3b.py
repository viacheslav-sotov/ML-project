"""Step 6 v2 stage 3B: confirmation seeds, pilot labelling, sample-identity and model-hash safeguards.

Fake models and explanations only (no LIME, no training). The real model files are hashed, never refitted.
"""

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

import experiments.unsw_step6_exhaustive as ex
import experiments.unsw_step6_runner as runner
import experiments.unsw_targeted_spoofing as ts
from experiments import unsw_step6_manifest as mf
from test_step6_v2_infrastructure import EX_SAMPLES, TrapSplit, _fake_lime, _models  # noqa: F401
from test_unsw_step6_runner import split  # noqa: F401  (fixture)

CONFIRM = [401, 402, 403, 404]
PINS = {"random_forest": "4e49f17d9b19747af9e4da472866c4385062d8018582bd7b4872983aefd56531",
        "xgboost": "83a34c617a41b886b628ed01257994360f36e8c6863edbc439ee4b201ac4f4b1"}


@pytest.fixture(scope="module")
def protocol_v2():
    return mf.load_protocol(mf.PROTOCOL_V2_PATH)


@pytest.fixture(scope="module")
def manifest_v2(protocol_v2):
    return mf.build_manifest(protocol_v2)


def _run(tmp_path, split, protocol, manifest, explain, **kw):
    return ex.run_exhaustive(protocol, tmp_path, split=TrapSplit(split), manifest=manifest, models=_models(),
                             explain_fn=explain, samples=kw.pop("samples", EX_SAMPLES), log=lambda m: None, **kw)


# ---------------------------------------------------------------------------------------------
# Protocol amendment
# ---------------------------------------------------------------------------------------------
def test_protocol_declares_pilot_confirmation_pins_and_identity(protocol_v2):
    spec = protocol_v2["exhaustive_evaluation"]
    assert spec["role"] == "pilot" and "not the final experimental design" in spec["role_note"]
    assert spec["confirmation"]["lime_seeds"] == CONFIRM
    assert protocol_v2["expected_model_sha256"] == PINS
    assert set(protocol_v2["sample_identity"]) == {str(s) for s in protocol_v2["sample_positions"]}
    # success criterion D8 and the primary analysis are unchanged
    assert protocol_v2["success_criteria"]["lime"].startswith("still attack (P >= 0.5) AND passes the enforced policy")
    assert protocol_v2["primary_analysis"]["fallback"] == "none adopted"


@pytest.mark.parametrize("change, message", [
    (lambda p: p["exhaustive_evaluation"]["confirmation"].update(lime_seeds=[401, 22, 403, 404]), "overlap another seed family"),
    (lambda p: p["exhaustive_evaluation"]["confirmation"].update(lime_seeds=[401, 301]), "overlap another seed family"),
    (lambda p: p["exhaustive_evaluation"]["confirmation"].update(lime_seeds=[401, 66]), "overlap another seed family"),
    (lambda p: p["exhaustive_evaluation"]["confirmation"].update(lime_seeds=[401]), "at least two distinct"),
    (lambda p: p["exhaustive_evaluation"].update(role="final_maybe"), "role"),
    (lambda p: p["expected_model_sha256"].update(xgboost="abc"), "expected_model_sha256"),
    (lambda p: p["sample_identity"].pop("2941"), "sample_identity"),
])
def test_protocol_validation_catches_stage3b_errors(protocol_v2, change, message):
    bad = copy.deepcopy(protocol_v2)
    change(bad)
    assert any(message in p for p in mf.validate_protocol(bad)), mf.validate_protocol(bad)


# ---------------------------------------------------------------------------------------------
# Selection: identity and role
# ---------------------------------------------------------------------------------------------
def test_sample_identity_matches_the_real_split(protocol_v2, split):
    ex.check_sample_identity(protocol_v2, split, protocol_v2["sample_positions"])   # all 10 pinned rows are attacks


def test_sample_identity_mismatch_is_refused(tmp_path, protocol_v2, manifest_v2, split):
    bad = copy.deepcopy(protocol_v2)
    bad["sample_identity"]["2941"]["official_train_row_index"] += 1
    with pytest.raises(ex.ExhaustiveError, match="sample identity"):
        _run(tmp_path, split, bad, manifest_v2, _fake_lime(split)[0])
    assert not list(Path(tmp_path).glob("exhaustive_*"))                          # refused before a run folder exists


def test_run_role_labels_pilot_and_smoke_test(tmp_path, protocol_v2, manifest_v2, split):
    assert ex.run_role(protocol_v2, protocol_v2["sample_positions"]) == "pilot"
    assert ex.run_role(protocol_v2, list(reversed(protocol_v2["sample_positions"]))) == "pilot"
    assert ex.run_role(protocol_v2, [2941]) == "smoke_test"
    out = _run(tmp_path, split, protocol_v2, manifest_v2, _fake_lime(split)[0])
    run_dir = Path(out["run_dir"])
    assert run_dir.name.startswith("exhaustive_smoke_test_")
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert config["run_role"] == "smoke_test" and config["header"]["run_role"] == "smoke_test"
    assert "(smoke_test)" in (run_dir / "README.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# Confirmation seeds
# ---------------------------------------------------------------------------------------------
def test_confirmation_runs_only_for_flagged_candidates_and_keeps_d8(tmp_path, protocol_v2, manifest_v2, split):
    explain, counter = _fake_lime(split)
    seen = []
    out = _run(tmp_path, split, protocol_v2, manifest_v2, lambda m, s, r, seed: (seen.append(seed), explain(m, s, r, seed))[1])
    summary, status = out["summary"], out["status"]
    flagged = summary[summary["exceeds_heldout_both"]]
    assert len(flagged) > 0                                                      # the fake shifts are large
    assert summary["confirmation_evaluated"].equals(summary["exceeds_heldout_both"].astype(bool))
    # cost: 4 seeds per flagged unique candidate and per original with a flagged candidate
    unique_pairs = flagged.groupby("model_key")["unique_candidate_uid"].nunique().sum()
    originals = flagged.groupby(["model_key", "sample_position"]).ngroups
    assert status["confirmation_explanations_computed"] == 4 * (unique_pairs + originals)
    assert status["confirmation_explanations_computed"] <= out["plan"]["confirmation_lime_calls_max"]
    assert sum(1 for s in seen if s in CONFIRM) == status["confirmation_explanations_computed"]
    assert counter["calls"] == out["plan"]["lime_calls_total"] + status["confirmation_explanations_computed"]
    # D8 unchanged: lime_success is exactly the combined held-out test AND consistent_with_checked_rules
    d8 = summary["exceeds_heldout_both"].astype(bool) & (summary["physical_status"] == "consistent_with_checked_rules")
    assert summary["lime_success"].equals(d8)
    assert (summary["confirmed_lime_success"] <= summary["lime_success"]).all()
    assert summary.loc[~summary["confirmation_evaluated"], "confirm_top10_jaccard_mean"].isna().all()
    assert set(pd.read_csv(Path(out["run_dir"]) / "confirmation_per_seed.csv")["seed"]) == set(CONFIRM)


def test_confirmation_can_fail_and_is_reported_separately(tmp_path, protocol_v2, manifest_v2, split):
    explain, _ = _fake_lime(split)
    X_val = split["X_val"]

    def unshifted_on_confirmation(model_key, sample, row, seed):
        if seed in CONFIRM:                                                      # no effect under the new seeds
            return explain(model_key, sample, X_val.iloc[[sample]].iloc[0], seed)
        return explain(model_key, sample, row, seed)

    out = _run(tmp_path, split, protocol_v2, manifest_v2, unshifted_on_confirmation)
    summary = out["summary"]
    assert summary["lime_success"].any()
    assert not summary["confirmation_passed"].any() and not summary["confirmed_lime_success"].any()
    overview = json.loads((Path(out["run_dir"]) / "pilot_summary.json").read_text(encoding="utf-8"))["by_model_and_status_group"]
    assert sum(r["lime_success"] for r in overview) == int(summary["lime_success"].sum())
    assert sum(r["confirmed_lime_success"] for r in overview) == 0


def test_status_groups_stay_separate_with_confirmation(tmp_path, protocol_v2, manifest_v2, split):
    out = _run(tmp_path, split, protocol_v2, manifest_v2, _fake_lime(split)[0])
    summary = out["summary"]
    unverified = summary[summary["physical_status"] == "unverified_dependencies"]
    assert len(unverified) and unverified["confirmation_evaluated"].any()      # confirmation is reported for them too
    assert not unverified["lime_success"].any() and not unverified["confirmed_lime_success"].any()
    groups = pd.read_csv(Path(out["run_dir"]) / "summary_by_status_group.csv")
    for column in ("confirmation_evaluated", "confirmation_passed", "confirmed_lime_success_count"):
        assert column in groups
    assert groups.loc[groups["physical_status_group"] == "unverified_dependencies", "confirmed_lime_success_count"].sum() == 0


# ---------------------------------------------------------------------------------------------
# Resume across the confirmation phase
# ---------------------------------------------------------------------------------------------
def test_resume_during_confirmation_is_identical(tmp_path, protocol_v2, manifest_v2, split):
    full = _run(tmp_path / "full", split, protocol_v2, manifest_v2, _fake_lime(split)[0])
    base = full["plan"]["lime_calls_total"]
    assert full["status"]["confirmation_explanations_computed"] > 3
    explain, counter = _fake_lime(split)
    with pytest.raises(ex.Interrupted):
        _run(tmp_path / "crash", split, protocol_v2, manifest_v2, explain, max_new_explanations=base + 3)
    run_dir = next((tmp_path / "crash").glob("exhaustive_*"))
    assert json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"] == "interrupted"
    resumed = _run(tmp_path / "crash", split, protocol_v2, manifest_v2, explain, resume_dir=run_dir)
    assert counter["calls"] == full["status"]["computed_now"]                  # nothing computed twice
    assert resumed["status"]["reused"] >= base + 3
    pd.testing.assert_frame_equal(full["summary"].reset_index(drop=True), resumed["summary"].reset_index(drop=True))


def test_resume_refuses_changed_confirmation_seeds(tmp_path, protocol_v2, manifest_v2, split):
    explain, _ = _fake_lime(split)
    with pytest.raises(ex.Interrupted):
        _run(tmp_path, split, protocol_v2, manifest_v2, explain, max_new_explanations=5)
    run_dir = next(Path(tmp_path).glob("exhaustive_*"))
    changed = copy.deepcopy(protocol_v2)
    changed["exhaustive_evaluation"]["confirmation"]["lime_seeds"] = [401, 402, 403, 405]
    manifest = copy.deepcopy(manifest_v2)
    with pytest.raises(ex.ExhaustiveError, match="confirmation_lime_seeds"):
        _run(tmp_path, split, changed, manifest, explain, resume_dir=run_dir)


# ---------------------------------------------------------------------------------------------
# Model-hash safeguards (real files hashed; the loader is never allowed to run on a bad file)
# ---------------------------------------------------------------------------------------------
@pytest.fixture
def forbid_loading(monkeypatch):
    def no_loading(*args, **kwargs):
        raise AssertionError("model loading must not be reached")
    monkeypatch.setattr(ts, "load_xai_model", no_loading)


def test_real_model_files_match_the_pins(protocol_v2, manifest_v2):
    shas = runner.verify_model_files(runner.make_selection(protocol_v2, models=list(PINS)), manifest_v2)
    assert shas == PINS


def test_wrong_pinned_hash_is_refused_before_loading(tmp_path, protocol_v2, manifest_v2, split, forbid_loading):
    bad = copy.deepcopy(protocol_v2)
    bad["expected_model_sha256"]["random_forest"] = "0" * 64
    manifest = copy.deepcopy(manifest_v2)
    manifest["protocol"] = bad
    with pytest.raises(ex.ExhaustiveError, match="pinned hash"):
        ex.run_exhaustive(bad, tmp_path, split=split, manifest=manifest, explain_fn=_fake_lime(split)[0],
                          samples=EX_SAMPLES, log=lambda m: None)
    assert not list(Path(tmp_path).glob("exhaustive_*"))


def test_missing_model_file_is_refused_never_refitted(tmp_path, protocol_v2, manifest_v2, split, forbid_loading):
    manifest = copy.deepcopy(manifest_v2)
    manifest["models"]["xgboost"]["file"] = "results/models/does_not_exist.joblib"
    with pytest.raises(ex.ExhaustiveError, match="missing"):
        ex.run_exhaustive(protocol_v2, tmp_path, split=split, manifest=manifest, explain_fn=_fake_lime(split)[0],
                          samples=EX_SAMPLES, log=lambda m: None)
    assert not list(Path(tmp_path).glob("exhaustive_*"))


def test_runner_refuses_a_model_that_differs_from_the_pin(protocol_v2, manifest_v2):
    manifest = copy.deepcopy(manifest_v2)
    manifest["protocol"]["expected_model_sha256"]["xgboost"] = "f" * 64
    with pytest.raises(runner.RunnerError, match="pinned hash"):
        runner.verify_model_files(runner.make_selection(protocol_v2, models=["xgboost"]), manifest)


def test_injected_manifest_hash_must_match_the_pin(tmp_path, protocol_v2, manifest_v2, split):
    manifest = copy.deepcopy(manifest_v2)
    manifest["models"]["xgboost"]["sha256"] = "1" * 64
    with pytest.raises(ex.ExhaustiveError, match="pinned hash"):
        _run(tmp_path, split, protocol_v2, manifest, _fake_lime(split)[0])


def test_model_change_during_the_run_is_detected(tmp_path):
    path = tmp_path / "model.joblib"
    path.write_bytes(b"model-a")
    shas = {"m": mf.file_sha256(path)}
    assert ex.assert_models_unchanged({"m": path}, shas) == shas
    path.write_bytes(b"model-b")
    with pytest.raises(ex.ExhaustiveError, match="changed during the run"):
        ex.assert_models_unchanged({"m": path}, shas)
    path.unlink()
    with pytest.raises(ex.ExhaustiveError, match="changed during the run"):
        ex.assert_models_unchanged({"m": path}, shas)
