"""E0 runner and frozen-sample loader: refusals and identities (no explanation is computed here)."""

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from experiments import unsw_v3_e0 as e0
from src.unsw import v3_sample as vs

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "experiments" / "step6_protocol_v3.json").read_text(encoding="utf-8"))
frozen = pytest.mark.skipif(not (vs.SAMPLE_DIR / "sample_manifest.csv").exists(), reason="frozen sample not present")


@frozen
def test_loader_accepts_only_the_pinned_sample():
    s = vs.load_frozen_sample(PROTOCOL)
    assert len(s) == 263 and vs.sample_digest(s) == PROTOCOL["attack_sample"]["sample_sha256"]


@frozen
def test_loader_refuses_a_changed_file_and_the_discarded_draw(tmp_path, monkeypatch):
    bad = copy.deepcopy(PROTOCOL)
    bad["attack_sample"]["sample_manifest_csv_sha256"] = "0" * 64
    with pytest.raises(vs.SampleError, match="pinned SHA-256"):
        vs.load_frozen_sample(bad)
    monkeypatch.setattr(vs, "DISCARDED_SAMPLE_DIGESTS", {PROTOCOL["attack_sample"]["sample_sha256"]})
    with pytest.raises(vs.SampleError, match="discarded draw"):
        vs.load_frozen_sample(PROTOCOL)


def test_the_discarded_first_draw_is_on_the_deny_list():
    assert "93a0f50e163f0c71d1d30a275ad26ad76322e5d4df94f8f6ac1cb00de6cd3522" in vs.DISCARDED_SAMPLE_DIGESTS


def test_e0_refuses_a_budget_the_protocol_does_not_approve(monkeypatch):
    p = copy.deepcopy(PROTOCOL)
    p["experiments"]["E0_exhaustive_physical_v1"]["approved_budgets"] = ["A"]
    monkeypatch.setattr(e0, "load_protocol", lambda: p)
    with pytest.raises(e0.E0Error, match="does not approve E0 Budget B"):
        e0.run("B", e0.PRIMARY)
    p["status"] = "PROPOSED"
    with pytest.raises(e0.E0Error, match="does not approve"):
        e0.run("A", e0.PRIMARY)


def test_variants_and_seeds_follow_the_protocol():
    assert [f"{m}_{fs}" for m, fs in e0.parse_variants(None)] == [
        "random_forest_full42", "xgboost_full42", "mlp_full42", "random_forest_no_ttl39", "xgboost_no_ttl39", "mlp_no_ttl39",
        "logistic_regression_full42", "logistic_regression_no_ttl39"]
    with pytest.raises(SystemExit):
        e0.parse_variants(["svm_full42"])
    seeds = PROTOCOL["seeds"]
    assert e0.SEARCH_SEED == seeds["lime_search"] and list(e0.HELDOUT) == seeds["lime_heldout"]
    assert list(e0.CONFIRM) == seeds["lime_confirmation"] and sorted(e0.PART_B) == sorted([seeds["lime_search"], *seeds["lime_heldout"]])


def test_e0_reuses_the_unchanged_v2_criteria():
    from experiments import unsw_step6_shap, unsw_targeted_heldout
    assert e0.ho is unsw_targeted_heldout and e0.sh is unsw_step6_shap
    assert unsw_targeted_heldout.HELDOUT_SEEDS == e0.HELDOUT


def test_candidate_identity_is_the_exact_42_value_vector():
    from src.unsw.v3_data import feature_order
    row = pd.Series({f: 1.0 for f in feature_order()} | {"proto": "tcp", "service": "-", "state": "FIN"})
    other = row.copy()
    other["sbytes"] = 1.0000000001
    assert e0.candidate_key(row) != e0.candidate_key(other) and e0.candidate_key(row) == e0.candidate_key(row.copy())
