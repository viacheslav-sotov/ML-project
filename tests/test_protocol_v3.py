"""Protocol v3 release candidate stays consistent with the files it pins."""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
P = json.loads((ROOT / "experiments" / "step6_protocol_v3.json").read_text(encoding="utf-8"))
sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def test_base_protocols_are_pinned_and_unchanged():
    assert sha(P["base"]["training_protocol"]["file"]) == P["base"]["training_protocol"]["sha256"]
    for f, h in P["base"]["prior_protocols_unchanged"].items():
        assert sha(f) == h, f


def test_model_pins_equal_the_registry():
    from src.unsw import v3_final_models as fm
    assert P["models"]["final_model_sha256"] == {f"{m}_{fs}": h for (m, fs), h in fm.PINNED_SHA256.items()}


@pytest.mark.skipif(not (ROOT / P["attack_sample"]["manifest"]).exists(), reason="frozen sample not present")
def test_sample_pins_equal_the_frozen_manifest():
    m = json.loads((ROOT / P["attack_sample"]["manifest"]).read_text(encoding="utf-8"))
    assert m["sample_sha256"] == P["attack_sample"]["sample_sha256"] and m["n_selected"] == P["attack_sample"]["flows"]
    assert sha("results/v3_prep/attack_sample/sample_manifest.csv") == P["attack_sample"]["sample_manifest_csv_sha256"]


def test_prior_success_criteria_are_kept_and_test_set_locked():
    assert "D8 unchanged" in P["success_criteria"]["lime"] and "D11 unchanged" in P["success_criteria"]["treeshap"]
    assert "locked" in P["data"]["official_test_partition"]
    assert P["status"].startswith("APPROVED") and P["protocol_version"] == "step6-v3"


def test_final_protocol_keeps_every_required_element():
    ex = P["experiments"]
    for k in ("E0_exhaustive_physical_v1", "G1_physical_search", "G2_multi_seed_adaptive", "G3_current_v1",
              "G4_unconstrained_diagnostic", "G5_treeshap_targeted", "G6_mlp_lr", "DEF1", "DEF2"):
        assert k in ex, k
    assert P["budgets"] == {"perturbation": ["A", "B"], "explanation_calls": [25, 50, 100]}
    assert set(P["research_questions"]) == {"RQ1", "RQ2", "RQ3", "RQ4", "RQ5"}
    assert any("packet-level validation" in x for x in P["limitations"])
    assert "exactly one predeclared evaluation" in P["data"]["final_official_test_evaluation"]
    for t in ("Wilson", "Clopper-Pearson"):
        assert t in P["statistics"]["tests"]["proportions"]
    assert "Wilcoxon" in P["statistics"]["tests"]["P2"] and "Holm" in P["statistics"]["tests"]["P2"]
    e0 = ex["E0_exhaustive_physical_v1"]
    assert e0["approved_budgets"] == ["A"] and e0["seeds"]["lime"] == [11, 22, 33, 44, 55] and "rule_d" in e0
    assert len(P["attack_sample"]["excluded_v2_pilot_official_ids"]) == 10
