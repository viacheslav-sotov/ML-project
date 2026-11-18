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
    assert P["status"].startswith("PROPOSED")
