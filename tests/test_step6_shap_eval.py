"""TreeSHAP evaluation of the physical_v1 lattice (experiments/unsw_step6_shap.py).

Fake models, SHAP and LIME (no explainer, no training); the real data split is used for the lattice.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import experiments.unsw_step6_exhaustive as ex
import experiments.unsw_step6_shap as sh
from experiments import unsw_step6_manifest as mf
from src.unsw import physical_repair as pr
from src.unsw.constraints import FEATURE_ORDER
from test_step6_v2_infrastructure import EX_SAMPLES, _fake_lime, _models  # noqa: F401
from test_unsw_step6_runner import split  # noqa: F401  (fixture)


@pytest.fixture(scope="module")
def protocol_v2():
    return mf.load_protocol(mf.PROTOCOL_V2_PATH)


@pytest.fixture(scope="module")
def manifest_v2(protocol_v2):
    return mf.build_manifest(protocol_v2)


def fake_shap(scale_sbytes=1.0):
    """SHAP stand-in: attributions depend on the row, sbytes edits move sbytes' attribution."""
    base = {f: (len(FEATURE_ORDER) - i) * 0.01 for i, f in enumerate(FEATURE_ORDER)}

    def shap_fn(model_key, frame):
        rows = []
        for _, r in frame.iterrows():
            w = dict(base)
            w["sbytes"] = scale_sbytes * float(r["sbytes"]) / 1000.0
            w["spkts"] = float(r["spkts"]) / 100.0
            rows.append(w)
        return pd.DataFrame(rows, columns=FEATURE_ORDER), "probability", 0.0
    return shap_fn


def _run(tmp_path, split, protocol, manifest, **kw):
    return sh.run_shap_evaluation(protocol, sh.load_spec(), tmp_path, split=split, manifest=manifest, models=_models(),
                                  shap_fn=kw.pop("shap_fn", fake_shap()), lime_fn=kw.pop("lime_fn", _fake_lime(split)[0]),
                                  samples=EX_SAMPLES, check_saved=False, log=lambda m: None, **kw)


# ---------------------------------------------------------------------------------------------
# Spec and D11 logic
# ---------------------------------------------------------------------------------------------
def test_spec_is_valid_and_matches_part_b_and_d11(protocol_v2):
    spec = sh.load_spec()
    assert sh.validate_spec(spec, protocol_v2) == []
    assert spec["background"] == {"source": "preprocessed validation split (X_val)", "size": 100, "seed": 42,
                                  "sampler": "numpy.random.default_rng(42).choice without replacement"}
    assert "J(x,x') <= J_ref AND rho(x,x') <= rho_ref" in spec["success_criterion_d11"]
    assert spec["base_protocol"]["sha256"] == mf.build_manifest(protocol_v2)["protocol_sha256"]   # v2 unchanged


@pytest.mark.parametrize("change, message", [
    (lambda s: s["background"].update(size=50), "background"),
    (lambda s: s.update(reference_lime_seeds=[11, 22, 33, 44]), "reference_lime_seeds"),
    (lambda s: s.update(role="final"), "role"),
])
def test_spec_validation_catches_errors(protocol_v2, change, message):
    bad = copy.deepcopy(sh.load_spec())
    change(bad)
    assert any(message in p for p in sh.validate_spec(bad, protocol_v2))


def test_shap_pair_and_d11_logic():
    a = {f: float(len(FEATURE_ORDER) - i) for i, f in enumerate(FEATURE_ORDER)}
    same = sh.shap_pair(a, dict(a))
    assert same["top1_agreement"] == 1.0 and same["top10_jaccard"] == 1.0 and same["spearman_abs_tieaware"] == pytest.approx(1.0)
    reversed_ = {f: float(i + 1) for i, f in enumerate(FEATURE_ORDER)}
    ref = sh.d11_reference(a, {22: reversed_, 33: reversed_, 44: a, 55: a}, (22, 33, 44, 55))
    assert ref["j_ref"] == pytest.approx(0.5) and ref["rho_ref"] == pytest.approx(0.0)
    m = sh.shap_pair(a, reversed_)
    flags = sh.d11_flags(m, ref, still_attack=True, physical_status="consistent_with_checked_rules")
    assert flags["d11_meets_both"] and flags["shap_success"]
    assert not sh.d11_flags(m, ref, True, "unverified_dependencies")["shap_success"]       # never for unverified
    assert not sh.d11_flags(m, ref, False, "consistent_with_checked_rules")["shap_success"]  # must stay attack
    assert not sh.d11_flags(same, ref, True, "consistent_with_checked_rules")["d11_meets_both"]


# ---------------------------------------------------------------------------------------------
# End to end (fake models / SHAP / LIME)
# ---------------------------------------------------------------------------------------------
def test_same_candidates_as_lime_pilot_and_groups_separate(tmp_path, split, protocol_v2, manifest_v2):
    lime_run = ex.run_exhaustive(protocol_v2, tmp_path / "lime", split=split, manifest=manifest_v2, models=_models(),
                                 explain_fn=_fake_lime(split)[0], samples=EX_SAMPLES, log=lambda m: None)
    out = _run(tmp_path / "shap", split, protocol_v2, manifest_v2, lime_pilot_dir=lime_run["run_dir"])
    summary = out["summary"]
    lime_uids = set(pd.read_csv(Path(lime_run["run_dir"]) / "unique_candidates.csv")["candidate_uid"])
    assert set(summary["candidate_uid"]) == lime_uids and len(summary) == 2 * len(lime_uids)
    checks = out["status"]["comparison_with_lime_pilot"]["checks"]
    assert checks["candidates_only_in_shap"] == 0 and checks["candidates_only_in_lime"] == 0
    assert checks["matched"] == len(summary) and checks["max_p_attack_gap"] == 0.0
    assert set(summary["physical_status"]) == set(sh.STATUS_GROUPS)
    assert not summary.loc[summary["physical_status"] != "consistent_with_checked_rules", "shap_success"].any()
    assert (summary["physically_valid_claimed"] == False).all()  # noqa: E712
    d11 = (summary["top10_jaccard"] <= summary["j_ref"]) & (summary["spearman_abs_tieaware"] <= summary["rho_ref"])
    assert summary["d11_meets_both"].equals(d11)
    groups = pd.read_csv(Path(out["run_dir"]) / "summary_by_status_group.csv")
    assert groups["candidates"].sum() == len(summary)
    run_dir = Path(out["run_dir"])
    for name in ("shap_summary.csv", "shap_values.csv", "d11_references.csv", "summary_by_flow.csv", "comparison_with_lime.csv", "README.md"):
        assert (run_dir / name).exists(), name
    assert run_dir.name.startswith("shap_smoke_test_")
    assert json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["official_test_used"] is False


def test_references_use_fresh_original_lime_on_heldout_seeds(tmp_path, split, protocol_v2, manifest_v2):
    explain, counter = _fake_lime(split)
    seen = []
    out = _run(tmp_path, split, protocol_v2, manifest_v2,
               lime_fn=lambda m, s, r, seed: (seen.append((m, s, seed)), explain(m, s, r, seed))[1])
    assert sorted({x[2] for x in seen}) == [22, 33, 44, 55]
    assert len(seen) == 2 * len(EX_SAMPLES) * 4 == out["status"]["reference_lime_calls"]
    assert len(out["refs"]) == 2 * len(EX_SAMPLES)


def test_candidate_set_mismatch_with_lime_pilot_is_refused(tmp_path, split, protocol_v2, manifest_v2):
    lime_run = ex.run_exhaustive(protocol_v2, tmp_path / "lime", split=split, manifest=manifest_v2, models=_models(),
                                 explain_fn=_fake_lime(split)[0], samples=[2941], log=lambda m: None)
    with pytest.raises(sh.ShapEvalError, match="candidate sets differ"):
        _run(tmp_path / "shap", split, protocol_v2, manifest_v2, lime_pilot_dir=lime_run["run_dir"])


def test_wrong_pinned_hash_is_refused(tmp_path, split, protocol_v2, manifest_v2):
    manifest = copy.deepcopy(manifest_v2)
    manifest["models"]["random_forest"]["sha256"] = "2" * 64
    with pytest.raises(ex.ExhaustiveError, match="pinned hash"):
        _run(tmp_path, split, protocol_v2, manifest)
