"""Report tables for an exhaustive run (experiments/unsw_step6_pilot_report.py), on a fake-LIME run."""

import json
from pathlib import Path

import pandas as pd
import pytest

import experiments.unsw_step6_exhaustive as ex
import experiments.unsw_step6_pilot_report as rep
from experiments import unsw_step6_manifest as mf
from test_step6_v2_infrastructure import EX_SAMPLES, _fake_lime, _models  # noqa: F401
from test_unsw_step6_runner import split  # noqa: F401  (fixture)


@pytest.mark.parametrize("k, n, cp, wi", [
    (0, 10, (0.0, 0.3085), (0.0, 0.2775)),
    (0, 2, (0.0, 0.8419), (0.0, 0.6576)),
    (3, 10, (0.0667, 0.6525), (0.1078, 0.6032)),
])
def test_interval_formulas(k, n, cp, wi):
    assert tuple(round(x, 4) for x in rep.clopper_pearson(k, n)) == cp
    assert tuple(round(x, 4) for x in rep.wilson(k, n)) == wi


def test_report_tables_on_a_fake_run(tmp_path, split, monkeypatch):
    protocol = mf.load_protocol(mf.PROTOCOL_V2_PATH)
    out = ex.run_exhaustive(protocol, tmp_path, split=split, manifest=mf.build_manifest(protocol), models=_models(),
                            explain_fn=_fake_lime(split)[0], samples=EX_SAMPLES, log=lambda m: None)
    run_dir = Path(out["run_dir"])
    monkeypatch.setattr("sys.argv", ["x", str(run_dir)])
    rep.main()
    report = run_dir / "report"
    for name in ("candidates_by_flow", "success_by_group", "success_intervals", "explanation_change", "attack_preservation",
                 "budget_effect", "closest_to_band", "feature_rank", "bin_crossings"):
        assert (report / f"{name}.csv").exists(), name
    summary = out["summary"]
    groups = pd.read_csv(report / "success_by_group.csv")
    assert groups["candidates"].sum() == len(summary)
    assert groups["lime_success"].sum() == summary["lime_success"].sum()
    assert groups["confirmed_lime_success"].sum() == summary["confirmed_lime_success"].sum()
    flows = pd.read_csv(report / "candidates_by_flow.csv")
    assert set(flows["model_key"]) == {"random_forest", "xgboost"} and (flows["rejected"] == 0).all()
    overview = json.loads((report / "run_overview.json").read_text(encoding="utf-8"))
    assert overview["run_role"] == "smoke_test" and overview["official_test_used"] is False
    assert sum(m["explanations"] for m in overview["per_model"].values()) == out["status"]["cached_explanations"]
    ranks = pd.read_csv(report / "feature_rank.csv")
    assert len(ranks) == 2 * len(EX_SAMPLES) and (ranks["seeds"] == 4).all()
