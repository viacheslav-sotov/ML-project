"""Step 6 analysis logic (small synthetic frames) plus one end-to-end run on the packaged results."""

import numpy as np
import pandas as pd
import pytest

from experiments import unsw_step6_analysis as an

SELECTION = {"models": ["random_forest"], "samples": [1, 2, 3], "budgets": ["A"], "explanation_budgets": [25, 50],
             "order_seeds": list(range(10)), "control_seeds": list(range(10)),
             "methods": ["canonical", "shuffled", "multistart", "random"]}


def test_expected_task_and_checkpoint_counts_match_the_runner_design():
    assert an.expected_tasks(SELECTION) == {"canonical": 3, "shuffled": 30, "multistart": 60, "random": 30}
    assert an.expected_checkpoints(SELECTION) == {"canonical": 6, "shuffled": 60, "multistart": 60, "random": 60}
    assert sum(an.expected_tasks(SELECTION).values()) == 123
    assert sum(an.expected_checkpoints(SELECTION).values()) == 186


def test_stratified_comparison_weights_strata_equally_and_counts_ties_as_half():
    rng = np.random.default_rng(0)
    a = {1: np.array([1.0, 1.0]), 2: np.array([0.0])}
    b = {1: np.array([0.0, 0.0, 0.0]), 2: np.array([0.0, 0.0])}
    out = an.stratified_comparison(a, b, rng)
    assert out["diff"] == pytest.approx(0.5)          # (1 - 0) and (0 - 0), averaged over 2 strata
    assert out["p_superiority"] == pytest.approx(0.75)  # stratum 1: 1.0; stratum 2: all ties -> 0.5
    assert out["n_strata"] == 2 and out["n_a"] == 3 and out["n_b"] == 5
    assert out["ci_low"] <= out["diff"] <= out["ci_high"]


def test_stratified_comparison_identical_groups_give_zero_difference_and_large_p():
    rng = np.random.default_rng(1)
    values = {1: np.arange(10.0), 2: np.arange(10.0) * 2}
    out = an.stratified_comparison(values, values, rng)
    assert out["diff"] == pytest.approx(0.0)
    assert out["p_superiority"] == pytest.approx(0.5)
    assert out["perm_p"] > 0.5


def test_stratified_comparison_ignores_strata_missing_from_one_group():
    out = an.stratified_comparison({1: np.array([2.0]), 9: np.array([5.0])}, {1: np.array([1.0])},
                                   np.random.default_rng(2))
    assert out["n_strata"] == 1 and out["diff"] == pytest.approx(1.0)


def test_split_rules_keeps_semicolons_inside_support_brackets():
    text = ("smean = round(sbytes / spkts) [100% of rows]; rate = (spkts + dpkts - 1) / dur (0 if dur = 0), within 1% "
            "[100% of rows with dur > 0; 99.85% for dur = 0]")
    assert an.split_rules(text) == ["smean = round(sbytes / spkts)",
                                    "rate = (spkts + dpkts - 1) / dur (0 if dur = 0), within 1%"]
    assert an.split_rules(np.nan) == [] and an.split_rules("") == []


def test_split_unverified_returns_feature_names():
    text = ("dur: dur also constrains sinpkt/sjit timing (unverified relation); sinpkt: sinpkt relation to dur/spkts "
            "unverified (sinpkt*(spkts-1)/1000 ~= dur holds for only 74.7% of rows)")
    assert an.split_unverified(text) == ["dur", "sinpkt"]


def _candidates(**overrides):
    base = {
        "checkpoint": [50, 50, 50], "group": ["targeted", "targeted", "random"],
        "explanation_changed_search_seed": [True, True, True], "experimental_valid": [True, True, True],
        "best_candidate_probability": [0.9, 0.9, 0.9],
        "exceeds_heldout_top10": [True, True, False], "exceeds_heldout_both": [True, False, False],
        "physical_status": ["consistent_with_checked_rules", "violates_checked_rule", "unverified_dependencies"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_funnel_is_cumulative_and_reports_each_requirement_alone():
    funnel = an.spoofing_funnel(_candidates())
    overall = funnel[funnel["checkpoint"] == "all"].set_index("stage")
    counts = overall["candidates"].tolist()
    assert counts == sorted(counts, reverse=True)
    assert counts[0] == 3 and counts[3] == 2 and counts[4] == 1 and counts[-1] == 1
    assert overall["requirement_alone"].iloc[5] == 2  # two candidates violate no checked rule


def test_funnel_drops_candidates_that_are_no_longer_attacks():
    funnel = an.spoofing_funnel(_candidates(best_candidate_probability=[0.4, 0.9, 0.9]))
    overall = funnel[funnel["checkpoint"] == "all"]
    assert overall.iloc[2]["candidates"] == 2


def test_full_analysis_on_the_packaged_step6_run(tmp_path):
    if not (an.DEFAULT_RUN_DIR / "results.csv").exists():
        pytest.skip("packaged Step 6 run not present")
    result = an.run_analysis(an.DEFAULT_RUN_DIR, tmp_path)
    checks = result["checks"]
    assert checks["passed"].all(), checks[~checks["passed"]].to_string()
    for name in result["metadata"]["tables"]:
        assert (tmp_path / name).exists()
    assert len(list((tmp_path / "figures").glob("*.png"))) >= 7
    cand = pd.read_csv(tmp_path / "step6_candidates.csv")
    assert len(cand) == 186
    assert int(cand["exceeds_heldout_both"].sum()) == 0
