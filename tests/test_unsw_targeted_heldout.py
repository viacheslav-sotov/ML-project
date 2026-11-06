"""Held-out LIME evaluation logic (fake explanations; no LIME, no models)."""

import itertools

import numpy as np
import pandas as pd
import pytest

from experiments import unsw_spoofing_explanations as partb
from experiments import unsw_targeted_heldout as ho
from experiments import unsw_targeted_spoofing as ts
from src.unsw.constraints import FEATURE_ORDER

BOOSTED = ["sjit", "djit", "swin", "trans_depth", "response_body_len"]  # low base rank -> move into the top 10


def make_row(**values):
    row = {feature: 0.0 for feature in FEATURE_ORDER}
    row.update({"proto": "udp", "service": "dns", "state": "INT"})
    row.update(values)
    return pd.Series(row)


ORIGINAL = make_row()


class FakeExplain:
    """Deterministic 'LIME': fixed descending base weights + seed noise + boosts for edited features."""

    def __init__(self, noise=0.6, boost=60.0):
        self.noise, self.boost, self.calls = noise, boost, 0

    def __call__(self, model_key, sample, row, seed):
        self.calls += 1
        rng = np.random.default_rng(1000 + seed)
        weights = {f: 100.0 - i for i, f in enumerate(FEATURE_ORDER)}
        weights[FEATURE_ORDER[0]] = 200.0  # stable top-1 for the original
        for f, eps in zip(FEATURE_ORDER, rng.normal(0.0, self.noise, len(FEATURE_ORDER))):
            weights[f] += eps
        for f in ts.NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES:
            weights[f] += self.boost * float(row[f])
        return weights


def candidate(uid, row, **extra):
    return {"candidate_uid": uid, "model_key": "random_forest", "sample_position": 2941,
            "original_row": ORIGINAL, "candidate_row": row, **extra}


def test_metrics_are_part_b_definitions():
    assert ho.pair_metrics is partb.pair_metrics


def test_seed_validation():
    assert ho.validate_seeds([22, 33, 44, 55]) == (22, 33, 44, 55)
    with pytest.raises(ValueError):
        ho.validate_seeds([11, 22, 33])  # search seed
    with pytest.raises(ValueError):
        ho.validate_seeds([22, 22])
    with pytest.raises(ValueError):
        ho.validate_seeds([22])


def test_noise_band_uses_all_seed_pairs():
    explain = FakeExplain()
    by_seed = {s: explain("m", 0, ORIGINAL, s) for s in (22, 33, 44, 55)}
    band = ho.noise_band(by_seed, (22, 33, 44, 55))
    pairs = [partb.pair_metrics(by_seed[a], by_seed[b]) for a, b in itertools.combinations((22, 33, 44, 55), 2)]
    assert band["n_pairs"] == 6
    for metric in ho.BAND_METRICS:
        values = [p[metric] for p in pairs]
        assert band[f"{metric}_min"] == pytest.approx(min(values))
        assert band[f"{metric}_mean"] == pytest.approx(np.mean(values))
        assert band[f"{metric}_max"] == pytest.approx(max(values))
    assert band["top1_stable_all_pairs"] is True


def test_identical_candidate_has_zero_shift_and_no_flags():
    result = ho.evaluate_candidates([candidate("same", ORIGINAL.copy())], FakeExplain())
    row = result["summary"][0]
    assert row["heldout_top10_jaccard_mean"] == 1.0 and row["heldout_top10_shift"] == 0.0
    assert not row["exceeds_heldout_top10"] and not row["exceeds_heldout_both"] and not row["partb_exceeds_seed_noise"]


def test_large_consistent_change_exceeds_heldout_noise():
    big = make_row(**{f: 1.0 for f in BOOSTED})
    row = ho.evaluate_candidates([candidate("big", big)], FakeExplain())["summary"][0]
    assert row["heldout_top10_jaccard_mean"] < row["heldout_noise_top10_jaccard_min"]
    assert row["exceeds_heldout_top10"] and row["exceeds_heldout_spearman"] and row["exceeds_heldout_both"]
    assert row["heldout_top10_shift"] == pytest.approx(1.0 - row["heldout_top10_jaccard_mean"])


def test_change_within_seed_noise_does_not_exceed():
    tiny = make_row(sjit=0.001)  # +0.06 weight, far below the seed noise
    row = ho.evaluate_candidates([candidate("tiny", tiny)], FakeExplain())["summary"][0]
    assert not row["exceeds_heldout_top10"]
    assert not row["exceeds_heldout_both"]


def test_top1_rule_needs_a_stable_original_and_a_majority_of_seeds():
    band = {"top5_jaccard_min": 0.5, "top10_jaccard_min": 0.5, "spearman_rho_min": 0.5, "top1_stable_all_pairs": True}
    means = {"top5_jaccard": 1.0, "top10_jaccard": 1.0, "spearman_rho": 1.0}
    assert ho.noise_flags(means, 3, band)["exceeds_top1_rule"] is True
    assert ho.noise_flags(means, 2, band)["exceeds_top1_rule"] is False
    assert ho.noise_flags(means, 4, {**band, "top1_stable_all_pairs": False})["exceeds_top1_rule"] is False
    # strict inequality: equal to the band minimum does not exceed
    assert ho.noise_flags({**means, "top10_jaccard": 0.5}, 0, band)["exceeds_top10_jaccard"] is False


def test_top1_change_in_all_heldout_seeds_is_flagged():
    takeover = make_row(sjit=10.0)  # +600 -> new top-1 under every seed
    row = ho.evaluate_candidates([candidate("top1", takeover)], FakeExplain())["summary"][0]
    assert row["heldout_top1_changed_seeds"] == 4 and row["exceeds_heldout_top1_rule"]


def test_explanations_are_cached_per_unique_candidate_and_seed():
    explain = FakeExplain()
    big = make_row(**{f: 1.0 for f in BOOSTED})
    cands = [candidate("a", big), candidate("b", big.copy()), candidate("c", ORIGINAL.copy()), candidate("d", make_row(sjit=0.5))]
    result = ho.evaluate_candidates(cands, explain)
    union = len(set(ho.HELDOUT_SEEDS) | set(ho.PART_B_SEEDS))
    assert explain.calls == result["explanations_computed"] == 3 * union  # original, big, sjit=0.5
    assert len(result["summary"]) == 4
    assert len([b for b in result["bands"] if b["band"] == "heldout"]) == 1


def test_part_b_criterion_is_reported_separately_and_marked_in_sample():
    big = make_row(**{f: 1.0 for f in BOOSTED})
    result = ho.evaluate_candidates([candidate("big", big)], FakeExplain())
    row = result["summary"][0]
    assert row["partb_includes_search_seed"] is True
    bands = {b["band"]: b for b in result["bands"]}
    assert bands["heldout"]["n_pairs"] == 6 and bands["part_b"]["n_pairs"] == 10
    seed_sets = [r["seed_set"] for r in result["per_seed"]]
    assert seed_sets.count("heldout") == 4 and seed_sets.count("search_seed") == 1
    assert {r["seed"] for r in result["per_seed"] if r["seed_set"] == "heldout"} == {22, 33, 44, 55}


def test_summary_never_claims_spoofing():
    big = make_row(**{f: 1.0 for f in BOOSTED})
    row = ho.evaluate_candidates([candidate("big", big)], FakeExplain())["summary"][0]
    assert not any("spoof" in key.lower() for key in row)
    assert "never labels" in ho.RULES["no_spoofing_claims"]


def test_inconsistent_originals_are_rejected():
    other = candidate("x", ORIGINAL.copy())
    other["original_row"] = make_row(dur=1.0)
    with pytest.raises(ValueError):
        ho.evaluate_candidates([candidate("a", ORIGINAL.copy()), other], FakeExplain())
