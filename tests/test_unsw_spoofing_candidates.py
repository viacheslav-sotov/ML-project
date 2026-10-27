import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.unsw_spoofing_candidates import (
    NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES,
    _candidate_for_budget,
    _relative_change,
    _serialize_values,
    generate_candidates,
    load_strict_model,
)
from src.unsw.constraints import FEATURE_METADATA, validate_candidate
from src.unsw.xai_setup import get_xai_data, load_shared_samples


@pytest.fixture(scope="module")
def split_and_shared():
    split = get_xai_data()
    shared = load_shared_samples(split)
    return split, shared


def test_budget_a_and_b_are_enforced(split_and_shared):
    split, _ = split_and_shared
    original = split["X_val"].iloc[[2941]].iloc[0].copy()

    for budget, max_features, max_rel in [("A", 5, 0.10), ("B", 10, 0.25)]:
        rng = np.random.default_rng(42)
        candidate = _candidate_for_budget(original, budget, rng)
        changed = [feature for feature in NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES if candidate[feature] != original[feature]]
        assert len(changed) <= max_features
        if changed:
            ds = [_relative_change(float(original[feature]), float(candidate[feature])) for feature in changed]
            assert max(ds) <= max_rel + 1e-12
            if budget == "B":
                assert sum(ds) <= 1.5 + 1e-12


def test_protected_features_remain_unchanged(split_and_shared):
    split, _ = split_and_shared
    original = split["X_val"].iloc[[2941]].iloc[0].copy()
    rng = np.random.default_rng(42)
    candidate = _candidate_for_budget(original, "A", rng)

    protected = [
        "is_sm_ips_ports",
        "is_ftp_login",
        "ct_ftp_cmd",
        "ct_flw_http_mthd",
        "ct_src_ltm",
        "ct_srv_dst",
        "rate",
        "sload",
        "dload",
    ]
    for feature in protected:
        assert candidate[feature] == original[feature]


def test_validator_rejection_for_invalid_candidate():
    split = get_xai_data()
    original = split["X_val"].iloc[[2941]].iloc[0].copy()
    candidate = original.copy()
    candidate["is_sm_ips_ports"] = 1 - int(original["is_sm_ips_ports"])
    result = validate_candidate(original.to_dict(), candidate.to_dict())
    assert result["valid"] is False
    assert any("frozen feature changed: is_sm_ips_ports" in violation for violation in result["violated_constraints"])


def test_prediction_preservation_and_budget_acceptance(split_and_shared):
    split, shared = split_and_shared
    accepted = generate_candidates(
        model_key="random_forest",
        sample_position=2941,
        budget_key="A",
        num_candidates=1,
        seed=42,
        split=split,
        shared_samples=shared,
    )
    assert len(accepted) == 1

    row = accepted[0]
    assert row["constraint_valid"] is True
    assert row["feature_edit_count"] <= 5
    assert row["original_attack_probability"] >= 0.5
    assert row["modified_attack_probability"] >= 0.5
    assert row["candidate_status"] == "accepted"
    assert row["seed"] == 42
    assert len(row["candidate_id"]) > 0

    original_vals = json.loads(row["original_feature_values_json"])
    modified_vals = json.loads(row["modified_feature_values_json"])
    assert set(original_vals) == set(modified_vals)
    assert len(original_vals) == len(modified_vals)
    assert row["model_name"] == "random_forest"
    assert row["validation_position"] == 2941


def test_json_round_trip_preserves_values_and_names(split_and_shared):
    split, shared = split_and_shared
    accepted = generate_candidates(
        model_key="xgboost",
        sample_position=3206,
        budget_key="A",
        num_candidates=1,
        seed=42,
        split=split,
        shared_samples=shared,
    )
    assert accepted

    original_vals = json.loads(accepted[0]["original_feature_values_json"])
    modified_vals = json.loads(accepted[0]["modified_feature_values_json"])
    assert set(original_vals) == set(modified_vals)
    for key in original_vals:
        assert key in modified_vals
        original_value = original_vals[key]
        modified_value = modified_vals[key]
        if isinstance(original_value, (int, float)) and isinstance(modified_value, (int, float)):
            assert np.isfinite(float(original_value))
            assert np.isfinite(float(modified_value))


def test_deterministic_candidate_generation_with_seed_42(split_and_shared):
    split, _ = split_and_shared
    original = split["X_val"].iloc[[2941]].iloc[0].copy()
    c1 = _candidate_for_budget(original, "A", np.random.default_rng(42))
    c2 = _candidate_for_budget(original, "A", np.random.default_rng(42))
    assert c1.equals(c2)


def test_missing_model_file_raises_instead_of_retraining(monkeypatch, split_and_shared):
    split, _ = split_and_shared
    from experiments import unsw_spoofing_candidates as mod

    monkeypatch.setattr(mod, "MODEL_DIR", Path("/definitely/not/here"))
    with pytest.raises(FileNotFoundError):
        load_strict_model("random_forest", split)
