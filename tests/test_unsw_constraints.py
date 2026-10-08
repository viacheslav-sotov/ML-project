from pathlib import Path

import pandas as pd

from src.unsw.constraints import OFFICIAL_TRAIN_DF, validate_candidate


def test_unchanged_sample_is_valid():
    sample = OFFICIAL_TRAIN_DF.iloc[0].to_dict()
    result = validate_candidate(sample, sample)
    assert result["valid"] is True
    assert result["violated_constraints"] == []


def test_negative_numeric_value_is_rejected():
    sample = OFFICIAL_TRAIN_DF.iloc[0].to_dict()
    sample["spkts"] = -1
    result = validate_candidate(OFFICIAL_TRAIN_DF.iloc[0].to_dict(), sample)
    assert result["valid"] is False
    assert any("negative value for spkts" in violation for violation in result["violated_constraints"])


def test_invalid_categorical_value_is_rejected():
    sample = OFFICIAL_TRAIN_DF.iloc[0].to_dict()
    sample["proto"] = "not-a-real-protocol"
    result = validate_candidate(OFFICIAL_TRAIN_DF.iloc[0].to_dict(), sample)
    assert result["valid"] is False
    assert any("invalid categorical value for proto" in violation for violation in result["violated_constraints"])


def test_noncontrollable_feature_change_is_rejected():
    sample = OFFICIAL_TRAIN_DF.iloc[0].to_dict()
    sample["is_sm_ips_ports"] = 1 if sample["is_sm_ips_ports"] == 0 else 0
    result = validate_candidate(OFFICIAL_TRAIN_DF.iloc[0].to_dict(), sample)
    assert result["valid"] is False
    assert any("frozen feature changed: is_sm_ips_ports" in violation for violation in result["violated_constraints"])


def test_dependency_violation_is_rejected():
    sample = OFFICIAL_TRAIN_DF.iloc[0].to_dict()
    sample["tcprtt"] = sample["synack"] + sample["ackdat"] + 1.0
    result = validate_candidate(OFFICIAL_TRAIN_DF.iloc[0].to_dict(), sample)
    assert result["valid"] is False
    assert any("dependency violation: tcprtt must equal synack + ackdat" in violation for violation in result["violated_constraints"])
