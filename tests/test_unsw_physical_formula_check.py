"""Unit tests for the read-only physical-formula check (synthetic rows; no data files, no models)."""

import json

import pandas as pd
import pytest

from experiments import unsw_physical_formula_check as pfc


def _row(**overrides):
    base = dict(proto="udp", service="dns", state="INT", dur=0.000003, spkts=2, dpkts=0, sbytes=114, dbytes=0,
                rate=1.0 / 0.000003, sttl=254, dttl=0, sload=(114 - 57) * 8 / 0.000003, dload=0.0, sloss=0, dloss=0,
                sinpkt=0.003, dinpkt=0.0, sjit=0.0, djit=0.0, swin=0, stcpb=0, dtcpb=0, dwin=0, tcprtt=0.0,
                synack=0.0, ackdat=0.0, smean=57, dmean=0, ct_state_ttl=2)
    base.update(overrides)
    return base


def _support(frame):
    table = pfc.check_rules(frame, "synthetic")
    return table.set_index("rule_id")


def test_consistent_two_packet_flow_satisfies_every_applicable_rule():
    table = _support(pd.DataFrame([_row(), _row(sbytes=200, smean=100, sload=(200 - 100) * 8 / 0.000003)]))
    applicable = table[table["applicable_rows"] > 0]
    assert (applicable["violating_rows"] == 0).all(), applicable[applicable["violating_rows"] > 0]
    assert applicable.loc["R08b", "satisfied_rows"] == 2  # sinpkt = 1000 * dur for 2-packet flows


@pytest.mark.parametrize("override, rule_id", [
    ({"smean": 58}, "R01"),
    ({"rate": 5.0}, "R03"),
    ({"sload": 1.0}, "R04"),
    ({"tcprtt": 0.5}, "R06"),
    ({"sinpkt": 0.009}, "R08b"),
    ({"dinpkt": 0.1}, "R10.4"),   # dinpkt must be 0 when dpkts = 0
    ({"swin": 255}, "R13.1"),     # TCP-only field set on a UDP flow
    ({"sjit": 0.2}, "R12"),
])
def test_each_rule_detects_its_violation(override, rule_id):
    table = _support(pd.DataFrame([_row(**override)]))
    assert table.loc[rule_id, "violating_rows"] == 1
    assert not table.loc[rule_id, "meets_repair_threshold"]


def test_rounding_tie_accepts_both_neighbours():
    frame = pd.DataFrame([_row(sbytes=115, smean=57, sload=(115 - 57) * 8 / 0.000003),
                          _row(sbytes=115, smean=58, sload=(115 - 58) * 8 / 0.000003)])
    assert _support(frame).loc["R01", "violating_rows"] == 0


def test_ttl_lookup_detects_conflicts():
    frame = pd.DataFrame([_row(), _row(), _row(ct_state_ttl=3), _row(sttl=252, ct_state_ttl=1)])
    lookup = pfc.ttl_lookup(frame)
    conflict = lookup[(lookup["sttl"] == 254)]
    assert int(conflict["n_values"].iloc[0]) == 2
    assert _support(frame).loc["R07", "violating_rows"] == 3  # all rows in the conflicting group


def test_sample_feasibility_lists_observed_ttl_alternatives_and_rounding():
    lookup = pfc.ttl_lookup(pd.DataFrame([_row(), _row(sttl=252, ct_state_ttl=1), _row(sttl=60, ct_state_ttl=0)]))
    sample = pd.DataFrame([_row()])
    feas = pfc.sample_feasibility(sample, [2941], lookup, pfc.build_rules()).iloc[0]
    assert feas["original_ttl_combo_in_lookup"]
    assert "252->ct_state_ttl 1" in feas["sttl_alternatives_A"]
    assert "60" not in feas["sttl_alternatives_B"]           # 254 -> 60 is far beyond 25%
    assert not feas["spkts_plus1_allowed_A"] and not feas["spkts_plus1_allowed_B"]  # +1 of 2 packets = 50%
    assert feas["dur_edit_repairable"]
    assert feas["rules_failing_on_original"] == ""


def test_run_reads_only_subtrain_and_validation(tmp_path):
    class Guarded(dict):
        def __getitem__(self, key):
            if key not in pfc.SPLITS_USED:
                raise AssertionError(f"script read {key}")
            return super().__getitem__(key)

    rows = pd.DataFrame([_row()] * 3)
    val = pd.DataFrame([_row()] * (max(pfc.V2_SAMPLES) + 1))
    out = pfc.run(tmp_path, split=Guarded(X_subtrain=rows, X_val=val, X_test=None))
    meta = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["official_test_used"] is False and set(meta["splits_read"]) == {"X_subtrain", "X_val"}
    assert len(out["feasibility"]) == len(pfc.V2_SAMPLES)
    for name in ("formula_support.csv", "ttl_lookup_subtrain.csv", "sample_feasibility.csv"):
        assert (tmp_path / name).exists()
