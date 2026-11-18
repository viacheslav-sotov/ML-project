"""WS3 final attack sample: rule implementation (synthetic) and the frozen manifest (real data, skipped if absent)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments import unsw_v3_attack_sample as s

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "v3_prep" / "attack_sample" / "sample_manifest.json"
POOL = ROOT / "results" / "v3_prep" / "attack_sample" / "pool.csv"
frozen = pytest.mark.skipif(not MANIFEST.exists(), reason="frozen sample not present")


def test_allocation_is_proportional_and_exact():
    assert s.allocate(30, {"tcp": 90, "other": 10}) == {"tcp": 27, "other": 3}
    assert s.allocate(30, {"tcp": 15, "other": 15}) == {"tcp": 15, "other": 15}
    assert s.allocate(30, {"tcp": 20, "other": 3}) == {"tcp": 20, "other": 3}       # fewer than n: take all
    assert s.allocate(30, {"tcp": 0, "other": 100}) == {"tcp": 0, "other": 30}
    assert sum(s.allocate(30, {"tcp": 2, "other": 1}).values()) == 3
    assert sum(s.allocate(30, {"tcp": 2041, "other": 665}).values()) == 30


def synthetic_pool(n1, n2, cat="Exploits", start=0, tcp_share=0.7, eligible=True):
    rows = []
    for i in range(n1):
        rows.append({"id": start + i, "attack_cat": cat, "tier": 1, "tier2_representative": False, "group_id": start + i,
                     "is_tcp": i < tcp_share * n1, "eligible": eligible})
    for g in range(n2):
        for k in range(2):  # two rows per duplicate group; the lower id represents it
            rows.append({"id": start + 10_000 + 2 * g + k, "attack_cat": cat, "tier": 2, "tier2_representative": k == 0,
                         "group_id": start + 5_000 + g, "is_tcp": g % 2 == 0, "eligible": eligible})
    p = pd.DataFrame(rows)
    p["flow_stratum"] = np.where(p["is_tcp"], "tcp", "other")
    return p


def test_tier1_only_when_enough_and_deterministic():
    pool = synthetic_pool(100, 20)
    a, per = s.draw_sample(pool, seed=2026)
    b, _ = s.draw_sample(pool, seed=2026)
    assert a["id"].tolist() == b["id"].tolist()
    assert len(a) == 30 and set(a["tier"]) == {1} and per["Exploits"]["shortfall"] == 0
    assert (a["is_tcp"].sum(), (~a["is_tcp"]).sum()) == (21, 9)                     # 70/30 of the eligible pool


def test_tier2_fills_only_below_30_with_one_representative_per_group():
    pool = synthetic_pool(12, 40)
    sample, per = s.draw_sample(pool, seed=2026)
    assert len(sample) == 30 and (sample["tier"] == 1).sum() == 12
    t2 = sample[sample["tier"] == 2]
    assert t2["tier2_representative"].all() and t2["group_id"].is_unique


def test_shortfall_recorded_never_filled_from_other_categories():
    pool = pd.concat([synthetic_pool(23, 0, cat="Worms"), synthetic_pool(200, 0, cat="Exploits", start=1_000)])
    sample, per = s.draw_sample(pool, seed=2026)
    assert per["Worms"] == {**per["Worms"], "selected": 23, "shortfall": 7}
    assert (sample["attack_cat"] == "Worms").sum() == 23 and (sample["attack_cat"] == "Exploits").sum() == 30


def test_ineligible_flows_are_never_drawn():
    pool = pd.concat([synthetic_pool(50, 0), synthetic_pool(50, 0, start=500, eligible=False)])
    sample, _ = s.draw_sample(pool, seed=2026)
    assert sample["eligible"].all() and sample["id"].max() < 500


# ---------------------------------------------------------------- frozen sample (real data)
@frozen
def test_frozen_sample_is_reproducible_from_the_pool():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    pool = pd.read_csv(POOL)
    pool["flow_stratum"] = np.where(pool["is_tcp"], "tcp", "other")
    redraw, per = s.draw_sample(pool, seed=m["seed"])
    frozen_csv = pd.read_csv(MANIFEST.parent / "sample_manifest.csv")
    assert redraw["id"].tolist() == frozen_csv["id"].tolist()
    assert {c: v["selected"] for c, v in per.items()} == {c: v["selected"] for c, v in m["per_category"].items()}


@frozen
def test_frozen_sample_obeys_the_rule():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sample = pd.read_csv(MANIFEST.parent / "sample_manifest.csv", float_precision="round_trip")
    assert sample["id"].is_unique and sample["group_id"].is_unique
    assert not set(sample["id"]) & set(m["excluded_v2_pilot_ids"])
    for model in ("random_forest", "xgboost", "mlp"):
        assert sample[f"attack_{model}_full42"].all() and (sample[f"p_{model}_full42"] >= 0.5).all()
    assert (sample.groupby("attack_cat").size() <= 30).all()
    assert set(sample.loc[sample["tier"] == 2, "attack_cat"]) <= {c for c, v in m["per_category"].items() if v["eligible_tier1"] < 30}
    from src.unsw.v3_final_models import file_sha256
    assert file_sha256(MANIFEST.parent / "sample_manifest.csv") == m["sample_manifest_csv_sha256"]
    assert m["official_test_partition_opened"] is False


@frozen
def test_sample_flows_are_validation_side_and_group_disjoint_from_training():
    from experiments import unsw_nested_cv as ncv
    from src.unsw import v3_data as vd
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    p = ncv.load_protocol()
    df = vd.load_official_train()
    split = vd.disjoint_split(df, p["data"]["split"]["validation_fraction"], p["data"]["split"]["seed"], label_conflicts=p["data"]["label_conflict_rule"])
    assert vd.split_digest(split) == m["split_digest"]
    info = split["info"]
    ids_val = set(info["id"].to_numpy()[split["validation"]])
    sample = pd.read_csv(MANIFEST.parent / "sample_manifest.csv")
    assert set(sample["id"]) <= ids_val
    train_hashes = set(info["hash"].to_numpy()[split["train"]])
    by_id = info.set_index("id")
    assert not set(by_id.loc[sample["id"], "hash"]) & train_hashes
    assert (by_id.loc[sample["id"], "stratum"].to_numpy() == sample["attack_cat"].to_numpy()).all()   # no MixedAttack, no conflicts
    rows = df.set_index("id").loc[sample["id"]]
    assert [s.vector_sha256(r) for _, r in rows.iterrows()] == sample["vector_sha256"].tolist()


def test_refuses_to_redraw_a_frozen_sample(tmp_path, monkeypatch):
    (tmp_path / "sample_manifest.json").write_text("{}")
    monkeypatch.setattr(s, "OUT", tmp_path)
    with pytest.raises(SystemExit, match="frozen"):
        s.main()


@frozen
def test_excluded_pilot_ids_are_the_official_ids_of_the_ten_pilot_rows():
    """Guards against using the reference table's 0-based row positions as official ids."""
    from src.unsw import v3_data as vd
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    df = vd.load_official_train()
    ref = pd.read_csv(ROOT / "results" / "reference" / "selected_flows.csv")
    rows = ref[ref["validation_position"].isin(s.PILOT_POSITIONS_V2)]["official_train_row_index"].astype(int)
    expected = sorted(int(df.iloc[r]["id"]) for r in rows)
    assert m["excluded_v2_pilot_ids"] == expected and len(expected) == 10
    assert sorted(int(r) for r in rows) != expected          # positions and ids differ by one
