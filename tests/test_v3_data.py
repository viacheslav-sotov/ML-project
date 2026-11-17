"""v3 duplicate-disjoint split builder (src/unsw/v3_data.py). Official training partition only."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.unsw import v3_data as vd


@pytest.fixture(scope="module")
def official():
    return vd.load_official_train()


def test_loader_refuses_the_test_partition_and_other_files(tmp_path):
    fake_test = tmp_path / vd.OFFICIAL_TEST_FILE_NAME
    with pytest.raises(vd.V3DataError, match="locked"):
        vd.load_official_train(fake_test)
    other = tmp_path / "other.csv"
    other.write_text("id,label\n1,0\n", encoding="utf-8")
    with pytest.raises(vd.V3DataError, match="not the official training partition"):
        vd.load_official_train(other)


def test_groups_and_conflicts_on_a_small_frame(official):
    df = official.iloc[:6].copy().reset_index(drop=True)
    df = pd.concat([df, df.iloc[[0, 0, 1]]], ignore_index=True)       # duplicates of rows 0 and 1
    df.loc[6, "label"] = 1 - int(df.loc[0, "label"])                   # row 6 = copy of row 0 with the other label
    df.loc[6, "attack_cat"] = "Normal" if df.loc[0, "attack_cat"] != "Normal" else "Fuzzers"
    info = vd.build_groups(df)
    assert info.loc[0, "group_id"] == info.loc[6, "group_id"] == info.loc[7, "group_id"]
    assert info.loc[0, "group_size"] == 3 and info.loc[0, "label_conflict"] and info.loc[0, "stratum"] == "LabelConflict"
    assert info.loc[1, "group_size"] == 2 and not info.loc[1, "label_conflict"]
    assert info["group_id"].nunique() == 6


def test_group_ids_do_not_depend_on_row_order(official):
    df = official.sample(3000, random_state=1)
    a = vd.build_groups(df.reset_index(drop=True)).set_index("id")["group_id"]
    b = vd.build_groups(df.sample(frac=1.0, random_state=2).reset_index(drop=True)).set_index("id")["group_id"]
    assert a.sort_index().equals(b.sort_index())


def test_real_split_is_duplicate_disjoint_and_balanced(official):
    split = vd.disjoint_split(official, 0.2, 42, "exclude")
    audit = vd.audit(split, official)
    assert audit["shared_feature_vectors"] == 0 and audit["shared_group_ids"] == 0 and audit["overlapping_rows"] == 0
    assert audit["covers_all_rows"]
    assert audit["rows_excluded"] == int(split["info"]["label_conflict"].sum()) == 940
    assert audit["label_conflict_rows_in_split"] == 0
    assert abs(audit["validation_share"] - 0.2) < 0.01
    assert vd.split_digest(split) == vd.split_digest(vd.disjoint_split(official, 0.2, 42, "exclude"))   # deterministic
    keep = vd.audit(vd.disjoint_split(official, 0.2, 42, "keep"), official)
    assert keep["rows_excluded"] == 0 and keep["shared_feature_vectors"] == 0


def test_audit_detects_leakage_of_a_naive_random_split(official):
    info = vd.build_groups(official)
    tr, va = train_test_split(np.arange(len(official)), test_size=0.2, random_state=42, stratify=official["label"])
    naive = {"info": info, "train": tr, "validation": va, "excluded": np.array([], int), "rule": {"validation_fraction": 0.2}}
    assert vd.audit(naive, official)["shared_feature_vectors"] > 1000


def test_bad_arguments_are_refused(official):
    with pytest.raises(vd.V3DataError):
        vd.disjoint_split(official.iloc[:100], 0.3, 42)
    with pytest.raises(vd.V3DataError):
        vd.disjoint_split(official.iloc[:100], 0.2, 42, label_conflicts="drop_some")
