"""Protocol v3 (draft): audit of the duplicate-disjoint split and its nested-CV folds (read-only; no training).

Uses ONLY the official training partition. Writes results/v3_prep/split_audit/:
    audit.json                 primary rule (label conflicts excluded) and sensitivity rule (kept), group statistics,
                               conflict statistics, outer/inner fold checks, attack-sample pool sizes
    split_assignment.csv       id, side (train/validation/excluded), group_id, group_size, stratum, label, attack_cat
    conflict_groups.csv        every label-conflict group: size, labels, categories
    fold_balance.csv           outer and inner folds: rows and category shares per fold

    python experiments/unsw_v3_split_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import unsw_nested_cv as ncv  # noqa: E402
from src.unsw import v3_data as vd  # noqa: E402

OUT = ROOT / "results" / "v3_prep" / "split_audit"
PILOT_POSITIONS_V2 = [2941, 3040, 3206, 27094, 15137, 4407, 30107, 34186, 15741, 25740]


def pilot_official_ids() -> list[int]:
    """Official ids of the v2 pilot flows (from the committed reference table; no data split is rebuilt)."""
    ref = pd.read_csv(ROOT / "results" / "reference" / "selected_flows.csv")
    rows = ref[ref["validation_position"].isin(PILOT_POSITIONS_V2)]["official_train_row_index"].astype(int).tolist()
    return rows


def fold_balance(name: str, strata, cats, groups, folds) -> tuple[list[dict], dict]:
    rows, worst = [], 0.0
    total = pd.Series(cats).value_counts(normalize=True)
    shared = 0
    for k, (a, b) in enumerate(folds):
        shared += len(set(np.asarray(groups)[a]) & set(np.asarray(groups)[b]))
        share = pd.Series(np.asarray(cats)[b]).value_counts(normalize=True)
        for c in total.index:
            dev = float(share.get(c, 0.0) - total[c])
            worst = max(worst, abs(dev) / total[c])
            rows.append({"level": name, "fold": k, "category": c, "rows_in_fold": int((np.asarray(cats)[b] == c).sum()),
                         "share_in_fold": round(float(share.get(c, 0.0)), 5), "share_overall": round(float(total[c]), 5)})
    return rows, {"groups_shared_across_any_fold": int(shared), "worst_relative_category_deviation": round(worst, 4)}


def attack_pool(df: pd.DataFrame, split: dict, exclude_rows: set[int]) -> pd.DataFrame:
    info = split["info"]
    va = np.sort(split["validation"])
    v = info.iloc[va]
    v = v[(v["label"] == 1) & (~v["label_conflict"]) & (v["stratum"] != "MixedAttack") & (~v["row"].isin(exclude_rows))]
    proto = df["proto"].to_numpy()
    rows = []
    for c, g in v.groupby("attack_cat"):
        tier1 = g[g["group_size"] == 1]
        tier2 = g[g["group_size"] > 1].drop_duplicates("group_id")
        rows.append({"attack_cat": c, "validation_attack_rows_eligible_before_models": int(len(g)),
                     "tier1_singletons": int(len(tier1)), "tier1_tcp": int((proto[tier1["row"]] == "tcp").sum()),
                     "tier2_group_representatives": int(len(tier2)),
                     "tier1_shortfall_vs_30": max(0, 30 - int(len(tier1))),
                     "tier1_plus_tier2_shortfall_vs_30": max(0, 30 - int(len(tier1)) - int(len(tier2)))})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = ncv.load_protocol()
    df = vd.load_official_train()
    s = protocol["data"]["split"]
    result: dict = {"official_test_partition_opened": False, "rule": None}
    primary = vd.disjoint_split(df, s["validation_fraction"], s["seed"], label_conflicts="exclude")
    sensitivity = vd.disjoint_split(df, s["validation_fraction"], s["seed"], label_conflicts="keep")
    result["rule"] = primary["rule"]
    result["primary"] = vd.audit(primary, df)
    result["sensitivity_keep_label_conflicts"] = vd.audit(sensitivity, df)
    result["split_digest_primary"] = vd.split_digest(primary)
    info = primary["info"]
    gs = info.drop_duplicates("group_id")
    result["groups"] = {
        "rows": int(len(info)), "groups": int(len(gs)), "singleton_groups": int((gs["group_size"] == 1).sum()),
        "rows_in_duplicate_groups": int((info["group_size"] > 1).sum()),
        "largest_group_sizes": sorted(gs["group_size"].tolist(), reverse=True)[:10],
        "label_conflict_groups": int(gs["label_conflict"].sum()), "label_conflict_rows": int(info["label_conflict"].sum()),
        "label_conflict_rows_by_category": {k: int(v) for k, v in info[info["label_conflict"]]["attack_cat"].value_counts().items()},
        "category_conflict_groups": int(gs["category_conflict"].sum()), "category_conflict_rows": int(info["category_conflict"].sum()),
        "category_conflict_rows_by_category": {k: int(v) for k, v in info[info["category_conflict"]]["attack_cat"].value_counts().items()},
        "strata": {k: int(v) for k, v in info["stratum"].value_counts().items()},
    }
    conflict = info[info["label_conflict"]].groupby("group_id").agg(size=("row", "size"), labels=("label", lambda x: "|".join(map(str, sorted(set(x))))),
                                                                     categories=("attack_cat", lambda x: "|".join(sorted(set(x)))))
    conflict.to_csv(OUT / "conflict_groups.csv", lineterminator="\n")
    # old (v1/v2) pilot flows: where do they fall in the v3 split?
    pilot_ids = set(pilot_official_ids())
    id_to_row = {int(i): r for r, i in enumerate(df.index)}
    pilot_rows = {id_to_row[i] for i in pilot_ids if i in id_to_row}
    side = np.full(len(info), "excluded", dtype=object)
    side[primary["train"]] = "train"
    side[primary["validation"]] = "validation"
    result["v2_pilot_flows_in_v3"] = {str(int(df.index[r])): side[r] for r in sorted(pilot_rows)}
    # nested-CV folds on the train side (primary)
    data = ncv.load_v3_train(protocol, df)
    outer = ncv.make_folds(data["strata"], data["groups"], protocol["nested_cv"]["outer"])
    rows_o, check_o = fold_balance("outer", data["strata"], data["categories"], data["groups"], outer)
    rows_i, worst_i, shared_i = [], 0.0, 0
    for k, (a, _) in enumerate(outer):
        inner = ncv.make_folds(data["strata"][a], data["groups"][a], protocol["nested_cv"]["inner"])
        r, c = fold_balance(f"inner_of_outer{k}", data["strata"][a], data["categories"][a], data["groups"][a], inner)
        rows_i += r
        worst_i = max(worst_i, c["worst_relative_category_deviation"])
        shared_i += c["groups_shared_across_any_fold"]
    result["nested_cv_folds"] = {"outer": check_o, "inner": {"groups_shared_across_any_fold": shared_i, "worst_relative_category_deviation": round(worst_i, 4)},
                                 "train_rows": int(len(data["y"])), "train_groups": int(len(set(data["groups"])))}
    pd.DataFrame(rows_o + rows_i).to_csv(OUT / "fold_balance.csv", index=False, lineterminator="\n")
    pool = attack_pool(df, primary, pilot_rows)
    pool.to_csv(OUT / "attack_pool_before_models.csv", index=False, lineterminator="\n")
    result["attack_pool_before_models"] = pool.to_dict(orient="records")
    pd.DataFrame({"id": info["id"], "side": side, "group_id": info["group_id"], "group_size": info["group_size"],
                  "stratum": info["stratum"], "label": info["label"], "attack_cat": info["attack_cat"]}).to_csv(
        OUT / "split_assignment.csv", index=False, lineterminator="\n")
    (OUT / "audit.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8", newline="\n")
    print(json.dumps({k: result[k] for k in ("primary", "groups", "nested_cv_folds", "v2_pilot_flows_in_v3")}, indent=2, default=str))


if __name__ == "__main__":
    main()
