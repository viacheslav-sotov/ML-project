"""Protocol v3 (draft) data layer: official TRAINING partition only, duplicate-aware groups, conflict rules and a
duplicate-disjoint, category-stratified train/validation split.

Nothing here reads the official test partition. The v1/v2 split (src.unsw.data.create_data_split) and every
earlier result stay unchanged; this module builds a NEW split for the final (v3) design.

Grouping rule (G-1)
    A group is the set of rows with an identical 42-feature vector (raw values, before preprocessing).
    A group is never split: all of its rows are on the same side of every split and every CV fold.

Conflict rules
    C-1 (label conflict): a group whose rows carry both labels (normal and attack) cannot be learned or
        evaluated consistently. Primary rule: such groups are EXCLUDED from the v3 split (counted and reported).
        Sensitivity variant (`label_conflicts="keep"`): kept whole, stratum "LabelConflict".
    C-2 (category conflict): an attack group whose rows carry several attack categories is kept (its label is
        consistent). Its stratum is "MixedAttack"; per-category metrics use each row's own category; such groups
        are not eligible for the attack sample (their category is ambiguous).

Stratum of a group (for stratification): its single attack category / "Normal" if uniform, else per C-1/C-2.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
# The partition files are named the wrong way round: this file holds the OFFICIAL TRAINING partition.
OFFICIAL_TRAIN_FILE = RAW / "UNSW_NB15_testing-set.csv"
OFFICIAL_TRAIN_ROWS = 175_341
OFFICIAL_TEST_FILE_NAME = "UNSW_NB15_training-set.csv"   # never opened by this module
TTL_FEATURES = ("sttl", "dttl", "ct_state_ttl")


class V3DataError(RuntimeError):
    pass


def feature_order() -> list[str]:
    from src.unsw.constraints import FEATURE_ORDER
    return list(FEATURE_ORDER)


def load_official_train(path: Path = OFFICIAL_TRAIN_FILE) -> pd.DataFrame:
    """The official training partition only; refuses any file that is not it."""
    path = Path(path)
    if path.name == OFFICIAL_TEST_FILE_NAME:
        raise V3DataError("the official test partition is locked")
    df = pd.read_csv(path)
    if len(df) != OFFICIAL_TRAIN_ROWS or "id" not in df.columns:
        raise V3DataError(f"{path.name} is not the official training partition ({len(df)} rows)")
    df["attack_cat"] = df["attack_cat"].fillna("Normal").astype(str).str.strip().replace({"Backdoors": "Backdoor", "": "Normal"})
    return df


def feature_hash(frame: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(frame[feature_order()], index=False).to_numpy()


def build_groups(df: pd.DataFrame) -> pd.DataFrame:
    """One row per input row: group id, group size, label/category conflict flags and the group stratum."""
    h = feature_hash(df)
    info = pd.DataFrame({"row": np.arange(len(df)), "id": df["id"].to_numpy(), "hash": h,
                         "label": df["label"].astype(int).to_numpy(), "attack_cat": df["attack_cat"].to_numpy()})
    g = info.groupby("hash")
    info["group_size"] = g["label"].transform("size").to_numpy()
    info["label_conflict"] = (g["label"].transform("nunique") > 1).to_numpy()
    info["category_conflict"] = ((g["attack_cat"].transform("nunique") > 1) & ~info["label_conflict"]).to_numpy()
    first_cat = g["attack_cat"].transform("first")
    info["stratum"] = np.where(info["label_conflict"], "LabelConflict",
                               np.where(info["category_conflict"], "MixedAttack", first_cat))
    # stable group ids, independent of row order: rank of the hash
    uniq = np.unique(h)                                  # exact uint64 sort; ids independent of row order
    info["group_id"] = np.searchsorted(uniq, h)
    return info


def disjoint_split(df: pd.DataFrame, validation_fraction: float = 0.2, seed: int = 42,
                   label_conflicts: str = "exclude") -> dict[str, Any]:
    """Duplicate-disjoint, stratum-stratified split. Returns row positions of train/validation/excluded."""
    if label_conflicts not in ("exclude", "keep"):
        raise V3DataError("label_conflicts must be 'exclude' or 'keep'")
    n_splits = int(round(1.0 / validation_fraction))
    if not np.isclose(1.0 / n_splits, validation_fraction):
        raise V3DataError("validation_fraction must be 1/k for an integer k")
    info = build_groups(df)
    excluded = info["label_conflict"].to_numpy() if label_conflicts == "exclude" else np.zeros(len(info), bool)
    use = info[~excluded]
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_pos, val_pos = next(sgkf.split(np.zeros(len(use)), use["stratum"].to_numpy(), groups=use["group_id"].to_numpy()))
    return {"info": info, "train": use["row"].to_numpy()[train_pos], "validation": use["row"].to_numpy()[val_pos],
            "excluded": info["row"].to_numpy()[excluded], "rule": {"grouping": "identical 42-feature vector",
            "label_conflicts": label_conflicts, "validation_fraction": validation_fraction, "seed": seed,
            "splitter": f"StratifiedGroupKFold(n_splits={n_splits}, shuffle=True, random_state={seed}), first fold = validation"}}


def audit(split: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    """Leakage and balance checks of a split (all must be read before the split is used)."""
    info = split["info"]
    tr, va, ex = split["train"], split["validation"], split["excluded"]
    h = info["hash"].to_numpy()
    shared = set(h[tr]) & set(h[va])
    rows = len(tr) + len(va)
    target = split["rule"]["validation_fraction"]
    cats = info["attack_cat"].to_numpy()
    by_cat = {}
    for c in sorted(set(cats)):
        n_tr, n_va = int((cats[tr] == c).sum()), int((cats[va] == c).sum())
        share = n_va / (n_tr + n_va) if n_tr + n_va else float("nan")
        by_cat[c] = {"train": n_tr, "validation": n_va, "validation_share": round(share, 4),
                     "deviation_from_target": round(share - target, 4)}
    strata = info["stratum"].to_numpy()
    by_stratum = {s: {"train": int((strata[tr] == s).sum()), "validation": int((strata[va] == s).sum())} for s in sorted(set(strata))}
    lab = info["label"].to_numpy()
    return {
        "rows_total": int(len(info)), "rows_train": int(len(tr)), "rows_validation": int(len(va)), "rows_excluded": int(len(ex)),
        "excluded_by_category": {c: int(n) for c, n in pd.Series(cats[ex]).value_counts().items()},
        "validation_share": round(len(va) / rows, 4),
        "groups_train": int(len(set(info["group_id"].to_numpy()[tr]))), "groups_validation": int(len(set(info["group_id"].to_numpy()[va]))),
        "shared_feature_vectors": len(shared), "shared_group_ids": len(set(info["group_id"].to_numpy()[tr]) & set(info["group_id"].to_numpy()[va])),
        "overlapping_rows": int(len(set(tr) & set(va))), "covers_all_rows": bool(len(set(tr) | set(va) | set(ex)) == len(info)),
        "attack_share_train": round(float(lab[tr].mean()), 4), "attack_share_validation": round(float(lab[va].mean()), 4),
        "by_category": by_cat, "by_stratum": by_stratum,
        "max_abs_category_deviation": round(max(abs(v["deviation_from_target"]) for v in by_cat.values()), 4),
        "label_conflict_rows_in_split": int(info["label_conflict"].to_numpy()[np.concatenate([tr, va])].sum()),
    }


def split_digest(split: dict[str, Any]) -> str:
    info = split["info"]
    ids = info["id"].to_numpy()
    payload = ",".join(map(str, sorted(ids[split["validation"]]))) + "|" + ",".join(map(str, sorted(ids[split["excluded"]])))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
