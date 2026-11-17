"""WS1b: can UNSW-NB15 support a valid time-based split? (read-only investigation; no training, no models)

Reads the four raw files data/raw/UNSW-NB15_{1..4}.csv (2.54M records with Stime/Ltime) and the OFFICIAL
TRAINING partition only (data/raw/UNSW_NB15_testing-set.csv, 175,341 rows; the file names are swapped).
The official TEST partition file is never opened.

Checks
1. Raw timestamps: range, capture periods, order within and across files, Ltime - Stime vs dur.
2. Raw-to-model feature mapping (45 raw columns -> 42 model features; `rate` is absent and is rebuilt
   with the verified formula), and agreement with the partitioned file on linked rows.
3. Category labels: spelling variants, distribution per capture period and per hour.
4. Duplicates on the 42 model features (and label conflicts), overall and across candidate time cuts.
5. Linkage: can official-training rows be matched to raw records (and so receive a timestamp)?
6. Chronological split feasibility for candidate cuts.

Note: the raw files contain every UNSW-NB15 record, including those later placed in the official test
partition. This script only counts and describes them; nothing here trains, tunes or selects anything.

    python experiments/unsw_time_split_investigation.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "ws1b_time_split"
OFFICIAL_TRAIN_FILE = RAW / "UNSW_NB15_testing-set.csv"   # 175,341 rows = official training partition (names swapped)
OFFICIAL_TRAIN_ROWS = 175_341
RAW_FILES = [RAW / f"UNSW-NB15_{i}.csv" for i in range(1, 5)]

RENAME = {"Spkts": "spkts", "Dpkts": "dpkts", "Sload": "sload", "Dload": "dload", "Sjit": "sjit", "Djit": "djit",
          "Sintpkt": "sinpkt", "Dintpkt": "dinpkt", "smeansz": "smean", "dmeansz": "dmean", "res_bdy_len": "response_body_len",
          "ct_src_ ltm": "ct_src_ltm", "ct_src_ltm": "ct_src_ltm", "Label": "label"}
MODEL_FEATURES = ["dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl", "sload", "dload",
                  "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat",
                  "smean", "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
                  "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
                  "is_sm_ips_ports"]
LINK_KEY = ["proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "swin", "dwin",
            "stcpb", "dtcpb", "smean", "dmean", "trans_depth", "response_body_len", "dur6"]
CHECK_AFTER_LINK = ["sload", "dload", "sinpkt", "dinpkt", "sjit", "djit", "tcprtt", "synack", "ackdat", "ct_srv_src", "ct_state_ttl",
                    "ct_dst_ltm", "ct_src_ltm", "ct_srv_dst", "ct_dst_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
                    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "is_sm_ips_ports"]
CATEGORY_FIX = {"Backdoors": "Backdoor"}


def clean_cat(series: pd.Series) -> pd.Series:
    s = series.fillna("Normal").astype(str).str.strip()
    s = s.replace({"": "Normal", "nan": "Normal"})
    return s.replace(CATEGORY_FIX)


def load_raw() -> pd.DataFrame:
    names = [n.strip() if n != "ct_src_ ltm" else n for n in pd.read_csv(RAW / "NUSW-NB15_features.csv", encoding="latin-1")["Name"]]
    frames = []
    for i, path in enumerate(RAW_FILES, 1):
        df = pd.read_csv(path, header=None, names=names, low_memory=False,
                         usecols=[n for n in names if n not in ("srcip", "sport", "dstip", "dsport")])
        df["raw_file"] = i
        df["raw_row"] = np.arange(len(df))
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True).rename(columns=RENAME)
    raw["attack_cat_raw"] = raw["attack_cat"]
    raw["_blank_ct_flw_http_mthd"] = raw["ct_flw_http_mthd"].isna() | (raw["ct_flw_http_mthd"].astype(str).str.strip() == "")
    raw["_blank_is_ftp_login"] = raw["is_ftp_login"].isna() | (raw["is_ftp_login"].astype(str).str.strip() == "")
    raw["_blank_ct_ftp_cmd"] = raw["ct_ftp_cmd"].isna() | (raw["ct_ftp_cmd"].astype(str).str.strip() == "")
    raw["attack_cat"] = clean_cat(raw["attack_cat"])
    for col in ("ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    # rate is not in the raw files: rebuild with the Stage 1 formula (100% support on the partition, dur > 0)
    raw["rate"] = np.where(raw["dur"] > 0, (raw["spkts"] + raw["dpkts"] - 1) / raw["dur"].where(raw["dur"] > 0, 1), 0.0)
    return raw


def iso(t: float) -> str:
    return datetime.fromtimestamp(float(t), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def timestamps(raw: pd.DataFrame) -> dict:
    out = {"rows": int(len(raw)), "per_file": []}
    for i, g in raw.groupby("raw_file"):
        st = g["Stime"]
        out["per_file"].append({"file": f"UNSW-NB15_{i}.csv", "rows": int(len(g)), "stime_min_utc": iso(st.min()), "stime_max_utc": iso(st.max()),
                                "stime_non_decreasing_share": round(float((st.diff().dropna() >= 0).mean()), 4)})
    gaps = np.sort(raw["Stime"].unique())
    d = np.diff(gaps)
    big = np.argsort(d)[-3:][::-1]
    out["largest_gaps"] = [{"after_utc": iso(gaps[i]), "before_utc": iso(gaps[i + 1]), "hours": round(float(d[i]) / 3600, 2)} for i in big]
    cut = gaps[big[0]] + d[big[0]] / 2
    out["period_cut_utc"] = iso(cut)
    raw["period"] = np.where(raw["Stime"] < cut, 1, 2)
    for p, g in raw.groupby("period"):
        out[f"period_{p}"] = {"rows": int(len(g)), "start_utc": iso(g["Stime"].min()), "end_utc": iso(g["Ltime"].max()),
                              "hours": round(float(g["Ltime"].max() - g["Stime"].min()) / 3600, 2), "files": sorted(g["raw_file"].unique().tolist())}
    span = raw["Ltime"] - raw["Stime"]
    out["ltime_minus_stime_vs_dur"] = {"share_within_1s": round(float((np.abs(span - raw["dur"]) <= 1.0).mean()), 4),
                                      "note": "Stime/Ltime have 1-second resolution"}
    out["stime_resolution_seconds"] = float(np.min(d[d > 0])) if (d > 0).any() else None
    return out


def categories(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    variants = raw["attack_cat_raw"].fillna("<empty>").astype(str).value_counts().to_dict()
    by_period = pd.crosstab(raw["attack_cat"], raw["period"]).rename(columns=lambda c: f"period_{c}")
    by_period["total"] = by_period.sum(axis=1)
    raw["hour_utc"] = pd.to_datetime(raw["Stime"], unit="s", utc=True).dt.strftime("%Y-%m-%d %H:00")
    by_hour = pd.crosstab(raw["hour_utc"], raw["attack_cat"])
    label_mismatch = int(((raw["attack_cat"] == "Normal") != (raw["label"] == 0)).sum())
    return by_period.reset_index(), by_hour.reset_index(), {"raw_label_variants": variants, "label_vs_category_mismatches": label_mismatch}


def duplicates(raw: pd.DataFrame) -> dict:
    feats = [f for f in MODEL_FEATURES]
    key = pd.util.hash_pandas_object(raw[feats], index=False)
    raw["feat_hash"] = key
    groups = raw.groupby("feat_hash")
    size = groups["label"].transform("size")
    n_dup_rows = int((size > 1).sum())
    labels_per_group = groups["label"].transform("nunique")
    periods_per_group = groups["period"].transform("nunique")
    return {
        "rows": int(len(raw)), "distinct_feature_vectors": int(key.nunique()),
        "rows_in_duplicate_groups": n_dup_rows, "share_rows_duplicated": round(n_dup_rows / len(raw), 4),
        "rows_in_groups_with_conflicting_labels": int((labels_per_group > 1).sum()),
        "rows_in_groups_spanning_both_periods": int((periods_per_group > 1).sum()),
        "attack_rows_duplicated_share": round(float((size[raw["label"] == 1] > 1).mean()), 4),
        "normal_rows_duplicated_share": round(float((size[raw["label"] == 0] > 1).mean()), 4),
    }


def missing_and_encoding(raw: pd.DataFrame) -> dict:
    """Blank values of the FTP/HTTP features per period, and cross-period duplicates without them."""
    out = {}
    for col in ("ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd"):
        out[col] = {f"blank_share_period_{p}": round(float(g[f"_blank_{col}"].mean()), 4) for p, g in raw.groupby("period")}
        out[col]["values_top"] = {str(k): int(v) for k, v in raw[col].value_counts(dropna=False).head(6).items()}
    feats = [f for f in MODEL_FEATURES if f not in ("ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd")]
    h = pd.util.hash_pandas_object(raw[feats], index=False)
    p1, p2 = set(h[raw["period"] == 1]), set(h[raw["period"] == 2])
    out["cross_period_shared_vectors_without_ftp_http_features"] = len(p1 & p2)
    feats2 = [f for f in feats if not f.startswith("ct_")]
    h2 = pd.util.hash_pandas_object(raw[feats2], index=False)
    q1, q2 = set(h2[raw["period"] == 1]), set(h2[raw["period"] == 2])
    out["cross_period_shared_vectors_without_any_ct_feature"] = len(q1 & q2)
    return out


def link_partition(raw: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    with OFFICIAL_TRAIN_FILE.open("rb") as fh:
        n = sum(1 for _ in fh) - 1
    if n != OFFICIAL_TRAIN_ROWS:
        raise SystemExit(f"{OFFICIAL_TRAIN_FILE.name} has {n} rows, expected the official training partition ({OFFICIAL_TRAIN_ROWS})")
    part = pd.read_csv(OFFICIAL_TRAIN_FILE)
    part["attack_cat"] = clean_cat(part["attack_cat"])
    for df in (raw, part):
        df["dur6"] = df["dur"].round(6)
    def keyed(df: pd.DataFrame) -> pd.Series:
        k = df[LINK_KEY].copy()
        for c in LINK_KEY:
            k[c] = k[c].astype(str).str.strip() if c in ("proto", "service", "state") else pd.to_numeric(k[c], errors="coerce").astype("float64")
        return pd.util.hash_pandas_object(k, index=False)

    rk, pk = keyed(raw), keyed(part)
    raw["_k"], part["_k"] = rk, pk
    counts = rk.value_counts()
    part["raw_matches"] = pk.map(counts).fillna(0).astype(int)
    periods = raw.groupby("_k")["period"].agg(lambda x: "".join(sorted(set(map(str, x)))))
    part["raw_periods"] = pk.map(periods).fillna("none")
    first = raw.drop_duplicates("_k").set_index("_k")
    uniq = part[part["raw_matches"] == 1].copy()
    linked = uniq.join(first[["Stime", "Ltime", "period", "raw_file", "label", "attack_cat"] + CHECK_AFTER_LINK + ["rate"]],
                       on="_k", rsuffix="_raw")
    agree = {}
    for col in CHECK_AFTER_LINK + ["rate"]:
        a, b = pd.to_numeric(linked[col], errors="coerce"), pd.to_numeric(linked[f"{col}_raw"], errors="coerce")
        agree[col] = round(float(np.isclose(a, b, rtol=1e-3, atol=1e-3, equal_nan=True).mean()), 4)
    stats = {
        "official_train_rows": int(len(part)),
        "matched_exactly_one_raw_record": int((part["raw_matches"] == 1).sum()),
        "matched_several_raw_records": int((part["raw_matches"] > 1).sum()),
        "matched_none": int((part["raw_matches"] == 0).sum()),
        "label_agreement_on_unique_links": round(float((linked["label"] == linked["label_raw"]).mean()), 4),
        "category_agreement_on_unique_links": round(float((linked["attack_cat"] == linked["attack_cat_raw"]).mean()), 4),
        "feature_agreement_on_unique_links": agree,
        "unique_links_by_category": linked["attack_cat"].value_counts().to_dict(),
        "all_rows_by_category": part["attack_cat"].value_counts().to_dict(),
        "unique_links_by_period": linked["period"].value_counts().to_dict(),
        "all_rows_by_matching_periods": part["raw_periods"].value_counts().to_dict(),
        "all_rows_by_matching_periods_and_label": {f"{k[0]}|label{k[1]}": int(v) for k, v in
                                                    part.groupby(["raw_periods", "label"]).size().items()},
        "partition_values_top": {c: {str(k): int(v) for k, v in part[c].value_counts(dropna=False).head(4).items()}
                                 for c in ("ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd")},
    }
    return stats, linked


def current_split_leakage() -> dict:
    """Duplicate leakage in the CURRENT random split (official training partition, 80/20, stratified, rs 42),
    reproduced from src.unsw.data.create_data_split without loading the official test partition."""
    sys.path.insert(0, str(ROOT))
    from sklearn.model_selection import train_test_split
    from src.unsw.data import prepare_features_target
    part = pd.read_csv(OFFICIAL_TRAIN_FILE)
    X, y = prepare_features_target(part)
    X_sub, X_val, y_sub, y_val = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    hs = set(pd.util.hash_pandas_object(X_sub, index=False))
    hv = pd.util.hash_pandas_object(X_val, index=False)
    in_sub = hv.isin(hs).to_numpy()
    cats = clean_cat(part.loc[X_val.index, "attack_cat"]).to_numpy()
    by_cat = {c: round(float(in_sub[cats == c].mean()), 4) for c in sorted(set(cats))}
    dup_within_sub = float(pd.util.hash_pandas_object(X_sub, index=False).duplicated(keep=False).mean())
    return {"subtrain_rows": int(len(X_sub)), "validation_rows": int(len(X_val)),
            "validation_rows_with_identical_vector_in_subtrain": int(in_sub.sum()),
            "share": round(float(in_sub.mean()), 4),
            "share_attack": round(float(in_sub[y_val.to_numpy() == 1].mean()), 4),
            "share_normal": round(float(in_sub[y_val.to_numpy() == 0].mean()), 4),
            "share_by_category": by_cat,
            "subtrain_rows_in_duplicate_groups_share": round(dup_within_sub, 4),
            "pilot_flows_with_identical_vector_in_subtrain": {
                int(p): bool(in_sub[list(X_val.index).index(X_val.index[p])]) for p in
                (2941, 3040, 3206, 27094, 15137, 4407, 30107, 34186, 15741, 25740)}}


def split_feasibility(raw: pd.DataFrame, linked: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def describe(name, early, late, note):
        cats = sorted(set(raw["attack_cat"]))
        e, l = early["attack_cat"].value_counts(), late["attack_cat"].value_counts()
        dup_cross = len(set(early["feat_hash"]) & set(late["feat_hash"])) if "feat_hash" in early else None
        for c in cats:
            rows.append({"split": name, "attack_cat": c, "early_rows": int(e.get(c, 0)), "late_rows": int(l.get(c, 0)), "note": note})
        rows.append({"split": name, "attack_cat": "_ALL_", "early_rows": int(len(early)), "late_rows": int(len(late)),
                     "note": f"{note}; distinct feature vectors in both sides: {dup_cross}"})

    describe("S1_capture_period", raw[raw["period"] == 1], raw[raw["period"] == 2], "raw records: period 1 vs period 2")
    for p in (1, 2):
        g = raw[raw["period"] == p]
        cut = g["Stime"].quantile(0.8)
        describe(f"S2_within_period_{p}_80_20", g[g["Stime"] < cut], g[g["Stime"] >= cut], f"raw records of period {p}: first 80% of time vs last 20%")
    if len(linked):
        cut = linked["Stime"].quantile(0.8)
        lk = linked.copy()
        lk["feat_hash"] = pd.util.hash_pandas_object(lk[[c for c in MODEL_FEATURES if c in lk]], index=False)
        describe("S3_official_train_linked_80_20", lk[lk["Stime"] < cut], lk[lk["Stime"] >= cut],
                 "official-training rows with a unique raw link: earliest 80% vs latest 20%")
    return pd.DataFrame(rows)


def main() -> None:
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "official_test_partition_opened": False,
               "mapping": {"raw_to_model_renames": RENAME, "rate": "absent in raw; rebuilt as (spkts+dpkts-1)/dur for dur>0, else 0",
                           "dropped_raw_columns": ["srcip", "sport", "dstip", "dsport", "Stime", "Ltime"],
                           "model_features_available": [f for f in MODEL_FEATURES if f in raw.columns]}}
    summary["timestamps"] = timestamps(raw)
    by_period, by_hour, cat_info = categories(raw)
    summary["categories"] = cat_info
    summary["duplicates"] = duplicates(raw)
    summary["missing_and_encoding"] = missing_and_encoding(raw)
    link_stats, linked = link_partition(raw)
    summary["linkage"] = link_stats
    feas = split_feasibility(raw, linked)
    summary["current_split_leakage"] = current_split_leakage()
    summary["elapsed_seconds"] = round(time.perf_counter() - start, 1)
    by_period.to_csv(OUT / "categories_by_period.csv", index=False, lineterminator="\n")
    by_hour.to_csv(OUT / "categories_by_hour.csv", index=False, lineterminator="\n")
    feas.to_csv(OUT / "split_feasibility.csv", index=False, lineterminator="\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
