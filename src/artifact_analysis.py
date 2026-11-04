from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Same sample, same split and same metrics as the earlier stages
from baseline import load_processed_data, prepare_data, train_baseline
from leakage_analysis import (
    CONSTANT_FEATURES,
    FINGERPRINT_FEATURES,
    evaluate,
    feature_group,
    value_purity,
)


RAW_DATA_PATH = Path("data/raw/cic.csv")
OUTPUT_DIR = Path("results/leakage_analysis")

HEADER_FEATURES = ["Fwd Header Len", "Bwd Header Len"]

TCP_FLAG_FEATURES = [
    "Fwd PSH Flags",
    "FIN Flag Cnt",
    "SYN Flag Cnt",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "ACK Flag Cnt",
    "URG Flag Cnt",
    "ECE Flag Cnt",
]

CORE_FLOW_FEATURES = [
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Fwd Pkts/s",
    "Bwd Pkts/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Down/Up Ratio",
    "Active Mean",
    "Idle Mean",
]

# A feature is a "signature" if one value covers most training attacks
# while almost never appearing in training benign traffic
SIGNATURE_ATTACK_SHARE = 0.50
SIGNATURE_BENIGN_SHARE = 0.01

# Flows closer than this (in standardised units) count as near-duplicates
NEAR_DUPLICATE_DISTANCE = 0.01

TIME_BLOCK = "5min"


def split_like_baseline(X, y):
    # Identical call to the one inside baseline.train_baseline
    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


def fit_pipeline(X_train, y_train, impute=False):
    steps = []

    if impute:
        steps.append(("imputer", SimpleImputer(strategy="median")))

    steps += [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ]

    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)

    return pipeline


def row_hashes(X):
    return pd.util.hash_pandas_object(X, index=False)


# ---------------------------------------------------------------- 1. duplicates


def analyse_duplicates(df, X_train, X_test, y_train, y_test, e_features):
    print("\n===== 1. DUPLICATES AND NEAR-DUPLICATES =====")

    all_features = list(X_train.columns)
    rows = []

    # 1a / 1b: test rows that exactly match a training row
    for name, cols in [("all_78", all_features), ("E_63", e_features)]:
        train_hash = row_hashes(X_train[cols])
        test_hash = row_hashes(X_test[cols])

        in_train = test_hash.isin(set(train_hash)).values

        # Do the matching training rows ever carry the other label?
        train_labels = pd.DataFrame({"h": train_hash.values, "y": y_train.values})
        labels_per_hash = train_labels.groupby("h")["y"].agg(["min", "max"])
        conflicting = set(labels_per_hash.index[labels_per_hash["min"] != labels_per_hash["max"]])

        for label, label_name in [(0, "Benign"), (1, "Attack")]:
            mask = (y_test == label).values
            rows.append({
                "check": f"test rows exactly in train ({name})",
                "class": label_name,
                "count": int(in_train[mask].sum()),
                "total": int(mask.sum()),
                "share": in_train[mask].mean(),
            })

        rows.append({
            "check": f"test rows matching a train row with both labels ({name})",
            "class": "All",
            "count": int(test_hash.isin(conflicting).sum()),
            "total": len(test_hash),
            "share": test_hash.isin(conflicting).mean(),
        })

    # 1c: duplicates across the full cleaned dataset
    for name, cols in [("all_78", all_features), ("E_63", e_features)]:
        dup = df.duplicated(subset=cols)
        for label, label_name in [(0, "Benign"), (1, "Attack")]:
            mask = df["BinaryLabel"] == label
            rows.append({
                "check": f"full dataset rows that duplicate an earlier row ({name})",
                "class": label_name,
                "count": int(dup[mask].sum()),
                "total": int(mask.sum()),
                "share": dup[mask].mean(),
            })

    # 1d: nearest training neighbour of every test row (E features)
    scaler = StandardScaler().fit(X_train[e_features])
    train_scaled = scaler.transform(X_train[e_features])
    test_scaled = scaler.transform(X_test[e_features])

    nn = NearestNeighbors(n_neighbors=1).fit(train_scaled)
    distance, index = nn.kneighbors(test_scaled)
    distance = distance[:, 0]
    neighbour_label = y_train.values[index[:, 0]]

    for label, label_name in [(0, "Benign"), (1, "Attack")]:
        mask = (y_test == label).values
        rows.append({
            "check": f"test rows with a train neighbour closer than {NEAR_DUPLICATE_DISTANCE} (E_63, scaled)",
            "class": label_name,
            "count": int((distance[mask] < NEAR_DUPLICATE_DISTANCE).sum()),
            "total": int(mask.sum()),
            "share": (distance[mask] < NEAR_DUPLICATE_DISTANCE).mean(),
        })
        rows.append({
            "check": "median distance to nearest train row (E_63, scaled)",
            "class": label_name,
            "count": np.nan,
            "total": int(mask.sum()),
            "share": np.median(distance[mask]),
        })
        rows.append({
            "check": "nearest train row has the same label (1-NN accuracy, E_63)",
            "class": label_name,
            "count": int((neighbour_label[mask] == label).sum()),
            "total": int(mask.sum()),
            "share": (neighbour_label[mask] == label).mean(),
        })

    results = pd.DataFrame(rows)

    output = OUTPUT_DIR / "stage3_duplicates.csv"
    results.to_csv(output, index=False)
    print(results.to_string(index=False))
    print(f"Saved: {output}")

    return results


def evaluate_on_unseen_rows(X, y, e_features):
    print("\n===== 1e. SCORES ON TEST ROWS WITH NO EXACT COPY IN TRAIN =====")

    X_train, X_test, y_train, y_test = split_like_baseline(X, y)
    rows = []

    for name, cols in [("A_all_78_features", list(X.columns)), ("E_behavior_focused", e_features)]:
        model = fit_pipeline(X_train[cols], y_train)

        unseen = ~row_hashes(X_test[cols]).isin(set(row_hashes(X_train[cols]))).values

        for subset, mask in [("full test set", np.ones(len(X_test), bool)), ("unseen rows only", unseen)]:
            rows.append({
                "experiment": name,
                "test_subset": subset,
                "n_test_rows": int(mask.sum()),
                **evaluate(model, X_test[cols][mask], y_test[mask]),
            })

    results = pd.DataFrame(rows)

    output = OUTPUT_DIR / "stage3_unseen_test_rows.csv"
    results.to_csv(output, index=False)
    print(results.to_string(index=False))
    print(f"Saved: {output}")


# ----------------------------------------------------------- 2. feature ranking


def rank_remaining_features(X_train, y_train, e_features):
    print("\n===== 2. RANKING OF THE 63 FEATURES LEFT AFTER EXPERIMENT E (train rows only) =====")

    rows = []

    for col in e_features:
        x = X_train[col]

        if x.nunique() > 1:
            auc = roc_auc_score(y_train, x)
            single_feature_auc = max(auc, 1 - auc)
        else:
            single_feature_auc = 0.5

        pure_coverage, top_value, top_share, benign_share = value_purity(x, y_train)

        rows.append({
            "feature": col,
            "group": feature_group(col),
            "n_unique": x.nunique(),
            "single_feature_auc": single_feature_auc,
            "pure_value_coverage": pure_coverage,
            "top_attack_value": top_value,
            "top_attack_value_share_of_attacks": top_share,
            "top_attack_value_share_of_benign": benign_share,
            "is_signature": (
                top_share >= SIGNATURE_ATTACK_SHARE
                and benign_share <= SIGNATURE_BENIGN_SHARE
            ),
        })

    results = pd.DataFrame(rows).sort_values("single_feature_auc", ascending=False)

    # Columns that are exact copies of an earlier column
    copies = []
    for i, a in enumerate(e_features):
        for b in e_features[i + 1:]:
            if X_train[a].equals(X_train[b]):
                copies.append(f"{b} = {a}")

    results["identical_to"] = results["feature"].map(
        {c.split(" = ")[0]: c.split(" = ")[1] for c in copies}
    )

    output = OUTPUT_DIR / "stage3_feature_ranking.csv"
    results.to_csv(output, index=False)
    print(results.round(4).to_string(index=False))
    print(f"\nIdentical column pairs: {copies}")
    print(f"Saved: {output}")

    return results


# ------------------------------------------------------------ 3. group ablation


def run_group_ablation(X, y, e_features, ranking):
    print("\n===== 3. GROUP ABLATION =====")

    f_features = [c for c in e_features if c not in HEADER_FEATURES]
    g_features = [c for c in f_features if c not in TCP_FLAG_FEATURES]

    signature = set(ranking.loc[ranking["is_signature"], "feature"])
    h_features = [c for c in g_features if c not in signature]

    print(f"Signature features removed in H: {sorted(signature & set(g_features))}")

    i_features = [c for c in CORE_FLOW_FEATURES if c in h_features]

    j_features = [
        c for c in h_features
        if feature_group(c) == "timing"
    ]
    k_features = [
        c for c in h_features
        if feature_group(c) == "packet/size"
    ]

    experiments = {
        "A_all_78_features": list(X.columns),
        "E_behavior_focused": e_features,
        "F_E_minus_header_len": f_features,
        "G_F_minus_tcp_flags": g_features,
        "H_G_minus_signature_values": h_features,
        "I_core_flow_only": i_features,
        "J_timing_only": j_features,
        "K_size_volume_only": k_features,
    }

    rows = []

    for name, cols in experiments.items():
        print(f"\n--- {name}: {len(cols)} features")
        print(cols)

        model, X_test, y_test = train_baseline(X[cols], y)

        rows.append({
            "experiment": name,
            "n_features": len(cols),
            "features": ", ".join(cols),
            **evaluate(model, X_test, y_test),
        })

    results = pd.DataFrame(rows)

    output = OUTPUT_DIR / "stage3_group_ablation.csv"
    results.to_csv(output, index=False)
    print(results.drop(columns="features").to_string(index=False))
    print(f"Saved: {output}")

    return experiments


# ------------------------------------------------------- raw rebuild (4 and 5)


def rebuild_without_imputation():
    # Same steps and order as preprocessing.py, but keep NaN and Timestamp
    print("\nRebuilding dataset from raw file without imputation...")

    raw = pd.read_csv(RAW_DATA_PATH)
    raw = raw.drop_duplicates()

    numeric_columns = raw.select_dtypes(include=np.number).columns
    raw[numeric_columns] = raw[numeric_columns].replace([np.inf, -np.inf], np.nan)

    timestamps = raw["Timestamp"]
    raw = raw.drop(columns=["Timestamp"])

    raw["BinaryLabel"] = raw["Label"].apply(lambda x: 0 if x == "Benign" else 1)

    # to_csv/read_csv in preprocessing.py resets the index, so do the same
    raw = raw.reset_index(drop=True)
    timestamps = timestamps.reset_index(drop=True)

    return raw, timestamps


def check_alignment(rebuilt, df):
    features = df.drop(columns=["Label", "BinaryLabel"]).columns

    same_shape = rebuilt.shape == df.shape
    same_labels = rebuilt["Label"].equals(df["Label"])

    a = rebuilt[features].to_numpy(dtype=float)
    b = df[features].to_numpy(dtype=float)
    known = ~np.isnan(a)
    same_values = np.allclose(a[known], b[known])

    print(f"Rebuilt data matches cleaned file: shape={same_shape}, labels={same_labels}, values={same_values}")

    if not (same_shape and same_labels and same_values):
        raise ValueError("Rebuilt dataset does not line up with cic_cleaned.csv")


# ---------------------------------------------------------------- 4. imputation


def compare_imputation(X_full_median, X_raw, y, e_features):
    print("\n===== 4. IMPUTATION: FULL-DATASET MEDIAN VS TRAIN-ONLY MEDIAN =====")

    Xf_train, Xf_test, y_train, y_test = split_like_baseline(X_full_median, y)
    Xr_train, Xr_test, _, _ = split_like_baseline(X_raw, y)

    missing = Xr_train.isnull().sum() + Xr_test.isnull().sum()
    missing = missing[missing > 0]

    print("\nMissing values in the 100k sample (before imputation):")
    for col in missing.index:
        # The value preprocessing.py filled in is the full-dataset median
        filled_rows = Xr_train[col].isnull()
        full_median = Xf_train.loc[filled_rows, col].iloc[0] if filled_rows.any() else np.nan

        print(
            f"{col}: train={filled_rows.sum()}, "
            f"test={Xr_test[col].isnull().sum()}, "
            f"full-dataset median={full_median}, "
            f"train-only median={Xr_train[col].median()}"
        )

    rows = []

    for name, cols in [("A_all_78_features", list(X_full_median.columns)), ("E_behavior_focused", e_features)]:
        full_model = fit_pipeline(Xf_train[cols], y_train)
        train_model = fit_pipeline(Xr_train[cols], y_train, impute=True)

        rows.append({
            "experiment": name,
            "imputation": "full-dataset median (current)",
            **evaluate(full_model, Xf_test[cols], y_test),
        })
        rows.append({
            "experiment": name,
            "imputation": "train-only median",
            **evaluate(train_model, Xr_test[cols], y_test),
        })

    results = pd.DataFrame(rows)

    output = OUTPUT_DIR / "stage3_imputation_comparison.csv"
    results.to_csv(output, index=False)
    print(results.to_string(index=False))
    print(f"Saved: {output}")


# ---------------------------------------------------------------- 5. splitting


def parse_timestamps(timestamps):
    # The raw file uses a 12-hour clock without AM/PM: hours 01-05 appear,
    # 13-23 never do, and capture starts at 08:31. Hours below 8 are afternoon.
    times = pd.to_datetime(timestamps, format="%d/%m/%Y %H:%M:%S")
    afternoon = times.dt.hour < 8

    return times + pd.to_timedelta(afternoon.astype(int) * 12, unit="h")


def describe_time_ranges(times, labels):
    print("\n===== 5. TIME RANGES PER LABEL (after 12-hour correction) =====")

    summary = pd.DataFrame({"time": times, "Label": labels}).groupby("Label")["time"].agg(["min", "max", "count"])
    print(summary)

    return summary


def time_ordered_split(times, labels):
    # Within each original label, the earliest 80% of flows train, the latest 20% test
    test_mask = pd.Series(False, index=times.index)

    for label in labels.unique():
        label_times = times[labels == label].sort_values(kind="stable")
        cutoff = int(len(label_times) * 0.8)
        test_mask[label_times.index[cutoff:]] = True

    return ~test_mask, test_mask


def time_block_split(times):
    blocks = times.dt.floor(TIME_BLOCK)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(times, groups=blocks))

    train_mask = pd.Series(False, index=times.index)
    train_mask.iloc[train_idx] = True

    return train_mask, ~train_mask


def compare_splits(X, y, times, labels, e_features):
    print("\n===== 5. RANDOM SPLIT VS TIME-AWARE SPLITS =====")

    random_train = split_like_baseline(X, y)[0].index
    random_mask = pd.Series(X.index.isin(random_train), index=X.index)

    splits = {
        "random stratified (baseline)": (random_mask, ~random_mask),
        "time-ordered within each label": time_ordered_split(times, labels),
        f"time blocks ({TIME_BLOCK}) grouped": time_block_split(times),
    }

    rows = []

    for split_name, (train_mask, test_mask) in splits.items():
        # How many test rows share a 5-minute window with a training row of the same label
        blocks = times.dt.floor(TIME_BLOCK).astype(str) + "|" + labels
        shared = blocks[test_mask].isin(set(blocks[train_mask])).mean()

        for name, cols in [("A_all_78_features", list(X.columns)), ("E_behavior_focused", e_features)]:
            model = fit_pipeline(X.loc[train_mask, cols], y[train_mask])

            rows.append({
                "split": split_name,
                "experiment": name,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "test_attack_share": y[test_mask].mean(),
                "test_rows_sharing_time_block_with_train": shared,
                **evaluate(model, X.loc[test_mask, cols], y[test_mask]),
            })

    results = pd.DataFrame(rows)

    output = OUTPUT_DIR / "stage3_split_comparison.csv"
    results.to_csv(output, index=False)
    print(results.to_string(index=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 80)

    df = load_processed_data()
    X, y = prepare_data(df)

    dropped = set(FINGERPRINT_FEATURES + CONSTANT_FEATURES)
    e_features = [c for c in X.columns if c not in dropped]

    X_train, X_test, y_train, y_test = split_like_baseline(X, y)

    # Sanity check: this is the same split train_baseline makes
    _, baseline_X_test, _ = train_baseline(X, y)
    assert X_test.index.equals(baseline_X_test.index), "Split differs from baseline"
    print("\nSplit check: identical test rows to baseline.train_baseline")

    analyse_duplicates(df, X_train, X_test, y_train, y_test, e_features)
    evaluate_on_unseen_rows(X, y, e_features)

    ranking = rank_remaining_features(X_train, y_train, e_features)
    run_group_ablation(X, y, e_features, ranking)

    rebuilt, timestamps = rebuild_without_imputation()
    check_alignment(rebuilt, df)
    del df

    # Same sample as the baseline, but without imputation
    X_raw, y_raw = prepare_data(rebuilt)
    assert X_raw.index.equals(X.index) and y_raw.equals(y), "Sample differs from baseline"
    print("Sample check: identical rows to baseline sample")

    compare_imputation(X, X_raw, y, e_features)

    times = parse_timestamps(timestamps)
    describe_time_ranges(times, rebuilt["Label"])

    sample_times = times.loc[X.index]
    sample_labels = rebuilt.loc[X.index, "Label"]
    compare_splits(X, y, sample_times, sample_labels, e_features)

    print("\nStage 3 complete.")
