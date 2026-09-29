from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

# Reuse the baseline's sampling and training so every experiment
# uses the exact same 100,000 rows and the same train/test split
from baseline import load_processed_data, prepare_data, train_baseline


OUTPUT_DIR = Path("results/leakage_analysis")

# A value only counts as "pure" if it appears often enough to matter
MIN_VALUE_ROWS = 100
PURITY_THRESHOLD = 0.99

# Single-valued across the whole cleaned dataset (found in Stage 1)
CONSTANT_FEATURES = [
    "Bwd URG Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd Blk Rate Avg",
    "Bwd Byts/b Avg",
    "Bwd Pkts/b Avg",
    "Fwd Pkts/b Avg",
    "Fwd Byts/b Avg",
    "CWE Flag Count",
    "Fwd Blk Rate Avg",
]

# Port/protocol plus TCP handshake values that identify the attacker/server setup
FINGERPRINT_FEATURES = [
    "Dst Port",
    "Protocol",
    "Fwd Seg Size Min",
    "Init Fwd Win Byts",
    "Init Bwd Win Byts",
]

EXPERIMENTS = {
    "A_all_78_features": [],
    "B_drop_dst_port": ["Dst Port"],
    "C_drop_protocol": ["Protocol"],
    "D_drop_dst_port_and_protocol": ["Dst Port", "Protocol"],
    "E_behavior_focused": FINGERPRINT_FEATURES + CONSTANT_FEATURES,
}


def feature_group(name):
    if name in ("Dst Port", "Protocol"):
        return "port/protocol"
    if "/s" in name:
        return "flow-rate"
    if "IAT" in name or "Duration" in name or "Active" in name or "Idle" in name:
        return "timing"
    if "Flag" in name or "Cnt" in name:
        return "flags"
    if "Pkts" in name or "Byts" in name or "Pkt" in name or "Seg" in name:
        return "packet/size"
    if "Header" in name or "Win" in name:
        return "header/window"
    return "other"


def value_purity(values, y):
    # Share of rows whose value is (>=99%) exclusive to one class,
    # counting only values that occur at least MIN_VALUE_ROWS times
    table = pd.crosstab(values, y)
    table = table.reindex(columns=[0, 1], fill_value=0)

    totals = table.sum(axis=1)
    frequent = table[totals >= MIN_VALUE_ROWS]
    frequent_totals = totals[totals >= MIN_VALUE_ROWS]

    pure = (frequent.max(axis=1) / frequent_totals) >= PURITY_THRESHOLD

    pure_coverage = frequent_totals[pure].sum() / len(values)

    # The single most common attack value and how exclusive it is
    top_attack_value = table[1].idxmax()
    top_attack_share = table.loc[top_attack_value, 1] / table[1].sum()
    benign_share_same_value = table.loc[top_attack_value, 0] / table[0].sum()

    return pure_coverage, top_attack_value, top_attack_share, benign_share_same_value


def analyse_features(df):
    print("\n===== FEATURE ASSOCIATION WITH BinaryLabel =====")

    y = df["BinaryLabel"]
    features = df.drop(columns=["Label", "BinaryLabel"]).columns

    rows = []

    for col in features:
        x = df[col]

        n_unique = x.nunique()

        if n_unique > 1:
            correlation = x.corr(y)
            auc = roc_auc_score(y, x)
            single_feature_auc = max(auc, 1 - auc)
        else:
            correlation = np.nan
            single_feature_auc = 0.5

        pure_coverage, top_value, top_share, benign_share = value_purity(x, y)

        rows.append({
            "feature": col,
            "group": feature_group(col),
            "n_unique": n_unique,
            "benign_median": x[y == 0].median(),
            "attack_median": x[y == 1].median(),
            "benign_mean": x[y == 0].mean(),
            "attack_mean": x[y == 1].mean(),
            "pearson_corr": correlation,
            "single_feature_auc": single_feature_auc,
            "pure_value_coverage": pure_coverage,
            "top_attack_value": top_value,
            "top_attack_value_share_of_attacks": top_share,
            "top_attack_value_share_of_benign": benign_share,
        })

    results = pd.DataFrame(rows).sort_values(
        "single_feature_auc",
        ascending=False
    )

    output = OUTPUT_DIR / "feature_association.csv"
    results.to_csv(output, index=False)
    print(f"Saved: {output}")

    return results


def analyse_categorical(df, column):
    print(f"\n===== {column.upper()} BY CLASS =====")

    table = pd.crosstab(df[column], df["Label"])
    table["Benign_pct"] = table["Benign"] / table["Benign"].sum() * 100

    attack_labels = [c for c in df["Label"].unique() if c != "Benign"]
    attack_total = table[attack_labels].sum(axis=1)
    table["Attack"] = attack_total
    table["Attack_pct"] = attack_total / attack_total.sum() * 100

    table = table.sort_values("Attack", ascending=False)

    name = column.lower().replace(" ", "_")
    output = OUTPUT_DIR / f"{name}_by_class.csv"
    table.to_csv(output)
    print(f"Saved: {output}")

    return table


def summarise_ports(port_table):
    print("\n===== DST PORT 21 / 22 SUMMARY =====")

    for port in [21, 22]:
        if port in port_table.index:
            row = port_table.loc[port]
            print(
                f"Port {port}: "
                f"{row['Attack_pct']:.2f}% of attack flows, "
                f"{row['Benign_pct']:.4f}% of benign flows "
                f"({int(row['Benign']):,} benign rows)"
            )

    ports_21_22 = port_table.loc[port_table.index.isin([21, 22])]
    print(
        f"Ports 21+22 combined: "
        f"{ports_21_22['Attack_pct'].sum():.2f}% of attack flows, "
        f"{ports_21_22['Benign_pct'].sum():.4f}% of benign flows"
    )


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "false_positive_rate": fp / (fp + tn),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def run_experiments(df):
    X, y = prepare_data(df)

    rows = []

    for name, dropped in EXPERIMENTS.items():
        print(f"\n===== EXPERIMENT {name} =====")

        X_experiment = X.drop(columns=dropped)

        model, X_test, y_test = train_baseline(X_experiment, y)
        metrics = evaluate(model, X_test, y_test)

        rows.append({
            "experiment": name,
            "dropped_features": ", ".join(dropped) or "none",
            "n_features": X_experiment.shape[1],
            **metrics,
        })

    results = pd.DataFrame(rows)

    output = OUTPUT_DIR / "leakage_experiments.csv"
    results.to_csv(output, index=False)
    print(f"\nSaved: {output}")

    return results


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_processed_data()

    features = df.drop(columns=["Label", "BinaryLabel"]).columns
    print(f"\n===== {len(features)} INPUT FEATURES =====")
    for i, col in enumerate(features, start=1):
        print(f"{i:2d}. {col}")

    association = analyse_features(df)

    port_table = analyse_categorical(df, "Dst Port")
    summarise_ports(port_table)

    analyse_categorical(df, "Protocol")

    experiments = run_experiments(df)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    print("\n===== EXPERIMENT SUMMARY =====")
    print(experiments.to_string(index=False))
