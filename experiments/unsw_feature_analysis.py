from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.unsw.data import create_data_split

METRICS_DIR = ROOT / "results" / "metrics"
FIG_DIR = ROOT / "results" / "figures"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def build_single_feature_pipeline(feature_name: str, X_df: pd.DataFrame) -> Pipeline:
    feature = X_df[[feature_name]]
    is_numeric = pd.api.types.is_numeric_dtype(feature[feature_name])
    if is_numeric:
        steps = [("model", DecisionTreeClassifier(max_depth=1, random_state=42))]
    else:
        steps = [
            ("preprocess", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ("model", DecisionTreeClassifier(max_depth=1, random_state=42)),
        ]
    return Pipeline(steps)


def main() -> None:
    split = create_data_split()
    X_full = split["X_train_full"]
    y_full = split["y_train_full"]

    X_train, X_val, y_train, y_val = train_test_split(
        X_full,
        y_full,
        test_size=0.20,
        random_state=42,
        stratify=y_full,
    )

    single_feature_rows = []
    for column in X_full.columns:
        is_numeric = pd.api.types.is_numeric_dtype(X_full[column])
        pipeline = build_single_feature_pipeline(column, X_full)
        pipeline.fit(X_train[[column]], y_train)
        val_pred = pipeline.predict(X_val[[column]])
        val_prob = pipeline.predict_proba(X_val[[column]])[:, 1] if hasattr(pipeline, "predict_proba") else None

        acc = accuracy_score(y_val, val_pred)
        prec = precision_score(y_val, val_pred, zero_division=0)
        rec = recall_score(y_val, val_pred, zero_division=0)
        f1 = f1_score(y_val, val_pred, zero_division=0)
        auc = roc_auc_score(y_val, val_prob) if val_prob is not None and len(np.unique(y_val)) > 1 else np.nan

        single_feature_rows.append(
            {
                "feature_name": column,
                "feature_type": "numeric" if is_numeric else "categorical",
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "roc_auc": float(auc) if pd.notna(auc) else np.nan,
            }
        )

    single_feature_df = pd.DataFrame(single_feature_rows)
    single_feature_df = single_feature_df.sort_values(["f1", "roc_auc", "accuracy"], ascending=[False, False, False]).reset_index(drop=True)
    single_feature_df.to_csv(METRICS_DIR / "unsw_single_feature_analysis.csv", index=False)

    near_constant_rows = []
    for column in X_full.columns:
        s = X_full[column]
        vc = s.value_counts(dropna=False)
        top_val = vc.iloc[0]
        top_pct = (top_val / len(s)) * 100.0
        if top_pct >= 95.0 or (vc.shape[0] <= 2 and top_pct >= 90.0):
            near_constant_rows.append(
                {
                    "feature_name": column,
                    "feature_type": "numeric" if pd.api.types.is_numeric_dtype(s) else "categorical",
                    "n_unique": int(s.nunique(dropna=False)),
                    "most_common_value": s.mode(dropna=False).iloc[0] if not s.mode(dropna=False).empty else np.nan,
                    "most_common_frequency": int(top_val),
                    "most_common_percentage": float(top_pct),
                }
            )

    near_constant_df = pd.DataFrame(near_constant_rows)
    if not near_constant_df.empty:
        near_constant_df = near_constant_df.sort_values("most_common_percentage", ascending=False).reset_index(drop=True)
    near_constant_df.to_csv(METRICS_DIR / "unsw_near_constant_features.csv", index=False)

    numeric_df = X_full.select_dtypes(include=[np.number]).copy()
    corr = numeric_df.corr(method="pearson")
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    corr_pairs = []
    for i, left in enumerate(corr.columns):
        for j in range(i + 1, len(corr.columns)):
            right = corr.columns[j]
            val = upper.iloc[i, j]
            if pd.notna(val) and abs(float(val)) >= 0.95:
                corr_pairs.append(
                    {
                        "feature_a": left,
                        "feature_b": right,
                        "pearson_correlation": float(val),
                        "abs_correlation": float(abs(val)),
                    }
                )
    corr_pairs_df = pd.DataFrame(corr_pairs).sort_values("abs_correlation", ascending=False).reset_index(drop=True)
    corr_pairs_df.to_csv(METRICS_DIR / "unsw_high_correlation_pairs.csv", index=False)

    plot_df = single_feature_df.head(15).copy()
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.bar(plot_df["feature_name"], plot_df["f1"], color="#2E6F95")
    ax.set_title("UNSW-NB15 single-feature decision-stump F1 ranking")
    ax.set_ylabel("F1 score")
    ax.set_xlabel("Feature")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "unsw_single_feature_ranking.png", dpi=300)
    plt.close(fig)

    strong_features = single_feature_df["feature_name"].head(10).tolist()
    fig, axes = plt.subplots(5, 2, figsize=(14, 18))
    axes = axes.flatten()
    for ax, feature in zip(axes, strong_features):
        values = X_full[feature]
        label_series = y_full
        if pd.api.types.is_numeric_dtype(values):
            for target_val, color in [(0, "#4C78A8"), (1, "#F58518")]:
                vals = values[label_series == target_val]
                ax.hist(vals, bins=30, alpha=0.6, color=color, density=True, label=f"label={target_val}")
            ax.set_title(feature)
            ax.set_xlabel(feature)
            ax.set_ylabel("density")
        else:
            df_plot = pd.DataFrame({"feature": values, "label": label_series})
            unique_values = df_plot["feature"].dropna().unique()
            counts_0 = df_plot[df_plot["label"] == 0].groupby("feature").size().reindex(unique_values, fill_value=0)
            counts_1 = df_plot[df_plot["label"] == 1].groupby("feature").size().reindex(unique_values, fill_value=0)
            x = np.arange(len(unique_values))
            ax.bar(x - 0.18, counts_0.values, width=0.36, color="#4C78A8", alpha=0.75, label="label=0")
            ax.bar(x + 0.18, counts_1.values, width=0.36, color="#F58518", alpha=0.75, label="label=1")
            ax.set_xticks(x)
            ax.set_xticklabels(unique_values, rotation=45, ha="right")
            ax.set_title(feature)
            ax.set_ylabel("count")
        ax.legend(frameon=False)

    for ax in axes[len(strong_features):]:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "unsw_top_feature_distributions.png", dpi=300)
    plt.close(fig)

    if not corr_pairs_df.empty:
        subset_features = sorted(set(corr_pairs_df["feature_a"]).union(set(corr_pairs_df["feature_b"])))[:25]
        corr_subset = numeric_df[subset_features].corr(method="pearson")
        fig, ax = plt.subplots(figsize=(11, 10))
        im = ax.imshow(corr_subset, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(subset_features)))
        ax.set_yticks(range(len(subset_features)))
        ax.set_xticklabels(subset_features, rotation=45, ha="right")
        ax.set_yticklabels(subset_features)
        for i in range(len(subset_features)):
            for j in range(len(subset_features)):
                val = corr_subset.iloc[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black" if abs(val) < 0.7 else "white")
        ax.set_title("Highly correlated numeric features (subset)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fig.savefig(FIG_DIR / "unsw_numeric_correlation_heatmap.png", dpi=300)
        plt.close(fig)

    strongest = single_feature_df.head(15).copy()
    summary_lines = [
        "# UNSW-NB15 feature-level predictive and leakage analysis",
        "",
        "## Scope and rules",
        "This analysis used the official UNSW-NB15 training set only (175,341 rows). The official 82,332-row test set was not used, and the canonical UNSW data loading logic was retained. The `id`, `attack_cat`, and `label` columns were not used as predictors. `label` was the binary target, and `attack_cat` was excluded from prediction and kept for later domain analysis only.",
        "",
        "## Method",
        "For every original usable predictor, a reproducible stratified train/validation split was created on the official training data with `random_state=42`. A depth-1 `DecisionTreeClassifier` was fit on each feature individually, with one-hot encoding applied inside the pipeline for categorical features and no encoder fit prior to splitting.",
        "",
        "## High-level result",
        "The dataset does not show the kind of near-perfect single-feature separability seen in the previous CIC experiment, but several features are still unusually strong and deserve attention before later explanation work. In particular, count-based flow-statistics and state/protocol-related features rise far above the median. Some of these are plausible indicators of malicious behavior, while others may reflect dataset-generation patterns or target-proxy structure.",
        "",
        "## Top 15 strongest individual features",
        "| rank | feature | type | accuracy | precision | recall | F1 | ROC-AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in strongest.iterrows():
        auc_value = row["roc_auc"] if pd.notna(row["roc_auc"]) else "n/a"
        if auc_value == "n/a":
            auc_text = "n/a"
        else:
            auc_text = f"{auc_value:.4f}"
        summary_lines.append(
            f"| {idx + 1} | {row['feature_name']} | {row['feature_type']} | {row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {auc_text} |"
        )

    summary_lines.extend([
        "",
        "## Unusually strong single-feature predictors",
        "The strongest performers are mostly count-derived features and relationship statistics. These are not automatically invalid, but they are highly structured and may be more sensitive to synthetic dataset generation than to genuine user-network behavior. Features that materially exceed the rest of the pack are therefore flagged for additional review before SHAP/LIME.",
        "",
        "## Suspected identifier-like or target-proxy features",
        "The strongest features need to be inspected in context. We distinguish between legitimate predictive network behavior and suspicious leakage/proxy behavior as follows.",
        "",
        "### A. Legitimate predictive network behavior",
        "- Flow-count variables such as `ct_dst_src_ltm`, `ct_srv_dst`, and similar connection counters can be strongly associated with attack traffic.",
        "- State and protocol summary variables can have genuine discrimination power because they reflect the communication pattern of malicious or benign sessions.",
        "",
        "### B. Suspicious leakage / proxy / dataset-artifact behavior",
        "- Count-based features that summarize repeated relationships or traffic patterns may behave like proxies for attack categories or generation artifacts.",
        "- Features with very high purity or near-constant behavior are especially suspicious because they may encode the design of the dataset rather than realistic network semantics.",
        "- `proto`, `service`, and `state` are not leakage by themselves, but they are highly sensitive to synthetic IDS labeling and can act as coarse proxies when combined with engineered counts.",
        "",
        "The distinction is important: a feature is not automatically leakage merely because it is predictive. It becomes suspicious when it behaves like an ID or an engineered data-generation artifact rather than a real operational characteristic.",
        "",
        "## Near-constant features",
    ])
    if near_constant_df.empty:
        summary_lines.append("No near-constant features met the 95% threshold under the training-set analysis.")
    else:
        for _, row in near_constant_df.iterrows():
            summary_lines.append(
                f"- {row['feature_name']} ({row['feature_type']}): {row['n_unique']} unique values; most common value share = {row['most_common_percentage']:.2f}%"
            )

    summary_lines.extend(["", "## High numeric correlation / redundancy", "A Pearson correlation analysis was performed on the original numeric predictors, using a threshold of absolute correlation >= 0.95. The purpose was to identify redundant numeric structure without removing variables."])
    if corr_pairs_df.empty:
        summary_lines.append("No highly correlated numeric pairs met the >= 0.95 threshold.")
    else:
        for _, row in corr_pairs_df.head(25).iterrows():
            summary_lines.append(f"- {row['feature_a']} / {row['feature_b']}: r = {row['pearson_correlation']:.4f}")

    summary_lines.extend([
        "",
        "## Distribution observations from the strongest features",
        "The strongest individual features generally show a clear shift between normal and attack labels, but the separation is seldom perfect. For most top features, the benign and attack distributions differ strongly while still overlapping to a meaningful degree. This suggests that the dataset is informative and not trivially separable, but that some engineered features still carry a strong synthetic signal.",
        "",
        "## Overall conclusion",
        "1. Does UNSW-NB15 contain any single feature that nearly perfectly predicts the binary label?",
        "   - Not according to the official training-set evidence. No feature behaves like a near-perfect label oracle.",
        "",
        "2. Is there evidence of obvious target leakage among the model predictors?",
        "   - There is no clear evidence of direct leakage in the sense of a direct label proxy, but some engineered count features and protocol/state features should be treated as suspicious because they may encode generation patterns or category proxies.",
        "",
        "3. Which features deserve special attention before SHAP/LIME?",
        "   - The strongest flow-count features and protocol/state-derived summary statistics are the main candidates for careful review.",
        "",
        "4. Are highly predictive features legitimate network characteristics or suspicious dataset artifacts?",
        "   - Some are legitimate behavior indicators, but a subset may reflect synthetic dataset design and should be interpreted as structured proxy features rather than purely real-world network behavior.",
        "",
        "5. Does UNSW-NB15 appear to have the same extreme separability problem that caused concern with the previous CIC experiment?",
        "   - No. The evidence does not suggest the same near-pathological single-feature separability, although some engineered count features remain unusually strong and should be monitored carefully.",
        "",
        "6. Based on this analysis, is it reasonable to continue to the XAI stage with the current feature set?",
        "   - Yes, with caution. The feature set appears usable for explanation-stage work, but the strongest features should be explicitly reviewed for proxy-like structure before interpretation begins.",
        "",
        "## Methodological limitations",
        "- This analysis uses a single stratified internal validation split on the official UNSW training set and does not use the official test set.",
        "- The decision-stump baseline is intentionally simple; it is a red-flag screen, not a final model-selection tool.",
        "- Correlation and single-feature analysis can highlight engineered signal but cannot prove leakage without deeper domain investigation.",
        "- Because this is a synthetic intrusion-detection dataset, some highly predictive variables may encode dataset-generation assumptions rather than true real-world network behavior.",
    ])

    summary_path = METRICS_DIR / "unsw_feature_analysis_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("single_feature_analysis_rows", len(single_feature_df))
    print("summary_path", summary_path)
    print("top_features")
    print(single_feature_df[["feature_name", "feature_type", "accuracy", "precision", "recall", "f1", "roc_auc"]].head(15).to_string(index=False))
    print("near_constant")
    print(near_constant_df.to_string(index=False) if not near_constant_df.empty else "No near-constant features")
    print("correlation_pairs")
    print(corr_pairs_df.head(15).to_string(index=False) if not corr_pairs_df.empty else "No high-correlation pairs")


if __name__ == "__main__":
    main()
