from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SHAP_DIR = ROOT / "results" / "metrics" / "shap"
FIG_DIR = ROOT / "results" / "figures" / "shap"
SHAP_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

METADATA_COLUMNS = {
    "model",
    "validation_position",
    "original_training_row_index",
    "true_label",
    "predicted_label",
    "predicted_probability",
    "attack_cat",
    "seed",
    "base_value",
    "reconstructed_model_output",
    "output_kind",
    "output_label",
}


def load_saved_artifacts():
    rf_ids = pd.read_csv(SHAP_DIR / "random_forest_selected_sample_identifiers.csv")
    xgb_ids = pd.read_csv(SHAP_DIR / "xgboost_selected_sample_identifiers.csv")
    rf_shap = pd.read_csv(SHAP_DIR / "random_forest_selected_shap_values.csv")
    xgb_shap = pd.read_csv(SHAP_DIR / "xgboost_selected_shap_values.csv")
    return rf_ids, xgb_ids, rf_shap, xgb_shap


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in METADATA_COLUMNS]
    return [c for c in cols if str(c) and not str(c).startswith("Unnamed")]


def compute_feature_importance(shap_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    feature_cols = get_feature_columns(shap_df)
    importance = pd.Series(
        {feature: float(np.abs(shap_df[feature]).mean()) for feature in feature_cols},
        name="mean_abs_shap",
    ).sort_values(ascending=False)
    importance_df = importance.reset_index()
    importance_df.columns = ["feature", "mean_abs_shap"]
    importance_df["feature_type"] = [
        "one_hot_encoded" if feature.startswith("cat__") else "transformed_numeric" for feature in importance_df["feature"].tolist()
    ]
    importance_df["model"] = model_name
    importance_df["overall_rank"] = np.arange(1, len(importance_df) + 1)
    importance_df = importance_df[["model", "overall_rank", "feature", "feature_type", "mean_abs_shap"]]
    return importance_df


def compute_top_features(shap_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    full_importance = compute_feature_importance(shap_df, model_name)
    top = full_importance.head(10).copy()
    top = top.rename(columns={"overall_rank": "rank"})
    top = top[["model", "rank", "feature", "feature_type", "mean_abs_shap"]]
    return top


def save_top10_csv(top_rf: pd.DataFrame, top_xgb: pd.DataFrame):
    top10 = pd.concat([top_rf, top_xgb], ignore_index=True)
    top10_path = SHAP_DIR / "feature_importance_top10.csv"
    top10.to_csv(top10_path, index=False)
    return top10_path


def validate_shared_sample_ids(rf_ids: pd.DataFrame, xgb_ids: pd.DataFrame):
    rf_set = set(rf_ids["validation_position"].astype(int).tolist())
    xgb_set = set(xgb_ids["validation_position"].astype(int).tolist())
    shared = sorted(rf_set & xgb_set)
    rf_only = sorted(rf_set - xgb_set)
    xgb_only = sorted(xgb_set - rf_set)

    overlap_summary = {
        "shared_validation_positions": shared,
        "shared_count": len(shared),
        "random_forest_only_positions": rf_only,
        "xgboost_only_positions": xgb_only,
        "random_forest_total": len(rf_set),
        "xgboost_total": len(xgb_set),
    }
    shared_path = SHAP_DIR / "shared_validation_sample_identifiers.csv"
    pd.DataFrame({"validation_position": shared}).to_csv(shared_path, index=False)
    return overlap_summary, shared_path


def compute_top10_overlap(top_rf: pd.DataFrame, top_xgb: pd.DataFrame):
    rf_set = set(top_rf["feature"].tolist())
    xgb_set = set(top_xgb["feature"].tolist())
    overlap = rf_set & xgb_set
    union = rf_set | xgb_set
    overlap_size = len(overlap)
    jaccard = overlap_size / len(union) if union else 0.0

    shared_features = sorted(overlap)
    rf_rank_map = {feature: int(row["rank"]) for _, row in top_rf.iterrows() for feature in [row["feature"]]}
    xgb_rank_map = {feature: int(row["rank"]) for _, row in top_xgb.iterrows() for feature in [row["feature"]]}

    if shared_features:
        rf_ranks = pd.Series([rf_rank_map[feat] for feat in shared_features], index=shared_features)
        xgb_ranks = pd.Series([xgb_rank_map[feat] for feat in shared_features], index=shared_features)
        spearman_rho = float(rf_ranks.corr(xgb_ranks, method="pearson"))
    else:
        spearman_rho = None

    summary = {
        "top10_overlap_count": overlap_size,
        "top10_jaccard_similarity": float(jaccard),
        "shared_features": shared_features,
        "random_forest_only_top10_features": sorted(rf_set - xgb_set),
        "xgboost_only_top10_features": sorted(xgb_set - rf_set),
        "spearman_rho_on_shared_top10_features": spearman_rho,
        "rankings_for_shared_features": {
            feature: {"random_forest_rank": int(rf_rank_map[feature]), "xgboost_rank": int(xgb_rank_map[feature])}
            for feature in shared_features
        },
    }
    return summary


def make_rank_compare_plot(top_rf: pd.DataFrame, top_xgb: pd.DataFrame):
    union = pd.Index(top_rf["feature"].tolist() + top_xgb["feature"].tolist()).unique().tolist()
    rank_table = []
    for feature in union:
        rf_rank = top_rf.loc[top_rf["feature"] == feature, "rank"].values[0] if feature in top_rf["feature"].values else np.nan
        xgb_rank = top_xgb.loc[top_xgb["feature"] == feature, "rank"].values[0] if feature in top_xgb["feature"].values else np.nan
        rank_table.append({"feature": feature, "random_forest_rank": rf_rank, "xgboost_rank": xgb_rank})
    rank_df = pd.DataFrame(rank_table).set_index("feature")
    rank_df = rank_df[["random_forest_rank", "xgboost_rank"]]
    rank_df.columns = ["Random Forest", "XGBoost"]

    matrix = rank_df.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap("viridis_r")
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap, vmin=1, vmax=10)
    fig.colorbar(im, ax=ax, label="rank within model top-10 (1 = highest)")

    ax.set_xticks(range(rank_df.shape[1]))
    ax.set_xticklabels(rank_df.columns)
    ax.set_yticks(range(rank_df.shape[0]))
    ax.set_yticklabels(rank_df.index)
    ax.set_title(
        "Top-10 feature ranking comparison\nNote: SHAP importance units differ across models; only rankings and membership are compared."
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Feature")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isnan(matrix[i, j]):
                continue
            txt_color = "white" if matrix[i, j] <= 5 else "black"
            ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center", color=txt_color)

    fig.tight_layout()
    out_path = FIG_DIR / "top10_rank_comparison.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def build_waterfall_summary(shap_df: pd.DataFrame, model_name: str, waterfall_positions: list[int]) -> dict:
    feature_cols = get_feature_columns(shap_df)
    rows = []
    for pos in waterfall_positions:
        sample = shap_df[shap_df["validation_position"] == pos]
        if sample.empty:
            raise ValueError(f"No SHAP row for {model_name} with validation_position={pos}")
        row = sample.iloc[0]
        contributions = row[feature_cols].copy()
        positive = contributions[contributions > 0].sort_values(ascending=False).head(5)
        negative = contributions[contributions < 0].sort_values(ascending=True).head(5)
        rows.append(
            {
                "validation_position": int(pos),
                "base_value": float(row["base_value"]),
                "reconstructed_model_output": float(row["reconstructed_model_output"]),
                "predicted_probability": float(row["predicted_probability"]),
                "positive_contributors": {str(k): float(v) for k, v in positive.items()},
                "negative_contributors": {str(k): float(v) for k, v in negative.items()},
                "net_feature_contribution": float(contributions.sum()),
            }
        )
    return {"model": model_name, "waterfall_positions": waterfall_positions, "samples": rows}


def build_research_summary(
    rf_importance: pd.DataFrame,
    xgb_importance: pd.DataFrame,
    overlap_summary: dict,
    top10_summary: dict,
):
    shared_count = overlap_summary["shared_count"]
    rf_top = rf_importance.head(10)
    xgb_top = xgb_importance.head(10)
    rf_top_names = list(rf_top["feature"])
    xgb_top_names = list(xgb_importance["feature"])
    summary = f"""We compared the saved SHAP results for a single canonical shared validation subset. Both models used the same 20 correctly classified malicious validation positions selected from the common attack pool: Random Forest and XGBoost each used {shared_count} shared validation samples, and the comparison below therefore reflects one aligned sample set rather than two different model-specific subsets.

We first computed mean absolute SHAP importance for each transformed feature separately within each model. We did not compare raw SHAP magnitudes across models because the output scales differ: Random Forest SHAP values explain attack-class probability, while XGBoost SHAP values explain raw margin/logit. Instead, we compared membership and ranking of the top 10 features.

The top 10 features for Random Forest were: {', '.join(rf_top_names)}.
The top 10 features for XGBoost were: {', '.join(xgb_top_names)}.
The top-10 overlap was {top10_summary['top10_overlap_count']} and the Jaccard similarity was {top10_summary['top10_jaccard_similarity']:.3f}. The shared top-10 features were {', '.join(top10_summary['shared_features']) if top10_summary['shared_features'] else 'none'}, while the missing features were {', '.join(top10_summary['random_forest_only_top10_features'] + top10_summary['xgboost_only_top10_features']) if (top10_summary['random_forest_only_top10_features'] or top10_summary['xgboost_only_top10_features']) else 'none'}.

For the shared feature set, the rank-based comparison (Spearman correlation on ranks) was {top10_summary['spearman_rho_on_shared_top10_features']:.3f} when both models ranked the same features. We treated missing features as absent from the rank comparison and did not impute them; this keeps the comparison honest and avoids comparing features that were not present in both top-10 sets. The fact that the overlap and rank correlation are modest suggests that the two models agree on a few recurrent features but still distribute importance differently across transformed features.

We also inspected the saved local waterfall explanations. The analysis was based on the actual SHAP contributions in the saved CSV files, not on model retraining or a new explanation pass. For the shared malicious validation subset, the main positive contributors remained the key transformed features for each attack decision, while several transformed features pushed the prediction downward; XGBoost showed a similar pattern but with different feature emphasis in the same malicious validation subset. In both models, the final decision remained clearly on the attack side, while the contribution patterns showed that a small subset of transformed features dominated each local explanation.

Importantly, we kept original UNSW-NB15 features and one-hot encoded features separate. Features such as cat__state_INT and cat__service_dns are encoded categorical indicators and were not merged into broader parent categories without an explicit grouping method. We therefore compared them at the transformed-feature level and kept one-hot features distinct from numeric transformed features.

This matters for our future explanation-spoofing experiments because a model may preserve the same attack label while concentrating importance on different transformed features or different feature families. That difference can affect which explanations are easiest to spoof and which feature-level manipulations are most likely to fool a downstream explanation method.
"""
    return summary


def main():
    rf_ids, xgb_ids, rf_shap, xgb_shap = load_saved_artifacts()

    overlap_summary, _ = validate_shared_sample_ids(rf_ids, xgb_ids)
    if overlap_summary["shared_count"] == 0:
        raise ValueError("The models do not share any validation identifiers; comparison is impossible without a shared sample set.")

    rf_full_importance = compute_feature_importance(rf_shap, "random_forest")
    xgb_full_importance = compute_feature_importance(xgb_shap, "xgboost")
    feature_importance_all = pd.concat([rf_full_importance, xgb_full_importance], ignore_index=True)
    feature_importance_all.to_csv(SHAP_DIR / "feature_importance_all.csv", index=False)

    rf_top = compute_top_features(rf_shap, "random_forest")
    xgb_top = compute_top_features(xgb_shap, "xgboost")

    save_top10_csv(rf_top, xgb_top)
    top10_summary = compute_top10_overlap(rf_top, xgb_top)

    overlap_path = SHAP_DIR / "feature_overlap_summary.json"
    overlap_path.write_text(json.dumps({"sample_overlap": overlap_summary, "top10_overlap": top10_summary}, indent=2), encoding="utf-8")

    rank_plot_path = make_rank_compare_plot(rf_top, xgb_top)

    shared_positions = overlap_summary["shared_validation_positions"]
    waterfall_rf = build_waterfall_summary(rf_shap, "random_forest", shared_positions[:3])
    waterfall_xgb = build_waterfall_summary(xgb_shap, "xgboost", shared_positions[:3])
    waterfall_summary = {"random_forest": waterfall_rf, "xgboost": waterfall_xgb}
    (SHAP_DIR / "waterfall_contribution_summary.json").write_text(json.dumps(waterfall_summary, indent=2), encoding="utf-8")

    rf_top_path = SHAP_DIR / "random_forest_top10_features.csv"
    xgb_top_path = SHAP_DIR / "xgboost_top10_features.csv"
    rf_top.to_csv(rf_top_path, index=False)
    xgb_top.to_csv(xgb_top_path, index=False)

    research_summary = build_research_summary(rf_top, xgb_top, overlap_summary, top10_summary)
    research_path = SHAP_DIR / "research_summary.md"
    research_path.write_text(research_summary, encoding="utf-8")

    print("Shared validation sample IDs:", overlap_summary["shared_validation_positions"])
    print("Shared count:", overlap_summary["shared_count"])
    print("Top-10 overlap:", top10_summary["top10_overlap_count"])
    print("Top-10 Jaccard:", top10_summary["top10_jaccard_similarity"])
    print("Spearman rho on shared features:", top10_summary["spearman_rho_on_shared_top10_features"])
    print("Rank comparison plot saved to:", rank_plot_path)
    print("Summary files saved to:", SHAP_DIR)


if __name__ == "__main__":
    main()
