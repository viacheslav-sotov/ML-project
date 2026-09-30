from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.unsw.data import OFFICIAL_TRAIN_ROWS, load_official_data, prepare_features_target
from src.unsw.models import get_initial_models

METRICS_DIR = ROOT / "results" / "metrics"
FIGURE_DIR = ROOT / "results" / "figures"


def ensure_directories() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def build_pipeline(model_name: str, X_train: pd.DataFrame) -> Pipeline:
    model = get_initial_models()[model_name]
    from src.unsw.data import build_preprocessor

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", clone(model)),
        ]
    )


def compute_fold_metrics(model_name: str, y_true: pd.Series, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def run_cross_validation() -> tuple[pd.DataFrame, pd.DataFrame]:
    official_train_df, _ = load_official_data()
    X_train_full, y_train_full = prepare_features_target(official_train_df)

    if len(official_train_df) != OFFICIAL_TRAIN_ROWS:
        raise ValueError(f"Expected official training rows={OFFICIAL_TRAIN_ROWS}, found {len(official_train_df)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for model_name in ["random_forest", "extra_trees"]:
        fold_metrics: list[dict[str, float]] = []
        for fold_index, (train_idx, valid_idx) in enumerate(cv.split(X_train_full, y_train_full), start=1):
            X_train, X_valid = X_train_full.iloc[train_idx], X_train_full.iloc[valid_idx]
            y_train, y_valid = y_train_full.iloc[train_idx], y_train_full.iloc[valid_idx]

            pipeline = build_pipeline(model_name, X_train)
            pipeline.fit(X_train, y_train)
            y_valid_prob = pipeline.predict_proba(X_valid)[:, 1]
            metrics = compute_fold_metrics(model_name, y_valid, y_valid_prob)
            metrics["model"] = model_name
            metrics["fold"] = fold_index
            fold_rows.append(metrics)
            fold_metrics.append(metrics)

            print(f"{model_name} fold {fold_index}: {metrics}")

        fold_df = pd.DataFrame(fold_metrics)
        summary = {
            "model": model_name,
            "accuracy_mean": fold_df["accuracy"].mean(),
            "accuracy_std": fold_df["accuracy"].std(ddof=0),
            "precision_mean": fold_df["precision"].mean(),
            "precision_std": fold_df["precision"].std(ddof=0),
            "recall_mean": fold_df["recall"].mean(),
            "recall_std": fold_df["recall"].std(ddof=0),
            "f1_mean": fold_df["f1"].mean(),
            "f1_std": fold_df["f1"].std(ddof=0),
            "roc_auc_mean": fold_df["roc_auc"].mean(),
            "roc_auc_std": fold_df["roc_auc"].std(ddof=0),
        }
        summary_rows.append(summary)

    per_fold_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(summary_rows)

    return per_fold_df, summary_df


def save_fold_results(per_fold_df: pd.DataFrame) -> None:
    output_path = METRICS_DIR / "unsw_cross_validation_folds.csv"
    per_fold_df.to_csv(output_path, index=False)
    print(f"Saved fold-level results to {output_path}")


def save_summary_results(summary_df: pd.DataFrame) -> None:
    output_path = METRICS_DIR / "unsw_cross_validation_results.csv"
    summary_df.to_csv(output_path, index=False)
    print(f"Saved cross-validation summary to {output_path}")


def save_comparison_figure(summary_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(summary_df))
    width = 0.35

    ax.bar(
        x - width / 2,
        summary_df["f1_mean"],
        width=width,
        yerr=summary_df["f1_std"],
        capsize=5,
        label="Mean F1",
        color="steelblue",
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        summary_df["roc_auc_mean"],
        width=width,
        yerr=summary_df["roc_auc_std"],
        capsize=5,
        label="Mean ROC-AUC",
        color="darkorange",
        alpha=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["model"].tolist())
    ax.set_ylabel("Score")
    ax.set_title("UNSW-NB15 5-fold cross-validation comparison")
    ax.legend()
    ax.set_ylim(0.0, 1.1)
    plt.tight_layout()

    output_path = FIGURE_DIR / "unsw_cross_validation_comparison.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved comparison figure to {output_path}")


def main() -> None:
    ensure_directories()

    per_fold_df, summary_df = run_cross_validation()
    save_fold_results(per_fold_df)
    save_summary_results(summary_df)
    save_comparison_figure(summary_df)

    print("\nCross-validation summary (mean ± std):")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
