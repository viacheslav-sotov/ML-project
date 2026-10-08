from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.unsw.data import OFFICIAL_TRAIN_ROWS, build_preprocessor, create_data_split, load_official_data, prepare_features_target
from src.unsw.evaluation import evaluate_model
from src.unsw.models import get_initial_models

METRICS_DIR = ROOT / "results" / "metrics"
FIGURE_DIR = ROOT / "results" / "figures"


def ensure_output_dirs() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def build_xgboost_pipeline(X_train: pd.DataFrame) -> Pipeline:
    model = get_initial_models()["xgboost"]
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", clone(model)),
        ]
    )


def save_validation_confusion_matrix(cm: list[list[int]]) -> None:
    cm_array = np.array(cm)
    display = ConfusionMatrixDisplay(confusion_matrix=cm_array, display_labels=["Benign", "Attack"])
    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, values_format="d")
    ax.set_title("XGBoost validation confusion matrix")
    plt.tight_layout()
    output_path = FIGURE_DIR / "unsw_xgboost_validation_confusion_matrix.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved validation confusion matrix to {output_path}")


def save_xgboost_validation_metrics(metrics: dict[str, object]) -> None:
    output_path = METRICS_DIR / "unsw_xgboost_validation_metrics.csv"
    pd.DataFrame([metrics]).to_csv(output_path, index=False)
    print(f"Saved validation metrics to {output_path}")


def save_comparison_with_baselines(metrics: dict[str, object]) -> None:
    baseline_path = METRICS_DIR / "unsw_baseline_model_comparison.csv"
    comparison_path = METRICS_DIR / "unsw_baseline_model_comparison_with_xgboost.csv"

    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)
    else:
        baseline_df = pd.DataFrame()

    xgb_df = pd.DataFrame([metrics])
    combined_df = pd.concat([baseline_df, xgb_df], ignore_index=True) if not baseline_df.empty else xgb_df
    combined_df.to_csv(comparison_path, index=False)
    print(f"Saved comparison dataset with XGBoost to {comparison_path}")

    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(combined_df))
    width = 0.75 / len(metric_names)

    for idx, metric in enumerate(metric_names):
        values = combined_df[metric].astype(float).to_numpy()
        offset = (idx - (len(metric_names) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(combined_df["model"], rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("UNSW-NB15 model comparison including XGBoost")
    ax.legend(loc="best")
    ax.set_ylim(0, 1.1)
    plt.tight_layout()

    plot_path = FIGURE_DIR / "unsw_baseline_model_comparison_with_xgboost.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved comparison figure with XGBoost to {plot_path}")


def run_validation() -> dict[str, object]:
    split = create_data_split()
    X_subtrain = split["X_subtrain"]
    X_val = split["X_val"]
    y_subtrain = split["y_subtrain"]
    y_val = split["y_val"]

    pipeline = build_xgboost_pipeline(X_subtrain)

    start_fit = time.perf_counter()
    pipeline.fit(X_subtrain, y_subtrain)
    training_time = time.perf_counter() - start_fit

    start_predict = time.perf_counter()
    val_prob = pipeline.predict_proba(X_val)[:, 1]
    prediction_time = time.perf_counter() - start_predict

    metrics = evaluate_model(
        pipeline,
        X_val,
        y_val,
        training_time=training_time,
        prediction_time=prediction_time,
        y_prob=val_prob,
    )
    metrics["model"] = "xgboost"

    print("\nXGBoost validation metrics:")
    print(metrics)

    save_validation_confusion_matrix(metrics["confusion_matrix"])
    save_xgboost_validation_metrics(metrics)
    save_comparison_with_baselines(metrics)

    return metrics


def run_cross_validation() -> tuple[pd.DataFrame, pd.DataFrame]:
    official_train_df, _ = load_official_data()
    X_train_full, y_train_full = prepare_features_target(official_train_df)

    if len(official_train_df) != OFFICIAL_TRAIN_ROWS:
        raise ValueError(f"Expected official training rows={OFFICIAL_TRAIN_ROWS}, found {len(official_train_df)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_rows: list[dict[str, object]] = []

    for fold_index, (train_idx, valid_idx) in enumerate(cv.split(X_train_full, y_train_full), start=1):
        X_train, X_valid = X_train_full.iloc[train_idx], X_train_full.iloc[valid_idx]
        y_train, y_valid = y_train_full.iloc[train_idx], y_train_full.iloc[valid_idx]

        pipeline = build_xgboost_pipeline(X_train)
        pipeline.fit(X_train, y_train)
        y_valid_prob = pipeline.predict_proba(X_valid)[:, 1]

        metrics = evaluate_model(
            pipeline,
            X_valid,
            y_valid,
            y_prob=y_valid_prob,
        )
        metrics["model"] = "xgboost"
        metrics["fold"] = fold_index

        fold_rows.append(
            {
                "model": "xgboost",
                "fold": fold_index,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
            }
        )

        print(f"XGBoost fold {fold_index}: accuracy={metrics['accuracy']:.6f}, precision={metrics['precision']:.6f}, recall={metrics['recall']:.6f}, f1={metrics['f1']:.6f}, roc_auc={metrics['roc_auc']:.6f}")

    per_fold_df = pd.DataFrame(fold_rows)
    summary_rows: list[dict[str, float | str]] = []
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        summary_rows.append(
            {
                "model": "xgboost",
                "metric": metric,
                "mean": float(per_fold_df[metric].mean()),
                "std": float(per_fold_df[metric].std(ddof=0)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    per_fold_output = METRICS_DIR / "unsw_xgboost_cv_folds.csv"
    summary_output = METRICS_DIR / "unsw_xgboost_cv_summary.csv"
    per_fold_df.to_csv(per_fold_output, index=False)
    summary_df.to_csv(summary_output, index=False)

    print(f"Saved fold-level results to {per_fold_output}")
    print(f"Saved CV summary to {summary_output}")

    return per_fold_df, summary_df


def save_cv_figure(summary_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(metrics))
    means = [summary_df.loc[summary_df["metric"] == metric, "mean"].iat[0] for metric in metrics]
    stds = [summary_df.loc[summary_df["metric"] == metric, "std"].iat[0] for metric in metrics]

    ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("XGBoost 5-fold stratified CV summary")
    ax.set_ylim(0.0, 1.1)
    plt.tight_layout()

    output_path = FIGURE_DIR / "unsw_xgboost_cv_summary.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved XGBoost CV summary figure to {output_path}")


def main() -> None:
    ensure_output_dirs()

    validation_metrics = run_validation()
    per_fold_df, summary_df = run_cross_validation()
    save_cv_figure(summary_df)

    print("\nAll 5 XGBoost CV folds:")
    print(per_fold_df.to_string(index=False))
    print("\nXGBoost CV mean ± std:")
    print(summary_df.to_string(index=False))

    print("\nXGBoost experiment complete.")
    print(f"Validation metrics: {validation_metrics}")


if __name__ == "__main__":
    main()
