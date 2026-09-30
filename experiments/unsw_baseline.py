from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.unsw.data import build_preprocessor, create_data_split, load_official_data
from src.unsw.evaluation import evaluate_model
from src.unsw.models import get_initial_models

FIGURE_DIR = ROOT / "results" / "figures"
METRICS_DIR = ROOT / "results" / "metrics"


def ensure_output_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def print_verification(split: dict[str, object]) -> None:
    official_train_df = split["official_train_df"]
    official_test_df = split["official_test_df"]
    X_train_full = split["X_train_full"]
    y_train_full = split["y_train_full"]
    X_subtrain = split["X_subtrain"]
    X_val = split["X_val"]

    print("===== UNSW-NB15 baseline verification =====")
    print(f"official training rows = {len(official_train_df)}")
    print(f"official final test rows = {len(official_test_df)}")
    print(f"sub-training rows = {len(X_subtrain)}")
    print(f"validation rows = {len(X_val)}")
    print(f"42 original X features before encoding = {X_train_full.shape[1]}")
    print(f"id in X = {'id' in X_train_full.columns}")
    print(f"attack_cat in X = {'attack_cat' in X_train_full.columns}")
    print(f"label in X = {'label' in X_train_full.columns}")
    print(f"official test set loaded = {len(official_test_df)} rows")
    print("official test set is loaded but not used for model evaluation")
    print(f"label distribution in training set = {y_train_full.value_counts().sort_index().to_dict()}")
    print("========================================")


def save_model_metrics(results: list[dict[str, object]]) -> None:
    metrics_df = pd.DataFrame(results)
    output_path = METRICS_DIR / "unsw_baseline_model_comparison.csv"
    metrics_df.to_csv(output_path, index=False)
    print(f"Saved model metrics to {output_path}")


def save_confusion_matrix(cm: list[list[int]], model_name: str) -> None:
    cm_array = np.array(cm)
    display = ConfusionMatrixDisplay(confusion_matrix=cm_array, display_labels=["Benign", "Attack"])
    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, values_format="d")
    ax.set_title(f"{model_name} - validation confusion matrix")
    plt.tight_layout()
    output_path = FIGURE_DIR / f"{model_name}_validation_confusion_matrix.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved confusion matrix to {output_path}")


def save_model_comparison_figure(results: list[dict[str, object]]) -> None:
    comparison_df = pd.DataFrame(results)
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(comparison_df))
    width = 0.75 / len(metric_names)

    for idx, metric in enumerate(metric_names):
        offset = (idx - (len(metric_names) - 1) / 2) * width
        ax.bar(x + offset, comparison_df[metric].astype(float), width=width, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["model"], rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("UNSW-NB15 baseline model comparison")
    ax.legend(loc="best")
    ax.set_ylim(0, 1.1)
    plt.tight_layout()

    output_path = FIGURE_DIR / "unsw_baseline_model_comparison.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved model comparison figure to {output_path}")


def build_logistic_regression_pipeline(X_train: pd.DataFrame) -> Pipeline:
    categorical_cols = ["proto", "service", "state"]
    numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
            (
                "num",
                StandardScaler(),
                numeric_cols,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=3000, random_state=42)),
        ]
    )


def main() -> None:
    ensure_output_dirs()

    split = create_data_split()
    print_verification(split)

    X_subtrain = split["X_subtrain"]
    X_val = split["X_val"]
    y_subtrain = split["y_subtrain"]
    y_val = split["y_val"]

    results: list[dict[str, object]] = []

    model_names = list(get_initial_models().keys())
    for model_name in model_names:
        if model_name == "logistic_regression":
            pipeline = build_logistic_regression_pipeline(X_subtrain)
        else:
            estimator = get_initial_models()[model_name]
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor(X_subtrain)),
                    ("model", estimator),
                ]
            )

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
        metrics["model"] = model_name
        results.append(metrics)

        print(f"\n{model_name}: {metrics}")
        save_confusion_matrix(metrics["confusion_matrix"], model_name)

    save_model_metrics(results)
    save_model_comparison_figure(results)

    print("\nUNSW baseline experiment complete.")


if __name__ == "__main__":
    main()
