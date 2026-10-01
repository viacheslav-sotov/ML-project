from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from unsw_nb15_preprocessing import (
    build_preprocessor,
    create_binary_target,
    load_and_validate_official_data,
)


RANDOM_STATE = 42

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "unsw_nb15"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURE_DIR = RESULTS_DIR / "figures"

DROP_COLUMNS = ["id", "attack_cat", "label"]

# Expected sizes from unsw_nb15_preprocessing.py, used to confirm we reproduce his split exactly.
EXPECTED_SUBTRAIN_ROWS = 140272
EXPECTED_VAL_ROWS = 35069
EXPECTED_TEST_ROWS = 82332


def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def create_directories() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_splits() -> dict:
    """Recreate Viacheslav's split: official training -> 80/20 stratified sub-train/validation.

    The official test set is returned separately and must only be used by the final test script.
    """
    train_df, test_df = load_and_validate_official_data()

    X_train_full = train_df.drop(columns=DROP_COLUMNS, errors="ignore")
    y_train_full = create_binary_target(train_df)

    X_test = test_df.drop(columns=DROP_COLUMNS, errors="ignore")
    y_test = create_binary_target(test_df)

    # Same parameters as unsw_nb15_preprocessing.main()
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    assert len(X_subtrain) == EXPECTED_SUBTRAIN_ROWS, len(X_subtrain)
    assert len(X_val) == EXPECTED_VAL_ROWS, len(X_val)
    assert len(X_test) == EXPECTED_TEST_ROWS, len(X_test)
    assert "attack_cat" not in X_subtrain.columns and "label" not in X_subtrain.columns

    return {
        "X_train_full": X_train_full,
        "y_train_full": y_train_full,
        "X_subtrain": X_subtrain,
        "y_subtrain": y_subtrain,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def to_dense_float32(X):
    # The preprocessor outputs a sparse matrix. HistGradientBoosting needs dense input, and for
    # XGBoost dense input keeps real zeros as zeros instead of treating them as missing values.
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def build_model_pipeline(X_reference: pd.DataFrame, model) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor(X_reference)),
        ("dense", FunctionTransformer(to_dense_float32, accept_sparse=True)),
        ("model", model),
    ])


def fit_timed(pipeline: Pipeline, X, y) -> float:
    start = time.perf_counter()
    pipeline.fit(X, y)
    return time.perf_counter() - start


def evaluate_model(pipeline: Pipeline, X, y, model_name: str, split_name: str) -> tuple[dict, np.ndarray, str]:
    start = time.perf_counter()
    y_prob = pipeline.predict_proba(X)[:, 1]
    predict_time = time.perf_counter() - start

    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model": model_name,
        "split": split_name,
        "rows": len(y),
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_prob),
        "false_positive_rate": fp / (fp + tn),
        "false_negative_rate": fn / (fn + tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predict_time_s": predict_time,
    }

    report = classification_report(y, y_pred, target_names=["Normal", "Attack"], digits=4)

    return metrics, cm, report


def print_metrics(metrics: dict) -> None:
    print(f"\n{metrics['model']} on {metrics['split']} ({metrics['rows']:,} rows)")
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate", "false_negative_rate"]:
        print(f"  {key:<20} {metrics[key]:.4f}")
    if "train_time_s" in metrics:
        print(f"  {'train_time_s':<20} {metrics['train_time_s']:.2f}")
    print(f"  {'predict_time_s':<20} {metrics['predict_time_s']:.2f}")
    print(f"  confusion matrix [[tn fp] [fn tp]]: [[{metrics['tn']} {metrics['fp']}] [{metrics['fn']} {metrics['tp']}]]")


def save_confusion_matrix(cm: np.ndarray, title: str, filename: str) -> Path:
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Attack"])

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)

    plt.tight_layout()
    output = FIGURE_DIR / filename
    plt.savefig(output, dpi=300)
    plt.close(fig)

    print(f"Saved: {output}")
    return output


def save_text(text: str, filename: str) -> Path:
    output = METRICS_DIR / filename
    output.write_text(text, encoding="utf-8")
    print(f"Saved: {output}")
    return output


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    output = METRICS_DIR / filename
    df.to_csv(output, index=False)
    print(f"Saved: {output}")
    return output
