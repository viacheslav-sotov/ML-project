from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.unsw.data import OFFICIAL_TRAIN_ROWS, build_preprocessor, load_official_data, prepare_features_target
from src.unsw.models import get_initial_models

METRICS_DIR = ROOT / "results" / "metrics"


def ensure_directories() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def build_search_pipeline(model_name: str, X_train: pd.DataFrame) -> Pipeline:
    model = get_initial_models()[model_name]
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", model),
        ]
    )


def get_param_distributions(model_name: str) -> dict[str, list]:
    if model_name == "random_forest":
        return {
            "model__n_estimators": [100, 200, 300, 500],
            "model__max_depth": [None, 10, 20, 30, 40],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
            "model__class_weight": [None, "balanced"],
        }

    if model_name == "extra_trees":
        return {
            "model__n_estimators": [100, 200, 300, 500],
            "model__max_depth": [None, 10, 20, 30, 40],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
            "model__class_weight": [None, "balanced"],
        }

    raise ValueError(f"Unsupported model for randomized search: {model_name}")


def tune_model(model_name: str) -> tuple[pd.DataFrame, dict[str, object]]:
    official_train_df, _ = load_official_data()
    X_train, y_train = prepare_features_target(official_train_df)

    if len(official_train_df) != OFFICIAL_TRAIN_ROWS:
        raise ValueError(f"Expected official training rows={OFFICIAL_TRAIN_ROWS}, found {len(official_train_df)}")

    pipeline = build_search_pipeline(model_name, X_train)
    param_distributions = get_param_distributions(model_name)

    print(f"\nStarting RandomizedSearchCV for {model_name}...")
    print(f"Search settings: n_iter=20, cv=3, scoring='f1', random_state=42, n_jobs=-1")

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        cv=3,
        scoring="f1",
        random_state=42,
        n_jobs=-1,
        return_train_score=True,
    )

    start_time = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed_time = time.perf_counter() - start_time

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results["model"] = model_name

    best_idx = search.best_index_
    best_row = cv_results.loc[best_idx]

    summary = {
        "model": model_name,
        "best_params": search.best_params_,
        "best_mean_cv_f1": float(search.best_score_),
        "best_cv_f1_std": float(best_row["std_test_score"]),
        "mean_train_f1_for_best_config": float(best_row["mean_train_score"]),
        "total_search_time_seconds": float(elapsed_time),
    }

    print(f"Completed {model_name} tuning.")
    print(f"Best mean CV F1: {summary['best_mean_cv_f1']:.6f}")
    print(f"Best params: {summary['best_params']}")

    return cv_results, summary


def save_all_results(all_results: pd.DataFrame) -> None:
    output_path = METRICS_DIR / "unsw_randomized_search_all_results.csv"
    all_results.to_csv(output_path, index=False)
    print(f"Saved all evaluated configurations to {output_path}")


def save_summary(summary_df: pd.DataFrame) -> None:
    output_path = METRICS_DIR / "unsw_randomized_search_summary.csv"
    summary_df.to_csv(output_path, index=False)
    print(f"Saved tuned-model summary to {output_path}")


def main() -> None:
    ensure_directories()

    all_results: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for model_name in ["random_forest", "extra_trees"]:
        cv_results, summary = tune_model(model_name)
        all_results.append(cv_results)
        summary_rows.append(summary)

    combined_results = pd.concat(all_results, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    save_all_results(combined_results)
    save_summary(summary_df)

    print("\nRandomizedSearchCV comparison:")
    for _, row in summary_df.iterrows():
        print(
            f"{row['model']}: best_params={row['best_params']}, "
            f"best_mean_cv_f1={row['best_mean_cv_f1']:.6f} ± {row['best_cv_f1_std']:.6f}, "
            f"mean_train_f1={row['mean_train_f1_for_best_config']:.6f}, "
            f"search_time_seconds={row['total_search_time_seconds']:.2f}"
        )


if __name__ == "__main__":
    main()
