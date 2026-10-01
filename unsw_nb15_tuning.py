import json
import time

import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from unsw_nb15_boosting_comparison import get_models
from unsw_nb15_model_utils import (
    METRICS_DIR,
    RANDOM_STATE,
    build_model_pipeline,
    create_directories,
    evaluate_model,
    fit_timed,
    load_splits,
    print_metrics,
    print_section,
    save_confusion_matrix,
    save_csv,
)


N_ITER = 20
CV_FOLDS = 3


def get_search_space(model_name: str) -> tuple[object, dict]:
    if model_name == "XGBoost":
        model = XGBClassifier(
            tree_method="hist",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        space = {
            "model__max_depth": randint(3, 11),
            "model__learning_rate": loguniform(0.03, 0.3),
            "model__n_estimators": randint(100, 601),
            "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        }
    elif model_name == "HistGradientBoosting":
        model = HistGradientBoostingClassifier(
            early_stopping=False,
            random_state=RANDOM_STATE,
        )
        space = {
            "model__learning_rate": loguniform(0.03, 0.3),
            "model__max_iter": randint(100, 601),
            "model__max_leaf_nodes": randint(15, 128),
            "model__max_depth": [None, 6, 10, 15],
            "model__min_samples_leaf": randint(10, 201),
            "model__l2_regularization": loguniform(1e-3, 10.0),
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, space


def main() -> None:
    create_directories()

    selection = json.loads((METRICS_DIR / "boosting_model_selection.json").read_text(encoding="utf-8"))
    model_name = selection["selected_model"]

    print_section(f"Tuning {model_name} (official test set is NOT used here)")
    splits = load_splits()
    X_subtrain, y_subtrain = splits["X_subtrain"], splits["y_subtrain"]
    X_val, y_val = splits["X_val"], splits["y_val"]

    base_model, space = get_search_space(model_name)
    pipeline = build_model_pipeline(X_subtrain, base_model)

    # The preprocessor is inside the pipeline, so it is refit on each CV training fold.
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=space,
        n_iter=N_ITER,
        scoring={"f1": "f1", "roc_auc": "roc_auc"},
        refit="f1",
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
    )

    start = time.perf_counter()
    search.fit(X_subtrain, y_subtrain)
    print(f"\nSearch time: {time.perf_counter() - start:.1f} s")

    cv_results = pd.DataFrame(search.cv_results_)
    keep = [c for c in cv_results.columns if c.startswith("param_")] + [
        "mean_test_f1", "std_test_f1", "mean_test_roc_auc", "std_test_roc_auc",
        "rank_test_f1", "mean_fit_time",
    ]
    cv_results = cv_results[keep].sort_values("rank_test_f1")
    print(cv_results.head(10).to_string(index=False))
    save_csv(cv_results, f"{model_name.lower()}_tuning_cv_results.csv")

    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    best_params = {k: (v.item() if hasattr(v, "item") else v) for k, v in best_params.items()}
    print_section("Best parameters (by mean CV F1)")
    print(best_params)
    print(f"Best mean CV F1: {search.best_score_:.4f}")

    # Compare tuned vs default on the same validation set.
    print_section("Default vs tuned on validation")
    default_pipeline = build_model_pipeline(X_subtrain, get_models()[model_name])
    default_train_time = fit_timed(default_pipeline, X_subtrain, y_subtrain)
    default_metrics, _, _ = evaluate_model(default_pipeline, X_val, y_val, f"{model_name} (default)", "validation")
    default_metrics["train_time_s"] = default_train_time

    tuned_pipeline = build_model_pipeline(X_subtrain, base_model.set_params(**best_params))
    tuned_train_time = fit_timed(tuned_pipeline, X_subtrain, y_subtrain)
    tuned_metrics, tuned_cm, _ = evaluate_model(tuned_pipeline, X_val, y_val, f"{model_name} (tuned)", "validation")
    tuned_metrics["train_time_s"] = tuned_train_time

    for metrics in [default_metrics, tuned_metrics]:
        print_metrics(metrics)

    comparison = pd.DataFrame([default_metrics, tuned_metrics])
    save_csv(comparison, f"{model_name.lower()}_tuned_vs_default_validation.csv")
    save_confusion_matrix(
        tuned_cm, f"{model_name} (tuned) - Validation", f"{model_name.lower()}_tuned_validation_confusion_matrix.png"
    )

    # Keep the tuned configuration only if it beats the default on validation F1.
    use_tuned = tuned_metrics["f1"] > default_metrics["f1"]
    if use_tuned:
        final_params = best_params
    else:
        final_params = {k.replace("model__", ""): v for k, v in default_pipeline.get_params().items()
                        if k.startswith("model__") and k[len("model__"):] in best_params}
    final_config = {
        "model": model_name,
        "configuration": "tuned" if use_tuned else "default",
        "params": final_params,
        "best_cv_f1": search.best_score_,
        "validation_f1_default": default_metrics["f1"],
        "validation_f1_tuned": tuned_metrics["f1"],
        "search": {"method": "RandomizedSearchCV", "n_iter": N_ITER, "cv_folds": CV_FOLDS,
                   "scoring": "f1", "random_state": RANDOM_STATE},
    }
    output = METRICS_DIR / "final_model_config.json"
    output.write_text(json.dumps(final_config, indent=2, default=str), encoding="utf-8")
    print(f"Saved: {output}")
    print_section(f"Final configuration for the test run: {final_config['configuration']} {model_name}")


if __name__ == "__main__":
    main()
