import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier

from unsw_nb15_model_utils import (
    FIGURE_DIR,
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
    save_text,
)


COMPARISON_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def get_models() -> dict:
    return {
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            tree_method="hist",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.1,
            early_stopping=False,
            random_state=RANDOM_STATE,
        ),
    }


def plot_comparison(results: pd.DataFrame) -> None:
    plot_df = results.set_index("model")[COMPARISON_METRICS].T

    ax = plot_df.plot(kind="bar", figsize=(9, 5))
    ax.set_title("UNSW-NB15 Validation: XGBoost vs HistGradientBoosting")
    ax.set_ylabel("Score")
    ax.set_ylim(max(0.0, plot_df.min().min() - 0.05), 1.0)
    ax.legend(loc="lower right")
    plt.xticks(rotation=0)

    plt.tight_layout()
    output = FIGURE_DIR / "boosting_validation_comparison.png"
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Saved: {output}")


def select_winner(results: pd.DataFrame) -> str:
    # Selection uses validation results only: F1 first, ROC-AUC as tie-breaker.
    ranked = results.sort_values(["f1", "roc_auc"], ascending=False)
    return ranked.iloc[0]["model"]


def main() -> None:
    create_directories()

    print_section("Loading UNSW-NB15 splits (official test set is NOT used here)")
    splits = load_splits()
    X_subtrain, y_subtrain = splits["X_subtrain"], splits["y_subtrain"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    print(f"Sub-train: {X_subtrain.shape}, Validation: {X_val.shape}")

    rows = []
    reports = []

    for model_name, model in get_models().items():
        print_section(f"Training {model_name} on sub-train")
        pipeline = build_model_pipeline(X_subtrain, model)
        train_time = fit_timed(pipeline, X_subtrain, y_subtrain)

        metrics, cm, report = evaluate_model(pipeline, X_val, y_val, model_name, "validation")
        metrics["train_time_s"] = train_time
        print_metrics(metrics)
        print(report)

        rows.append(metrics)
        reports.append(f"===== {model_name} (validation) =====\n{report}")

        file_stub = model_name.lower()
        save_confusion_matrix(cm, f"{model_name} - Validation", f"{file_stub}_validation_confusion_matrix.png")

    print_section("Validation comparison")
    results = pd.DataFrame(rows)
    column_order = [
        "model", "split", "rows", "accuracy", "precision", "recall", "f1", "roc_auc",
        "false_positive_rate", "false_negative_rate", "tn", "fp", "fn", "tp",
        "train_time_s", "predict_time_s",
    ]
    results = results[column_order]
    print(results.to_string(index=False))

    save_csv(results, "boosting_validation_comparison.csv")
    save_text("\n\n".join(reports), "boosting_validation_classification_reports.txt")
    plot_comparison(results)

    winner = select_winner(results)
    print_section(f"Selected model (validation F1, then ROC-AUC): {winner}")

    selection = {
        "selected_model": winner,
        "selection_rule": "highest validation F1, ROC-AUC as tie-breaker",
        "validation_f1": {row["model"]: round(row["f1"], 6) for row in rows},
        "validation_roc_auc": {row["model"]: round(row["roc_auc"], 6) for row in rows},
    }
    output = METRICS_DIR / "boosting_model_selection.json"
    output.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
