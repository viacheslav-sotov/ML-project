import json

import pandas as pd

from unsw_nb15_model_utils import (
    METRICS_DIR,
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
from unsw_nb15_tuning import get_search_space


# This is the ONLY script that uses the official UNSW-NB15 test set.
# Run it once, after model selection and tuning are finished.


def main() -> None:
    create_directories()

    config = json.loads((METRICS_DIR / "final_model_config.json").read_text(encoding="utf-8"))
    model_name = config["model"]
    params = config["params"]

    print_section(f"Final model: {config['configuration']} {model_name}")
    print(params)

    splits = load_splits()
    X_train_full, y_train_full = splits["X_train_full"], splits["y_train_full"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    # Refit the chosen configuration on the full official training set (sub-train + validation).
    model, _ = get_search_space(model_name)
    model.set_params(**params)
    pipeline = build_model_pipeline(X_train_full, model)

    print_section(f"Training on full official training set {X_train_full.shape}")
    train_time = fit_timed(pipeline, X_train_full, y_train_full)

    print_section(f"Evaluating once on official test set {X_test.shape}")
    metrics, cm, report = evaluate_model(pipeline, X_test, y_test, f"{model_name} ({config['configuration']})", "official_test")
    metrics["train_time_s"] = train_time
    print_metrics(metrics)
    print(report)

    file_stub = model_name.lower()
    save_csv(pd.DataFrame([metrics]), f"{file_stub}_final_official_test.csv")
    save_text(report, f"{file_stub}_final_official_test_classification_report.txt")
    save_confusion_matrix(cm, f"{model_name} ({config['configuration']}) - Official Test",
                          f"{file_stub}_final_official_test_confusion_matrix.png")


if __name__ == "__main__":
    main()
