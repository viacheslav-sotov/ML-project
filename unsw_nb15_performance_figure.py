from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
UNSW_METRICS_DIR = ROOT / "results" / "unsw_nb15" / "metrics"
FIGURE_DIR = ROOT / "results" / "unsw_nb15" / "figures"

# Viacheslav's Random Forest / Extra Trees results (copied from origin/main).
# Same split as ours: official training -> 80/20 stratified, random_state=42, threshold 0.5.
TEAM_BASELINE_CSV = ROOT / "results" / "metrics" / "unsw_baseline_model_comparison.csv"
BOOSTING_CSV = UNSW_METRICS_DIR / "boosting_validation_comparison.csv"
TUNED_CSV = UNSW_METRICS_DIR / "histgradientboosting_tuned_vs_default_validation.csv"
FINAL_TEST_CSV = UNSW_METRICS_DIR / "histgradientboosting_final_official_test.csv"

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
}

MODEL_ORDER = [
    "Random Forest",
    "Extra Trees",
    "XGBoost",
    "HistGradientBoosting",
    "Tuned HistGradientBoosting",
]

# Categorical palette, fixed order (blue, orange, aqua, yellow).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
ROW_HIGHLIGHT = "#eef4fc"

VALIDATION_NOTE = (
    "All models evaluated on the same validation set: 35,069 rows "
    "(20% stratified hold-out of the official training data, random_state=42, threshold 0.5)."
)


def load_validation_results() -> pd.DataFrame:
    team = pd.read_csv(TEAM_BASELINE_CSV)
    team = team[team["model"].isin(["random_forest", "extra_trees"])].copy()
    team["model"] = team["model"].map({"random_forest": "Random Forest", "extra_trees": "Extra Trees"})

    boosting = pd.read_csv(BOOSTING_CSV)
    boosting = boosting[boosting["split"] == "validation"]

    tuned = pd.read_csv(TUNED_CSV)
    tuned = tuned[tuned["model"] == "HistGradientBoosting (tuned)"].copy()
    tuned["model"] = "Tuned HistGradientBoosting"

    summary = pd.concat(
        [team[["model", *METRICS]], boosting[["model", *METRICS]], tuned[["model", *METRICS]]],
        ignore_index=True,
    )
    summary["model"] = pd.Categorical(summary["model"], categories=MODEL_ORDER, ordered=True)
    summary = summary.sort_values("model").reset_index(drop=True)
    summary["model"] = summary["model"].astype(str)

    missing = set(MODEL_ORDER) - set(summary["model"])
    if missing:
        raise ValueError(f"Missing results for: {sorted(missing)}")

    return summary


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=10, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def draw_table(ax, df: pd.DataFrame, highlight_row: str | None, first_col_label: str = "Model",
               bold_best: bool = True) -> None:
    """Draw a clean summary table; optionally the best value in each metric column is bold."""
    ax.axis("off")

    columns = [first_col_label] + [METRIC_LABELS[m] for m in METRICS]
    col_x = [0.0, 0.40, 0.52, 0.64, 0.76, 0.88]
    n_rows = len(df)
    row_h = 1.0 / (n_rows + 1)

    best = {m: df[m].max() for m in METRICS}

    def y_of(i: int) -> float:
        return 1.0 - (i + 0.5) * row_h

    for x, label in zip(col_x, columns):
        ax.text(x + (0 if x == 0 else 0.05), y_of(0), label, ha="left" if x == 0 else "center", va="center",
                fontsize=10.5, fontweight="bold", color=TEXT_PRIMARY, transform=ax.transAxes)
    ax.plot([0, 1], [1.0 - row_h, 1.0 - row_h], color=TEXT_SECONDARY, linewidth=1.0, transform=ax.transAxes)

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        if row["model"] == highlight_row:
            ax.add_patch(plt.Rectangle((0, 1.0 - (i + 1) * row_h), 1, row_h, transform=ax.transAxes,
                                       facecolor=ROW_HIGHLIGHT, edgecolor="none", zorder=0))
        ax.text(0.01, y_of(i), row["model"], ha="left", va="center", fontsize=10.5,
                color=TEXT_PRIMARY, transform=ax.transAxes)
        for x, metric in zip(col_x[1:], METRICS):
            is_best = bold_best and np.isclose(row[metric], best[metric])
            ax.text(x + 0.05, y_of(i), f"{row[metric]:.4f}", ha="center", va="center", fontsize=10.5,
                    fontweight="bold" if is_best else "normal", color=TEXT_PRIMARY, transform=ax.transAxes)

    ax.plot([0, 1], [0, 0], color=GRID, linewidth=1.0, transform=ax.transAxes)


def draw_grouped_bars(ax, df: pd.DataFrame, metrics: list[str], y_min: float, label_col: str = "model") -> None:
    style_axes(ax)

    n_series = len(metrics)
    x = np.arange(len(df))
    group_width = 0.8
    bar_w = group_width / n_series
    gap = 0.02  # small surface gap between adjacent bars

    for idx, metric in enumerate(metrics):
        offset = (idx - (n_series - 1) / 2) * bar_w
        ax.bar(x + offset, df[metric], width=bar_w - gap, color=SERIES_COLORS[idx],
               label=METRIC_LABELS[metric], edgecolor=SURFACE, linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([name.replace("Tuned ", "Tuned\n") for name in df[label_col]], fontsize=10,
                       color=TEXT_PRIMARY)
    ax.set_ylim(y_min, 1.0)
    ax.set_ylabel("Score", fontsize=10.5, color=TEXT_SECONDARY)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=n_series, frameon=False, fontsize=10.5,
              labelcolor=TEXT_PRIMARY, handlelength=1.2)


def build_figure(df: pd.DataFrame, metrics: list[str], title: str, subtitle: str, y_min: float,
                 output_name: str, footnote: str, highlight_row: str | None = "Tuned HistGradientBoosting",
                 label_col: str = "model", first_col_label: str = "Model") -> Path:
    fig = plt.figure(figsize=(12, 9.5), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.0], hspace=0.16, left=0.07, right=0.97, top=0.86, bottom=0.12)

    fig.text(0.07, 0.95, title, fontsize=17, fontweight="bold", color=TEXT_PRIMARY, ha="left")
    fig.text(0.07, 0.915, subtitle, fontsize=11, color=TEXT_SECONDARY, ha="left")

    table_df = df.rename(columns={label_col: "model"}) if label_col != "model" else df
    draw_table(fig.add_subplot(grid[0]), table_df, highlight_row, first_col_label)
    draw_grouped_bars(fig.add_subplot(grid[1]), df, metrics, y_min, label_col)

    fig.text(0.07, 0.035, footnote, fontsize=9, color=TEXT_SECONDARY, ha="left", wrap=True)

    output = FIGURE_DIR / output_name
    fig.savefig(output, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved: {output}")
    return output


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    summary = load_validation_results()
    print(summary.to_string(index=False))

    summary_out = UNSW_METRICS_DIR / "model_performance_summary.csv"
    summary.assign(split="validation", rows=35069).to_csv(summary_out, index=False)
    print(f"Saved: {summary_out}")

    bold_note = "Bold = best value in column. Highlighted row = final selected model."

    build_figure(
        summary,
        ["f1", "roc_auc"],
        "UNSW-NB15 Model Performance: F1 and ROC-AUC",
        "Binary intrusion detection (Normal vs Attack), validation set",
        y_min=0.96,
        output_name="unsw_model_performance_f1_rocauc.png",
        footnote=f"{VALIDATION_NOTE}\n{bold_note} Note: y-axis starts at 0.96 to make small differences visible.",
    )

    build_figure(
        summary,
        ["accuracy", "precision", "recall", "f1"],
        "UNSW-NB15 Model Performance: Accuracy, Precision, Recall, F1",
        "Binary intrusion detection (Normal vs Attack), validation set",
        y_min=0.94,
        output_name="unsw_model_performance_all_metrics.png",
        footnote=f"{VALIDATION_NOTE}\n{bold_note} Note: y-axis starts at 0.94 to make small differences visible.",
    )

    # Tuned HistGradientBoosting: validation vs the single official test run.
    tuned_val = summary[summary["model"] == "Tuned HistGradientBoosting"].iloc[0]
    test = pd.read_csv(FINAL_TEST_CSV).iloc[0]
    val_vs_test = pd.DataFrame([
        {"split": "Validation (35,069 rows)", **{m: tuned_val[m] for m in METRICS}},
        {"split": "Official test (82,332 rows)", **{m: test[m] for m in METRICS}},
    ])

    # Here the series are the two evaluation sets and the groups are the metrics.
    long = val_vs_test.set_index("split")[METRICS].T.reset_index().rename(columns={"index": "metric"})
    long["metric"] = long["metric"].map(METRIC_LABELS)
    fig = plt.figure(figsize=(12, 7.5), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 1, height_ratios=[0.6, 2.0], hspace=0.12, left=0.07, right=0.97, top=0.84, bottom=0.16)
    fig.text(0.07, 0.94, "Tuned HistGradientBoosting: Validation vs Official Test", fontsize=17,
             fontweight="bold", color=TEXT_PRIMARY, ha="left")
    fig.text(0.07, 0.9, "Final selected model on UNSW-NB15 (binary label)", fontsize=11,
             color=TEXT_SECONDARY, ha="left")

    table_df = val_vs_test.rename(columns={"split": "model"})
    draw_table(fig.add_subplot(grid[0]), table_df, highlight_row=None, first_col_label="Evaluation set",
               bold_best=False)

    ax = fig.add_subplot(grid[1])
    style_axes(ax)
    x = np.arange(len(METRICS))
    bar_w = 0.38
    for idx, split in enumerate(val_vs_test["split"]):
        ax.bar(x + (idx - 0.5) * bar_w, long[split], width=bar_w - 0.02, color=SERIES_COLORS[idx],
               label=split, edgecolor=SURFACE, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(long["metric"], fontsize=10.5, color=TEXT_PRIMARY)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score", fontsize=10.5, color=TEXT_SECONDARY)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=10.5,
              labelcolor=TEXT_PRIMARY, handlelength=1.2)

    fig.text(
        0.07, 0.03,
        "Validation: model trained on 80% sub-train. Official test: same configuration retrained on the full official "
        "training set, evaluated once.\nThe official UNSW-NB15 test set has a different distribution from the "
        f"training set, so lower test scores are expected. Test false positive rate: {test['false_positive_rate']:.4f}.",
        fontsize=9, color=TEXT_SECONDARY, ha="left",
    )
    output = FIGURE_DIR / "unsw_tuned_hgb_validation_vs_test.png"
    fig.savefig(output, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
