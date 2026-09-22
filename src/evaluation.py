from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


RAW_DATA_PATH = Path("data/raw/cic.csv")
CLEAN_DATA_PATH = Path("data/processed/cic_cleaned.csv")

FIGURE_DIR = Path("results/figures")
METRICS_DIR = Path("results/metrics")


def create_directories():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def plot_original_class_distribution():
    print("Loading raw dataset for original class distribution...")

    df = pd.read_csv(
        RAW_DATA_PATH,
        usecols=["Label"]
    )

    counts = df["Label"].value_counts()

    print("\nOriginal class distribution:")
    print(counts)

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")

    plt.title("Original CIC Class Distribution")
    plt.xlabel("Traffic Class")
    plt.ylabel("Number of Flows")
    plt.xticks(rotation=0)

    plt.tight_layout()

    output = FIGURE_DIR / "original_class_distribution.png"
    plt.savefig(output, dpi=300)
    plt.close()

    print(f"Saved: {output}")


def plot_cleaned_class_distribution():
    print("\nLoading cleaned dataset...")

    df = pd.read_csv(
        CLEAN_DATA_PATH,
        usecols=["Label", "BinaryLabel"]
    )

    counts = df["Label"].value_counts()

    print("\nClass distribution after preprocessing:")
    print(counts)

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")

    plt.title("CIC Class Distribution After Duplicate Removal")
    plt.xlabel("Traffic Class")
    plt.ylabel("Number of Flows")
    plt.xticks(rotation=0)

    plt.tight_layout()

    output = FIGURE_DIR / "cleaned_class_distribution.png"
    plt.savefig(output, dpi=300)
    plt.close()

    print(f"Saved: {output}")


def plot_binary_class_distribution():
    df = pd.read_csv(
        CLEAN_DATA_PATH,
        usecols=["BinaryLabel"]
    )

    counts = df["BinaryLabel"].value_counts().sort_index()

    counts.index = ["Benign", "Attack"]

    print("\nBinary class distribution:")
    print(counts)

    plt.figure(figsize=(7, 5))
    counts.plot(kind="bar")

    plt.title("Binary IDS Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Flows")
    plt.xticks(rotation=0)

    plt.tight_layout()

    output = FIGURE_DIR / "binary_class_distribution.png"
    plt.savefig(output, dpi=300)
    plt.close()

    print(f"Saved: {output}")


def plot_baseline_confusion_matrix():
    # Results from the first Logistic Regression baseline
    cm = np.array([
        [16199, 4],
        [0, 3797]
    ])

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Attack"]
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    display.plot(
        ax=ax,
        values_format="d"
    )

    ax.set_title("Logistic Regression Baseline")

    plt.tight_layout()

    output = FIGURE_DIR / "baseline_confusion_matrix.png"
    plt.savefig(output, dpi=300)
    plt.close()

    print(f"\nSaved: {output}")


def save_baseline_metrics():
    metrics = {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
            "False Positive Rate"
        ],
        "Value": [
            0.9998,
            0.9989,
            1.0000,
            0.9995,
            0.9999,
            0.0002
        ]
    }

    df = pd.DataFrame(metrics)

    output = METRICS_DIR / "logistic_regression_baseline.csv"

    df.to_csv(output, index=False)

    print(f"\nSaved: {output}")

    print("\nBaseline metrics:")
    print(df)


if __name__ == "__main__":
    create_directories()

    plot_original_class_distribution()
    plot_cleaned_class_distribution()
    plot_binary_class_distribution()
    plot_baseline_confusion_matrix()
    save_baseline_metrics()

    print("\nAll evaluation outputs created successfully.")