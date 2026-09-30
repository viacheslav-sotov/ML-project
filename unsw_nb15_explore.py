from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"
TRAIN_PATH = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_PATH = DATA_DIR / "UNSW_NB15_testing-set.csv"


def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def summarize_dataframe(df: pd.DataFrame, dataset_name: str) -> None:
    print_section(f"{dataset_name}: dataset overview")
    print(f"Shape: {df.shape}")

    print_section(f"{dataset_name}: column names")
    for col in df.columns:
        print(col)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    print_section(f"{dataset_name}: numeric vs. categorical columns")
    print("Numeric columns:")
    print(numeric_cols)
    print("\nCategorical columns:")
    print(categorical_cols)

    print_section(f"{dataset_name}: data types")
    print(df.dtypes)

    if "label" in df.columns:
        print_section(f"{dataset_name}: label distribution")
        print(df["label"].value_counts(dropna=False))

    if "attack_cat" in df.columns:
        print_section(f"{dataset_name}: attack category distribution")
        print(df["attack_cat"].value_counts(dropna=False))

    print_section(f"{dataset_name}: missing values")
    missing = df.isna().sum().sort_values(ascending=False)
    print(missing)

    print_section(f"{dataset_name}: duplicate rows")
    print(f"Exact duplicate rows: {int(df.duplicated().sum())}")

    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        print_section(f"{dataset_name}: infinity check")
        pos_inf = np.isposinf(numeric_df.to_numpy()).sum(axis=0)
        neg_inf = np.isneginf(numeric_df.to_numpy()).sum(axis=0)
        inf_summary = pd.DataFrame({
            "positive_infinity": pos_inf,
            "negative_infinity": neg_inf,
        }, index=numeric_df.columns)
        print(inf_summary)

    print_section(f"{dataset_name}: basic statistics")
    print(df.describe(include="all").transpose())

    print_section(f"{dataset_name}: first 5 rows")
    print(df.head())


def main() -> None:
    print_section("UNSW-NB15 exploration")
    print(f"Training file: {TRAIN_PATH}")
    print(f"Testing file: {TEST_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    summarize_dataframe(train_df, "UNSW-NB15 training set")
    summarize_dataframe(test_df, "UNSW-NB15 testing set")


if __name__ == "__main__":
    main()
