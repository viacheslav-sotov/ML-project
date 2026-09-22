from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/raw/cic.csv")


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print(f"Loading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    print("\nDataset loaded successfully.")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    return df


def explore_data(df):
    print("\n===== FIRST 5 ROWS =====")
    print(df.head())

    print("\n===== COLUMN NAMES =====")
    for col in df.columns:
        print(col)

    print("\n===== DATA TYPES =====")
    print(df.dtypes)

    print("\n===== MISSING VALUES =====")
    missing = df.isnull().sum()
    print(missing[missing > 0].sort_values(ascending=False))

    print("\n===== DUPLICATE ROWS =====")
    print(df.duplicated().sum())

    print("\n===== BASIC STATISTICS =====")
    print(df.describe(include="all").transpose())

    print("\n===== POSSIBLE LABEL COLUMNS =====")
    for col in df.columns:
        name = col.lower().strip()
        if "label" in name or "class" in name or "attack" in name:
            print(f"\nColumn: {col}")
            print(df[col].value_counts(dropna=False).head(20))


if __name__ == "__main__":
    df = load_data()
    explore_data(df)