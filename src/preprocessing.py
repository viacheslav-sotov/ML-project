from pathlib import Path
import numpy as np
import pandas as pd


DATA_PATH = Path("data/raw/cic.csv")
OUTPUT_PATH = Path("data/processed/cic_cleaned.csv")


def preprocess_data():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"\nOriginal shape: {df.shape}")

    print("\nOriginal class distribution:")
    print(df["Label"].value_counts())

    # 1. Remove exact duplicate rows
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows found: {duplicates:,}")

    df = df.drop_duplicates()

    print(f"Shape after removing duplicates: {df.shape}")


    # 2. Replace infinity values with NaN
    numeric_columns = df.select_dtypes(include=np.number).columns

    inf_count = np.isinf(df[numeric_columns]).sum().sum()
    print(f"\nInfinite values found: {inf_count:,}")

    df[numeric_columns] = df[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # 3. Check missing values
    missing_before = df.isnull().sum().sum()

    print(f"Missing values before cleaning: {missing_before:,}")

    # Fill numeric missing values with column median
    df[numeric_columns] = df[numeric_columns].fillna(
        df[numeric_columns].median()
    )

    missing_after = df.isnull().sum().sum()

    print(f"Missing values after cleaning: {missing_after:,}")

    # 4. Remove timestamp for the first baseline
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])
        print("\nTimestamp removed.")

    
    # 5. Create binary target
    df["BinaryLabel"] = df["Label"].apply(
        lambda x: 0 if x == "Benign" else 1
    )

    print("\nBinary class distribution:")
    print(df["BinaryLabel"].value_counts())

    print("\nOriginal labels after duplicate removal:")
    print(df["Label"].value_counts())

    # 6. Save cleaned dataset
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nCleaned dataset saved to:")
    print(OUTPUT_PATH)

    print(f"\nFinal shape: {df.shape}")


if __name__ == "__main__":
    preprocess_data()