from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

OFFICIAL_TRAIN_ROWS = 175_341
OFFICIAL_TEST_ROWS = 82_332
TARGET_COLUMN = "label"
CATEGORICAL_FEATURES = ["proto", "service", "state"]
DROP_COLUMNS = ["id", "attack_cat", TARGET_COLUMN]


def load_official_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the official UNSW-NB15 training and final test datasets.

    The filenames in this repo are reversed relative to their actual roles,
    so assignment is based on the known official row counts.
    """
    data_dir = Path(__file__).resolve().parents[2] / "data" / "raw"

    train_candidate = pd.read_csv(data_dir / "UNSW_NB15_training-set.csv")
    test_candidate = pd.read_csv(data_dir / "UNSW_NB15_testing-set.csv")

    if len(train_candidate) == OFFICIAL_TRAIN_ROWS and len(test_candidate) == OFFICIAL_TEST_ROWS:
        official_train_df = train_candidate
        official_test_df = test_candidate
    elif len(train_candidate) == OFFICIAL_TEST_ROWS and len(test_candidate) == OFFICIAL_TRAIN_ROWS:
        official_train_df = test_candidate
        official_test_df = train_candidate
    else:
        raise ValueError(
            "UNSW-NB15 data files do not match the expected official sizes: "
            f"training candidate rows={len(train_candidate)}, "
            f"test candidate rows={len(test_candidate)}"
        )

    required_columns = {TARGET_COLUMN, "attack_cat"}
    for name, df in (("official training", official_train_df), ("official test", official_test_df)):
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"{name} data is missing required columns: {sorted(missing)}")

    return official_train_df, official_test_df


def prepare_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create X and y using the binary label target."""
    X = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns], errors="ignore")

    y = df[TARGET_COLUMN].copy()
    if not pd.api.types.is_numeric_dtype(y):
        y = y.map({"normal": 0, "attack": 1, "Normal": 0, "Attack": 1, "benign": 0, "malicious": 1})
    y = y.astype(int)

    return X, y


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing pipeline for the UNSW features."""
    categorical_cols = [col for col in CATEGORICAL_FEATURES if col in X_train.columns]
    numerical_cols = [col for col in X_train.columns if col not in categorical_cols]

    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
            (
                "num",
                "passthrough",
                numerical_cols,
            ),
        ],
        remainder="drop",
    )


def create_data_split() -> dict[str, pd.DataFrame | pd.Series]:
    """Create the official split and the internal validation setup for model selection."""
    official_train_df, official_test_df = load_official_data()

    X_train_full, y_train_full = prepare_features_target(official_train_df)
    X_test, y_test = prepare_features_target(official_test_df)

    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        random_state=42,
        stratify=y_train_full,
    )

    return {
        "official_train_df": official_train_df,
        "official_test_df": official_test_df,
        "X_train_full": X_train_full,
        "X_test": X_test,
        "y_train_full": y_train_full,
        "y_test": y_test,
        "X_subtrain": X_subtrain,
        "X_val": X_val,
        "y_subtrain": y_subtrain,
        "y_val": y_val,
    }
