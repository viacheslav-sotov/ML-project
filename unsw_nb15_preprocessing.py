from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"
TRAIN_PATH = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_PATH = DATA_DIR / "UNSW_NB15_testing-set.csv"


def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def load_and_validate_official_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_candidate = pd.read_csv(TRAIN_PATH)
    test_candidate = pd.read_csv(TEST_PATH)

    # The filenames in this repo do not correspond to the correct official roles,
    # so we assign training and test by the confirmed dataset sizes instead.
    expected_train_rows = 175341
    expected_test_rows = 82332
    expected_columns = 45
    required_columns = {"label", "attack_cat"}

    if len(train_candidate) == expected_train_rows and len(test_candidate) == expected_test_rows:
        train_df, test_df = train_candidate, test_candidate
    elif len(train_candidate) == expected_test_rows and len(test_candidate) == expected_train_rows:
        train_df, test_df = test_candidate, train_candidate
    else:
        raise ValueError(
            "UNSW-NB15 file sizes do not match the expected official roles: "
            f"training candidate rows={len(train_candidate)}, test candidate rows={len(test_candidate)}"
        )

    for name, df in [("training", train_df), ("test", test_df)]:
        if df.shape[1] != expected_columns:
            raise ValueError(f"Expected {expected_columns} columns in {name} data, found {df.shape[1]}")

        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"{name} data is missing required columns: {sorted(missing)}")

    return train_df, test_df


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    categorical_features = ["proto", "service", "state"]
    numerical_features = [
        col for col in X_train.columns if col not in ["id", "attack_cat", "label", *categorical_features]
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "num",
                "passthrough",
                numerical_features,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


def create_binary_target(df: pd.DataFrame) -> pd.Series:
    # 0 = Normal, 1 = Attack
    return df["label"].astype(int)


def print_split_summary(name: str, X: pd.DataFrame, y: pd.Series) -> None:
    print_section(f"{name} split")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print("Binary label distribution:")
    print(y.value_counts().sort_index())


def main() -> None:
    print_section("UNSW-NB15 preprocessing setup")
    train_df, test_df = load_and_validate_official_data()

    # Keep official training and official test sets separate.
    # We only use the official training set to build the internal validation split.
    X_train_full = train_df.drop(columns=["id", "attack_cat", "label"], errors="ignore")
    y_train_full = create_binary_target(train_df)

    X_test = test_df.drop(columns=["id", "attack_cat", "label"], errors="ignore")
    y_test = create_binary_target(test_df)

    # Internal stratified split for model selection.
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        random_state=42,
        stratify=y_train_full,
    )

    # Fit preprocessing only on the sub-training data.
    preprocessor = build_preprocessor(X_subtrain)
    X_subtrain_pre = preprocessor.fit_transform(X_subtrain)
    X_val_pre = preprocessor.transform(X_val)
    X_test_pre = preprocessor.transform(X_test)

    print_section("Official dataset sizes")
    print(f"Training rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")

    print_split_summary("Sub-training", X_subtrain, y_subtrain)
    print_split_summary("Validation", X_val, y_val)
    print_split_summary("Official test", X_test, y_test)

    print_section("Preprocessor check")
    print("Columns in X_subtrain:")
    print(X_subtrain.columns.tolist())
    print("\nVerification: id, attack_cat, and label are not present in X.")
    print(f"id in X_subtrain: {'id' in X_subtrain.columns}")
    print(f"attack_cat in X_subtrain: {'attack_cat' in X_subtrain.columns}")
    print(f"label in X_subtrain: {'label' in X_subtrain.columns}")

    print_section("Preprocessed array shapes")
    print(f"Sub-train transformed shape: {X_subtrain_pre.shape}")
    print(f"Validation transformed shape: {X_val_pre.shape}")
    print(f"Official test transformed shape: {X_test_pre.shape}")

    print_section("Pipeline-ready preprocessing")
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
    ])
    print("Pipeline created successfully. It is ready to be combined with a model later.")


if __name__ == "__main__":
    main()
