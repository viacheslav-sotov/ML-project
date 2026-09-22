from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


DATA_PATH = Path("data/processed/cic_cleaned.csv")


def load_processed_data():
    print("Loading cleaned dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    return df


def prepare_data(df, sample_size=100000):
    print("\nPreparing data...")

    # we are using a subset for the first baseline
    if len(df) > sample_size:
        df = df.sample(
            n=sample_size,
            random_state=42
        )

    print(f"Rows used for baseline: {len(df):,}")

    # removing original text labels and keeping only the binary target
    X = df.drop(columns=["Label", "BinaryLabel"])
    y = df["BinaryLabel"]

    print(f"Number of input features: {X.shape[1]}")

    print("\nClass distribution in subset:")
    print(y.value_counts())

    return X, y


def train_baseline(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    print("\nTraining Logistic Regression baseline...")
    pipeline.fit(X_train, y_train)

    print("Training complete.")

    return pipeline, X_test, y_test


def evaluate_model(model, X_test, y_test):
    print("\nEvaluating model...")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    false_positive_rate = fp / (fp + tn)

    print("\n===== BASELINE RESULTS =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"False Positive Rate: {false_positive_rate:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    df = load_processed_data()

    X, y = prepare_data(df)

    model, X_test, y_test = train_baseline(X, y)

    evaluate_model(model, X_test, y_test)