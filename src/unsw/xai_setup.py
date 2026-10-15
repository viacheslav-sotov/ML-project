"""Shared setup for the XAI experiments: the SHAP models and the SHAP shared sample set.

LIME must explain the *same* fitted models and the *same* malicious flows as SHAP, so this
module only reads what the SHAP experiment (experiments/unsw_shap_explain.py) defined:

- Models: the "validation pipelines", i.e. ``get_initial_models()`` + ``build_preprocessor``
  fitted on the internal sub-training split. They are loaded from ``results/models``; a
  missing file is refitted with exactly the SHAP recipe and saved in the SHAP bundle format.
- Samples: ``results/metrics/shap/shared_validation_sample_identifiers.csv`` is the single
  source of truth. No samples are selected here.

How a shared sample ID maps to a flow
    validation_position  -> row position in X_val (80/20 stratified split of the official
                            training set, random_state=42, from create_data_split())
    -> original_training_row_index = X_val.index[validation_position]
    -> official_id = the 'id' column of that row in the official training file.
The shared samples come from the validation split, not from the official test set.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from src.unsw.data import build_preprocessor, create_data_split
from src.unsw.models import get_initial_models

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "results" / "models"
SHAP_DIR = ROOT / "results" / "metrics" / "shap"
SHARED_SAMPLES_CSV = SHAP_DIR / "shared_validation_sample_identifiers.csv"
SAMPLE_KEY = "validation_position"

# The two models covered by the LIME/SHAP comparison (keys of get_initial_models()).
XAI_MODELS = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

DECISION_THRESHOLD = 0.5
CLASS_NAMES = ["Normal", "Attack"]


def get_xai_data() -> dict[str, pd.DataFrame | pd.Series]:
    """Return the project's official split (create_data_split), unchanged."""
    return create_data_split()


def _shap_model_signature(model_key: str, split: dict) -> str:
    """Same signature as make_model_signature() in experiments/unsw_shap_explain.py.

    Writing it lets the SHAP script recognise a pipeline refitted here instead of retraining it.
    """
    signature = {
        "model_name": model_key,
        "split": {
            "train_rows": int(len(split["X_train_full"])),
            "val_rows": int(len(split["X_val"])),
            "feature_names": [str(name) for name in split["X_train_full"].columns],
            "random_state": 42,
            "test_size": 0.20,
            "stratified": True,
        },
        "estimator_params": get_initial_models()[model_key].get_params(),
        "preprocessor": "ColumnTransformer(OneHotEncoder(handle_unknown='ignore', sparse_output=False), passthrough)",
    }
    return hashlib.md5(json.dumps(signature, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def load_xai_model(model_key: str, split: dict) -> Pipeline:
    """Load SHAP's fitted validation pipeline (fitted on the sub-training split)."""
    if model_key not in XAI_MODELS:
        raise ValueError(f"Unknown XAI model {model_key!r}; expected one of {list(XAI_MODELS)}")

    path = MODEL_DIR / f"{model_key}_validation_pipeline.joblib"
    signature = _shap_model_signature(model_key, split)

    if path.exists():
        bundle = joblib.load(path)
        if bundle.get("signature") != signature:
            raise ValueError(f"{path} was fitted with a different split or settings than the SHAP recipe")
        print(f"Loaded SHAP model: {path}")
        return bundle["pipeline"]

    # Not in the repository (e.g. too large for git): refit exactly as unsw_shap_explain.py does.
    X_subtrain, y_subtrain = split["X_subtrain"], split["y_subtrain"]
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_subtrain)),
            ("model", clone(get_initial_models()[model_key])),
        ]
    )
    print(f"{path.name} not found; fitting {XAI_MODELS[model_key]} on the sub-training split {X_subtrain.shape} (SHAP recipe) ...")
    pipeline.fit(X_subtrain, y_subtrain)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": model_key,
        "random_state": 42,
        "model_signature": signature,
        "train_rows": int(len(X_subtrain)),
        "val_rows": int(len(split["X_val"])),
        "preprocessor_type": "ColumnTransformer",
        "feature_count": int(X_subtrain.shape[1]),
        "created_by": "src/unsw/xai_setup.py (same recipe as experiments/unsw_shap_explain.py)",
    }
    joblib.dump({"pipeline": pipeline, "signature": signature, "metadata": metadata}, path)
    print(f"Saved: {path}")
    return pipeline


def load_shared_samples(split: dict) -> pd.DataFrame:
    """Read the SHAP shared sample set and map every ID to its flow in the validation split.

    Raises if an ID is missing, duplicated, not an integer, or outside the validation split.
    """
    if not SHARED_SAMPLES_CSV.exists():
        raise FileNotFoundError(f"Shared sample file not found: {SHARED_SAMPLES_CSV}")

    shared = pd.read_csv(SHARED_SAMPLES_CSV)
    if SAMPLE_KEY not in shared.columns:
        raise ValueError(f"{SHARED_SAMPLES_CSV.name} has no '{SAMPLE_KEY}' column (columns: {list(shared.columns)})")

    ids = shared[SAMPLE_KEY]
    if ids.isna().any() or not np.all(np.mod(ids.to_numpy(dtype=float), 1) == 0):
        raise ValueError(f"'{SAMPLE_KEY}' must contain whole numbers only")
    ids = ids.astype(int)
    if ids.duplicated().any():
        raise ValueError(f"Duplicate shared IDs: {sorted(ids[ids.duplicated()].unique())}")

    X_val = split["X_val"]
    out_of_range = ids[(ids < 0) | (ids >= len(X_val))]
    if len(out_of_range):
        raise ValueError(f"Shared IDs not found in the validation split ({len(X_val)} rows): {out_of_range.tolist()}")

    official_train_df = split["official_train_df"]
    train_index = X_val.index[ids.to_numpy()]
    samples = pd.DataFrame({
        SAMPLE_KEY: ids.to_numpy(),
        "original_training_row_index": np.asarray(train_index, dtype=int),
        "official_id": official_train_df.loc[train_index, "id"].astype(int).to_numpy(),
        "attack_cat": official_train_df.loc[train_index, "attack_cat"].astype(str).str.strip().to_numpy(),
        "label": split["y_val"].iloc[ids.to_numpy()].to_numpy(),
    })

    # If the shared file also lists the training row index, it must agree with the split.
    if "original_training_row_index" in shared.columns:
        if not (shared["original_training_row_index"].astype(int).to_numpy() == samples["original_training_row_index"].to_numpy()).all():
            raise ValueError("original_training_row_index in the shared file does not match the validation split")

    print(f"Loaded {len(samples)} shared samples from {SHARED_SAMPLES_CSV}; all found in the validation split.")
    return samples


def load_shap_reference_probabilities(model_key: str) -> pd.Series | None:
    """P(Attack) per validation_position as saved by the SHAP run (used to confirm the same model)."""
    path = SHAP_DIR / f"{model_key}_selected_sample_identifiers.csv"
    if not path.exists():
        return None
    ref = pd.read_csv(path)
    return ref.set_index(SAMPLE_KEY)["predicted_probability"]
