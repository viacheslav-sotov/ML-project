"""Protocol v3 (draft) model pipelines: RF, XGBoost, MLP (primary) and logistic regression (secondary baseline),
for the full 42-feature set and the 39-feature no-TTL set.

Every pipeline takes the usual 42-column frame (so the flow, constraint, LIME and SHAP code is unchanged) and
has the steps "preprocessor" and "model" (the names the existing XAI code expects):
    preprocessor = ColumnTransformer(one-hot proto/service/state, numeric columns passed through or scaled,
                                     remainder="drop")
The no-TTL set simply leaves sttl, dttl and ct_state_ttl out of the ColumnTransformer, so the model never sees
them. MLP and LR get a StandardScaler on the numeric columns (trees do not need scaling).

The v1/v2 models (src.unsw.models, results/models/*) are not used or changed here.
"""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.unsw.v3_data import TTL_FEATURES, feature_order

CATEGORICAL = ("proto", "service", "state")
PRIMARY_MODELS = ("random_forest", "xgboost", "mlp")
SECONDARY_MODELS = ("logistic_regression",)
MODELS = PRIMARY_MODELS + SECONDARY_MODELS
FEATURE_SETS = ("full42", "no_ttl39")
SCALED = {"mlp", "logistic_regression"}


def features_for(feature_set: str) -> list[str]:
    if feature_set == "full42":
        return feature_order()
    if feature_set == "no_ttl39":
        return [f for f in feature_order() if f not in TTL_FEATURES]
    raise ValueError(f"unknown feature set {feature_set!r}; expected one of {FEATURE_SETS}")


def build_preprocessor(feature_set: str, scale: bool) -> ColumnTransformer:
    feats = features_for(feature_set)
    cats = [f for f in feats if f in CATEGORICAL]
    nums = [f for f in feats if f not in CATEGORICAL]
    return ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
                      ("num", StandardScaler() if scale else "passthrough", nums)],
        remainder="drop",
    )


def base_estimator(model_key: str, random_state: int = 42) -> Any:
    if model_key == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    if model_key == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_estimators=300, max_depth=6, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=random_state, n_jobs=-1, tree_method="hist")
    if model_key == "mlp":
        return MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=1e-4, learning_rate_init=1e-3,
                             max_iter=300, early_stopping=True, validation_fraction=0.1, n_iter_no_change=15, random_state=random_state)
    if model_key == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=random_state)
    raise ValueError(f"unknown model {model_key!r}; expected one of {MODELS}")


def build_pipeline(model_key: str, feature_set: str = "full42", params: dict[str, Any] | None = None,
                   random_state: int = 42) -> Pipeline:
    """params use the search-space names without the 'model__' prefix."""
    estimator = base_estimator(model_key, random_state)
    if params:
        estimator.set_params(**{k: (tuple(v) if k == "hidden_layer_sizes" and isinstance(v, list) else v) for k, v in params.items()})
    return Pipeline([("preprocessor", build_preprocessor(feature_set, model_key in SCALED)), ("model", estimator)])


def search_space(protocol: dict[str, Any], model_key: str) -> dict[str, list]:
    """The protocol's space for a model, with the 'model__' prefix RandomizedSearchCV needs."""
    space = protocol["nested_cv"]["search_spaces"][model_key]["space"]
    out = {}
    for k, values in space.items():
        vals = [tuple(v) if k == "hidden_layer_sizes" and isinstance(v, list) else v for v in values]
        out[f"model__{k}"] = vals
    return out
