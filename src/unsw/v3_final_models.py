"""The eight final protocol-v3 models (nested-CV final refits), pinned by run, file and SHA-256.

A model is only ever loaded through load_final_model(), which refuses it unless
  - the file's SHA-256 equals the pinned hash AND the hash recorded by the training run (units/<model>__<fs>__final.json),
  - the stored best_params equal the training record's,
  - the pipeline's preprocessing uses exactly the protocol feature set (42 or 39 features, one-hot categoricals,
    scaling only for MLP/LR) and was fitted on the 42-column frame in the protocol feature order.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.unsw.v3_data import TTL_FEATURES, feature_order
from src.unsw.v3_pipelines import CATEGORICAL, MODELS, PRIMARY_MODELS, SCALED, features_for

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "v3_runs"

# Training runs (protocol step6-v3-draft-1): Stage 1 no_ttl39 trees/LR, Stage 2 full42 trees/LR, Stage 3 MLP (both sets).
STAGE_RUNS = {
    ("xgboost", "no_ttl39"): "nested_cv_20261117_221501",
    ("random_forest", "no_ttl39"): "nested_cv_20261117_221501",
    ("logistic_regression", "no_ttl39"): "nested_cv_20261117_221501",
    ("xgboost", "full42"): "nested_cv_20261118_002142",
    ("random_forest", "full42"): "nested_cv_20261118_002142",
    ("logistic_regression", "full42"): "nested_cv_20261118_002142",
    ("mlp", "no_ttl39"): "nested_cv_20261118_020018",
    ("mlp", "full42"): "nested_cv_20261118_020018",
}
PINNED_SHA256 = {
    ("xgboost", "no_ttl39"): "ec41c67035f845541f5c27ba2ed9efc702789b1042045b39b57c7309ea876c0c",
    ("random_forest", "no_ttl39"): "8f270ee3597289f22320d25908d866cd0d21194ce7a7a06b574b182313b1c7d4",
    ("logistic_regression", "no_ttl39"): "cc1ae514afa7fe2d9f7b7043d4490fc3054ecd3a2e14feb6e1dcd41db68d642d",
    ("xgboost", "full42"): "670fbcf2b1ac6c7610452ed07bd447f05568e899dd1630da4c9d3e05cf6332a2",
    ("random_forest", "full42"): "abda2ca069cc6c9e1d6ce90f9bf5d2ee8e2e8f9374a7840c7c7c6bbf4d4ed757",
    ("logistic_regression", "full42"): "ba24a396fcd383002ca5e2889e5aa1956b60dbd04b3abfe14485c61cf5eaac01",
    ("mlp", "no_ttl39"): "99c614981ca17306b81d668b8f931aa7a1cb373d438b6ed561ea1144ea9a55ff",
    ("mlp", "full42"): "6e16abd95d0a240738d2faac7325d0fd2098322cfa4b2c4cafc040523feccba7",
}
FEATURE_SETS = ("full42", "no_ttl39")
DECISION_THRESHOLD = 0.5


class FinalModelError(RuntimeError):
    pass


@dataclass
class FinalModel:
    model_key: str
    feature_set: str
    pipeline: Any
    sha256: str
    best_params: dict[str, Any]
    file: str            # path relative to the repository root
    run: str
    features: list[str]  # raw features the model uses (42 or 39), in protocol order

    def predict_attack_proba(self, X) -> Any:
        return np.asarray(self.pipeline.predict_proba(X[feature_order()].infer_objects())[:, 1], dtype=np.float64)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def model_path(model_key: str, feature_set: str) -> Path:
    return RUNS / STAGE_RUNS[(model_key, feature_set)] / "models" / f"{model_key}_{feature_set}.joblib"


def training_record(model_key: str, feature_set: str) -> dict[str, Any]:
    p = RUNS / STAGE_RUNS[(model_key, feature_set)] / "units" / f"{model_key}__{feature_set}__final.json"
    return json.loads(p.read_text(encoding="utf-8"))


def check_preprocessing(pipeline, feature_set: str, model_key: str) -> None:
    """The fitted preprocessor must use exactly the protocol features; nothing else may reach the model."""
    pre = pipeline.named_steps["preprocessor"]
    expected = features_for(feature_set)
    names_in = [str(c) for c in getattr(pre, "feature_names_in_", [])]
    if names_in != feature_order():
        raise FinalModelError(f"{model_key}/{feature_set}: preprocessor was fitted on {len(names_in)} columns, not the 42-column protocol order")
    cols = {name: list(c) for name, _, c in pre.transformers_ if name != "remainder"}
    used = cols.get("cat", []) + cols.get("num", [])
    if sorted(used) != sorted(expected) or cols.get("cat") != [f for f in expected if f in CATEGORICAL]:
        raise FinalModelError(f"{model_key}/{feature_set}: preprocessing columns differ from the protocol feature set")
    if feature_set == "no_ttl39" and set(TTL_FEATURES) & set(used):
        raise FinalModelError(f"{model_key}/no_ttl39: a TTL feature reaches the model")
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    num_step = dict((n, t) for n, t, _ in pre.transformers_)["num"]
    scaled = isinstance(num_step, StandardScaler)
    if not scaled and not (num_step == "passthrough" or isinstance(num_step, FunctionTransformer) and num_step.func is None):
        raise FinalModelError(f"{model_key}/{feature_set}: unexpected numeric transformer {num_step!r}")
    if scaled != (model_key in SCALED):
        raise FinalModelError(f"{model_key}/{feature_set}: numeric scaling is {scaled}, expected {model_key in SCALED}")
    if pre.remainder != "drop":
        raise FinalModelError(f"{model_key}/{feature_set}: remainder must be 'drop'")


DETERMINISTIC_INFERENCE = {"random_forest": {"n_jobs": 1}}
# RandomForest.predict_proba with n_jobs=-1 sums tree probabilities in thread order, so repeated predictions differ by
# up to ~2e-16; with n_jobs=1 they are bit-identical (checked). Inference setting only: the fitted trees, the model file
# and its hash are unchanged. XGBoost, MLP and LR predictions were checked to be bit-identical as loaded.


def load_final_model(model_key: str, feature_set: str, deterministic: bool = True) -> FinalModel:
    import joblib
    if (model_key, feature_set) not in STAGE_RUNS:
        raise FinalModelError(f"no final model for {model_key}/{feature_set}")
    path = model_path(model_key, feature_set)
    if not path.exists():
        raise FinalModelError(f"model file missing: {path}")
    record = training_record(model_key, feature_set)
    actual = file_sha256(path)
    if actual != PINNED_SHA256[(model_key, feature_set)] or actual != record["model_sha256"]:
        raise FinalModelError(f"{model_key}/{feature_set}: SHA-256 {actual} differs from the pinned/recorded hash")
    payload = joblib.load(path)
    if payload.get("best_params") != record["best_params"]:
        raise FinalModelError(f"{model_key}/{feature_set}: stored best_params differ from the training record")
    pipeline = payload["pipeline"]
    check_preprocessing(pipeline, feature_set, model_key)
    if deterministic:
        pipeline.named_steps["model"].set_params(**DETERMINISTIC_INFERENCE.get(model_key, {}))
    return FinalModel(model_key, feature_set, pipeline, actual, record["best_params"],
                      str(path.relative_to(ROOT)).replace("\\", "/"), STAGE_RUNS[(model_key, feature_set)], features_for(feature_set))


def manifest_entries() -> list[dict[str, Any]]:
    """Identity of all eight final models (no model is loaded)."""
    out = []
    for (m, fs), run in STAGE_RUNS.items():
        rec = training_record(m, fs)
        out.append({"model": m, "feature_set": fs, "primary": m in PRIMARY_MODELS, "run": run,
                    "file": str(model_path(m, fs).relative_to(ROOT)).replace("\\", "/"),
                    "sha256": PINNED_SHA256[(m, fs)], "recorded_sha256": rec["model_sha256"],
                    "best_params": rec["best_params"], "best_inner_f1": rec["best_inner_f1"]})
    return out


assert set(STAGE_RUNS) == set(PINNED_SHA256) == {(m, fs) for m in MODELS for fs in FEATURE_SETS}
