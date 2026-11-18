"""Explanations for the protocol-v3 final models: LIME (all models) and TreeSHAP (RF, XGBoost), on the raw features
each model actually uses, plus the RQ5 matched ranking-change metrics and an identity-keyed explanation cache.

Conventions (identical to the v1/v2 Step 6 code unless stated):
  LIME     LimeTabularExplainer, quartile discretizer, 5,000 samples, kernel width sqrt(n_features) * 0.75,
           feature_selection "none", Ridge surrogate, explained label 1 (attack), one RandomState(seed) shared by
           explainer, discretizer and surrogate and reset before every explanation, categoricals as integer codes.
           Changed for v3: the explainer's feature space is the model's own feature set (42, or 39 for no_ttl39),
           and its training statistics come from the v3 TRAINING side (the data the models were fitted on).
  TreeSHAP RF: shap.TreeExplainer(model, data=background) (interventional), probability of attack;
           XGBoost: shap.TreeExplainer(booster, data=background, feature_perturbation="tree_path_dependent",
           model_output="raw"), log-odds. One-hot columns are summed back into the raw categorical feature.
           Changed for v3: the 100-row background is drawn from the v3 TRAINING side with default_rng(42)
           (v2 drew it from the old validation split, which now hosts the attack sample).
  Rankings by |weight|, ties broken by feature name (as v2). Only features the model uses are ranked, so a
  no_ttl39 explanation never contains the three TTL features as artificial zero-weight ties.

Prior success criteria D8, D11 and D14 are not changed by this module; it only produces explanations and metrics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.unsw.v3_data import feature_order
from src.unsw.v3_pipelines import CATEGORICAL

NUM_SAMPLES = 5000
DISCRETIZER = "quartile"
LIME_KERNEL_FACTOR = 0.75
EXPLAINED_LABEL = 1
SHAP_BACKGROUND_SIZE = 100
SHAP_BACKGROUND_SEED = 42
ADDITIVITY_TOL = 1e-2      # abort above this (the v2 threshold, unsw_shap_explain.validate_shap_for_model)
ADDITIVITY_REPORT = 1e-4   # gaps above this are recorded and reported (single-tree threshold ties, see E0 smoke)


def lime_settings(n_features: int) -> dict[str, Any]:
    return {"explainer": "lime.lime_tabular.LimeTabularExplainer", "num_samples": NUM_SAMPLES, "num_features": n_features,
            "discretize_continuous": True, "discretizer": DISCRETIZER, "kernel_width": float(np.sqrt(n_features) * LIME_KERNEL_FACTOR),
            "feature_selection": "none", "sample_around_instance": False, "model_regressor": "Ridge(alpha=1.0) (LIME default)",
            "distance_metric": "euclidean", "explained_label": EXPLAINED_LABEL, "training_statistics": "v3 training side"}


def settings_hash(settings: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------------------------------------------
# LIME
# ----------------------------------------------------------------------------------------------------------------
class Codec:
    """Model features <-> LIME's numeric matrix. Categoricals become integer codes (sorted training categories)."""

    def __init__(self, X_train: pd.DataFrame, features: list[str]):
        self.features = list(features)
        self.categorical = [c for c in self.features if c in CATEGORICAL]
        self.categorical_idx = [self.features.index(c) for c in self.categorical]
        self.categories = {c: sorted(X_train[c].astype(str).unique().tolist()) for c in self.categorical}

    def encode(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.features].copy()
        for c in self.categorical:
            codes = pd.Categorical(X[c].astype(str), categories=self.categories[c]).codes
            if (codes < 0).any():
                raise ValueError(f"{c} contains a value not seen in training")
            X[c] = codes
        return X.to_numpy(dtype=np.float64)

    def decode(self, Z: np.ndarray, base_row: pd.Series) -> pd.DataFrame:
        """LIME rows -> the 42-column model frame. Features outside the model's set (the TTL features of a
        no_ttl39 model) are copied from base_row; the model never reads them (checked by check_ttl_invariance)."""
        X = pd.DataFrame(Z, columns=self.features)
        for c in self.categorical:
            X[c] = np.asarray(self.categories[c], dtype=object)[np.rint(X[c].to_numpy()).astype(int)]
        for f in feature_order():
            if f not in X.columns:
                X[f] = base_row[f]
        return X[feature_order()]


def build_lime_explainer(X_train: pd.DataFrame, codec: Codec):
    from lime.lime_tabular import LimeTabularExplainer
    n = len(codec.features)
    return LimeTabularExplainer(
        training_data=codec.encode(X_train), mode="classification", feature_names=codec.features,
        class_names=["Normal", "Attack"], categorical_features=codec.categorical_idx,
        categorical_names={i: codec.categories[c] for i, c in zip(codec.categorical_idx, codec.categorical)},
        discretize_continuous=True, discretizer=DISCRETIZER, kernel_width=np.sqrt(n) * LIME_KERNEL_FACTOR,
        feature_selection="none", sample_around_instance=False, random_state=0)


def make_predict_fn(pipeline, codec: Codec, base_row: pd.Series) -> Callable[[np.ndarray], np.ndarray]:
    def predict_fn(Z: np.ndarray) -> np.ndarray:
        return pipeline.predict_proba(codec.decode(Z, base_row))
    return predict_fn


def reseed(explainer, seed: int) -> None:
    rs = np.random.RandomState(seed)
    explainer.random_state = rs
    explainer.base.random_state = rs
    if explainer.discretizer is not None:
        explainer.discretizer.random_state = rs


def lime_weights(explainer, codec: Codec, pipeline, row: pd.Series, seed: int) -> dict[str, float]:
    """Signed LIME weights for the attack class over exactly the model's features."""
    reseed(explainer, seed)
    exp = explainer.explain_instance(
        data_row=codec.encode(row.to_frame().T)[0], predict_fn=make_predict_fn(pipeline, codec, row),
        labels=(EXPLAINED_LABEL,), num_features=len(codec.features), num_samples=NUM_SAMPLES,
        distance_metric="euclidean", model_regressor=None)
    weights = {codec.features[i]: float(w) for i, w in exp.as_map()[EXPLAINED_LABEL]}
    if set(weights) != set(codec.features):
        raise ValueError("LIME did not return a weight for every model feature")
    return weights


# ----------------------------------------------------------------------------------------------------------------
# TreeSHAP
# ----------------------------------------------------------------------------------------------------------------
def column_to_feature(column: str) -> str:
    if column.startswith("num__"):
        return column[len("num__"):]
    for c in CATEGORICAL:
        if column.startswith(f"cat__{c}_"):
            return c
    raise ValueError(f"cannot map transformed column {column!r} to a raw feature")


def background_positions(n_train: int) -> np.ndarray:
    """Row positions (in the v3 training side, sorted official order) of the 100 TreeSHAP background rows."""
    rng = np.random.default_rng(SHAP_BACKGROUND_SEED)
    return np.sort(rng.choice(n_train, size=min(SHAP_BACKGROUND_SIZE, n_train), replace=False))


def shap_background(X_train: pd.DataFrame, pipeline) -> np.ndarray:
    pre = pipeline.named_steps["preprocessor"]
    return pre.transform(X_train.iloc[background_positions(len(X_train))][feature_order()])


def array_sha256(a: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    return hashlib.sha256(a.tobytes() + str(a.shape).encode()).hexdigest()


def build_tree_explainer(model_key: str, pipeline, background: np.ndarray):
    import shap
    est = pipeline.named_steps["model"]
    if model_key == "xgboost":
        return shap.TreeExplainer(est.get_booster(), data=background, feature_perturbation="tree_path_dependent", model_output="raw"), "log_odds"
    if model_key == "random_forest":
        return shap.TreeExplainer(est, data=background), "probability"
    raise ValueError(f"TreeSHAP is only defined for random_forest and xgboost, not {model_key}")


def shap_weights(model_key: str, pipeline, explainer, output_kind: str, X: pd.DataFrame, features: list[str],
                 return_gaps: bool = False):
    """SHAP values of the attack class summed into the model's raw features; additivity checked per row."""
    pre, est = pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]
    Xp = pre.transform(X[feature_order()])
    cols = [str(c) for c in pre.get_feature_names_out()]
    ex = explainer(Xp)
    vals = np.asarray(ex.values)
    base = np.asarray(ex.base_values)
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    base = np.asarray(base, dtype=float)
    if base.ndim == 2:
        base = base[:, 1]
    elif base.ndim == 1 and len(base) == 2 and len(Xp) != 2:
        base = np.full(len(Xp), base[1])
    base = np.broadcast_to(base.reshape(-1) if base.ndim else base, (len(Xp),))
    p = est.predict_proba(Xp)[:, 1]
    target = p if output_kind == "probability" else np.log(np.clip(p, 1e-12, 1 - 1e-12) / (1 - np.clip(p, 1e-12, 1 - 1e-12)))
    gaps = np.abs(vals.sum(axis=1) + base - target)
    gap = float(np.max(gaps))
    if gap > ADDITIVITY_TOL:
        raise ValueError(f"{model_key}: SHAP additivity gap {gap:.2e} > {ADDITIVITY_TOL}")
    mapping = [column_to_feature(c) for c in cols]
    if sorted(set(mapping)) != sorted(features):
        raise ValueError(f"{model_key}: transformed columns map to {sorted(set(mapping))}, not the model features")
    out = []
    for r in range(len(Xp)):
        w = dict.fromkeys(features, 0.0)
        for c, f in zip(range(len(cols)), mapping):
            w[f] += float(vals[r, c])
        out.append(w)
    return (out, [float(g) for g in gaps]) if return_gaps else out


# ----------------------------------------------------------------------------------------------------------------
# Ranking metrics (identical definitions to experiments.unsw_spoofing_explanations.pair_metrics)
# ----------------------------------------------------------------------------------------------------------------
def ranked(weights: dict[str, float]) -> list[str]:
    return sorted(weights, key=lambda f: (-abs(weights[f]), f))


def pair_metrics(a: dict[str, float], b: dict[str, float]) -> dict[str, Any]:
    if set(a) != set(b):
        raise ValueError("explanations must cover the same features (compare only features the model uses)")
    order_a, order_b = ranked(a), ranked(b)
    result = {"orig_top1": order_a[0], "mod_top1": order_b[0], "top1_agreement": float(order_a[0] == order_b[0])}
    for k in (5, 10):
        top_a, top_b = set(order_a[:k]), set(order_b[:k])
        result[f"top{k}_jaccard"] = len(top_a & top_b) / len(top_a | top_b)
    rank_a = pd.Series({f: i + 1 for i, f in enumerate(order_a)})
    rank_b = pd.Series({f: i + 1 for i, f in enumerate(order_b)})[rank_a.index]
    rho = float(np.corrcoef(rank_a.to_numpy(), rank_b.to_numpy())[0, 1])
    result["spearman_rho"] = 0.0 if np.isnan(rho) else rho
    feats = list(a)
    tie = spearmanr([abs(a[f]) for f in feats], [abs(b[f]) for f in feats]).statistic
    result["spearman_abs_tieaware"] = float(tie) if np.isfinite(tie) else np.nan
    shared = sorted(set(order_a[:10]) & set(order_b[:10]))
    result["direction_agreement_top10"] = sum(np.sign(a[f]) == np.sign(b[f]) for f in shared) / len(shared) if shared else 0.0
    result["shared_top10_features"] = len(shared)
    return result


def exact_zero_count(w: dict[str, float]) -> int:
    return int(sum(1 for v in w.values() if v == 0.0))


RQ5_METRICS = ("top10_jaccard", "spearman_rho", "spearman_abs_tieaware", "top1_agreement")


def rq5_matched(lime_orig: dict[int, dict[str, float]], lime_cand: dict[int, dict[str, float]],
                shap_orig: dict[str, float], shap_cand: dict[str, float]) -> dict[str, Any]:
    """Primary descriptive RQ5 comparison for one (model, flow, candidate): the same ranking metrics for both
    explainers, each against its own explanation of the original. LIME: mean over the given (held-out) seeds,
    original and candidate explained with the same seed. Positive *_diff = LIME changed less than TreeSHAP."""
    seeds = sorted(lime_orig)
    if sorted(lime_cand) != seeds or not seeds:
        raise ValueError("LIME original and candidate must use the same seeds")
    if set(shap_orig) != set(next(iter(lime_orig.values()))):
        raise ValueError("LIME and TreeSHAP must cover the same model features")
    per_seed = [pair_metrics(lime_orig[s], lime_cand[s]) for s in seeds]
    lime = {m: float(np.nanmean([p[m] for p in per_seed])) for m in RQ5_METRICS}
    shap_m = pair_metrics(shap_orig, shap_cand)
    out = {"lime_seeds": seeds, "n_features": len(shap_orig),
           "shap_exact_zero_features_orig": exact_zero_count(shap_orig), "shap_exact_zero_features_cand": exact_zero_count(shap_cand)}
    for m in RQ5_METRICS:
        out[f"lime_{m}"] = lime[m]
        out[f"shap_{m}"] = float(shap_m[m]) if shap_m[m] == shap_m[m] else np.nan
        out[f"{m}_diff"] = out[f"lime_{m}"] - out[f"shap_{m}"]
    out["lime_top1_changed"] = 1.0 - lime["top1_agreement"]
    out["shap_top1_changed"] = 1.0 - float(shap_m["top1_agreement"])
    return out


# ----------------------------------------------------------------------------------------------------------------
# Identity-keyed explanation cache (append-only JSON lines; resumable)
# ----------------------------------------------------------------------------------------------------------------
def vector_key(row: pd.Series, features: list[str]) -> str:
    vals = [str(row[f]) if isinstance(row[f], str) else repr(float(row[f])) for f in features]
    return hashlib.sha256("|".join(vals).encode("utf-8")).hexdigest()


class ExplanationCache:
    """Key = (explainer, model SHA-256, feature set, settings hash, seed, vector hash over the model's features).
    A hit is only returned for exactly that identity; a stored entry with the same key but other content is an error."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries: dict[str, dict[str, float]] = {}
        self.meta: dict[str, dict[str, Any]] = {}
        self.hits = self.misses = 0
        self.dropped_lines = 0
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        rec = json.loads(line)
                        self.entries[rec["key"]] = rec["weights"]
                        self.meta[rec["key"]] = {k: v for k, v in rec.items() if k not in ("key", "weights")}
                    except (json.JSONDecodeError, KeyError):
                        self.dropped_lines += 1          # a line cut by an interruption; recomputed on resume

    @staticmethod
    def key(explainer: str, model_sha: str, feature_set: str, settings_sha: str, seed: int | None, vec: str) -> str:
        return hashlib.sha256(json.dumps([explainer, model_sha, feature_set, settings_sha, seed, vec]).encode()).hexdigest()

    def get_or_compute(self, key: str, compute: Callable[[], dict[str, float]], meta: dict[str, Any]) -> dict[str, float]:
        if key in self.entries:
            self.hits += 1
            return self.entries[key]
        self.misses += 1
        w = compute()
        if isinstance(w, tuple):            # (weights, extra meta) from the caller
            w, extra = w
            meta = {**meta, **extra}
        self.entries[key] = w
        self.meta[key] = dict(meta)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"key": key, **meta, "weights": w}) + "\n")
            fh.flush()
        return w
