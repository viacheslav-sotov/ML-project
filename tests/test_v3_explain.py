"""Protocol v3 explanation layer: metric definitions, feature sets, cache identity/resume (synthetic), and the eight
final models (hash pins, preprocessing, TTL invariance; real files, skipped if absent)."""

import numpy as np
import pandas as pd
import pytest

from src.unsw import v3_explain as vx
from src.unsw import v3_final_models as fm
from src.unsw.v3_data import TTL_FEATURES, feature_order

models_present = pytest.mark.skipif(not all(fm.model_path(*k).exists() for k in fm.STAGE_RUNS), reason="final model files not present")


def rand_weights(features, rng):
    return {f: float(rng.normal()) for f in features}


def test_pair_metrics_identical_to_the_v2_definition():
    from experiments.unsw_spoofing_explanations import pair_metrics as v2_pair_metrics
    rng = np.random.default_rng(0)
    feats = feature_order()
    for _ in range(20):
        a, b = rand_weights(feats, rng), rand_weights(feats, rng)
        b[feats[3]] = 0.0
        mine, ref = vx.pair_metrics(a, b), v2_pair_metrics(a, b)
        for k in ref:
            assert mine[k] == ref[k] or (np.isnan(mine[k]) and np.isnan(ref[k])), k


def test_rankings_compare_only_the_same_feature_set():
    rng = np.random.default_rng(1)
    f42 = feature_order()
    f39 = [f for f in f42 if f not in TTL_FEATURES]
    with pytest.raises(ValueError, match="same features"):
        vx.pair_metrics(rand_weights(f42, rng), rand_weights(f39, rng))


def test_rq5_matched_uses_each_explainers_own_original_and_the_same_seeds():
    rng = np.random.default_rng(2)
    f = [x for x in feature_order() if x not in TTL_FEATURES]
    lo = {22: rand_weights(f, rng), 33: rand_weights(f, rng)}
    so = rand_weights(f, rng)
    same = vx.rq5_matched(lo, lo, so, so)
    assert same["lime_top10_jaccard"] == same["shap_top10_jaccard"] == 1.0 and same["top10_jaccard_diff"] == 0.0
    assert same["n_features"] == 39 and same["lime_top1_changed"] == 0.0
    with pytest.raises(ValueError, match="same seeds"):
        vx.rq5_matched(lo, {22: lo[22]}, so, so)
    with pytest.raises(ValueError, match="same model features"):
        vx.rq5_matched(lo, lo, rand_weights(feature_order(), rng), rand_weights(feature_order(), rng))


def test_column_mapping_for_one_hot_and_numeric_columns():
    assert vx.column_to_feature("num__sbytes") == "sbytes"
    assert vx.column_to_feature("cat__proto_tcp") == "proto" and vx.column_to_feature("cat__state_FIN") == "state"
    with pytest.raises(ValueError):
        vx.column_to_feature("sbytes")


def test_codec_roundtrip_fills_unused_features_from_the_flow():
    feats = feature_order()
    rng = np.random.default_rng(3)
    train = pd.DataFrame({f: rng.integers(0, 5, 20) for f in feats})
    train["proto"], train["service"], train["state"] = "tcp", "-", "FIN"
    f39 = [f for f in feats if f not in TTL_FEATURES]
    codec = vx.Codec(train, f39)
    row = train.iloc[0].copy()
    row["sttl"] = 254
    Z = codec.encode(row.to_frame().T)
    assert Z.shape == (1, 39)
    back = codec.decode(Z, row)
    assert list(back.columns) == feats and back.iloc[0]["sttl"] == 254 and back.iloc[0]["proto"] == "tcp"


def test_cache_identity_hit_and_resume(tmp_path):
    p = tmp_path / "c.jsonl"
    calls = []
    c1 = vx.ExplanationCache(p)
    k = c1.key("lime", "sha", "full42", "st", 22, "vec")
    w = c1.get_or_compute(k, lambda: calls.append(1) or {"a": 1.0}, {"m": 1})
    assert w == {"a": 1.0} and c1.misses == 1
    c2 = vx.ExplanationCache(p)
    assert c2.get_or_compute(k, lambda: calls.append(1) or {"a": 9.0}, {}) == {"a": 1.0} and c2.hits == 1 and len(calls) == 1
    for other in (("treeshap", "sha", "full42", "st", 22, "vec"), ("lime", "sha2", "full42", "st", 22, "vec"),
                  ("lime", "sha", "no_ttl39", "st", 22, "vec"), ("lime", "sha", "full42", "st2", 22, "vec"),
                  ("lime", "sha", "full42", "st", 33, "vec"), ("lime", "sha", "full42", "st", 22, "vec2")):
        assert c2.key(*other) != k


def test_lime_settings_follow_the_feature_count():
    assert vx.lime_settings(42)["kernel_width"] == pytest.approx(np.sqrt(42) * 0.75)
    assert vx.lime_settings(39)["num_features"] == 39
    assert vx.settings_hash(vx.lime_settings(42)) != vx.settings_hash(vx.lime_settings(39))


# ---------------------------------------------------------------- final models (real files)
@models_present
def test_all_eight_final_models_load_with_pinned_hashes_and_protocol_preprocessing():
    for m, fs in fm.STAGE_RUNS:
        x = fm.load_final_model(m, fs)
        assert x.sha256 == fm.PINNED_SHA256[(m, fs)] and len(x.features) == (42 if fs == "full42" else 39)


@models_present
def test_a_wrong_pin_is_refused(monkeypatch):
    monkeypatch.setitem(fm.PINNED_SHA256, ("xgboost", "full42"), "0" * 64)
    with pytest.raises(fm.FinalModelError, match="SHA-256"):
        fm.load_final_model("xgboost", "full42")


@models_present
def test_no_ttl39_models_ignore_the_ttl_features():
    from experiments import unsw_nested_cv as ncv
    X = ncv.load_v3_train(ncv.load_protocol())["X"].iloc[:300]
    for m in ("random_forest", "xgboost", "mlp", "logistic_regression"):
        x = fm.load_final_model(m, "no_ttl39")
        base = x.predict_attack_proba(X)
        for v in (0, 255):
            assert np.array_equal(x.predict_attack_proba(X.assign(**{f: v for f in TTL_FEATURES})), base)


@models_present
def test_final_model_predictions_are_bit_reproducible():
    from experiments import unsw_nested_cv as ncv
    X = ncv.load_v3_train(ncv.load_protocol())["X"].iloc[:2000]
    for m in ("random_forest", "xgboost"):
        x = fm.load_final_model(m, "full42")
        a, b = x.predict_attack_proba(X), x.predict_attack_proba(X)
        assert a.dtype == np.float64 and np.array_equal(a, b)
        assert np.array_equal(x.predict_attack_proba(X.iloc[:7]), a[:7])        # independent of batch size
