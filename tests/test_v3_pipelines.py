"""v3 model pipelines (src/unsw/v3_pipelines.py): 42- and 39-feature sets, MLP/LR scaling. Tiny fits only."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.unsw import v3_data as vd
from src.unsw import v3_pipelines as vp


@pytest.fixture(scope="module")
def small():
    df = vd.load_official_train().sample(3000, random_state=0).reset_index(drop=True)
    return df[vd.feature_order()], df["label"].astype(int).to_numpy()


def test_feature_sets():
    assert len(vp.features_for("full42")) == 42
    no_ttl = vp.features_for("no_ttl39")
    assert len(no_ttl) == 39 and not set(vd.TTL_FEATURES) & set(no_ttl)
    with pytest.raises(ValueError):
        vp.features_for("full41")


@pytest.mark.parametrize("model", vp.MODELS)
@pytest.mark.parametrize("fs", vp.FEATURE_SETS)
def test_pipeline_structure(model, fs, small):
    pipe = vp.build_pipeline(model, fs)
    assert list(pipe.named_steps) == ["preprocessor", "model"]
    names = [str(c) for c in pipe.named_steps["preprocessor"].fit(small[0]).get_feature_names_out()]
    if fs == "no_ttl39":
        assert not any(n.endswith(("__sttl", "__dttl", "__ct_state_ttl")) for n in names)
    else:
        assert any(n.endswith("__sttl") for n in names)
    num = {name: tr for name, tr, _ in pipe.named_steps["preprocessor"].transformers}["num"]
    assert isinstance(num, StandardScaler) == (model in vp.SCALED)


def test_no_ttl_model_ignores_ttl_features(small):
    X, y = small
    # n_jobs=1: RandomForest predict_proba with n_jobs=-1 sums trees in thread order (last-bit differences between
    # calls); single-thread inference is bit-reproducible, so exact equality tests TTL-blindness only.
    pipe = vp.build_pipeline("random_forest", "no_ttl39", {"n_estimators": 10, "n_jobs": 1}).fit(X, y)
    changed = X.copy()
    for f in vd.TTL_FEATURES:
        changed[f] = changed[f] + 37
    assert np.array_equal(pipe.predict_proba(X), pipe.predict_proba(changed))


def test_scaled_mlp_fits_and_params_are_set(small):
    X, y = small
    pipe = vp.build_pipeline("mlp", "full42", {"hidden_layer_sizes": [16], "max_iter": 30})
    assert pipe.named_steps["model"].hidden_layer_sizes == (16,)
    pipe.fit(X, y)
    Z = pipe.named_steps["preprocessor"].transform(X)
    n_cat = len(pipe.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out())
    assert np.allclose(Z[:, n_cat:].mean(axis=0), 0, atol=1e-6)


def test_search_space_prefix_and_tuples():
    import json
    protocol = json.loads((vd.ROOT / "experiments" / "step6_protocol_v3_draft.json").read_text(encoding="utf-8"))
    space = vp.search_space(protocol, "mlp")
    assert all(k.startswith("model__") for k in space)
    assert (128, 64) in space["model__hidden_layer_sizes"]
    pipe = vp.build_pipeline("xgboost", "full42")
    pipe.set_params(**{k: v[0] for k, v in vp.search_space(protocol, "xgboost").items()})
