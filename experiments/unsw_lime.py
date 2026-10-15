"""LIME explanations for Random Forest and XGBoost on the SHAP shared UNSW-NB15 samples.

Steps
1. Load the official split, SHAP's two fitted validation pipelines, and the shared samples
   from results/metrics/shap/shared_validation_sample_identifiers.csv (the single source of
   truth; LIME never selects samples itself).
2. Build one LIME explainer on the sub-training split (the data the models were fitted on),
   in the original 42-feature space (proto/service/state stay single categorical features
   instead of one-hot columns).
3. For every model and sample: run the validation checks, explain the Attack class,
   save a bar plot, and collect the LIME weights.
4. Save CSVs (all feature weights + top features), the LIME settings, and a summary, then
   check that the saved outputs contain exactly the shared sample IDs.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from importlib.metadata import version
from lime.lime_tabular import LimeTabularExplainer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.unsw.constraints import FEATURE_ORDER
from src.unsw.data import CATEGORICAL_FEATURES
from src.unsw.xai_setup import (
    CLASS_NAMES,
    DECISION_THRESHOLD,
    SAMPLE_KEY,
    SHARED_SAMPLES_CSV,
    XAI_MODELS,
    get_xai_data,
    load_shap_reference_probabilities,
    load_shared_samples,
    load_xai_model,
)

OUTPUT_DIR = ROOT / "results" / "unsw_nb15" / "lime"
FIGURE_DIR = OUTPUT_DIR / "figures"

# ---- LIME settings (identical for Random Forest and XGBoost) ----
LIME_SEED = 42                 # base seed; each explanation uses LIME_SEED + validation_position
NUM_SAMPLES = 5000             # perturbation samples per explanation
NUM_FEATURES = len(FEATURE_ORDER)  # weights are estimated for all 42 features ...
TOP_K = 10                     # ... and the 10 largest |weights| are reported and plotted
DISCRETIZE_CONTINUOUS = True
DISCRETIZER = "quartile"
FEATURE_SELECTION = "none"     # fit the surrogate on all features (no forward/greedy selection)
KERNEL_WIDTH = np.sqrt(len(FEATURE_ORDER)) * 0.75  # LIME default, written out for the record
EXPLAINED_LABEL = 1            # explain the Attack class
PROB_ROUNDTRIP_TOL = 1e-9      # LIME prediction function vs. direct model call (must be exact)
SAVED_PROB_TOL = 1e-6          # current model vs. P(Attack) saved by the SHAP run

# Diverging pair from the dataviz reference palette: red = towards Attack, blue = towards Normal.
COLOR_ATTACK = "#e34948"
COLOR_NORMAL = "#2a78d6"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"


# --------------------------------------------------------------------------------------
# Encoding between the model's DataFrame input and LIME's numeric array
# --------------------------------------------------------------------------------------
class CategoricalCodec:
    """LIME needs a numeric matrix, the pipeline needs the original DataFrame.

    Categorical columns are turned into integer codes (index in the sorted list of training
    categories) for LIME, and turned back into strings before calling the model. Numeric
    columns pass through unchanged, so the model sees exactly the same values.
    """

    def __init__(self, X_train: pd.DataFrame):
        self.feature_names = list(X_train.columns)
        self.categorical_idx = [self.feature_names.index(c) for c in CATEGORICAL_FEATURES]
        self.categories = {c: sorted(X_train[c].unique().tolist()) for c in CATEGORICAL_FEATURES}
        self.numeric_dtypes = {c: X_train[c].dtype for c in self.feature_names if c not in CATEGORICAL_FEATURES}

    def encode(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_names].copy()
        for col in CATEGORICAL_FEATURES:
            codes = pd.Categorical(X[col], categories=self.categories[col]).codes
            if (codes < 0).any():
                raise ValueError(f"{col} contains a value not seen in training")
            X[col] = codes
        return X.to_numpy(dtype=np.float64)

    def decode(self, Z: np.ndarray) -> pd.DataFrame:
        X = pd.DataFrame(Z, columns=self.feature_names)
        for col in CATEGORICAL_FEATURES:
            X[col] = np.asarray(self.categories[col], dtype=object)[np.rint(X[col].to_numpy()).astype(int)]
        return X

    def categorical_names(self) -> dict[int, list[str]]:
        return {i: self.categories[c] for i, c in zip(self.categorical_idx, CATEGORICAL_FEATURES)}


def make_predict_fn(pipeline, codec: CategoricalCodec):
    """Prediction function given to LIME: numeric LIME rows -> [P(Normal), P(Attack)]."""

    def predict_fn(Z: np.ndarray) -> np.ndarray:
        return pipeline.predict_proba(codec.decode(Z))

    return predict_fn


def build_explainer(train_encoded: np.ndarray, codec: CategoricalCodec) -> LimeTabularExplainer:
    return LimeTabularExplainer(
        training_data=train_encoded,
        mode="classification",
        feature_names=codec.feature_names,
        class_names=CLASS_NAMES,
        categorical_features=codec.categorical_idx,
        categorical_names=codec.categorical_names(),
        discretize_continuous=DISCRETIZE_CONTINUOUS,
        discretizer=DISCRETIZER,
        kernel_width=KERNEL_WIDTH,
        feature_selection=FEATURE_SELECTION,
        sample_around_instance=False,
        random_state=LIME_SEED,
    )


def reseed_explainer(explainer: LimeTabularExplainer, seed: int) -> None:
    """Give one explanation its own random stream.

    LIME shares one RandomState between the explainer, the discretizer and the ridge
    surrogate, so without this the result for a sample would depend on which samples were
    explained before it. Reseeding makes each explanation reproducible on its own, and the
    same sample gets the identical perturbation set for Random Forest and XGBoost.
    """
    rs = np.random.RandomState(seed)
    explainer.random_state = rs
    explainer.base.random_state = rs
    if explainer.discretizer is not None:
        explainer.discretizer.random_state = rs


# --------------------------------------------------------------------------------------
# Validation checks
# --------------------------------------------------------------------------------------
def check_global_feature_order(pipelines: dict, codec: CategoricalCodec, X_val: pd.DataFrame) -> None:
    """Feature order must be identical everywhere: data, model input, LIME."""
    assert list(X_val.columns) == FEATURE_ORDER, "X_val column order differs from FEATURE_ORDER"
    assert codec.feature_names == FEATURE_ORDER, "LIME feature names differ from FEATURE_ORDER"
    for key, pipe in pipelines.items():
        fitted_order = list(pipe.named_steps["preprocessor"].feature_names_in_)
        assert fitted_order == FEATURE_ORDER, f"{key}: model was fitted with a different feature order"
    print(f"Feature order check passed: {len(FEATURE_ORDER)} features, identical in data, models and LIME.")


def validate_sample(model_key, pipeline, codec, x_row, y_true, attack_cat, expected_prob, predict_fn) -> dict:
    """Run all per-sample checks. Returns the predicted class/probability or raises ValueError."""
    if y_true != 1 or attack_cat == "Normal":
        raise ValueError(f"sample is not malicious (label={y_true}, attack_cat={attack_cat})")

    numeric = x_row.drop(columns=CATEGORICAL_FEATURES).to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("raw numeric features contain NaN or infinite values")
    if x_row[CATEGORICAL_FEATURES].isna().any(axis=None):
        raise ValueError("categorical features contain missing values")

    model_input = pipeline.named_steps["preprocessor"].transform(x_row)
    if not np.isfinite(np.asarray(model_input, dtype=np.float64)).all():
        raise ValueError("preprocessed model input contains NaN or infinite values")

    prob_attack = float(pipeline.predict_proba(x_row)[0, 1])
    predicted = int(prob_attack >= DECISION_THRESHOLD)
    print(f"    predicted class: {CLASS_NAMES[predicted]}   P(Attack) = {prob_attack:.6f}")
    if predicted != 1:
        raise ValueError("model does not classify the sample as Attack")

    # Same model as SHAP: P(Attack) must equal the probability saved by the SHAP run.
    if expected_prob is None:
        raise ValueError("no SHAP reference probability for this sample, cannot confirm it is the SHAP model")
    if abs(prob_attack - expected_prob) > SAVED_PROB_TOL:
        raise ValueError(
            f"P(Attack) {prob_attack:.6f} differs from the SHAP run ({expected_prob:.6f}); "
            "the model is not the same one"
        )

    # The LIME prediction function must give the model's own answer for the unperturbed flow;
    # this confirms the encode/decode step keeps the feature order and values intact.
    encoded = codec.encode(x_row)
    if not np.isfinite(encoded).all():
        raise ValueError("LIME-encoded row contains NaN or infinite values")
    lime_prob = float(predict_fn(encoded)[0, 1])
    if abs(lime_prob - prob_attack) > PROB_ROUNDTRIP_TOL:
        raise ValueError(f"LIME prediction function gives {lime_prob:.6f}, model gives {prob_attack:.6f}")

    print("    checks passed: malicious, correctly classified, matches SHAP P(Attack), no NaN/inf, "
          "feature order and prediction round-trip OK")
    return {"predicted_class": CLASS_NAMES[predicted], "prob_attack": prob_attack, "encoded": encoded[0]}


# --------------------------------------------------------------------------------------
# Explanation, plot and tables
# --------------------------------------------------------------------------------------
def explain(explainer, encoded_row, predict_fn, seed):
    reseed_explainer(explainer, seed)
    return explainer.explain_instance(
        data_row=encoded_row,
        predict_fn=predict_fn,
        labels=(EXPLAINED_LABEL,),
        num_features=NUM_FEATURES,
        num_samples=NUM_SAMPLES,
        distance_metric="euclidean",
        model_regressor=None,  # LIME default: Ridge(alpha=1.0)
    )




def explanation_rows(exp, model_key, sample, x_row) -> list[dict]:
    """One row per feature: the LIME rule (e.g. 'sttl > 62.00'), its weight and its rank."""
    weights = exp.as_map()[EXPLAINED_LABEL]  # [(feature index, weight)] sorted by |weight|
    rules = exp.domain_mapper.map_exp_ids(weights)  # same order: [(rule text, weight)]

    rows = []
    for rank, ((feat_idx, weight), (rule, _)) in enumerate(zip(weights, rules), start=1):
        feature = FEATURE_ORDER[feat_idx]
        rows.append({
            "model": model_key,
            SAMPLE_KEY: int(sample.validation_position),
            "original_training_row_index": int(sample.original_training_row_index),
            "official_id": int(sample.official_id),
            "attack_cat": sample.attack_cat,
            "rank": rank,
            "feature": feature,
            "feature_value": x_row.iloc[0][feature],
            "lime_rule": rule,
            "lime_weight": weight,
            "abs_weight": abs(weight),
            "direction": "supports Attack" if weight > 0 else "pushes towards Normal" if weight < 0 else "neutral",
            "in_top_k": rank <= TOP_K,
        })
    return rows


def plot_path_for(model_key: str, position: int) -> Path:
    # Same naming as the SHAP waterfall plots: <model>_..._sample_<validation_position>.png
    return FIGURE_DIR / f"lime_{model_key}_sample_{position}.png"


def save_plot(rows: list[dict], model_key: str, sample, prob_attack: float, exp) -> Path:
    top = rows[:TOP_K][::-1]  # largest weight at the top of the chart
    weights = np.array([r["lime_weight"] for r in top])
    labels = [r["lime_rule"] for r in top]
    colors = [COLOR_ATTACK if w > 0 else COLOR_NORMAL for w in weights]

    fig, ax = plt.subplots(figsize=(9, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh(np.arange(len(top)), weights, height=0.62, color=colors, edgecolor=SURFACE, linewidth=2)
    ax.axvline(0, color=INK_SECONDARY, linewidth=1)

    span = np.abs(weights).max() or 1.0
    for i, w in enumerate(weights):
        ax.text(w + (0.02 * span if w >= 0 else -0.02 * span), i, f"{w:+.3f}",
                va="center", ha="left" if w >= 0 else "right", fontsize=8.5, color=INK_PRIMARY)
    ax.set_xlim(-1.3 * span, 1.3 * span)

    ax.set_yticks(np.arange(len(top)), labels, fontsize=9, color=INK_PRIMARY)
    ax.set_xlabel("LIME weight for P(Attack)", color=INK_SECONDARY)
    ax.tick_params(axis="x", colors=INK_SECONDARY)
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")

    ax.set_title(
        f"{XAI_MODELS[model_key]} - LIME, validation sample {sample.validation_position} ({sample.attack_cat})\n"
        f"P(Attack) = {prob_attack:.4f}   local surrogate R² = {exp.score:.3f}   "
        f"{NUM_SAMPLES:,} perturbations",
        loc="left", fontsize=10.5, color=INK_PRIMARY,
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR_ATTACK), plt.Rectangle((0, 0), 1, 1, color=COLOR_NORMAL)]
    ax.legend(handles, ["supports Attack (+)", "pushes towards Normal (−)"],
              loc="lower right", fontsize=8.5, frameon=False, labelcolor=INK_SECONDARY)

    plt.tight_layout()
    path = plot_path_for(model_key, int(sample.validation_position))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def check_output_ids(shared_ids: list[int]) -> list[tuple[str, bool]]:
    """Read the saved outputs back and confirm they contain exactly the shared sample IDs."""
    expected = set(shared_ids)
    results = []

    def record(name: str, ok: bool) -> None:
        results.append((name, ok))
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")

    weights = pd.read_csv(OUTPUT_DIR / "lime_feature_weights_all.csv")
    top = pd.read_csv(OUTPUT_DIR / "lime_top_features.csv")
    summary = pd.read_csv(OUTPUT_DIR / "lime_explanation_summary.csv")

    record(f"shared file has {len(shared_ids)} unique IDs, all found in the validation split", len(expected) == len(shared_ids))
    record(f"explanations generated = {len(shared_ids)} shared IDs x {len(XAI_MODELS)} models",
           len(summary) == len(shared_ids) * len(XAI_MODELS))
    id_sets = {}
    for key in XAI_MODELS:
        id_sets[key] = set(summary.loc[summary["model"] == key, SAMPLE_KEY])
        record(f"{XAI_MODELS[key]}: exactly the shared IDs (no extra, none missing)",
               id_sets[key] == expected and (summary["model"] == key).sum() == len(shared_ids))
    record("Random Forest and XGBoost use the same IDs", len(set(map(frozenset, id_sets.values()))) == 1)
    for name, df in (("lime_feature_weights_all.csv", weights), ("lime_top_features.csv", top)):
        ok = all(set(df.loc[df["model"] == key, SAMPLE_KEY]) == expected for key in XAI_MODELS)
        record(f"{name}: IDs exactly match the shared file for both models", ok)
    record(f"lime_feature_weights_all.csv: {len(FEATURE_ORDER)} features per explanation",
           (weights.groupby(["model", SAMPLE_KEY]).size() == len(FEATURE_ORDER)).all())
    plots = {p.name for p in FIGURE_DIR.glob("*.png")}
    expected_plots = {plot_path_for(key, i).name for key in XAI_MODELS for i in expected}
    record(f"figures: exactly {len(expected_plots)} plots, one per model and shared ID", plots == expected_plots)
    return results


def lime_settings(codec: CategoricalCodec, n_train: int) -> dict:
    return {
        "library": f"lime {version('lime')}",
        "explainer": "lime.lime_tabular.LimeTabularExplainer",
        "mode": "classification",
        "explained_class": f"{EXPLAINED_LABEL} = {CLASS_NAMES[EXPLAINED_LABEL]}",
        "class_names": CLASS_NAMES,
        "random_seed": LIME_SEED,
        "random_state_per_explanation": "numpy RandomState(LIME_SEED + validation_position), shared by explainer, discretizer and ridge surrogate; reset before every explanation",
        "num_samples": NUM_SAMPLES,
        "num_features_estimated": NUM_FEATURES,
        "num_features_shown": TOP_K,
        "feature_selection": FEATURE_SELECTION,
        "discretize_continuous": DISCRETIZE_CONTINUOUS,
        "discretizer": DISCRETIZER,
        "discretizer_fitted_on": f"internal sub-training split ({n_train:,} rows), the data the models were fitted on",
        "sample_around_instance": False,
        "kernel": "exponential (LIME default)",
        "kernel_width": float(KERNEL_WIDTH),
        "distance_metric": "euclidean",
        "surrogate_model": "Ridge(alpha=1.0) (LIME default)",
        "training_data": f"internal sub-training split ({n_train:,} rows), 42 original features, categorical columns integer-coded",
        "feature_space": "original 42 UNSW-NB15 features (before one-hot encoding)",
        "feature_names": codec.feature_names,
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_feature_indices": codec.categorical_idx,
        "categorical_encoding": "integer code = position in the sorted list of training-set categories",
        "num_categories": {c: len(v) for c, v in codec.categories.items()},
        "prediction_function": "Z -> decode integer codes back to category strings -> DataFrame in FEATURE_ORDER -> fitted_pipeline.predict_proba (same preprocessor + model as SHAP)",
        "decision_threshold": DECISION_THRESHOLD,
        "models": {
            key: f"{name}: results/models/{key}_validation_pipeline.joblib (SHAP's pipeline, fitted on the sub-training split)"
            for key, name in XAI_MODELS.items()
        },
        "samples_file": SHARED_SAMPLES_CSV.relative_to(ROOT).as_posix(),
        "sample_id": f"{SAMPLE_KEY} = row position in the internal validation split (X_val), as used by SHAP",
        "same_settings_for_all_models": True,
    }


def write_summary(summary: pd.DataFrame, top: pd.DataFrame, settings: dict, problems: list[str],
                  id_checks: list[tuple[str, bool]]) -> Path:
    lines = [
        "# LIME results - UNSW-NB15 (Random Forest and XGBoost)",
        "",
        f"Samples: the {summary[SAMPLE_KEY].nunique()} malicious validation flows in "
        f"`{settings['samples_file']}` (SHAP's shared sample set). Models: SHAP's validation pipelines. "
        "LIME weights explain P(Attack): **positive = supports the Attack prediction, negative = pushes towards Normal**.",
        "",
        "## Settings",
        "",
        f"- lime: {settings['library']}; seed {LIME_SEED} (per explanation: {LIME_SEED} + validation_position)",
        f"- perturbation samples: {NUM_SAMPLES}; weights estimated for all {NUM_FEATURES} features; top {TOP_K} shown",
        f"- discretizer: {DISCRETIZER} (fitted on the sub-training split); feature_selection: {FEATURE_SELECTION}; "
        f"kernel width {KERNEL_WIDTH:.3f}; surrogate Ridge(alpha=1)",
        "- feature space: original 42 features; proto/service/state as categorical features",
        f"- reproducibility check: {settings['reproducibility_check']}",
        "- full settings: `lime_settings.json`",
        "",
        "## Per-sample results (top 3 features)",
        "",
        "| Model | Validation position | Official id | Attack category | Prediction | P(Attack) | Local R² | Top 3 LIME features (weight) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in summary.itertuples():
        lines.append(
            f"| {XAI_MODELS[r.model]} | {r.validation_position} | {r.official_id} | {r.attack_cat} | {r.predicted_class} | "
            f"{r.prob_attack:.4f} | {r.lime_score:.3f} | {r.top3} |"
        )

    lines += ["", "## Features appearing most often in the top 5", ""]
    for key in XAI_MODELS:
        counts = top[(top["model"] == key) & (top["rank"] <= 5)]["feature"].value_counts().head(8)
        lines.append(f"- **{XAI_MODELS[key]}**: " + ", ".join(f"{f} ({n})" for f, n in counts.items()))

    lines += ["", "## Sample ID checks", ""]
    lines += [f"- {'PASS' if ok else 'FAIL'}: {name}" for name, ok in id_checks]

    lines += [
        "",
        "## Notes for interpreting and comparing with SHAP",
        "",
        "- Join LIME and SHAP results on `validation_position` (and `model`).",
        "- A LIME weight belongs to a *rule* (e.g. `62.00 < sttl <= 254.00`), i.e. the quartile bin the flow "
        "falls in, not to the raw value. Compare with SHAP on the `feature` column of `lime_feature_weights_all.csv`.",
        "- LIME works on the 42 original features. SHAP on the pipeline sees one-hot columns for proto/service/state; "
        "sum those SHAP values per original feature before comparing.",
        "- TTL values (`sttl`, `ct_state_ttl`) are a known UNSW-NB15 testbed artifact, so a large TTL weight "
        "reflects what the models learned, not necessarily real attack behaviour.",
        "- Local surrogate R² is moderate (see table), so LIME is only an approximation of each model near the flow.",
        "- Random Forest probabilities move in steps of 1/200 (200 trees).",
    ]
    lines += ["", "## Errors and problems", ""]
    lines += [f"- {p}" for p in problems] if problems else ["- None. All validation checks passed for every sample."]

    path = OUTPUT_DIR / "lime_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
def main() -> None:
    # Start from an empty output folder so no result from an older sample set can survive.
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    FIGURE_DIR.mkdir(parents=True)

    print("Step 1: loading the official split, SHAP's fitted models and the SHAP shared samples")
    split = get_xai_data()
    X_subtrain, X_val, y_val = split["X_subtrain"], split["X_val"], split["y_val"]
    pipelines = {key: load_xai_model(key, split) for key in XAI_MODELS}
    samples = load_shared_samples(split)
    shared_ids = samples[SAMPLE_KEY].tolist()
    shap_probs = {key: load_shap_reference_probabilities(key) for key in XAI_MODELS}

    print("\nStep 2: building the LIME explainer on the sub-training split")
    codec = CategoricalCodec(X_subtrain)
    explainer = build_explainer(codec.encode(X_subtrain), codec)
    check_global_feature_order(pipelines, codec, X_val)
    settings = lime_settings(codec, len(X_subtrain))

    print("\nStep 3: validating and explaining every sample")
    all_rows, summary_rows, problems = [], [], []
    for key, pipeline in pipelines.items():
        predict_fn = make_predict_fn(pipeline, codec)
        for sample in samples.itertuples():
            pos = int(sample.validation_position)
            print(f"\n[{XAI_MODELS[key]}] validation sample {pos} (official id {sample.official_id}, {sample.attack_cat})")
            x_row = X_val.iloc[[pos]]
            ref = shap_probs[key]
            expected_prob = float(ref[pos]) if ref is not None and pos in ref.index else None
            try:
                check = validate_sample(
                    key, pipeline, codec, x_row, int(y_val.iloc[pos]), sample.attack_cat, expected_prob, predict_fn,
                )
            except ValueError as err:
                print(f"    VALIDATION FAILED: {err}")
                problems.append(f"{XAI_MODELS[key]}, validation sample {pos}: skipped - {err}")
                continue

            seed = LIME_SEED + pos
            exp = explain(explainer, check["encoded"], predict_fn, seed)
            rows = explanation_rows(exp, key, sample, x_row)
            plot_path = save_plot(rows, key, sample, check["prob_attack"], exp)
            all_rows.extend(rows)

            local_pred = float(np.ravel(exp.local_pred)[0])
            summary_rows.append({
                "model": key,
                SAMPLE_KEY: pos,
                "original_training_row_index": int(sample.original_training_row_index),
                "official_id": int(sample.official_id),
                "attack_cat": sample.attack_cat,
                "predicted_class": check["predicted_class"],
                "prob_attack": check["prob_attack"],
                "lime_score": float(exp.score),
                "lime_local_pred": local_pred,
                "lime_intercept": float(exp.intercept[EXPLAINED_LABEL]),
                "lime_seed": seed,
                "top3": "; ".join(f"{r['feature']} ({r['lime_weight']:+.3f})" for r in rows[:3]),
                "plot": plot_path.relative_to(ROOT).as_posix(),
            })
            for r in rows[:TOP_K]:
                print(f"    {r['rank']:>2}. {r['lime_rule']:<40} {r['lime_weight']:+.4f}  {r['direction']}")
            print(f"    local surrogate R^2 = {exp.score:.3f}, surrogate P(Attack) = {local_pred:.4f}")

    print("\nStep 4: reproducibility check (re-explain the first sample with the same seed)")
    first = int(samples.iloc[0][SAMPLE_KEY])
    for key, pipeline in pipelines.items():
        predict_fn = make_predict_fn(pipeline, codec)
        encoded = codec.encode(X_val.iloc[[first]])[0]
        again = dict(explain(explainer, encoded, predict_fn, LIME_SEED + first).as_map()[EXPLAINED_LABEL])
        before = {FEATURE_ORDER.index(r["feature"]): r["lime_weight"] for r in all_rows
                  if r["model"] == key and r[SAMPLE_KEY] == first}
        if not before:
            print(f"    {XAI_MODELS[key]}: not run (sample {first} was skipped)")
            problems.append(f"{XAI_MODELS[key]}: reproducibility check not run, sample {first} was skipped")
            continue
        same = before.keys() == again.keys() and all(np.isclose(before[i], again[i], rtol=0, atol=1e-12) for i in before)
        print(f"    {XAI_MODELS[key]}: {'identical weights' if same else 'DIFFERENT weights'}")
        if not same:
            problems.append(f"{XAI_MODELS[key]}: re-running sample {first} with the same seed gave different weights")
    settings["reproducibility_check"] = (
        "passed" if not any("same seed" in p or "check not run" in p for p in problems) else "failed or not run"
    )

    print("\nStep 5: saving results")
    weights_df = pd.DataFrame(all_rows)
    top_df = weights_df[weights_df["in_top_k"]].drop(columns="in_top_k")
    summary_df = pd.DataFrame(summary_rows)

    outputs = {
        "lime_feature_weights_all.csv": weights_df,
        "lime_top_features.csv": top_df,
        "lime_explanation_summary.csv": summary_df,
    }
    for name, df in outputs.items():
        # 12 significant digits: removes ~1e-16 run-to-run noise from Random Forest's parallel
        # predict_proba, so repeated runs write byte-identical files.
        df.to_csv(OUTPUT_DIR / name, index=False, float_format="%.12g")
        print(f"Saved: {OUTPUT_DIR / name}")

    print("\nStep 6: checking that the saved outputs contain exactly the shared sample IDs")
    id_checks = check_output_ids(shared_ids)
    failed = [name for name, ok in id_checks if not ok]
    problems += [f"sample ID check failed: {name}" for name in failed]
    settings["sample_id_checks"] = {name: ("passed" if ok else "failed") for name, ok in id_checks}

    (OUTPUT_DIR / "lime_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUTPUT_DIR / 'lime_settings.json'}")
    print(f"Saved: {write_summary(summary_df, top_df, settings, problems, id_checks)}")
    print(f"Saved {len(summary_df)} plots in {FIGURE_DIR}")

    if problems:
        print("\nProblems encountered:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("\nAll validation, sample ID and reproducibility checks passed.")


if __name__ == "__main__":
    main()
