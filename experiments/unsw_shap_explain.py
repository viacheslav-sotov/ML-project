from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_MODELS_DIR = ROOT / "results" / "models"
RESULTS_SHAP_DIR = ROOT / "results" / "metrics" / "shap"
RESULTS_FIGURES_SHAP_DIR = ROOT / "results" / "figures" / "shap"

from src.unsw.data import build_preprocessor, create_data_split, load_official_data
from src.unsw.models import get_initial_models

EXPECTED_F1 = {
    "random_forest": 0.9710,
    "xgboost": 0.9692,
}


def ensure_directories() -> None:
    RESULTS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_SHAP_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES_SHAP_DIR.mkdir(parents=True, exist_ok=True)


def build_pipeline_for_model(model_name: str, X_train: pd.DataFrame) -> Pipeline:
    estimator = get_initial_models()[model_name]
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", clone(estimator)),
        ]
    )


def make_split_signature(split: dict[str, object]) -> dict[str, object]:
    X_train_full = split["X_train_full"]
    X_val = split["X_val"]
    return {
        "train_rows": int(len(X_train_full)),
        "val_rows": int(len(X_val)),
        "feature_names": [str(name) for name in X_train_full.columns],
        "random_state": 42,
        "test_size": 0.20,
        "stratified": True,
    }


def make_model_signature(model_name: str, split: dict[str, object]) -> str:
    estimator = get_initial_models()[model_name]
    signature = {
        "model_name": model_name,
        "split": make_split_signature(split),
        "estimator_params": estimator.get_params(),
        "preprocessor": "ColumnTransformer(OneHotEncoder(handle_unknown='ignore', sparse_output=False), passthrough)",
    }
    serialized = json.dumps(signature, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()


def load_or_train_pipeline(model_name: str, split: dict[str, object]) -> tuple[Pipeline, dict[str, object], Path, str]:
    X_subtrain = split["X_subtrain"]
    y_subtrain = split["y_subtrain"]
    artifact_path = RESULTS_MODELS_DIR / f"{model_name}_validation_pipeline.joblib"
    desired_signature = make_model_signature(model_name, split)

    if artifact_path.exists():
        existing_bundle = joblib.load(artifact_path)
        existing_signature = existing_bundle.get("signature")
        if existing_signature == desired_signature:
            pipeline = existing_bundle["pipeline"]
            print(f"Loaded existing {model_name} pipeline from {artifact_path}")
            return pipeline, existing_bundle["metadata"], artifact_path, "loaded"
        print(
            f"Existing {model_name} artifact at {artifact_path} does not match the current split/settings; retraining."
        )

    pipeline = build_pipeline_for_model(model_name, X_subtrain)
    start_fit = time.perf_counter()
    pipeline.fit(X_subtrain, y_subtrain)
    training_time = time.perf_counter() - start_fit

    metadata = {
        "model_name": model_name,
        "random_state": 42,
        "split": make_split_signature(split),
        "model_signature": desired_signature,
        "training_time_seconds": float(training_time),
        "train_rows": int(len(X_subtrain)),
        "val_rows": int(len(split["X_val"])),
        "preprocessor_type": "ColumnTransformer",
        "feature_count": int(X_subtrain.shape[1]),
    }
    bundle = {"pipeline": pipeline, "signature": desired_signature, "metadata": metadata}
    joblib.dump(bundle, artifact_path)
    print(f"Saved trained {model_name} pipeline to {artifact_path}")
    return pipeline, metadata, artifact_path, "trained"


def verify_model_against_saved_results(model_name: str, pipeline: Pipeline, split: dict[str, object]) -> float:
    X_val = split["X_val"]
    y_val = split["y_val"]
    val_prob = pipeline.predict_proba(X_val)[:, 1]
    y_pred = (val_prob >= 0.5).astype(int)
    f1 = f1_score(y_val, y_pred)
    target = EXPECTED_F1[model_name]
    tolerance = 0.025
    if abs(f1 - target) > tolerance:
        print(
            f"WARNING: {model_name} F1={f1:.6f} differs from saved validation result {target:.6f} by {abs(f1-target):.6f}; "
            "no settings were changed automatically"
        )
    else:
        print(f"Verified {model_name} F1={f1:.6f} is consistent with the saved validation result {target:.6f}.")
    return float(f1)


def get_correct_malicious_candidate_positions(model_name: str, pipeline: Pipeline, split: dict[str, object]) -> np.ndarray:
    X_val = split["X_val"]
    y_val = split["y_val"]
    val_prob = pipeline.predict_proba(X_val)[:, 1]
    y_pred = (val_prob >= 0.5).astype(int)
    attack_mask = (y_val.to_numpy() == 1) & (y_pred == 1)
    candidate_positions = np.flatnonzero(attack_mask)
    if len(candidate_positions) == 0:
        raise ValueError(f"No correctly classified malicious validation samples were found for {model_name}.")
    return candidate_positions


def choose_shared_malicious_positions(
    model_names: list[str],
    pipelines: dict[str, Pipeline],
    split: dict[str, object],
    seed: int = 42,
    sample_count: int = 20,
) -> np.ndarray:
    candidate_sets = {
        model_name: set(get_correct_malicious_candidate_positions(model_name, pipelines[model_name], split).tolist())
        for model_name in model_names
    }
    shared_positions = sorted(set.intersection(*[candidate_sets[name] for name in model_names]))
    if len(shared_positions) < sample_count:
        raise ValueError(
            f"Not enough shared correctly classified malicious validation samples across models: "
            f"found {len(shared_positions)}, needed {sample_count}."
        )

    rng = np.random.default_rng(seed)
    selected_positions = np.sort(rng.choice(np.asarray(shared_positions, dtype=int), size=sample_count, replace=False))

    shared_df = pd.DataFrame(
        {
            "validation_position": selected_positions,
            "original_training_row_index": np.asarray(split["X_val"].index[selected_positions], dtype=int),
            "seed": seed,
            "source_models": ",".join(model_names),
        }
    )
    shared_path = RESULTS_SHAP_DIR / "shared_validation_sample_identifiers.csv"
    shared_df.to_csv(shared_path, index=False)
    print(f"Saved canonical shared validation sample set to {shared_path}: {selected_positions.tolist()}")
    return selected_positions


def select_correct_malicious_samples(
    model_name: str,
    pipeline: Pipeline,
    split: dict[str, object],
    seed: int = 42,
    sample_count: int = 20,
    override_positions: np.ndarray | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    X_val = split["X_val"]
    y_val = split["y_val"]
    val_prob = pipeline.predict_proba(X_val)[:, 1]
    y_pred = (val_prob >= 0.5).astype(int)
    attack_mask = (y_val.to_numpy() == 1) & (y_pred == 1)
    candidate_positions = np.flatnonzero(attack_mask)
    if override_positions is not None:
        selected_positions = np.asarray(override_positions, dtype=int)
        missing = sorted(set(selected_positions.tolist()) - set(candidate_positions.tolist()))
        if missing:
            raise ValueError(
                f"The canonical shared sample set contains validation positions not in {model_name}'s malicious candidate pool: {missing[:10]}"
            )
    else:
        if len(candidate_positions) < sample_count:
            raise ValueError(
                f"Not enough correctly classified malicious validation samples for {model_name}: "
                f"found {len(candidate_positions)}, needed {sample_count}"
            )
        rng = np.random.default_rng(seed)
        selected_positions = rng.choice(candidate_positions, size=sample_count, replace=False)
        selected_positions = np.sort(selected_positions)

    val_lookup = X_val.index
    train_row_indices = np.asarray(val_lookup[selected_positions], dtype=int)
    true_labels = y_val.iloc[selected_positions].to_numpy(dtype=int)
    predicted_labels = y_pred[selected_positions]
    predicted_probs = val_prob[selected_positions]

    official_train_df, _ = load_official_data()
    attack_cat_values = []
    for train_row_index in train_row_indices:
        row = official_train_df.loc[train_row_index]
        attack_cat_values.append(row.get("attack_cat", np.nan))

    selected_df = pd.DataFrame(
        {
            "model": model_name,
            "validation_position": selected_positions,
            "original_training_row_index": train_row_indices,
            "true_label": true_labels,
            "predicted_label": predicted_labels,
            "predicted_probability": predicted_probs,
            "attack_cat": attack_cat_values,
            "seed": seed,
        }
    )
    return selected_positions, selected_df


def extract_class_one_values(values: np.ndarray | list[np.ndarray]) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        return arr[:, :, 1]
    if arr.ndim == 2:
        return arr
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    raise ValueError(f"Unsupported SHAP value shape for class-1 extraction: {arr.shape}")


def normalize_base_values(base_values: np.ndarray | list[np.ndarray]) -> np.ndarray:
    arr = np.asarray(base_values)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr[:, 1]
    if arr.ndim == 1:
        return arr
    raise ValueError(f"Unsupported SHAP base value shape: {arr.shape}")


def determine_output_scale(model_name: str, values: np.ndarray, base_values: np.ndarray, probability_reference: np.ndarray) -> tuple[str, str, np.ndarray]:
    if values.ndim == 3 and values.shape[-1] == 2:
        reconstruction = values[:, :, 1].sum(axis=1) + base_values[:, 1]
        return (
            "probability",
            "probability scale for attack class (label=1)",
            np.asarray(reconstruction),
        )

    reconstruction = values.sum(axis=1) + base_values
    logit_reference = np.log(np.clip(probability_reference, 1e-9, 1 - 1e-9) / (1 - np.clip(probability_reference, 1e-9, 1 - 1e-9)))
    return (
        "raw_margin",
        "raw model-output scale (logit/margin) for attack class (label=1); probabilities are sigmoid(raw_output)",
        np.asarray(reconstruction),
    )


def validate_shap_for_model(
    model_name: str,
    model: object,
    X_selected_pre: np.ndarray,
    feature_names: list[str],
    shap_explanation: object,
    expected_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    raw_values = np.asarray(shap_explanation.values)
    raw_base_values = np.asarray(shap_explanation.base_values)
    values = extract_class_one_values(raw_values)
    base_values = normalize_base_values(raw_base_values)

    if values.shape[0] != X_selected_pre.shape[0]:
        raise ValueError(
            f"{model_name} SHAP row count mismatch: values {values.shape[0]} vs selected samples {X_selected_pre.shape[0]}"
        )
    if values.shape[1] != len(feature_names):
        raise ValueError(
            f"{model_name} SHAP feature count mismatch: values {values.shape[1]} vs feature names {len(feature_names)}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{model_name} SHAP values contain non-finite entries")

    shaps_feature_names = getattr(shap_explanation, "feature_names", None)
    if shaps_feature_names is not None:
        if list(shaps_feature_names) != feature_names:
            print(
                f"WARNING: {model_name} SHAP feature order differs from transformed feature names; "
                f"SHAP names={list(shaps_feature_names)[:5]} vs preprocessor names={feature_names[:5]}"
            )

    output_kind, output_label, reconstruction = determine_output_scale(model_name, raw_values, raw_base_values, expected_probability)
    reference = expected_probability if output_kind == "probability" else np.log(
        np.clip(expected_probability, 1e-9, 1 - 1e-9) / (1 - np.clip(expected_probability, 1e-9, 1 - 1e-9))
    )
    gap = float(np.max(np.abs(reconstruction - reference)))
    if gap > 1e-2:
        print(
            f"WARNING: {model_name} additivity gap is {gap:.6f}. SHAP output scale is {output_kind}, and the reconstructed output should match "
            f"{output_label}."
        )
    else:
        print(
            f"Validated {model_name} SHAP additivity on the {output_kind} scale; max reconstruction gap={gap:.6e}."
        )

    return values, base_values, reconstruction, output_kind, output_label


def explain_selected_samples(
    model_name: str,
    pipeline: Pipeline,
    selected_positions: np.ndarray,
    split: dict[str, object],
    seed: int = 42,
) -> tuple[pd.DataFrame, str, np.ndarray, np.ndarray, np.ndarray, str, str, dict[str, object]]:
    X_val = split["X_val"]
    y_val = split["y_val"]
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    explainable_model = estimator.get_booster() if hasattr(estimator, "get_booster") else estimator

    X_val_pre = preprocessor.transform(X_val)
    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    X_selected_pre = X_val_pre[selected_positions]
    expected_probability = estimator.predict_proba(X_selected_pre)[:, 1]

    background_size = min(100, X_val_pre.shape[0])
    rng = np.random.default_rng(seed)
    background_index = rng.choice(np.arange(X_val_pre.shape[0]), size=background_size, replace=False)
    background_data = X_val_pre[background_index]

    if model_name == "xgboost":
        explainer = shap.TreeExplainer(
            explainable_model,
            data=background_data,
            feature_perturbation="tree_path_dependent",
            model_output="raw",
        )
    else:
        explainer = shap.TreeExplainer(estimator, data=background_data)

    shap_explanation = explainer(X_selected_pre)
    class_values, base_values, reconstruction, output_kind, output_label = validate_shap_for_model(
        model_name,
        estimator,
        X_selected_pre,
        feature_names,
        shap_explanation,
        expected_probability,
    )

    shap_df = pd.DataFrame(class_values, columns=feature_names)
    shap_df.insert(0, "model", model_name)
    shap_df.insert(1, "validation_position", selected_positions)
    shap_df.insert(2, "true_label", y_val.iloc[selected_positions].to_numpy(dtype=int))
    shap_df.insert(3, "predicted_probability", expected_probability)
    shap_df.insert(4, "base_value", base_values)
    shap_df.insert(5, "reconstructed_model_output", reconstruction)
    shap_df.insert(6, "output_kind", output_kind)
    shap_df.insert(7, "output_label", output_label)

    shap_path = RESULTS_SHAP_DIR / f"{model_name}_selected_shap_values.csv"
    shap_df.to_csv(shap_path, index=False)
    print(f"Saved {model_name} SHAP values to {shap_path}")

    metadata = {
        "model": model_name,
        "feature_count": len(feature_names),
        "sample_count": X_selected_pre.shape[0],
        "shape": list(class_values.shape),
        "units": output_label,
        "output_kind": output_kind,
        "background_size": background_size,
        "shap_file": str(shap_path),
        "output_label": output_label,
        "selected_validation_positions": selected_positions.tolist(),
        "feature_names": feature_names,
    }

    return (
        shap_df,
        str(shap_path),
        class_values,
        base_values,
        reconstruction,
        output_kind,
        output_label,
        metadata,
    )


def _save_readable_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    try:
        import PIL.Image

        with PIL.Image.open(output_path) as img:
            img.verify()
        print(f"Verified readable image: {output_path}")
    except Exception as exc:  # pragma: no cover
        print(f"WARNING: Could not verify readability of {output_path}: {exc}")


def save_shap_plot_metadata(metadata_by_model: dict[str, dict[str, object]]) -> None:
    metadata_path = RESULTS_SHAP_DIR / "shap_plot_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata_by_model, fh, indent=2, sort_keys=True, default=str)
    print(f"Saved SHAP plot metadata to {metadata_path}")


def plot_global_feature_importance(
    model_name: str,
    feature_names: list[str],
    values: np.ndarray,
    output_label: str,
) -> Path:
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    top_features = [feature_names[i] for i in order[:10]]
    top_values = [float(mean_abs[i]) for i in order[:10]]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(top_features[::-1], top_values[::-1], color="steelblue")
    ax.set_title(
        f"{model_name.replace('_', ' ').title()} global SHAP feature importance\n"
        f"(subset: 20 correctly classified malicious validation samples; {output_label})"
    )
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("Transformed feature")
    plt.tight_layout()
    output_path = RESULTS_FIGURES_SHAP_DIR / f"{model_name}_global_shap_importance.png"
    _save_readable_figure(fig, output_path)
    print(f"Saved {model_name} global SHAP bar plot to {output_path}")
    return output_path


def plot_beeswarm(
    model_name: str,
    feature_names: list[str],
    values: np.ndarray,
    data: np.ndarray,
    base_values: np.ndarray,
    output_label: str,
) -> Path:
    explanation = shap.Explanation(
        values=values,
        base_values=base_values,
        data=data,
        feature_names=feature_names,
    )
    fig = plt.figure(figsize=(14, 10))
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    fig = plt.gcf()
    fig.suptitle(
        f"{model_name.replace('_', ' ').title()} SHAP beeswarm\n"
        f"(20 malicious validation samples; feature values vs contribution; {output_label})"
    )
    fig.tight_layout()
    output_path = RESULTS_FIGURES_SHAP_DIR / f"{model_name}_shap_beeswarm.png"
    _save_readable_figure(fig, output_path)
    print(f"Saved {model_name} SHAP beeswarm plot to {output_path}")
    return output_path


def plot_waterfalls(
    model_name: str,
    feature_names: list[str],
    values: np.ndarray,
    data: np.ndarray,
    base_values: np.ndarray,
    output_label: str,
    selected_positions: list[int],
) -> list[Path]:
    output_paths: list[Path] = []
    for idx, position in enumerate(selected_positions[:3]):
        sample_values = values[idx : idx + 1, :]
        sample_data = data[idx : idx + 1, :]
        sample_base = np.asarray([base_values[idx]])
        explanation = shap.Explanation(
            values=sample_values,
            base_values=sample_base,
            data=sample_data,
            feature_names=feature_names,
        )
        fig = plt.figure(figsize=(12, 8))
        shap.plots.waterfall(explanation[0], max_display=15, show=False)
        fig = plt.gcf()
        fig.suptitle(
            f"{model_name.replace('_', ' ').title()} SHAP waterfall for suspicious validation sample #{position}\n"
            f"({output_label})"
        )
        fig.tight_layout()
        output_path = RESULTS_FIGURES_SHAP_DIR / f"{model_name}_waterfall_sample_{position}.png"
        _save_readable_figure(fig, output_path)
        output_paths.append(output_path)
        print(f"Saved {model_name} waterfall plot to {output_path}")
    return output_paths


def generate_shap_plots_for_model(
    model_name: str,
    split: dict[str, object],
    pipeline: Pipeline,
    selected_positions: np.ndarray,
) -> dict[str, object]:
    selected_df = pd.read_csv(RESULTS_SHAP_DIR / f"{model_name}_selected_sample_identifiers.csv")
    shap_df = pd.read_csv(RESULTS_SHAP_DIR / f"{model_name}_selected_shap_values.csv")
    feature_names = [
        col
        for col in shap_df.columns
        if col not in {"model", "validation_position", "true_label", "predicted_probability", "base_value", "reconstructed_model_output", "output_kind", "output_label"}
    ]
    values = shap_df[feature_names].to_numpy(dtype=float)
    base_values = shap_df["base_value"].to_numpy(dtype=float)
    output_kind = shap_df["output_kind"].iloc[0] if "output_kind" in shap_df.columns else "probability"
    output_label = shap_df["output_label"].iloc[0] if "output_label" in shap_df.columns else output_kind

    X_val = split["X_val"]
    X_val_pre = pipeline.named_steps["preprocessor"].transform(X_val)
    selected_pre = X_val_pre[selected_positions]

    feature_plot_path = plot_global_feature_importance(model_name, feature_names, values, output_label)
    beeswarm_path = plot_beeswarm(model_name, feature_names, values, selected_pre, base_values, output_label)
    waterfall_paths = plot_waterfalls(model_name, feature_names, values, selected_pre, base_values, output_label, selected_positions.tolist())

    model_summary = {
        "model": model_name,
        "selected_validation_positions": selected_positions.tolist(),
        "selected_sample_count": int(len(selected_positions)),
        "output_kind": output_kind,
        "output_label": output_label,
        "feature_count": len(feature_names),
        "global_feature_importance": feature_plot_path.as_posix(),
        "beeswarm": beeswarm_path.as_posix(),
        "waterfalls": [str(path) for path in waterfall_paths],
        "top_features": [
            {"feature": feature_names[i], "mean_abs_shap": float(np.abs(values[:, i]).mean())}
            for i in np.argsort(np.abs(values).mean(axis=0))[::-1][:5]
        ],
    }
    print(f"Generated {model_name} SHAP plots and metadata.")
    return model_summary


def main() -> None:
    ensure_directories()
    start_total = time.perf_counter()
    split = create_data_split()
    model_names = ["random_forest", "xgboost"]
    model_summaries: dict[str, dict[str, object]] = {}
    pipelines: dict[str, Pipeline] = {}

    for model_name in model_names:
        pipeline, _, artifact_path, _ = load_or_train_pipeline(model_name, split)
        pipelines[model_name] = pipeline
        f1_score_value = verify_model_against_saved_results(model_name, pipeline, split)
        model_summaries[model_name] = {"model": model_name, "artifact_path": str(artifact_path), "f1": float(f1_score_value)}

    shared_positions = choose_shared_malicious_positions(model_names, pipelines, split, seed=42, sample_count=20)

    for model_name in model_names:
        pipeline = pipelines[model_name]
        selected_positions, selected_df = select_correct_malicious_samples(
            model_name,
            pipeline,
            split,
            seed=42,
            sample_count=20,
            override_positions=shared_positions,
        )
        selected_path = RESULTS_SHAP_DIR / f"{model_name}_selected_sample_identifiers.csv"
        selected_df.to_csv(selected_path, index=False)
        print(f"Saved {model_name} selected sample IDs to {selected_path}")

        shap_df, shap_path, class_values, base_values, reconstruction, output_kind, output_label, metadata = explain_selected_samples(
            model_name,
            pipeline,
            selected_positions,
            split,
            seed=42,
        )

        model_summaries[model_name].update(
            {
                "selected_samples": int(len(selected_positions)),
                "transformed_features": int(shap_df.shape[1] - 7),
                "shap_shape": str(class_values.shape),
                "output_kind": output_kind,
                "output_label": output_label,
                "shap_file": shap_path,
                "selection_file": str(selected_path),
                "plot_summary": generate_shap_plots_for_model(model_name, split, pipeline, selected_positions),
            }
        )

    save_shap_plot_metadata(model_summaries)

    elapsed = time.perf_counter() - start_total
    print("\n=== Final SHAP analysis summary ===")
    for model_name in model_names:
        summary = model_summaries[model_name]
        print(
            f"{model_name}: selected={summary['selected_samples']}, transformed_features={summary['transformed_features']}, "
            f"F1={summary['f1']:.6f}, SHAP_shape={summary['shap_shape']}, output_kind={summary['output_kind']}, "
            f"output_label={summary['output_label']}, artifact={summary['artifact_path']}"
        )
        top_features = summary["plot_summary"]["top_features"]
        print(f"Top five SHAP features for {model_name}: {top_features}")
    print(f"Total runtime: {elapsed:.2f}s")
    print("Done. No official test set was used.")


if __name__ == "__main__":
    main()
