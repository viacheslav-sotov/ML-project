from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.unsw_lime import (
    CategoricalCodec,
    EXPLAINED_LABEL,
    NUM_FEATURES,
    NUM_SAMPLES,
    build_explainer,
    make_predict_fn,
    reseed_explainer,
    validate_sample,
)
from src.unsw.constraints import FEATURE_ORDER
from src.unsw.xai_setup import (
    SAMPLE_KEY,
    SHARED_SAMPLES_CSV,
    XAI_MODELS,
    get_xai_data,
    load_shap_reference_probabilities,
    load_shared_samples,
    load_xai_model,
)

OUTPUT_DIR = ROOT / "results" / "lime_stability"
SEEDS = [11, 22, 33, 44, 55]


def ensure_safe_output_dir() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"{OUTPUT_DIR} already exists. This script will not overwrite existing stability artifacts. "
            "Please move or delete the directory before rerunning."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)


def check_model_and_sample_compatibility() -> tuple[dict[str, object], pd.DataFrame, dict[str, pd.Series | None]]:
    split = get_xai_data()
    pipelines = {key: load_xai_model(key, split) for key in XAI_MODELS}
    samples = load_shared_samples(split)
    if len(samples) != 20:
        raise ValueError(f"Expected 20 shared validation samples but found {len(samples)}.")

    shap_probs = {key: load_shap_reference_probabilities(key) for key in XAI_MODELS}
    print(f"Loaded {len(samples)} shared validation samples from {SHARED_SAMPLES_CSV.relative_to(ROOT).as_posix()}.")
    for key in XAI_MODELS:
        if shap_probs[key] is None:
            raise ValueError(f"Missing SHAP reference probabilities for model '{key}' required for compatibility checks.")
    return pipelines, samples, shap_probs


def explanation_ranking(exp) -> list[tuple[str, float, int]]:
    raw = exp.as_map()[EXPLAINED_LABEL]
    weights = {FEATURE_ORDER[idx]: float(weight) for idx, weight in raw}
    ranked = sorted(weights.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
    return [(feature, weight, rank + 1) for rank, (feature, weight) in enumerate(ranked)]


def compute_pairwise_metrics(run_df: pd.DataFrame, model_key: str, sample_id: int) -> list[dict]:
    seed_to_weights = {}
    for seed in SEEDS:
        subset = run_df[
            (run_df["model"] == model_key)
            & (run_df["validation_position"] == sample_id)
            & (run_df["seed"] == seed)
        ].copy()
        if subset.empty:
            raise ValueError(f"No rows for model={model_key}, sample={sample_id}, seed={seed}")
        seed_to_weights[seed] = subset.set_index("feature")["signed_weight"].to_dict()

    pairwise = []
    for seed_a, seed_b in itertools.combinations(SEEDS, 2):
        a = seed_to_weights[seed_a]
        b = seed_to_weights[seed_b]

        def topk_features(weights: dict[str, float], k: int) -> set[str]:
            return set(sorted(weights, key=lambda feat: (-abs(weights[feat]), feat))[:k])

        top1_a = topk_features(a, 1)
        top1_b = topk_features(b, 1)
        top5_a = topk_features(a, 5)
        top5_b = topk_features(b, 5)
        top10_a = topk_features(a, 10)
        top10_b = topk_features(b, 10)

        top1_agreement = 1.0 if top1_a == top1_b else 0.0
        top5_overlap = top5_a & top5_b
        top5_union = top5_a | top5_b
        top5_jaccard = len(top5_overlap) / len(top5_union) if top5_union else 0.0
        top10_overlap = top10_a & top10_b
        top10_union = top10_a | top10_b
        top10_jaccard = len(top10_overlap) / len(top10_union) if top10_union else 0.0

        rank_a = pd.Series({feat: idx + 1 for idx, feat in enumerate(sorted(a, key=lambda feat: (-abs(a[feat]), feat)))})
        rank_b = pd.Series({feat: idx + 1 for idx, feat in enumerate(sorted(b, key=lambda feat: (-abs(b[feat]), feat)))})
        common_features = sorted(set(rank_a.index) & set(rank_b.index))
        if len(common_features) >= 2:
            rho = float(np.corrcoef(rank_a.loc[common_features].to_numpy(), rank_b.loc[common_features].to_numpy())[0, 1])
            rho = 0.0 if np.isnan(rho) else rho
        else:
            rho = 0.0

        shared_top10 = sorted(top10_a & top10_b)
        if shared_top10:
            direction_hits = sum(1 for feat in shared_top10 if np.sign(a[feat]) == np.sign(b[feat]))
            direction_agreement = direction_hits / len(shared_top10)
        else:
            direction_agreement = 0.0

        pairwise.append(
            {
                "model": model_key,
                "validation_position": int(sample_id),
                "seed_a": int(seed_a),
                "seed_b": int(seed_b),
                "top1_agreement": float(top1_agreement),
                "top5_overlap_count": int(len(top5_overlap)),
                "top5_jaccard": float(top5_jaccard),
                "top10_overlap_count": int(len(top10_overlap)),
                "top10_jaccard": float(top10_jaccard),
                "spearman_rho": float(rho),
                "direction_agreement": float(direction_agreement),
                "shared_top10_features": int(len(shared_top10)),
            }
        )
    return pairwise


def build_stability_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    split = get_xai_data()
    pipelines, samples, shap_probs = check_model_and_sample_compatibility()
    X_subtrain, X_val = split["X_subtrain"], split["X_val"]
    codec = CategoricalCodec(X_subtrain)

    run_rows: list[dict] = []
    pairwise_rows: list[dict] = []
    per_sample_rows: list[dict] = []
    model_probabilities: dict[str, dict[int, float]] = {model: {} for model in XAI_MODELS}

    for model_key, pipeline in pipelines.items():
        predict_fn = make_predict_fn(pipeline, codec)
        expected_probs = shap_probs[model_key]
        for sample in samples.itertuples(index=False):
            pos = int(sample.validation_position)
            x_row = X_val.iloc[[pos]]
            ref_prob = float(expected_probs[pos]) if pos in expected_probs.index else None
            base = validate_sample(
                model_key,
                pipeline,
                codec,
                x_row,
                int(split["y_val"].iloc[pos]),
                str(sample.attack_cat),
                ref_prob,
                predict_fn,
            )
            prob_attack = float(base["prob_attack"])
            model_probabilities[model_key][pos] = prob_attack
            for seed in SEEDS:
                explainer = build_explainer(codec.encode(X_subtrain), codec)
                reseed_explainer(explainer, seed)
                encoded_row = np.asarray(base["encoded"], dtype=np.float64)
                exp = explainer.explain_instance(
                    data_row=encoded_row,
                    predict_fn=predict_fn,
                    labels=(EXPLAINED_LABEL,),
                    num_features=NUM_FEATURES,
                    num_samples=NUM_SAMPLES,
                    distance_metric="euclidean",
                    model_regressor=None,
                )
                ranked = explanation_ranking(exp)
                for feature_name, signed_weight, rank in ranked:
                    run_rows.append(
                        {
                            "model": model_key,
                            "validation_position": int(pos),
                            "original_training_row_index": int(sample.original_training_row_index),
                            "official_id": int(sample.official_id),
                            "attack_cat": str(sample.attack_cat),
                            "seed": int(seed),
                            "feature": feature_name,
                            "signed_weight": float(signed_weight),
                            "abs_weight": float(abs(signed_weight)),
                            "rank": int(rank),
                            "top5": bool(rank <= 5),
                            "top10": bool(rank <= 10),
                            "direction": "supports_attack" if signed_weight > 0 else "pushes_towards_normal" if signed_weight < 0 else "neutral",
                            "model_probability": float(prob_attack),
                        }
                    )

    run_df = pd.DataFrame(run_rows)
    if run_df.empty:
        raise ValueError("No LIME stability runs were generated.")

    expected_run_rows = len(XAI_MODELS) * len(samples) * len(SEEDS) * len(FEATURE_ORDER)
    if len(run_df) != expected_run_rows:
        raise ValueError(f"Unexpected run-row count: {len(run_df)} != {expected_run_rows}.")

    for model_key in XAI_MODELS:
        for sample in samples.itertuples(index=False):
            pos = int(sample.validation_position)
            pair_rows = compute_pairwise_metrics(run_df, model_key, pos)
            pairwise_rows.extend(pair_rows)
            if len(pair_rows) != 10:
                raise ValueError(f"Expected 10 pairwise seed comparisons for {model_key} sample {pos}, got {len(pair_rows)}.")

    pairwise_df = pd.DataFrame(pairwise_rows)
    if len(pairwise_df) != len(XAI_MODELS) * len(samples) * 10:
        raise ValueError(f"Unexpected pairwise comparison count: {len(pairwise_df)} != {len(XAI_MODELS) * len(samples) * 10}.")

    for model_key in XAI_MODELS:
        for sample in samples.itertuples(index=False):
            pos = int(sample.validation_position)
            subset = pairwise_df[
                (pairwise_df["model"] == model_key) & (pairwise_df["validation_position"] == pos)
            ].copy()
            if subset.empty:
                raise ValueError(f"No pairwise rows for {model_key} sample {pos}.")
            per_sample_rows.append(
                {
                    "model": model_key,
                    "validation_position": int(pos),
                    "original_training_row_index": int(sample.original_training_row_index),
                    "official_id": int(sample.official_id),
                    "attack_cat": str(sample.attack_cat),
                    "avg_top1_agreement": float(subset["top1_agreement"].mean()),
                    "avg_top5_overlap_count": float(subset["top5_overlap_count"].mean()),
                    "avg_top5_jaccard": float(subset["top5_jaccard"].mean()),
                    "avg_top10_overlap_count": float(subset["top10_overlap_count"].mean()),
                    "avg_top10_jaccard": float(subset["top10_jaccard"].mean()),
                    "avg_spearman_rho": float(subset["spearman_rho"].mean()),
                    "avg_direction_agreement": float(subset["direction_agreement"].mean()),
                    "probability_mean": float(model_probabilities[model_key][pos]),
                    "probability_std": 0.0,
                    "probability_min": float(model_probabilities[model_key][pos]),
                    "probability_max": float(model_probabilities[model_key][pos]),
                    "seed_count": int(len(SEEDS)),
                }
            )

    per_sample_df = pd.DataFrame(per_sample_rows)
    if len(per_sample_df) != len(XAI_MODELS) * len(samples):
        raise ValueError(f"Unexpected per-sample stability count: {len(per_sample_df)}")

    summary_rows = []
    for model_key in XAI_MODELS:
        model_subset = per_sample_df[per_sample_df["model"] == model_key]
        summary_rows.append(
            {
                "model": model_key,
                "n_samples": int(len(model_subset)),
                "mean_top1_agreement": float(model_subset["avg_top1_agreement"].mean()),
                "std_top1_agreement": float(model_subset["avg_top1_agreement"].std(ddof=1)) if len(model_subset) > 1 else 0.0,
                "mean_top5_overlap_count": float(model_subset["avg_top5_overlap_count"].mean()),
                "mean_top5_jaccard": float(model_subset["avg_top5_jaccard"].mean()),
                "mean_top10_overlap_count": float(model_subset["avg_top10_overlap_count"].mean()),
                "mean_top10_jaccard": float(model_subset["avg_top10_jaccard"].mean()),
                "mean_spearman_rho": float(model_subset["avg_spearman_rho"].mean()),
                "std_spearman_rho": float(model_subset["avg_spearman_rho"].std(ddof=1)) if len(model_subset) > 1 else 0.0,
                "mean_direction_agreement": float(model_subset["avg_direction_agreement"].mean()),
                "std_direction_agreement": float(model_subset["avg_direction_agreement"].std(ddof=1)) if len(model_subset) > 1 else 0.0,
                "mean_probability": float(model_subset["probability_mean"].mean()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    metadata = {
        "experiment": "LIME seed stability",
        "goal": "Measure how much LIME rankings and directions change under different random seeds for the same trained UNSW-NB15 model and flow.",
        "seeds": SEEDS,
        "n_models": len(XAI_MODELS),
        "n_samples": int(len(samples)),
        "n_explanations_expected": len(XAI_MODELS) * len(samples) * len(SEEDS),
        "n_pairwise_expected": len(XAI_MODELS) * len(samples) * (len(SEEDS) * (len(SEEDS) - 1) // 2),
        "random_seed_rule": "Each explanation resets the explainer, discretizer and surrogate RNG with numpy RandomState(seed), keeping training data, model, sample and perturbation count fixed while changing only the random seed.",
        "limesettings": {
            "library": f"lime {version('lime')}",
            "num_samples": NUM_SAMPLES,
            "num_features": NUM_FEATURES,
            "explained_label": int(EXPLAINED_LABEL),
            "feature_space": "42 original UNSW-NB15 features (before one-hot encoding)",
            "background_data": "internal sub-training split used by the SHAP validation pipeline",
            "distance_metric": "euclidean",
            "feature_selection": "none",
            "discretize_continuous": True,
            "discretizer": "quartile",
            "kernel_width": np.sqrt(len(FEATURE_ORDER)) * 0.75,
            "prediction_function": "decode integer-coded categorical features -> DataFrame -> pipeline.predict_proba",
            "sample_id_rule": "validation_position in the internal validation split, not raw dataset row ID",
        },
        "model_artifacts": {
            model_key: {
                "path": f"results/models/{model_key}_validation_pipeline.joblib",
                "model_type": type(pipeline.named_steps["model"]).__name__,
                "has_pipeline": pipeline is not None,
            }
            for model_key, pipeline in pipelines.items()
        },
        "sample_manifest": {
            "path": SHARED_SAMPLES_CSV.relative_to(ROOT).as_posix(),
            "sample_count": int(len(samples)),
            "sample_validation_positions": [int(v) for v in samples[SAMPLE_KEY].tolist()],
        },
        "feature_universe_rule": "Ranks are computed by absolute LIME weight across all 42 original feature weights for each seed. For pairwise comparison, we use the intersection of ranked features from the two seeds; missing values are dropped. Spearman-like correlation is computed on the shared ranked feature universe and set to 0.0 when fewer than 2 shared features exist.",
        "direction_rule": "For features present in both top-10 sets, direction agreement is the fraction of common features with the same sign (+/-) in both seed runs; when no shared top-10 features exist, agreement is 0.0.",
        "package_versions": {
            "lime": version("lime"),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit_learn": version("scikit-learn"),
            "xgboost": version("xgboost"),
            "matplotlib": version("matplotlib"),
        },
    }

    return run_df, pairwise_df, per_sample_df, summary_df, metadata


def render_plots(per_sample_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    metrics = [
        ("avg_top5_jaccard", "Top-5 Jaccard similarity"),
        ("avg_top10_jaccard", "Top-10 Jaccard similarity"),
        ("avg_spearman_rho", "Rank correlation (shared universe)"),
        ("avg_top1_agreement", "Top-1 agreement rate"),
    ]

    for ax, (col, title) in zip(axes, metrics):
        values = []
        labels = []
        for model_key in XAI_MODELS:
            series = per_sample_df.loc[per_sample_df["model"] == model_key, col]
            values.append(series.to_numpy())
            labels.append(model_key)
        positions = np.arange(1, len(labels) + 1)
        bp = ax.boxplot(values, positions=positions, patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], ["#4c72b0", "#dd8452"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.set_ylabel("value")
        ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--")
        if col == "avg_spearman_rho":
            ax.set_ylim(-1.05, 1.05)
        else:
            ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "lime_seed_stability_summary.png", dpi=300)
    plt.close(fig)


def validate_outputs(run_df: pd.DataFrame, pairwise_df: pd.DataFrame, per_sample_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    assert not run_df.empty
    assert not pairwise_df.empty
    assert not per_sample_df.empty
    assert not summary_df.empty

    expected_run_rows = len(XAI_MODELS) * len(run_df["validation_position"].drop_duplicates()) * len(SEEDS) * len(FEATURE_ORDER)
    assert len(run_df) == expected_run_rows, (len(run_df), expected_run_rows)

    expected_pairwise_rows = len(XAI_MODELS) * len(run_df["validation_position"].drop_duplicates()) * 10
    assert len(pairwise_df) == expected_pairwise_rows, (len(pairwise_df), expected_pairwise_rows)

    assert run_df["feature"].notna().all()
    assert run_df["signed_weight"].notna().all()
    assert not (run_df["feature"] == "").any()
    assert run_df["seed"].isin(SEEDS).all()
    assert run_df["validation_position"].between(0, len(get_xai_data()["X_val"]) - 1).all()
    assert np.isfinite(run_df["signed_weight"]).all()

    for col in ["top1_agreement", "top5_jaccard", "top10_jaccard", "direction_agreement"]:
        assert np.isfinite(pairwise_df[col]).all(), col
        assert ((pairwise_df[col] >= 0.0) & (pairwise_df[col] <= 1.0)).all(), col
    assert np.isfinite(pairwise_df["spearman_rho"]).all(), "spearman_rho"
    assert ((pairwise_df["spearman_rho"] >= -1.0) & (pairwise_df["spearman_rho"] <= 1.0)).all(), "spearman_rho"

    probability_sets = run_df.groupby(["model", "validation_position"])["model_probability"].nunique()
    assert (probability_sets == 1).all(), probability_sets[probability_sets != 1]


def main() -> None:
    ensure_safe_output_dir()
    run_df, pairwise_df, per_sample_df, summary_df, metadata = build_stability_outputs()
    validate_outputs(run_df, pairwise_df, per_sample_df, summary_df)

    run_df.to_csv(OUTPUT_DIR / "lime_stability_runs.csv", index=False, float_format="%.12g")
    pairwise_df.to_csv(OUTPUT_DIR / "lime_stability_pairwise.csv", index=False, float_format="%.12g")
    per_sample_df.to_csv(OUTPUT_DIR / "lime_stability_per_sample.csv", index=False, float_format="%.12g")
    summary_df.to_csv(OUTPUT_DIR / "lime_stability_summary.csv", index=False, float_format="%.12g")
    (OUTPUT_DIR / "lime_stability_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    render_plots(per_sample_df)

    print(f"Saved stability outputs to: {OUTPUT_DIR}")
    print(f"Run rows: {len(run_df)}")
    print(f"Pairwise rows: {len(pairwise_df)}")
    print(f"Per-sample rows: {len(per_sample_df)}")
    print(f"Summary rows: {len(summary_df)}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
