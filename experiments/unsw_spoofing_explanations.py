"""Part B of the UNSW-NB15 explanation-spoofing pilot: explanation sensitivity of the Part A candidates.

The Part A candidates are RANDOM constrained perturbations, so this script measures
random-perturbation explanation sensitivity. It is not a targeted attack and no matched random
control is run here.

Steps
1. Load the accepted candidates of a Part A run (read-only, never altered) and select the rows to
   evaluate (all, a fixed list of IDs, or the first N per group by candidate_id).
2. Re-check every input: file hashes, each original vector equals its validation row, and each
   model reproduces the stored P(Attack) for the original and modified vectors.
3. SHAP: same explainer setup as experiments/unsw_shap_explain.py (100 background rows of the
   preprocessed validation split, numpy default_rng(42)); one-hot columns are summed into the 42
   original features. SHAP is deterministic, so each candidate is compared once with its original.
4. LIME: same explainer as experiments/unsw_lime.py and the five-seed protocol of
   experiments/unsw_lime_stability.py (seeds 11, 22, 33, 44, 55). Each candidate is explained under
   all five seeds and compared with its original under the SAME seed (5 paired comparisons). The
   original's 10 seed-to-seed comparisons give the same-input noise band.
5. Metrics follow unsw_lime_stability.compute_pairwise_metrics: top-1 agreement, top-5/top-10
   Jaccard, Spearman correlation of the 42 rank positions (ranking by |weight|, ties by feature
   name), and sign agreement on the shared top-10 features. Raw SHAP magnitudes are never compared
   between Random Forest (probability scale) and XGBoost (logit scale); only rank-based metrics.

Outputs go to a NEW directory (default results/spoofing_pilot/part_b); the script refuses to write
into a non-empty directory, so Part A and the existing SHAP/LIME/stability outputs are never touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.unsw_lime import (
    CATEGORICAL_FEATURES,
    DISCRETIZER,
    KERNEL_WIDTH,
    NUM_SAMPLES,
    CategoricalCodec,
    build_explainer,
    explain,
    make_predict_fn,
    validate_sample,
)
from experiments.unsw_lime_stability import SEEDS, compute_pairwise_metrics, explanation_ranking
from experiments.unsw_shap_explain import validate_shap_for_model
from src.unsw.constraints import FEATURE_ORDER
from src.unsw.xai_setup import XAI_MODELS, get_xai_data, load_shap_reference_probabilities, load_xai_model

warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_RUN_DIR = ROOT / "results" / "spoofing_pilot" / "run_20261027_164146"
DEFAULT_OUT_DIR = ROOT / "results" / "spoofing_pilot" / "part_b"
SHAP_BACKGROUND_SIZE = 100       # as in unsw_shap_explain.explain_selected_samples
SHAP_BACKGROUND_SEED = 42
PROB_TOL = 1e-6                  # model vs. P(Attack) stored by Part A / the SHAP run
SHAP_REPRO_TOL = 1e-6            # recomputed original SHAP vs. results/metrics/shap
LIME_REPRO_TOL = 1e-6            # recomputed original LIME vs. results/lime_stability
TOP1_MAJORITY = 3                # LIME top-1 counts as changed if it differs in >= 3 of 5 seeds
SIMILARITY_METRICS = ["top5_jaccard", "top10_jaccard", "spearman_rho"]

# Declared before any candidate was explained. Do not edit after results exist.
THRESHOLD_RULE = {
    "noise_band": "For each model x original sample: the 10 seed-pair comparisons (seeds 11,22,33,44,55) "
                  "of the ORIGINAL sample's LIME explanation with itself (same input, different seed).",
    "observed_shift": "For each candidate: the mean over the 5 seeds of the comparison original(seed s) vs. "
                      "modified(seed s), i.e. same seed, different input.",
    "per_metric_rule": "For top5_jaccard, top10_jaccard and spearman_rho (higher = more similar) the candidate "
                       "exceeds seed noise on that metric if its mean observed value is strictly LOWER than the "
                       "MINIMUM of the 10 seed-pair values of its own original sample.",
    "top1_rule": "The top-1 change exceeds seed noise only if the original's top-1 feature agreed in all 10 seed "
                 "pairs AND the candidate's top-1 differs from the original's in >= 3 of the 5 paired seeds.",
    "primary_flag": "shift_exceeds_seed_noise = exceeds on top10_jaccard AND on spearman_rho.",
    "shap": "SHAP is deterministic (no seed noise), so no noise flag is defined for SHAP; any SHAP change is "
            "caused by the input change.",
}


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit_hash() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ids", help="comma-separated candidate_ids to evaluate")
    selection.add_argument("--per-group", type=int, help="first N candidates per model x budget x sample, by candidate_id")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def select_candidates(candidates: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    if args.ids:
        ids = [s.strip() for s in args.ids.split(",") if s.strip()]
        missing = sorted(set(ids) - set(candidates["candidate_id"]))
        if missing:
            raise ValueError(f"Unknown candidate_ids: {missing}")
        return candidates[candidates["candidate_id"].isin(ids)].copy(), f"explicit list of {len(ids)} candidate_ids"
    if args.per_group:
        selected = (candidates.sort_values("candidate_id")
                    .groupby(["model_name", "perturbation_budget", "validation_position"], sort=True)
                    .head(args.per_group))
        return selected.copy(), (f"first {args.per_group} per model x budget x sample group, ordered by candidate_id "
                                 "(chosen before any explanation was computed)")
    return candidates.copy(), "all accepted candidates"


def prepare_out_dir(out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty; this script never overwrites existing outputs.")
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)


def vectors_frame(json_column: pd.Series) -> pd.DataFrame:
    rows = [json.loads(s) for s in json_column]
    if any(list(r.keys()) != FEATURE_ORDER for r in rows):
        raise ValueError("A stored feature vector does not use the canonical 42-feature order")
    return pd.DataFrame(rows, columns=FEATURE_ORDER)


def check_originals(selected: pd.DataFrame, X_val: pd.DataFrame) -> None:
    for (pos, js) in selected[["validation_position", "original_feature_values_json"]].drop_duplicates().itertuples(index=False):
        stored, row = json.loads(js), X_val.iloc[int(pos)]
        for feature in FEATURE_ORDER:
            a, b = stored[feature], row[feature]
            same = str(a) == str(b) if feature in CATEGORICAL_FEATURES else np.isclose(float(a), float(b))
            if not same:
                raise ValueError(f"Stored original for position {pos} differs from X_val on {feature}: {a!r} vs {b!r}")


# --------------------------------------------------------------------------------------
# Metrics (same definitions as unsw_lime_stability.compute_pairwise_metrics)
# --------------------------------------------------------------------------------------
def ranked(weights: dict[str, float]) -> list[str]:
    return sorted(weights, key=lambda feat: (-abs(weights[feat]), feat))


def pair_metrics(a: dict[str, float], b: dict[str, float]) -> dict:
    order_a, order_b = ranked(a), ranked(b)
    result = {"orig_top1": order_a[0], "mod_top1": order_b[0], "top1_agreement": float(order_a[0] == order_b[0])}
    for k in (5, 10):
        top_a, top_b = set(order_a[:k]), set(order_b[:k])
        result[f"top{k}_jaccard"] = len(top_a & top_b) / len(top_a | top_b)
    rank_a = pd.Series({feat: i + 1 for i, feat in enumerate(order_a)})
    rank_b = pd.Series({feat: i + 1 for i, feat in enumerate(order_b)})[rank_a.index]
    rho = float(np.corrcoef(rank_a.to_numpy(), rank_b.to_numpy())[0, 1])
    result["spearman_rho"] = 0.0 if np.isnan(rho) else rho
    # Secondary, tie-aware version (matters when many attributions are exactly zero).
    feats = list(a)
    tie_rho = spearmanr([abs(a[f]) for f in feats], [abs(b[f]) for f in feats]).statistic
    result["spearman_abs_tieaware"] = float(tie_rho) if np.isfinite(tie_rho) else np.nan
    shared = sorted(set(order_a[:10]) & set(order_b[:10]))
    result["direction_agreement_top10"] = (
        sum(np.sign(a[f]) == np.sign(b[f]) for f in shared) / len(shared) if shared else 0.0
    )
    result["shared_top10_features"] = len(shared)
    return result


def check_pair_metrics_match_stability(run_df: pd.DataFrame, model_key: str, pos: int) -> None:
    """pair_metrics must reproduce compute_pairwise_metrics exactly on the original's seed pairs."""
    reference = compute_pairwise_metrics(run_df, model_key, pos)
    for row in reference:
        w = {s: run_df[(run_df.model == model_key) & (run_df.validation_position == pos) & (run_df.seed == s)]
             .set_index("feature")["signed_weight"].to_dict() for s in (row["seed_a"], row["seed_b"])}
        mine = pair_metrics(w[row["seed_a"]], w[row["seed_b"]])
        for key in ("top1_agreement", "top5_jaccard", "top10_jaccard", "spearman_rho"):
            if not np.isclose(mine[key], row[key]):
                raise ValueError(f"pair_metrics differs from compute_pairwise_metrics on {key}: {mine[key]} vs {row[key]}")
        if not np.isclose(mine["direction_agreement_top10"], row["direction_agreement"]):
            raise ValueError("pair_metrics differs from compute_pairwise_metrics on direction agreement")


# --------------------------------------------------------------------------------------
# SHAP
# --------------------------------------------------------------------------------------
def shap_column_to_feature(column: str) -> str:
    """'num__sttl' -> 'sttl'; 'cat__proto_tcp' -> 'proto'. From experiments/unsw_shap_lime_comparison.py."""
    if column.startswith("num__"):
        return column[len("num__"):]
    for cat in CATEGORICAL_FEATURES:
        if column.startswith(f"cat__{cat}_"):
            return cat
    raise ValueError(f"SHAP column {column!r} cannot be mapped to an original feature")


def build_shap_explainer(model_key: str, pipeline, X_val: pd.DataFrame):
    """Explainer configured exactly as unsw_shap_explain.explain_selected_samples."""
    preprocessor, estimator = pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]
    X_val_pre = preprocessor.transform(X_val)
    rng = np.random.default_rng(SHAP_BACKGROUND_SEED)
    background = X_val_pre[rng.choice(np.arange(X_val_pre.shape[0]), size=min(SHAP_BACKGROUND_SIZE, X_val_pre.shape[0]), replace=False)]
    if model_key == "xgboost":
        return shap.TreeExplainer(estimator.get_booster(), data=background,
                                  feature_perturbation="tree_path_dependent", model_output="raw")
    return shap.TreeExplainer(estimator, data=background)


def shap_grouped(model_key: str, pipeline, explainer, X: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """SHAP values for the attack class, summed into the 42 original features (rows follow X)."""
    preprocessor, estimator = pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]
    X_pre = preprocessor.transform(X)
    columns = [str(c) for c in preprocessor.get_feature_names_out()]
    expected = estimator.predict_proba(X_pre)[:, 1]
    values, _, _, output_kind, _ = validate_shap_for_model(model_key, estimator, X_pre, columns, explainer(X_pre), expected)
    raw = pd.DataFrame(values, columns=columns)
    grouped = raw.T.groupby({c: shap_column_to_feature(c) for c in columns}).sum().T
    return grouped[FEATURE_ORDER], output_kind


def check_shap_reproduces_saved(model_key: str, pipeline, explainer, X_val: pd.DataFrame, positions: list[int]) -> float:
    saved = pd.read_csv(ROOT / "results" / "metrics" / "shap" / f"{model_key}_selected_shap_values.csv").set_index("validation_position")
    preprocessor = pipeline.named_steps["preprocessor"]
    columns = [str(c) for c in preprocessor.get_feature_names_out()]
    X_pre = preprocessor.transform(X_val.iloc[positions])
    values = np.asarray(explainer(X_pre).values)
    values = values[:, :, 1] if values.ndim == 3 else values
    gap = float(np.max(np.abs(values - saved.loc[positions, columns].to_numpy())))
    if gap > SHAP_REPRO_TOL:
        raise ValueError(f"{model_key}: recomputed original SHAP differs from results/metrics/shap by {gap:.2e}")
    return gap


# --------------------------------------------------------------------------------------
# LIME
# --------------------------------------------------------------------------------------
def lime_weights(explainer, codec, predict_fn, X_row: pd.DataFrame, seed: int) -> dict[str, float]:
    exp = explain(explainer, codec.encode(X_row)[0], predict_fn, seed)
    return {feature: weight for feature, weight, _ in explanation_ranking(exp)}


def check_lime_reproduces_saved(run_df: pd.DataFrame) -> float:
    saved = pd.read_csv(ROOT / "results" / "lime_stability" / "lime_stability_runs.csv")
    keys = ["model", "validation_position", "seed", "feature"]
    merged = run_df.merge(saved[keys + ["signed_weight"]], on=keys, suffixes=("", "_saved"), validate="one_to_one")
    if len(merged) != len(run_df):
        raise ValueError("Saved LIME stability runs do not cover the selected originals")
    gap = float((merged["signed_weight"] - merged["signed_weight_saved"]).abs().max())
    if gap > LIME_REPRO_TOL:
        raise ValueError(f"Recomputed original LIME differs from results/lime_stability by {gap:.2e}")
    return gap


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------
def noise_band(run_df: pd.DataFrame, model_key: str, pos: int) -> dict:
    pairs = pd.DataFrame(compute_pairwise_metrics(run_df, model_key, pos))
    band = {}
    for col, name in [("top1_agreement", "top1_agreement"), ("top5_jaccard", "top5_jaccard"),
                      ("top10_jaccard", "top10_jaccard"), ("spearman_rho", "spearman_rho"),
                      ("direction_agreement", "direction_agreement_top10")]:
        band[f"noise_{name}_min"] = float(pairs[col].min())
        band[f"noise_{name}_mean"] = float(pairs[col].mean())
        band[f"noise_{name}_max"] = float(pairs[col].max())
    band["noise_pair_values_top10_jaccard"] = pairs["top10_jaccard"].round(6).tolist()
    return band


def build_summary(comparison: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    flags = stability.set_index("candidate_id")[["shift_exceeds_seed_noise", "exceeds_top10_jaccard",
                                                 "exceeds_spearman_rho", "exceeds_top1"]].astype(float)
    df = comparison.join(flags, on="candidate_id")
    for col in flags.columns:
        df.loc[df["explainer"] == "shap", col] = np.nan

    def aggregate(group: pd.DataFrame) -> pd.Series:
        lime = group["shift_exceeds_seed_noise"].notna().all()  # noise flags exist only for LIME rows
        return pd.Series({
            "n_evaluated": len(group),
            "n_unique_modified_vectors": group["modified_vector_hash"].nunique(),
            "mean_changed_features": group["n_changed_features"].mean(),
            "top1_changed_n": int(group["top1_changed"].sum()),
            "top1_changed_rate": group["top1_changed"].mean(),
            "top5_jaccard_mean": group["top5_jaccard"].mean(),
            "top5_jaccard_min": group["top5_jaccard"].min(),
            "top10_jaccard_mean": group["top10_jaccard"].mean(),
            "top10_jaccard_min": group["top10_jaccard"].min(),
            "spearman_rho_mean": group["spearman_rho"].mean(),
            "spearman_rho_min": group["spearman_rho"].min(),
            "direction_agreement_top10_mean": group["direction_agreement_top10"].mean(),
            "prob_original": group["prob_original"].mean(),
            "prob_modified_min": group["prob_modified"].min(),
            "delta_prob_mean": group["delta_prob"].mean(),
            "delta_prob_min": group["delta_prob"].min(),
            "exceeds_seed_noise_n": int(group["shift_exceeds_seed_noise"].sum()) if lime else np.nan,
            "exceeds_seed_noise_rate": group["shift_exceeds_seed_noise"].astype(float).mean() if lime else np.nan,
            "exceeds_top1_rule_n": int(group["exceeds_top1"].sum()) if lime else np.nan,
        })

    keys = ["model", "budget", "explainer"]
    per_sample = df.groupby(keys + ["validation_position", "attack_cat"]).apply(aggregate, include_groups=False).reset_index()
    pooled = df.groupby(keys).apply(aggregate, include_groups=False).reset_index()
    pooled["validation_position"], pooled["attack_cat"] = "all", "all"
    summary = pd.concat([per_sample, pooled], ignore_index=True)
    return summary.sort_values(["explainer", "model", "budget", "validation_position"], key=lambda s: s.astype(str)).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
INK, INK_2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#a3a29c", "#fcfcfb"
SERIES = {"A": "#2a78d6", "B": "#eb6834"}
MOVED = "#2a78d6"


def style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)


def plot_rank_change(example: dict, path: Path) -> None:
    """Bump chart: original vs. modified top-10 ranking for SHAP and LIME (one seed) of one candidate."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), facecolor=SURFACE)
    changed = set(example["changed_features"])
    for ax, (title, orig, mod) in zip(axes, example["panels"]):
        order_o, order_m = ranked(orig)[:10], ranked(mod)[:10]
        out_rank = 11.4
        for feature in sorted(set(order_o) | set(order_m)):
            r_o = order_o.index(feature) + 1 if feature in order_o else out_rank
            r_m = order_m.index(feature) + 1 if feature in order_m else out_rank
            moved = r_o != r_m
            ax.plot([0, 1], [r_o, r_m], color=MOVED if moved else MUTED, lw=2 if moved else 1.2, zorder=2)
            ax.scatter([0, 1], [r_o, r_m], s=36, color=MOVED if moved else MUTED, edgecolor=SURFACE, linewidth=2, zorder=3)
            weight = "bold" if feature in changed else "normal"
            if r_o <= 10:
                ax.text(-0.05, r_o, feature, ha="right", va="center", fontsize=9, color=INK, fontweight=weight)
            if r_m <= 10:
                ax.text(1.05, r_m, feature, ha="left", va="center", fontsize=9, color=INK, fontweight=weight)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(out_rank + 0.6, 0.4)
        ax.set_xticks([0, 1], ["Original", "Modified"])
        ax.set_yticks(list(range(1, 11)) + [out_rank], [str(i) for i in range(1, 11)] + ["out of\ntop 10"])
        ax.set_ylabel("Rank by |attribution|", color=INK_2, fontsize=9)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        style_axis(ax)
    fig.suptitle(example["suptitle"], fontsize=12, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01, "Blue = feature changed rank; grey = same rank. Bold = feature edited by the perturbation.",
             fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_shift_vs_noise(stability: pd.DataFrame, path: Path) -> None:
    """Strip plot: candidate LIME top-10 Jaccard (orig vs. modified) vs. the original's seed-to-seed band."""
    groups = stability[["model", "validation_position", "attack_cat"]].drop_duplicates().sort_values(["model", "validation_position"])
    fig, ax = plt.subplots(figsize=(9, 0.75 * len(groups) + 1.8), facecolor=SURFACE)
    rng = np.random.default_rng(0)  # jitter only
    labels = []
    for y, g in enumerate(groups.itertuples(index=False)):
        sub = stability[(stability.model == g.model) & (stability.validation_position == g.validation_position)]
        noise = sub["noise_pair_values_top10_jaccard"].iloc[0]
        noise = json.loads(noise) if isinstance(noise, str) else noise
        values, counts = np.unique(np.round(noise, 6), return_counts=True)
        ax.scatter(values, np.full(len(values), y + 0.28), marker="|", s=140, color=INK_2, linewidth=2, zorder=3,
                   label="Seed-to-seed, same input (10 pairs)" if y == 0 else None)
        for value, count in zip(values, counts):
            if count > 1:
                ax.text(value, y + 0.28 + 0.06, f"×{count}", ha="center", va="top", fontsize=7, color=INK_2)
        ax.plot([min(noise)] * 2, [y - 0.42, y + 0.42], color=INK_2, lw=1, ls="--", zorder=2)
        for offset, budget in ((-0.26, "A"), (0.0, "B")):
            vals = sub.loc[sub.budget == budget, "obs_top10_jaccard_mean"].to_numpy()
            ax.scatter(vals, y + offset + rng.uniform(-0.07, 0.07, len(vals)), s=18, color=SERIES[budget], alpha=0.8,
                       edgecolor=SURFACE, linewidth=0.6, zorder=4, label=f"Candidates, budget {budget}" if y == 0 else None)
        labels.append(f"{XAI_MODELS[g.model]}\n{g.validation_position} {g.attack_cat}")
    ax.set_yticks(range(len(groups)), labels)
    ax.set_ylim(len(groups) - 0.5, -0.5)
    ax.set_xlabel("LIME top-10 Jaccard similarity (1 = same top-10 set)", color=INK_2, fontsize=9)
    ax.grid(axis="x", color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    style_axis(ax)
    ax.legend(loc="lower left", fontsize=8, frameon=False, ncol=3, bbox_to_anchor=(0, 1.0))
    ax.set_title("Dashed line = least similar seed pair; points left of it exceed seed noise on this metric",
                 fontsize=9, color=INK_2, loc="left", pad=26)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    t_start = time.perf_counter()
    run_dir, out_dir = args.run_dir.resolve(), args.out_dir.resolve()
    if run_dir in out_dir.parents or out_dir == run_dir:
        raise ValueError("Output directory must not be inside the Part A run directory")
    prepare_out_dir(out_dir)

    candidates_path = run_dir / "candidates.csv"
    candidates = pd.read_csv(candidates_path)
    part_a_meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    selected, selection_rule = select_candidates(candidates, args)
    selected = selected.sort_values("candidate_id").reset_index(drop=True)
    print(f"Selected {len(selected)} of {len(candidates)} candidates ({selection_rule}).")

    split = get_xai_data()
    X_subtrain, X_val, y_val = split["X_subtrain"], split["X_val"], split["y_val"]
    check_originals(selected, X_val)
    codec = CategoricalCodec(X_subtrain)
    lime_explainer = build_explainer(codec.encode(X_subtrain), codec)
    positions = sorted(selected["validation_position"].unique().tolist())
    checks: dict[str, object] = {"originals_equal_validation_rows": True}

    comparison_rows, stability_rows, timings = [], [], {}
    example_store: dict[str, dict] = {}
    for model_key in sorted(selected["model_name"].unique()):
        t_model = time.perf_counter()
        pipeline = load_xai_model(model_key, split)
        predict_fn = make_predict_fn(pipeline, codec)
        shap_explainer = build_shap_explainer(model_key, pipeline, X_val)
        rows = selected[selected["model_name"] == model_key].reset_index(drop=True)
        model_positions = sorted(rows["validation_position"].unique().tolist())
        reference_probs = load_shap_reference_probabilities(model_key)

        # Originals: SHAP once, LIME under the five seeds (= the stability protocol).
        checks[f"{model_key}_shap_original_max_gap_vs_saved"] = check_shap_reproduces_saved(
            model_key, pipeline, shap_explainer, X_val, model_positions)
        shap_orig, shap_kind = shap_grouped(model_key, pipeline, shap_explainer, X_val.iloc[model_positions])
        shap_orig.index = model_positions
        run_rows = []
        for pos in model_positions:
            x_row = X_val.iloc[[pos]]
            validate_sample(model_key, pipeline, codec, x_row, int(y_val.iloc[pos]),
                            str(rows.loc[rows.validation_position == pos, "attack_cat"].iloc[0]),
                            float(reference_probs[pos]), predict_fn)
            for seed in SEEDS:
                for feature, weight in lime_weights(lime_explainer, codec, predict_fn, x_row, seed).items():
                    run_rows.append({"model": model_key, "validation_position": pos, "seed": seed,
                                     "feature": feature, "signed_weight": weight})
        run_df = pd.DataFrame(run_rows)
        checks[f"{model_key}_lime_original_max_gap_vs_saved"] = check_lime_reproduces_saved(run_df)
        for pos in model_positions:
            check_pair_metrics_match_stability(run_df, model_key, pos)
        checks[f"{model_key}_pair_metrics_match_compute_pairwise_metrics"] = True
        lime_orig = {(pos, seed): run_df[(run_df.validation_position == pos) & (run_df.seed == seed)]
                     .set_index("feature")["signed_weight"].to_dict() for pos in model_positions for seed in SEEDS}
        bands = {pos: noise_band(run_df, model_key, pos) for pos in model_positions}
        t_originals = time.perf_counter() - t_model

        # Candidates.
        X_mod = vectors_frame(rows["modified_feature_values_json"])
        prob_orig = pipeline.predict_proba(vectors_frame(rows["original_feature_values_json"]))[:, 1]
        prob_mod = pipeline.predict_proba(X_mod)[:, 1]
        gap = max(np.abs(prob_orig - rows["original_attack_probability"]).max(),
                  np.abs(prob_mod - rows["modified_attack_probability"]).max())
        if gap > PROB_TOL:
            raise ValueError(f"{model_key}: P(Attack) differs from Part A's stored values by {gap:.2e}")
        checks[f"{model_key}_prob_max_gap_vs_part_a"] = float(gap)
        t_shap = time.perf_counter()
        shap_mod, _ = shap_grouped(model_key, pipeline, shap_explainer, X_mod)
        t_shap = time.perf_counter() - t_shap

        t_lime = time.perf_counter()
        for i, row in rows.iterrows():
            pos = int(row.validation_position)
            x_mod = X_mod.iloc[[i]]
            validate_sample(model_key, pipeline, codec, x_mod, 1, str(row.attack_cat), float(row.modified_attack_probability), predict_fn)
            changed = json.loads(row.changed_features_json)
            base = {
                "candidate_id": row.candidate_id, "validation_position": pos, "attack_cat": row.attack_cat,
                "model": model_key, "budget": row.perturbation_budget,
                "n_changed_features": len(changed), "changed_features": ";".join(changed),
                "modified_vector_hash": hashlib.sha256(row.modified_feature_values_json.encode()).hexdigest()[:16],
                "prob_original": float(prob_orig[i]), "prob_modified": float(prob_mod[i]),
                "delta_prob": float(prob_mod[i] - prob_orig[i]), "part_a_generation_seed": int(row.seed),
            }

            shap_m = pair_metrics(shap_orig.loc[pos].to_dict(), shap_mod.iloc[i].to_dict())
            comparison_rows.append({**base, "explainer": "shap", **shap_m, "top1_changed": shap_m["top1_agreement"] == 0.0,
                                    "top1_changed_seed_count": np.nan, "explainer_seeds": "",
                                    "explainer_config": f"TreeExplainer; background={SHAP_BACKGROUND_SIZE} X_val rows "
                                                        f"(default_rng({SHAP_BACKGROUND_SEED})); output={shap_kind}; "
                                                        "one-hot summed to 42 features"})

            per_seed, mod_weights = [], {}
            for seed in SEEDS:
                mod_weights[seed] = lime_weights(lime_explainer, codec, predict_fn, x_mod, seed)
                per_seed.append(pair_metrics(lime_orig[(pos, seed)], mod_weights[seed]))
            seeds_df = pd.DataFrame(per_seed)
            numeric = ["top1_agreement", "top5_jaccard", "top10_jaccard", "spearman_rho", "spearman_abs_tieaware",
                       "direction_agreement_top10", "shared_top10_features"]
            lime_m = seeds_df[numeric].mean().to_dict()
            top1_changed_count = int((seeds_df["top1_agreement"] == 0.0).sum())
            comparison_rows.append({**base, "explainer": "lime", **lime_m,
                                    "orig_top1": seeds_df["orig_top1"].mode().iloc[0],
                                    "mod_top1": seeds_df["mod_top1"].mode().iloc[0],
                                    "top1_changed": top1_changed_count >= TOP1_MAJORITY,
                                    "top1_changed_seed_count": top1_changed_count,
                                    "explainer_seeds": ";".join(map(str, SEEDS)),
                                    "explainer_config": f"LimeTabularExplainer; num_samples={NUM_SAMPLES}; 42 features; "
                                                        f"{DISCRETIZER} discretizer; kernel_width={KERNEL_WIDTH:.4f}; "
                                                        "background=X_subtrain; metrics = mean of 5 same-seed pairs"})

            band = bands[pos]
            stab = {"candidate_id": row.candidate_id, "model": model_key, "budget": row.perturbation_budget,
                    "validation_position": pos, "attack_cat": row.attack_cat, **band}
            for metric in SIMILARITY_METRICS:
                stab[f"obs_{metric}_mean"] = float(seeds_df[metric].mean())
                stab[f"obs_{metric}_min"] = float(seeds_df[metric].min())
                stab[f"exceeds_{metric}"] = bool(stab[f"obs_{metric}_mean"] < band[f"noise_{metric}_min"])
            stab["obs_direction_agreement_top10_mean"] = float(seeds_df["direction_agreement_top10"].mean())
            stab["obs_top1_changed_seed_count"] = top1_changed_count
            stab["original_top1_seed_stable"] = band["noise_top1_agreement_min"] == 1.0
            stab["exceeds_top1"] = bool(stab["original_top1_seed_stable"] and top1_changed_count >= TOP1_MAJORITY)
            stab["shift_exceeds_seed_noise"] = bool(stab["exceeds_top10_jaccard"] and stab["exceeds_spearman_rho"])
            stab["criterion"] = THRESHOLD_RULE["primary_flag"]
            stability_rows.append(stab)

            example_store[row.candidate_id] = {
                "changed_features": changed, "lime_seed": SEEDS[0],
                "panels": [("SHAP", shap_orig.loc[pos].to_dict(), shap_mod.iloc[i].to_dict()),
                           (f"LIME (seed {SEEDS[0]})", lime_orig[(pos, SEEDS[0])], mod_weights[SEEDS[0]])],
            }
        t_lime = time.perf_counter() - t_lime
        timings[model_key] = {"originals_s": round(t_originals, 2), "shap_candidates_s": round(t_shap, 2),
                              "lime_candidates_s": round(t_lime, 2), "n_candidates": len(rows),
                              "lime_s_per_candidate": round(t_lime / max(len(rows), 1), 3)}
        print(f"{model_key}: {len(rows)} candidates, timings {timings[model_key]}")

    comparison = pd.DataFrame(comparison_rows)
    stability = pd.DataFrame(stability_rows)
    summary = build_summary(comparison, stability)

    lead = ["candidate_id", "validation_position", "attack_cat", "model", "budget", "explainer", "n_changed_features",
            "changed_features", "orig_top1", "mod_top1", "top1_changed", "top1_changed_seed_count", "top5_jaccard",
            "top10_jaccard", "spearman_rho", "spearman_abs_tieaware", "direction_agreement_top10", "shared_top10_features",
            "prob_original", "prob_modified", "delta_prob"]
    comparison = comparison[lead + [c for c in comparison.columns if c not in lead]]
    comparison.to_csv(out_dir / "explanation_comparison.csv", index=False)
    stability.to_csv(out_dir / "lime_stability_comparison.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    figures = {}
    if not args.no_figures:
        lime_rows = stability.sort_values(["obs_top10_jaccard_mean", "candidate_id"])
        largest = lime_rows.iloc[0]
        n_tied = int(np.isclose(lime_rows["obs_top10_jaccard_mean"], largest.obs_top10_jaccard_mean).sum())
        example = example_store[largest.candidate_id]
        example["suptitle"] = (f"Largest observed LIME shift: {largest.candidate_id} "
                               f"({XAI_MODELS[largest.model]}, budget {largest.budget}, {largest.attack_cat})")
        figures["rank_change_example.png"] = (
            f"Original vs. modified top-10 feature ranking for {largest.candidate_id}, the candidate with the lowest "
            f"mean LIME top-10 Jaccard ({largest.obs_top10_jaccard_mean:.3f}; {n_tied} candidates share this value, "
            f"first by candidate_id shown). Largest observed shift, not the typical candidate. Left: SHAP; right: LIME, "
            f"seed {SEEDS[0]}. Random constrained perturbation, not a targeted attack.")
        plot_rank_change(example, out_dir / "figures" / "rank_change_example.png")
        figures["lime_shift_vs_seed_noise.png"] = (
            "Per candidate: mean LIME top-10 Jaccard between original and modified sample (same seed, 5 seeds). "
            "Grey ticks: the 10 seed-to-seed comparisons of the unmodified original (same input). Points left of the "
            "dashed line are less similar than the least similar seed pair.")
        plot_shift_vs_noise(stability, out_dir / "figures" / "lime_shift_vs_seed_noise.png")

    rf_expected = part_a_meta["model_sha256"]["random_forest"]
    files = {
        "candidates_csv": candidates_path, "part_a_run_metadata": run_dir / "run_metadata.json",
        "training_csv": ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv",
        "testing_csv_not_used_for_evaluation": ROOT / "data" / "raw" / "UNSW_NB15_testing-set.csv",
        "shared_validation_manifest": ROOT / "results" / "metrics" / "shap" / "shared_validation_sample_identifiers.csv",
        "random_forest_pipeline": ROOT / "results" / "models" / "random_forest_validation_pipeline.joblib",
        "xgboost_pipeline": ROOT / "results" / "models" / "xgboost_validation_pipeline.joblib",
    }
    checksums = {name: file_sha256(path) for name, path in files.items()}
    metadata = {
        "experiment": "Part B: random-perturbation explanation sensitivity (UNSW-NB15 validation split only)",
        "git_commit_hash_at_run": git_commit_hash(),
        "part_a_run_id": part_a_meta.get("run_id", run_dir.name),
        "part_a_run_dir": run_dir.relative_to(ROOT).as_posix(),
        "selection_rule": selection_rule,
        "n_candidates_available": int(len(candidates)),
        "n_candidates_evaluated": int(len(selected)),
        "n_unique_modified_vectors_evaluated": int(selected["modified_feature_values_json"].nunique()),
        "evaluated_candidate_ids": selected["candidate_id"].tolist(),
        "seeds": {"lime_stability_seeds": SEEDS, "shap_background_seed": SHAP_BACKGROUND_SEED,
                  "part_a_generation_seed": part_a_meta.get("random_seed"), "data_split_random_state": 42},
        "threshold_rule": THRESHOLD_RULE,
        "lime_config": {"num_samples": NUM_SAMPLES, "num_features": len(FEATURE_ORDER), "discretizer": DISCRETIZER,
                        "kernel_width": KERNEL_WIDTH, "background": "X_subtrain", "explained_label": 1},
        "shap_config": {"background_size": SHAP_BACKGROUND_SIZE, "background_source": "preprocessed X_val",
                        "random_forest": "TreeExplainer(model, data=background); probability scale",
                        "xgboost": "TreeExplainer(booster, background, tree_path_dependent, raw); logit scale",
                        "grouping": "one-hot SHAP columns summed into the 42 original features"},
        "checksums_sha256": checksums,
        "handoff_deviation": {
            "random_forest_pipeline_sha256_expected_by_part_a": rf_expected,
            "random_forest_pipeline_sha256_used": checksums["random_forest_pipeline"],
            "matches": checksums["random_forest_pipeline"] == rf_expected,
            "note": "Part A's verifier fails on the RF hash. The RF file supplied by the Part A author was checked to be "
                    "the same model: all 200 trees identical (feature/threshold/value), identical preprocessing output, "
                    "max |dP| = 2.2e-16 on all 35,069 validation rows, and exact reproduction of Part A's stored RF "
                    "probabilities. The byte difference comes from bundle metadata fields.",
        },
        "input_checks": checks,
        "package_versions": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "xgboost", "shap", "lime", "scipy", "matplotlib"]},
        "python_version": sys.version.split()[0],
        "runtime_seconds": {"total": round(time.perf_counter() - t_start, 1), **timings},
        "figures": figures,
        "official_test_set_used_for_evaluation": False,
        "random_control_run": False,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"Wrote outputs to {out_dir} in {metadata['runtime_seconds']['total']} s")


if __name__ == "__main__":
    main()
