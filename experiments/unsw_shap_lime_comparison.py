"""Compare the saved SHAP and LIME explanations for Random Forest and XGBoost.

Only saved results are read: no model is loaded or retrained and no explanation is recomputed.

Steps
1. Load SHAP values (results/metrics/shap/<model>_selected_shap_values.csv) and LIME weights
   (results/unsw_nb15/lime/lime_feature_weights_all.csv), and check that both cover exactly the
   samples in shared_validation_sample_identifiers.csv and come from the same models.
2. Put both on the same 42 original features: SHAP's one-hot columns for proto/service/state
   are summed per original feature (allowed because SHAP values are additive), and the
   'num__' prefix is removed from numeric columns.
3. Per model and sample, compare rankings and signs (not raw values, because the scales
   differ: RF SHAP = probability, XGBoost SHAP = logit, LIME = local surrogate weight).
4. Compare the global ranking (mean |attribution| over the 20 samples).
5. Save CSVs, two figures and a summary.
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
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
SHAP_DIR = ROOT / "results" / "metrics" / "shap"
LIME_DIR = ROOT / "results" / "unsw_nb15" / "lime"
OUTPUT_DIR = ROOT / "results" / "unsw_nb15" / "shap_vs_lime"
FIGURE_DIR = OUTPUT_DIR / "figures"

SHARED_SAMPLES_CSV = SHAP_DIR / "shared_validation_sample_identifiers.csv"
SAMPLE_KEY = "validation_position"
MODELS = {"random_forest": "Random Forest", "xgboost": "XGBoost"}
CATEGORICAL_FEATURES = ["proto", "service", "state"]
SHAP_META_COLUMNS = [
    "model", SAMPLE_KEY, "true_label", "predicted_probability", "base_value",
    "reconstructed_model_output", "output_kind", "output_label",
]
TOP_KS = (1, 5, 10)
PROB_TOL = 1e-6        # SHAP vs LIME P(Attack): confirms both used the same model
ADDITIVITY_TOL = 1e-9  # base_value + sum(SHAP) must equal the saved model output

# Categorical slots 1 and 2 of the dataviz reference palette (validated together).
COLOR_SHAP = "#2a78d6"
COLOR_LIME = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8984"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


# --------------------------------------------------------------------------------------
# Loading and alignment
# --------------------------------------------------------------------------------------
def shap_column_to_feature(column: str) -> str:
    """'num__sttl' -> 'sttl'; 'cat__proto_tcp' -> 'proto'."""
    if column.startswith("num__"):
        return column[len("num__"):]
    for cat in CATEGORICAL_FEATURES:
        if column.startswith(f"cat__{cat}_"):
            return cat
    raise ValueError(f"SHAP column {column!r} cannot be mapped to an original feature")


def load_shap(model_key: str, feature_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (meta per sample, SHAP values per sample x 42 original features)."""
    raw = pd.read_csv(SHAP_DIR / f"{model_key}_selected_shap_values.csv")
    value_cols = [c for c in raw.columns if c not in SHAP_META_COLUMNS]

    # Sanity check that the file was read correctly: SHAP values must add up to the model output.
    additivity_error = (raw[value_cols].sum(axis=1) + raw["base_value"] - raw["reconstructed_model_output"]).abs().max()
    if additivity_error > ADDITIVITY_TOL:
        raise ValueError(f"{model_key}: SHAP values do not add up to the model output (max error {additivity_error})")

    mapping = {c: shap_column_to_feature(c) for c in value_cols}
    grouped = raw.set_index(SAMPLE_KEY)[value_cols].T.groupby(mapping).sum().T
    if set(grouped.columns) != set(feature_names):
        raise ValueError(
            f"{model_key}: SHAP features {sorted(set(grouped.columns) ^ set(feature_names))} do not match LIME's 42 features"
        )
    meta = raw.set_index(SAMPLE_KEY)[["predicted_probability", "output_kind"]]
    return meta, grouped[feature_names]


def load_lime(feature_names: list[str]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    weights = pd.read_csv(LIME_DIR / "lime_feature_weights_all.csv")
    summary = pd.read_csv(LIME_DIR / "lime_explanation_summary.csv")
    per_model = {}
    for key in MODELS:
        table = weights[weights["model"] == key].pivot(index=SAMPLE_KEY, columns="feature", values="lime_weight")
        if set(table.columns) != set(feature_names):
            raise ValueError(f"{key}: LIME weights do not cover the 42 features")
        per_model[key] = table[feature_names]
    return summary, per_model


def check_inputs(shared_ids, shap_values, shap_meta, lime_values, lime_summary) -> list[tuple[str, bool]]:
    checks = []

    def record(name: str, ok: bool) -> None:
        checks.append((name, bool(ok)))
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}")

    expected = set(shared_ids)
    record(f"shared file has {len(shared_ids)} unique IDs", len(expected) == len(shared_ids))
    for key, name in MODELS.items():
        record(f"{name}: SHAP IDs exactly match the shared file", list(shap_values[key].index) == list(shared_ids))
        record(f"{name}: LIME IDs exactly match the shared file", set(lime_values[key].index) == expected
               and len(lime_values[key]) == len(shared_ids))
        record(f"{name}: no missing values in SHAP or LIME",
               not shap_values[key].isna().any(axis=None) and not lime_values[key].isna().any(axis=None))

        lime_prob = lime_summary[lime_summary["model"] == key].set_index(SAMPLE_KEY)["prob_attack"]
        diff = (lime_prob.loc[shared_ids] - shap_meta[key]["predicted_probability"].loc[shared_ids]).abs().max()
        record(f"{name}: same model in SHAP and LIME (max P(Attack) difference {diff:.1e})", diff <= PROB_TOL)
    record("SHAP values add up to the model output for both models (checked while loading)", True)
    return checks


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def importance_order(values: pd.Series) -> list[str]:
    """Features sorted by |attribution|, largest first; ties keep the original feature order."""
    return list(values.abs().sort_values(ascending=False, kind="stable").index)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def compare_sample(shap_row: pd.Series, lime_row: pd.Series) -> dict:
    shap_order, lime_order = importance_order(shap_row), importance_order(lime_row)
    result = {"shap_top1": shap_order[0], "lime_top1": lime_order[0], "top1_same": shap_order[0] == lime_order[0]}
    for k in TOP_KS[1:]:
        shap_top, lime_top = set(shap_order[:k]), set(lime_order[:k])
        result[f"top{k}_overlap"] = len(shap_top & lime_top)
        result[f"top{k}_jaccard"] = jaccard(shap_top, lime_top)

    # Do both methods rank all 42 features in a similar order of importance?
    result["spearman_abs_all"] = float(spearmanr(shap_row.abs(), lime_row.abs()).statistic)

    # Among features both methods put in their top 10: do they agree on the direction?
    shared_top = [f for f in shap_order[:10] if f in set(lime_order[:10])]
    signed = [f for f in shared_top if shap_row[f] != 0 and lime_row[f] != 0]
    agree = [f for f in signed if np.sign(shap_row[f]) == np.sign(lime_row[f])]
    result["top10_shared_features"] = "; ".join(shared_top)
    result["top10_sign_agreement"] = len(agree) / len(signed) if signed else np.nan
    result["top10_sign_disagreements"] = "; ".join(f for f in signed if f not in agree)
    return result


def aligned_long(key: str, shap_df: pd.DataFrame, lime_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pos in shap_df.index:
        shap_rank = {f: r for r, f in enumerate(importance_order(shap_df.loc[pos]), start=1)}
        lime_rank = {f: r for r, f in enumerate(importance_order(lime_df.loc[pos]), start=1)}
        for f in shap_df.columns:
            s, l = shap_df.at[pos, f], lime_df.at[pos, f]
            rows.append({
                "model": key, SAMPLE_KEY: pos, "feature": f,
                "shap_value": s, "lime_weight": l,
                "shap_rank": shap_rank[f], "lime_rank": lime_rank[f],
                "same_sign": bool(np.sign(s) == np.sign(l)) if s != 0 and l != 0 else None,
            })
    return pd.DataFrame(rows)


def global_comparison(key: str, shap_df: pd.DataFrame, lime_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    shap_mean, lime_mean = shap_df.abs().mean(), lime_df.abs().mean()
    table = pd.DataFrame({
        "model": key,
        "feature": shap_df.columns,
        "shap_mean_abs": shap_mean.values,
        "lime_mean_abs": lime_mean.values,
        # Share of the total importance, so the two methods are on a comparable 0-1 scale.
        "shap_share": (shap_mean / shap_mean.sum()).values,
        "lime_share": (lime_mean / lime_mean.sum()).values,
    })
    shap_order, lime_order = importance_order(shap_mean), importance_order(lime_mean)
    table["shap_global_rank"] = table["feature"].map({f: r for r, f in enumerate(shap_order, 1)})
    table["lime_global_rank"] = table["feature"].map({f: r for r, f in enumerate(lime_order, 1)})
    table = table.sort_values("shap_global_rank").reset_index(drop=True)
    stats = {
        "shap_top10": shap_order[:10],
        "lime_top10": lime_order[:10],
        "top10_overlap": len(set(shap_order[:10]) & set(lime_order[:10])),
        "top10_jaccard": jaccard(set(shap_order[:10]), set(lime_order[:10])),
        "spearman_mean_abs": float(spearmanr(shap_mean, lime_mean).statistic),
    }
    return table, stats


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")
    ax.tick_params(colors=INK_SECONDARY)


def plot_per_sample_agreement(per_sample: pd.DataFrame) -> Path:
    """One dot per sample for each agreement measure; the black bar is the median."""
    metrics = [("top5_jaccard", "Top-5 overlap (Jaccard)"), ("top10_jaccard", "Top-10 overlap (Jaccard)"),
               ("spearman_abs_all", "Rank correlation, all 42 features"), ("top10_sign_agreement", "Sign agreement, shared top-10")]
    fig, axes = plt.subplots(1, len(MODELS), figsize=(11, 4.2), sharey=True, facecolor=SURFACE)
    rng = np.random.default_rng(0)  # only for vertical jitter of the dots
    for ax, (key, name) in zip(axes, MODELS.items()):
        style_axis(ax)
        data = per_sample[per_sample["model"] == key]
        for i, (col, _) in enumerate(metrics):
            vals = data[col].dropna().to_numpy()
            ax.scatter(vals, i + rng.uniform(-0.15, 0.15, len(vals)), s=36, color=COLOR_SHAP, alpha=0.55,
                       edgecolors=SURFACE, linewidths=1)
            med = np.median(vals)
            ax.plot([med, med], [i - 0.3, i + 0.3], color=INK_PRIMARY, linewidth=2)
            ax.text(med, i + 0.36, f"median {med:.2f}", ha="center", va="bottom", fontsize=8, color=INK_PRIMARY)
        ax.set_yticks(range(len(metrics)), [label for _, label in metrics], fontsize=9, color=INK_PRIMARY)
        ax.set_ylim(-0.6, len(metrics) - 0.25)
        ax.invert_yaxis()
        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel("Agreement (1 = SHAP and LIME identical)", color=INK_SECONDARY, fontsize=9)
        ax.set_title(name, loc="left", fontsize=11, color=INK_PRIMARY)
    fig.suptitle(f"SHAP vs LIME agreement per sample ({per_sample[SAMPLE_KEY].nunique()} shared malicious flows; one dot per flow)",
                 x=0.01, ha="left", fontsize=11.5, color=INK_PRIMARY)
    plt.tight_layout()
    path = FIGURE_DIR / "shap_vs_lime_per_sample_agreement.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_global_importance(global_tables: dict[str, pd.DataFrame]) -> Path:
    """Share of total |attribution| for the features in either method's global top 10."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(12, 6.2), facecolor=SURFACE)
    for ax, (key, name) in zip(axes, MODELS.items()):
        style_axis(ax)
        t = global_tables[key]
        t = t[(t["shap_global_rank"] <= 10) | (t["lime_global_rank"] <= 10)].sort_values("shap_global_rank")
        y = np.arange(len(t))
        h = 0.38
        ax.barh(y - h / 2, t["shap_share"], height=h, color=COLOR_SHAP, edgecolor=SURFACE, linewidth=2, label="SHAP")
        ax.barh(y + h / 2, t["lime_share"], height=h, color=COLOR_LIME, edgecolor=SURFACE, linewidth=2, label="LIME")
        labels = [f"{f}  (#{int(s)} / #{int(l)})" for f, s, l in zip(t["feature"], t["shap_global_rank"], t["lime_global_rank"])]
        ax.set_yticks(y, labels, fontsize=8.5, color=INK_PRIMARY)
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("Share of total importance (mean |attribution|)", color=INK_SECONDARY, fontsize=9)
        ax.set_title(name, loc="left", fontsize=11, color=INK_PRIMARY)
        ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    fig.suptitle("Global feature importance: SHAP vs LIME (labels show SHAP rank / LIME rank)",
                 x=0.01, ha="left", fontsize=11.5, color=INK_PRIMARY)
    plt.tight_layout()
    path = FIGURE_DIR / "shap_vs_lime_global_importance.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------------
def model_summary(per_sample: pd.DataFrame, global_stats: dict) -> pd.DataFrame:
    rows = []
    for key, name in MODELS.items():
        d = per_sample[per_sample["model"] == key]
        g = global_stats[key]
        rows.append({
            "model": key,
            "samples": len(d),
            "top1_same_count": int(d["top1_same"].sum()),
            "top5_overlap_mean": d["top5_overlap"].mean(),
            "top5_jaccard_mean": d["top5_jaccard"].mean(),
            "top10_overlap_mean": d["top10_overlap"].mean(),
            "top10_jaccard_mean": d["top10_jaccard"].mean(),
            "spearman_abs_all_mean": d["spearman_abs_all"].mean(),
            "spearman_abs_all_min": d["spearman_abs_all"].min(),
            "spearman_abs_all_max": d["spearman_abs_all"].max(),
            "top10_sign_agreement_mean": d["top10_sign_agreement"].mean(),
            "global_top10_overlap": g["top10_overlap"],
            "global_top10_jaccard": g["top10_jaccard"],
            "global_spearman_mean_abs": g["spearman_mean_abs"],
        })
    return pd.DataFrame(rows)


def write_summary(summary: pd.DataFrame, per_sample: pd.DataFrame, global_stats: dict,
                  checks: list[tuple[str, bool]], shap_kinds: dict) -> Path:
    n = per_sample[SAMPLE_KEY].nunique()
    lines = [
        "# SHAP vs LIME - UNSW-NB15 (Random Forest and XGBoost)",
        "",
        f"Both methods explain the same {n} correctly classified malicious validation flows "
        "(`results/metrics/shap/shared_validation_sample_identifiers.csv`) with the same fitted models. "
        "This comparison only reads saved results; nothing was retrained or re-explained.",
        "",
        "## How the comparison works",
        "",
        "- **Same features:** SHAP's one-hot columns for proto/service/state were summed into one value per "
        "original feature (SHAP values are additive), so both methods cover the same 42 original features.",
        f"- **Ranks and signs, not raw values:** SHAP output scale is {shap_kinds['random_forest']} for Random Forest "
        f"and {shap_kinds['xgboost']} (logit) for XGBoost; LIME weights come from a local linear surrogate. "
        "Raw magnitudes are therefore not comparable.",
        "- **Measures per flow:** same top-1 feature; top-5 / top-10 overlap (Jaccard = shared / union); "
        "Spearman rank correlation of |attribution| over all 42 features; sign agreement on the features "
        "both methods put in their top 10.",
        "- **Global:** features ranked by mean |attribution| over the flows, compared as share of total importance.",
        "",
        "## Input checks",
        "",
    ]
    lines += [f"- {'PASS' if ok else 'FAIL'}: {name}" for name, ok in checks]

    lines += [
        "",
        "## Results per model",
        "",
        "| Measure | Random Forest | XGBoost |",
        "|---|---|---|",
    ]
    s = summary.set_index("model")

    def row(label: str, fmt) -> None:
        lines.append(f"| {label} | {fmt(s.loc['random_forest'])} | {fmt(s.loc['xgboost'])} |")

    row("Same top-1 feature", lambda r: f"{r.top1_same_count} / {r.samples} flows")
    row("Top-5 overlap (mean)", lambda r: f"{r.top5_overlap_mean:.1f} of 5 (Jaccard {r.top5_jaccard_mean:.2f})")
    row("Top-10 overlap (mean)", lambda r: f"{r.top10_overlap_mean:.1f} of 10 (Jaccard {r.top10_jaccard_mean:.2f})")
    row("Spearman, all 42 features (mean, min-max)",
        lambda r: f"{r.spearman_abs_all_mean:.2f} ({r.spearman_abs_all_min:.2f} to {r.spearman_abs_all_max:.2f})")
    row("Sign agreement, shared top-10 (mean)", lambda r: f"{r.top10_sign_agreement_mean:.0%}")
    row("Global top-10 overlap", lambda r: f"{r.global_top10_overlap} of 10 (Jaccard {r.global_top10_jaccard:.2f})")
    row("Global Spearman (mean abs)", lambda r: f"{r.global_spearman_mean_abs:.2f}")

    lines += ["", "## Global top 10", ""]
    for key, name in MODELS.items():
        g = global_stats[key]
        both = [f for f in g["shap_top10"] if f in g["lime_top10"]]
        lines += [
            f"**{name}**",
            "",
            f"- SHAP: {', '.join(g['shap_top10'])}",
            f"- LIME: {', '.join(g['lime_top10'])}",
            f"- In both: {', '.join(both) if both else 'none'}",
            "",
        ]

    lines += ["## Flows where the top-1 feature differs", ""]
    diff = per_sample[~per_sample["top1_same"]]
    if diff.empty:
        lines.append("- None: SHAP and LIME name the same top-1 feature for every flow.")
    else:
        for key, name in MODELS.items():
            d = diff[diff["model"] == key]
            if len(d):
                pairs = ", ".join(f"{r[SAMPLE_KEY]} (SHAP {r.shap_top1} / LIME {r.lime_top1})" for _, r in d.iterrows())
                lines.append(f"- **{name}** ({len(d)}): {pairs}")

    lines += [
        "",
        "## Caveats",
        "",
        "- A LIME weight is about the quartile *range* the value falls in (e.g. `62 < sttl <= 254`); "
        "SHAP is about the exact value. Some disagreement is expected even when both are correct.",
        "- The reference data differs: SHAP used 100 validation rows as background, LIME used the "
        "sub-training split for its perturbation statistics.",
        "- LIME's local surrogate fit (R²) is only moderate, so LIME is an approximation of the model.",
        "- With 20 flows these numbers describe this sample set; they are a baseline for how much the two "
        "explainers naturally disagree, which the spoofing threshold must exceed.",
    ]
    path = OUTPUT_DIR / "shap_vs_lime_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
def main() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    FIGURE_DIR.mkdir(parents=True)

    print("Step 1: loading saved SHAP and LIME results")
    feature_names = json.loads((LIME_DIR / "lime_settings.json").read_text(encoding="utf-8"))["feature_names"]
    shared_ids = pd.read_csv(SHARED_SAMPLES_CSV)[SAMPLE_KEY].astype(int).tolist()
    lime_summary, lime_values = load_lime(feature_names)
    shap_meta, shap_values = {}, {}
    for key in MODELS:
        shap_meta[key], shap_values[key] = load_shap(key, feature_names)
        lime_values[key] = lime_values[key].loc[shap_values[key].index]  # same row order
    print(f"    {len(shared_ids)} shared IDs; SHAP grouped from one-hot columns into {len(feature_names)} original features")

    print("\nStep 2: input checks")
    checks = check_inputs(shared_ids, shap_values, shap_meta, lime_values, lime_summary)
    if not all(ok for _, ok in checks):
        raise SystemExit("Input checks failed; SHAP and LIME results are not comparable.")

    print("\nStep 3: comparing per sample and globally")
    per_sample_rows, aligned, global_tables, global_stats = [], [], {}, {}
    for key in MODELS:
        for pos in shared_ids:
            per_sample_rows.append({"model": key, SAMPLE_KEY: pos,
                                    **compare_sample(shap_values[key].loc[pos], lime_values[key].loc[pos])})
        aligned.append(aligned_long(key, shap_values[key], lime_values[key]))
        global_tables[key], global_stats[key] = global_comparison(key, shap_values[key], lime_values[key])
    per_sample = pd.DataFrame(per_sample_rows)
    summary = model_summary(per_sample, global_stats)
    print(summary.T.to_string(header=False))

    print("\nStep 4: saving results")
    outputs = {
        "shap_lime_aligned_attributions.csv": pd.concat(aligned, ignore_index=True),
        "shap_lime_per_sample_metrics.csv": per_sample,
        "shap_lime_global_importance.csv": pd.concat(global_tables.values(), ignore_index=True),
        "shap_lime_model_summary.csv": summary,
    }
    for name, df in outputs.items():
        df.to_csv(OUTPUT_DIR / name, index=False, float_format="%.12g")
        print(f"Saved: {OUTPUT_DIR / name}")
    print(f"Saved: {plot_per_sample_agreement(per_sample)}")
    print(f"Saved: {plot_global_importance(global_tables)}")
    shap_kinds = {key: shap_meta[key]["output_kind"].iloc[0] for key in MODELS}
    print(f"Saved: {write_summary(summary, per_sample, global_stats, checks, shap_kinds)}")


if __name__ == "__main__":
    main()
