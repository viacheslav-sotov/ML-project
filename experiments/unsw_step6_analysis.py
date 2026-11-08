"""Step 6 analysis: targeted vs random explanation-spoofing search (tables, figures, checks).

Only saved results are read: no model is loaded, no LIME or SHAP explanation is recomputed and the
Step 6 run directory is never written to.

Inputs (a packaged Step 6 run, e.g. results/step6_shared/run_20261106_203918)
    results.csv, checkpoints.csv, best_so_far.csv, run_config.json, status.json,
    heldout/heldout_summary.csv, heldout/heldout_per_seed.csv, heldout/noise_bands.csv,
    heldout/run_metadata.json
Optional context (skipped with a note when missing)
    results/lime_stability/lime_stability_pairwise.csv            original-vs-original LIME seed pairs
    results/unsw_nb15/shap_vs_lime/shap_lime_per_sample_metrics.csv SHAP vs LIME on the original samples

Terms used in every output
    targeted     canonical (greedy_baseline_ref_v1), shuffled (greedy_shuffled_v1),
                 multistart (multistart_greedy_v1)
    random       the random control (best-of-budget random valid candidates, same objective)
    checkpoint   explanation budget c (25 or 50 LIME calls, baseline included): the best candidate
                 found within the first c calls. Rows of checkpoints.csv / heldout_summary.csv.
    search-seed objective   the score the search maximised, under LIME seed 11 only:
                            0.7*(1 - top10_jaccard) + 0.3*(1 - spearman_rho)/2 (in-sample)
    held-out top-10 shift   1 - mean top-10 Jaccard(original, candidate) over LIME seeds 22/33/44/55
    held-out Spearman shift 1 - mean Spearman rho(original, candidate) over the same seeds
    seed-noise test         Part B's rule on held-out seeds: the candidate's mean is strictly below
                            the MINIMUM of the original sample's own seed-pair values
                            (exceeds_heldout_top10 / _spearman; "both" = the combined test)

An explanation change is never called successful spoofing here. A candidate would only count as a
spoofing candidate if it (1) changed the explanation, (2) stayed an attack and met the enforced
constraints, (3) passed the combined held-out seed-noise test and (4) violated no checked physical
rule. The funnel table reports how many candidates reach each stage.

The targeted-vs-random statistics are exploratory: 3 samples, 10 seeds per method and sample, no
multiplicity correction. Differences are averaged within sample (each sample weighted equally),
bootstrap intervals resample within sample, and permutation p-values permute labels within sample.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "results" / "step6_shared" / "run_20261106_203918"
LIME_STABILITY_PAIRWISE = ROOT / "results" / "lime_stability" / "lime_stability_pairwise.csv"
SHAP_LIME_PER_SAMPLE = ROOT / "results" / "unsw_nb15" / "shap_vs_lime" / "shap_lime_per_sample_metrics.csv"

SEARCH_ORDER = ["canonical", "shuffled", "multistart", "random"]
TARGETED = ("canonical", "shuffled", "multistart")
PREFIX_METHODS = ("canonical", "shuffled", "random")  # 25-call results read from the 50-call run
SEARCH_LABELS = {
    "canonical": "Canonical greedy",
    "shuffled": "Shuffled greedy",
    "multistart": "Multi-start greedy",
    "random": "Random control",
}
SEED_COLUMN = {"canonical": None, "shuffled": "order_seed", "multistart": "order_seed", "random": "control_seed"}

METRICS = {  # column -> label; larger = bigger explanation change for every metric listed here
    "search_seed_objective": "Search-seed objective (LIME seed 11)",
    "heldout_top10_shift": "Held-out top-10 shift (1 - Jaccard)",
    "heldout_spearman_shift": "Held-out Spearman shift (1 - rho)",
}
FLAGS = {
    "exceeds_heldout_top10": "held-out top-10 beyond seed noise",
    "exceeds_heldout_spearman": "held-out Spearman beyond seed noise",
    "exceeds_heldout_both": "combined held-out test (top-10 AND Spearman)",
    "exceeds_heldout_top1_rule": "held-out top-1 rule",
    "partb_exceeds_seed_noise": "Part B combined test (seeds 11-55, not held-out)",
}
PHYSICAL_STATUS_ORDER = ["violates_checked_rule", "unverified_dependencies", "consistent_with_checked_rules"]

REQUIRED_COLUMNS = {
    "results": ["task_id", "search", "method", "strategy", "model_key", "sample_position", "budget", "explanation_budget",
                "order_seed", "control_seed", "objective_score", "explanation_calls", "still_attack", "constraint_valid",
                "stopped_reason", "accounting_consistent", "best_probability", "original_probability"],
    "checkpoints": ["task_id", "search", "checkpoint", "checkpoint_valid", "budget_reached", "best_score",
                    "candidate_json", "candidate_is_original", "best_candidate_probability", "explanation_calls_at_best",
                    "sample_position", "order_seed", "control_seed"],
    "best_so_far": ["task_id", "search", "explanation_budget", "explanation_calls", "score", "best_so_far"],
    "heldout_summary": ["candidate_uid", "task_id", "search", "checkpoint", "sample_position", "order_seed", "control_seed",
                        "search_seed_objective", "candidate_is_original", "changed_features", "experimental_valid",
                        "teammate_table_status", "physical_status", "physical_violations", "physical_unverified",
                        "heldout_top10_jaccard_mean", "heldout_top10_shift", "heldout_noise_top10_jaccard_min",
                        "heldout_spearman_mean", "heldout_top5_jaccard_mean", "heldout_top1_agreement_mean",
                        "heldout_sign_agreement_top10_mean", "heldout_top1_changed_seeds", *FLAGS,
                        "search_score_reproduced"],
    "heldout_per_seed": ["candidate_uid", "seed_set", "seed", "top10_jaccard", "spearman_rho", "spearman_abs_tieaware",
                         "top1_agreement"],
    "noise_bands": ["model_key", "sample_position", "band", "n_pairs", "top10_jaccard_min", "top10_jaccard_mean",
                    "top10_jaccard_max", "spearman_rho_min", "spearman_rho_mean", "spearman_rho_max",
                    "top1_stable_all_pairs", "top10_jaccard_pair_values"],
}
CORE_METRIC_COLUMNS = ["search_seed_objective", "heldout_top10_jaccard_mean", "heldout_spearman_mean", *FLAGS]

BOOTSTRAP_REPS = 10_000
PERMUTATION_REPS = 10_000
STAT_SEED = 20261106
SCORE_TOL = 1e-9

# Categorical slots 1 and 2 of the dataviz reference palette (validated together, all pairs).
COLOR_TARGETED = "#2a78d6"
COLOR_RANDOM = "#eb6834"
COLOR_NOISE = "#8a8984"
STATUS_COLORS = {"violates_checked_rule": "#d03b3b", "unverified_dependencies": "#fab219",
                 "consistent_with_checked_rules": "#0ca30c"}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SAMPLE_MARKERS = {2941: "o", 3206: "s", 4407: "^"}


class AnalysisInputError(ValueError):
    """Raised when a required input file or column is missing."""


# ---------------------------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------------------------
def load_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    files = {
        "results": run_dir / "results.csv",
        "checkpoints": run_dir / "checkpoints.csv",
        "best_so_far": run_dir / "best_so_far.csv",
        "heldout_summary": run_dir / "heldout" / "heldout_summary.csv",
        "heldout_per_seed": run_dir / "heldout" / "heldout_per_seed.csv",
        "noise_bands": run_dir / "heldout" / "noise_bands.csv",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    for name in ("run_config.json", "status.json", "heldout/run_metadata.json"):
        if not (run_dir / name).exists():
            missing.append(str(run_dir / name))
    if missing:
        raise AnalysisInputError(f"missing input files: {missing}")
    data: dict[str, Any] = {name: pd.read_csv(path) for name, path in files.items()}
    data["run_config"] = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    data["status"] = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    data["heldout_metadata"] = json.loads((run_dir / "heldout" / "run_metadata.json").read_text(encoding="utf-8"))
    data["run_dir"] = run_dir
    return data


def check_required_columns(data: dict[str, Any]) -> None:
    problems = {name: sorted(set(cols) - set(data[name].columns)) for name, cols in REQUIRED_COLUMNS.items()}
    problems = {k: v for k, v in problems.items() if v}
    if problems:
        raise AnalysisInputError(f"required columns missing: {problems}")


def expected_tasks(selection: dict[str, Any]) -> dict[str, int]:
    """Task count per search implied by the runner's selection (see experiments/unsw_step6_runner.py)."""
    configs = len(selection["models"]) * len(selection["samples"]) * len(selection["budgets"])
    counts = {
        "canonical": configs,
        "shuffled": configs * len(selection["order_seeds"]),
        "multistart": configs * len(selection["order_seeds"]) * len(selection["explanation_budgets"]),
        "random": configs * len(selection["control_seeds"]),
    }
    return {m: n for m, n in counts.items() if m in selection["methods"]}


def expected_checkpoints(selection: dict[str, Any]) -> dict[str, int]:
    """Prefix methods give one checkpoint per explanation budget; multi-start only its own budget."""
    tasks = expected_tasks(selection)
    per_task = len(selection["explanation_budgets"])
    return {m: (n if m == "multistart" else n * per_task) for m, n in tasks.items()}


def validate_run(data: dict[str, Any]) -> pd.DataFrame:
    """Completeness and consistency checks. Returns one row per check; never drops data silently."""
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    results, cps, summary = data["results"], data["checkpoints"], data["heldout_summary"]
    per_seed, status, selection = data["heldout_per_seed"], data["status"], data["run_config"]["selection"]

    add("status.json: completed == planned, no failures",
        status["completed"] == status["planned"] and not status["failed"],
        f"completed={status['completed']}, planned={status['planned']}, failed={status['failed']}")
    exp_tasks = expected_tasks(selection)
    got_tasks = results.groupby("search")["task_id"].nunique().to_dict()
    add("results.csv: task count per search matches the selection", got_tasks == exp_tasks,
        f"expected {exp_tasks}, found {got_tasks} (total {results['task_id'].nunique()})")
    add("results.csv: task_id unique", results["task_id"].is_unique, f"{len(results)} rows")
    for col, want in (("stopped_reason", "explanation_budget_exhausted"),):
        bad = int((results[col] != want).sum())
        add(f"results.csv: every task {col} == {want}", bad == 0, f"{bad} task(s) differ")
    for col in ("still_attack", "constraint_valid", "accounting_consistent"):
        bad = int((~results[col].astype(bool)).sum())
        add(f"results.csv: {col} is True for every task", bad == 0, f"{bad} task(s) False")
    over = int((results["explanation_calls"] > results["explanation_budget"]).sum())
    add("results.csv: explanation calls within budget", over == 0, f"{over} task(s) over budget")

    exp_cps = expected_checkpoints(selection)
    got_cps = cps.groupby("search").size().to_dict()
    add("checkpoints.csv: checkpoint rows per search match the selection", got_cps == exp_cps,
        f"expected {exp_cps}, found {got_cps}")
    add("checkpoints.csv: all checkpoints valid and reached",
        bool(cps["checkpoint_valid"].all() and cps["budget_reached"].all()),
        f"invalid={int((~cps['checkpoint_valid']).sum())}, unreached={int((~cps['budget_reached']).sum())}")
    add("checkpoints.csv: (task_id, checkpoint) unique", not cps.duplicated(["task_id", "checkpoint"]).any(), "")

    merged = cps.merge(summary[["task_id", "checkpoint", "search_seed_objective"]], on=["task_id", "checkpoint"],
                       how="outer", indicator=True)
    unmatched = int((merged["_merge"] != "both").sum())
    add("heldout_summary.csv joins checkpoints.csv 1:1 on (task_id, checkpoint)",
        unmatched == 0 and len(summary) == len(cps), f"{len(summary)} held-out rows, {len(cps)} checkpoints, {unmatched} unmatched")
    gap = float((merged["best_score"] - merged["search_seed_objective"]).abs().max())
    add("held-out search_seed_objective equals checkpoint best_score", gap <= SCORE_TOL, f"max gap {gap:.2e}")
    add("held-out evaluation reproduced every search score",
        bool(summary["search_score_reproduced"].astype(bool).all()),
        f"{int((~summary['search_score_reproduced'].astype(bool)).sum())} not reproduced")

    final = cps[cps["checkpoint"] == cps["task_id"].map(results.set_index("task_id")["explanation_budget"])]
    final = final.merge(results[["task_id", "objective_score"]], on="task_id")
    gap = float((final["best_score"] - final["objective_score"]).abs().max())
    add("checkpoint at the run budget equals results.csv objective_score", gap <= SCORE_TOL, f"max gap {gap:.2e}")

    heldout_seeds = data["heldout_metadata"]["heldout_seeds"]
    counts = per_seed.groupby(["candidate_uid", "seed_set"]).size().unstack(fill_value=0)
    ok = (counts.get("heldout", 0) == len(heldout_seeds)).all() and (counts.get("search_seed", 0) == 1).all()
    add("heldout_per_seed.csv: 4 held-out rows + 1 search-seed row per candidate",
        bool(ok) and len(counts) == len(summary), f"{len(counts)} candidates")
    recomputed = per_seed[per_seed["seed_set"] == "heldout"].groupby("candidate_uid")["top10_jaccard"].mean()
    gap = float((summary.set_index("candidate_uid")["heldout_top10_jaccard_mean"] - recomputed).abs().max())
    add("heldout_top10_jaccard_mean equals the mean of the per-seed rows", gap <= SCORE_TOL, f"max gap {gap:.2e}")

    na = summary[CORE_METRIC_COLUMNS].isna().sum()
    add("no missing values in core metric columns", int(na.sum()) == 0,
        "; ".join(f"{k}={v}" for k, v in na.items() if v) or "none")
    add("no checkpoint candidate is the unchanged original",
        not bool(summary["candidate_is_original"].astype(bool).any()),
        f"{int(summary['candidate_is_original'].astype(bool).sum())} original")
    bands = data["noise_bands"]
    add("noise bands present for every sample (held-out and Part B)",
        set(map(tuple, bands[["sample_position", "band"]].values)) ==
        {(s, b) for s in selection["samples"] for b in ("heldout", "part_b")}, f"{len(bands)} rows")
    return pd.DataFrame(checks)


def missing_value_report(data: dict[str, Any]) -> pd.DataFrame:
    """Missing values per file and column, with the structural reason where one is known."""
    expected_reason = {
        "order_seed": "not seeded by an order seed (canonical, random)",
        "control_seed": "only the random control has a control seed",
        "restarts": "only multi-start has restarts",
        "restart_id": "only multi-start/shuffled histories have restart ids",
        "attempts": "only the random control records attempts",
        "experimental_reasons": "empty = no experimental-policy violation",
        "constraint_experimental_reasons": "empty = no experimental-policy violation",
        "validator_warnings": "empty = no validator warning",
        "constraint_validator_warnings": "empty = no validator warning",
        "physical_violations": "empty = no checked physical rule violated",
        "constraint_physical_violations": "empty = no checked physical rule violated",
        "physical_unverified": "empty = no changed feature with an unverified dependency",
        "constraint_physical_unverified": "empty = no changed feature with an unverified dependency",
        "physical_rules_failing_on_original": "empty = every checked rule holds on the original",
        "constraint_physical_rules_failing_on_original": "empty = every checked rule holds on the original",
        "best_changed_features": "restart that found no improvement",
        **{col: "multi-start/shuffled bookkeeping; not defined for canonical and random"
           for col in ("cache_hits", "cache_hits_rejected", "cache_hits_scored", "deferred_at_cap",
                       "restarts_completed", "restarts_started", "unique_candidates")},
    }
    rows = []
    for name in ("results", "checkpoints", "best_so_far", "heldout_summary", "heldout_per_seed", "noise_bands"):
        frame = data[name]
        for col, n in frame.isna().sum().items():
            if n:
                rows.append({"file": name, "column": col, "missing": int(n), "rows": len(frame),
                             "explanation": expected_reason.get(col, "UNEXPECTED - inspect")})
    return pd.DataFrame(rows, columns=["file", "column", "missing", "rows", "explanation"])


# ---------------------------------------------------------------------------------------------
# Candidate table
# ---------------------------------------------------------------------------------------------
def build_candidates(data: dict[str, Any]) -> pd.DataFrame:
    """One row per checkpoint candidate (held-out summary + checkpoint fields + per-seed spread)."""
    summary = data["heldout_summary"].copy()
    cps = data["checkpoints"][["task_id", "checkpoint", "candidate_json", "best_candidate_probability",
                               "explanation_calls_at_best"]]
    cand = summary.merge(cps, on=["task_id", "checkpoint"], how="left", validate="one_to_one")
    per_seed = data["heldout_per_seed"]
    held = per_seed[per_seed["seed_set"] == "heldout"].groupby("candidate_uid").agg(
        heldout_top10_jaccard_sd=("top10_jaccard", "std"),
        heldout_spearman_sd=("spearman_rho", "std"),
        heldout_spearman_abs_tieaware_mean=("spearman_abs_tieaware", "mean"),
    )
    cand = cand.merge(held, left_on="candidate_uid", right_index=True, how="left")
    band = data["noise_bands"]
    band = band[band["band"] == "heldout"].set_index("sample_position")
    cand["heldout_noise_spearman_min"] = cand["sample_position"].map(band["spearman_rho_min"])
    cand["heldout_spearman_shift"] = 1.0 - cand["heldout_spearman_mean"]
    cand["top10_margin_to_noise"] = cand["heldout_top10_jaccard_mean"] - cand["heldout_noise_top10_jaccard_min"]
    cand["spearman_margin_to_noise"] = cand["heldout_spearman_mean"] - cand["heldout_noise_spearman_min"]
    cand["group"] = np.where(cand["search"].isin(TARGETED), "targeted", "random")
    cand["seed"] = [row[SEED_COLUMN[row["search"]]] if SEED_COLUMN[row["search"]] else np.nan
                    for _, row in cand.iterrows()]
    cand["explanation_changed_search_seed"] = cand["search_seed_objective"] > 0
    cand["unique_candidate_key"] = cand["sample_position"].astype(str) + "|" + cand["candidate_json"].astype(str)
    cand["search"] = pd.Categorical(cand["search"], SEARCH_ORDER, ordered=True)
    return cand.sort_values(["checkpoint", "search", "sample_position", "seed"]).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------------------------
def describe(values: pd.Series) -> dict[str, float]:
    v = pd.to_numeric(values, errors="coerce").dropna()
    return {"n": int(len(v)), "mean": float(v.mean()) if len(v) else np.nan,
            "median": float(v.median()) if len(v) else np.nan,
            "sd": float(v.std(ddof=1)) if len(v) > 1 else np.nan,
            "min": float(v.min()) if len(v) else np.nan, "max": float(v.max()) if len(v) else np.nan}


def method_summary(cand: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for keys, grp in cand.groupby(by, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row: dict[str, Any] = dict(zip(by, keys))
        row["n_candidates"] = len(grp)
        row["n_samples"] = grp["sample_position"].nunique()
        for col in METRICS:
            for stat, value in describe(grp[col]).items():
                if stat != "n":
                    row[f"{col}_{stat}"] = value
        for flag in FLAGS:
            row[f"{flag}_count"] = int(grp[flag].astype(bool).sum())
            row[f"{flag}_rate"] = float(grp[flag].astype(bool).mean())
        row["changed_explanation_search_seed_count"] = int(grp["explanation_changed_search_seed"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def seed_summary(cand: pd.DataFrame) -> pd.DataFrame:
    """Per order/control seed (pooled over the 3 samples): does any seed stand out?"""
    seeded = cand[cand["seed"].notna()]
    rows = []
    for (search, checkpoint, seed), grp in seeded.groupby(["search", "checkpoint", "seed"], observed=True):
        rows.append({"search": search, "checkpoint": checkpoint, "seed_kind": SEED_COLUMN[search], "seed": int(seed),
                     "n_samples": len(grp),
                     **{f"{c}_mean": float(grp[c].mean()) for c in METRICS},
                     "exceeds_heldout_top10_count": int(grp["exceeds_heldout_top10"].sum()),
                     "exceeds_heldout_both_count": int(grp["exceeds_heldout_both"].sum())})
    return pd.DataFrame(rows)


def stratified_comparison(values_a: dict[Any, np.ndarray], values_b: dict[Any, np.ndarray],
                          rng: np.random.Generator) -> dict[str, float]:
    """Group A vs group B within strata (samples), each stratum weighted equally.

    diff            mean over strata of (mean A - mean B)
    ci_low/high     2.5/97.5% of the stratified bootstrap (resample A and B within each stratum)
    p_superiority   mean over strata of P(a > b) + 0.5 P(a = b) over all within-stratum pairs
    perm_p          two-sided stratified permutation p-value for diff (labels permuted within stratum)
    """
    strata = sorted(set(values_a) & set(values_b))
    if not strata:
        return {"n_strata": 0, "n_a": 0, "n_b": 0, "diff": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "p_superiority": np.nan, "perm_p": np.nan}
    a = [np.asarray(values_a[s], float) for s in strata]
    b = [np.asarray(values_b[s], float) for s in strata]
    diff = float(np.mean([x.mean() - y.mean() for x, y in zip(a, b)]))
    sup = float(np.mean([((x[:, None] > y[None, :]) + 0.5 * (x[:, None] == y[None, :])).mean() for x, y in zip(a, b)]))
    boots = np.zeros(BOOTSTRAP_REPS)
    for x, y in zip(a, b):
        bx = x[rng.integers(0, len(x), (BOOTSTRAP_REPS, len(x)))].mean(axis=1)
        by = y[rng.integers(0, len(y), (BOOTSTRAP_REPS, len(y)))].mean(axis=1)
        boots += (bx - by) / len(strata)
    perm = np.zeros(PERMUTATION_REPS)
    for x, y in zip(a, b):
        pooled = np.concatenate([x, y])
        idx = np.argsort(rng.random((PERMUTATION_REPS, len(pooled))), axis=1)
        shuffled = pooled[idx]
        perm += (shuffled[:, :len(x)].mean(axis=1) - shuffled[:, len(x):].mean(axis=1)) / len(strata)
    perm_p = float((np.sum(np.abs(perm) >= abs(diff) - 1e-12) + 1) / (PERMUTATION_REPS + 1))
    return {"n_strata": len(strata), "n_a": int(sum(len(x) for x in a)), "n_b": int(sum(len(y) for y in b)),
            "diff": diff, "ci_low": float(np.quantile(boots, 0.025)), "ci_high": float(np.quantile(boots, 0.975)),
            "p_superiority": sup, "perm_p": perm_p}


def targeted_vs_random(cand: pd.DataFrame, seed: int = STAT_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for checkpoint in sorted(cand["checkpoint"].unique()):
        at = cand[cand["checkpoint"] == checkpoint]
        rand = at[at["search"] == "random"]
        for method in (*TARGETED, "all_targeted"):
            targ = at[at["search"].isin(TARGETED)] if method == "all_targeted" else at[at["search"] == method]
            for metric, label in METRICS.items():
                va = {s: g[metric].to_numpy() for s, g in targ.groupby("sample_position")}
                vb = {s: g[metric].to_numpy() for s, g in rand.groupby("sample_position")}
                stats = stratified_comparison(va, vb, rng)
                rows.append({"checkpoint": checkpoint, "targeted_method": method, "metric": metric, "metric_label": label,
                             "targeted_mean": float(targ[metric].mean()), "random_mean": float(rand[metric].mean()),
                             "targeted_median": float(targ[metric].median()), "random_median": float(rand[metric].median()),
                             **stats,
                             "samples_targeted_mean_higher": int(sum(va[s].mean() > vb[s].mean() for s in va if s in vb)),
                             "samples_compared": int(sum(s in vb for s in va))})
    out = pd.DataFrame(rows)
    out["ci_excludes_zero"] = (out["ci_low"] > 0) | (out["ci_high"] < 0)
    return out


def targeted_vs_random_by_sample(cand: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (checkpoint, sample), at in cand.groupby(["checkpoint", "sample_position"]):
        rand = at[at["search"] == "random"]
        for method in TARGETED:
            targ = at[at["search"] == method]
            for metric in METRICS:
                rows.append({"checkpoint": checkpoint, "sample_position": sample, "targeted_method": method,
                             "metric": metric, "n_targeted": len(targ), "n_random": len(rand),
                             "targeted_mean": float(targ[metric].mean()), "random_mean": float(rand[metric].mean()),
                             "difference": float(targ[metric].mean() - rand[metric].mean()),
                             "random_rank_pct_of_targeted_mean": float((rand[metric] < targ[metric].mean()).mean()
                                                                       + 0.5 * (rand[metric] == targ[metric].mean()).mean())})
    return pd.DataFrame(rows)


def budget_effect(cand: pd.DataFrame) -> pd.DataFrame:
    """25 -> 50 calls, paired by (sample, seed). Prefix methods: same task (search-seed objective cannot
    decrease by construction). Multi-start: separate runs with the same order seed."""
    rows = []
    key = ["sample_position", "seed_key"]
    frame = cand.assign(seed_key=cand["seed"].fillna(-1),
                        exceeds_heldout_top10=cand["exceeds_heldout_top10"].astype(int))
    for method in SEARCH_ORDER:
        sub = frame[frame["search"] == method]
        wide = sub.pivot_table(index=key, columns="checkpoint", values=list(METRICS) + ["exceeds_heldout_top10"],
                               aggfunc="first", observed=True)
        for metric in [*METRICS, "exceeds_heldout_top10"]:
            if 25 not in wide[metric] or 50 not in wide[metric]:
                continue
            pair = wide[metric][[25, 50]].dropna().astype(float)
            delta = pair[50] - pair[25]
            rows.append({"search": method, "metric": metric, "pairing": "same task" if method in PREFIX_METHODS
                         else "separate runs, same order seed", "n_pairs": len(pair),
                         "mean_at_25": float(pair[25].mean()), "mean_at_50": float(pair[50].mean()),
                         "mean_change": float(delta.mean()), "median_change": float(delta.median()),
                         "n_increased": int((delta > 1e-12).sum()), "n_unchanged": int((delta.abs() <= 1e-12).sum()),
                         "n_decreased": int((delta < -1e-12).sum())})
    return pd.DataFrame(rows)


def best_so_far_curves(data: dict[str, Any]) -> pd.DataFrame:
    """Mean/quantiles of the best search-seed objective after c explanation calls, 50-call tasks only."""
    bsf = data["best_so_far"]
    bsf = bsf[bsf["explanation_budget"] == 50]
    rows = []
    for (search, calls), grp in bsf.groupby(["search", "explanation_calls"]):
        rows.append({"search": search, "explanation_calls": int(calls), "n_tasks": grp["task_id"].nunique(),
                     "mean": float(grp["best_so_far"].mean()), "median": float(grp["best_so_far"].median()),
                     "q25": float(grp["best_so_far"].quantile(0.25)), "q75": float(grp["best_so_far"].quantile(0.75))})
    return pd.DataFrame(rows)


def best_so_far_is_monotone(data: dict[str, Any]) -> bool:
    bsf = data["best_so_far"].sort_values(["task_id", "explanation_calls"])
    return bool((bsf.groupby("task_id")["best_so_far"].diff().fillna(0) >= -1e-15).all())


def lime_stability_tables(data: dict[str, Any], cand: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(1) original-vs-original seed noise per sample next to the candidates' held-out change;
    (2) the original's held-out seed pairs (from results/lime_stability) for plotting, if available."""
    bands = data["noise_bands"].copy()
    rows = []
    for _, band in bands.iterrows():
        sub = cand[cand["sample_position"] == band["sample_position"]]
        rows.append({
            "sample_position": band["sample_position"], "band": band["band"], "seed_pairs": band["n_pairs"],
            "noise_top1_stable_all_pairs": band["top1_stable_all_pairs"],
            "noise_top10_jaccard_min": band["top10_jaccard_min"], "noise_top10_jaccard_mean": band["top10_jaccard_mean"],
            "noise_spearman_min": band["spearman_rho_min"], "noise_spearman_mean": band["spearman_rho_mean"],
            "noise_top5_jaccard_mean": band["top5_jaccard_mean"],
            "noise_top1_agreement_mean": band["top1_agreement_mean"],
            "candidates": len(sub),
            "candidate_top10_jaccard_mean_median": float(sub["heldout_top10_jaccard_mean"].median()),
            "candidate_top10_jaccard_mean_min": float(sub["heldout_top10_jaccard_mean"].min()),
            "candidate_spearman_mean_median": float(sub["heldout_spearman_mean"].median()),
            "candidate_spearman_mean_min": float(sub["heldout_spearman_mean"].min()),
            "candidate_top10_sd_across_seeds_mean": float(sub["heldout_top10_jaccard_sd"].mean()),
            "candidate_spearman_sd_across_seeds_mean": float(sub["heldout_spearman_sd"].mean()),
        })
    noise = pd.DataFrame(rows)
    pairs = pd.DataFrame()
    if LIME_STABILITY_PAIRWISE.exists():
        pw = pd.read_csv(LIME_STABILITY_PAIRWISE)
        held = set(data["heldout_metadata"]["heldout_seeds"])
        model = cand["model_key"].iloc[0]
        pairs = pw[(pw["model"] == model) & pw["validation_position"].isin(cand["sample_position"].unique())
                   & pw["seed_a"].isin(held) & pw["seed_b"].isin(held)].copy()
    return noise, pairs


def noise_pairs_match_bands(pairs: pd.DataFrame, data: dict[str, Any]) -> tuple[bool, float]:
    """The saved Part B seed pairs must reproduce the held-out noise bands (same LIME, same seeds)."""
    if pairs.empty:
        return False, np.nan
    bands = data["noise_bands"].query("band == 'heldout'").set_index("sample_position")
    gap = 0.0
    for sample, grp in pairs.groupby("validation_position"):
        gap = max(gap, abs(grp["spearman_rho"].min() - bands.loc[sample, "spearman_rho_min"]),
                  abs(grp["top10_jaccard"].min() - bands.loc[sample, "top10_jaccard_min"]))
    return gap <= 1e-6, float(gap)


def shap_lime_context(cand: pd.DataFrame, noise: pd.DataFrame) -> pd.DataFrame:
    """Top-10 Jaccard distance (same definition everywhere) for three sources of explanation change."""
    if not SHAP_LIME_PER_SAMPLE.exists():
        return pd.DataFrame()
    sl = pd.read_csv(SHAP_LIME_PER_SAMPLE)
    model = cand["model_key"].iloc[0]
    sl = sl[(sl["model"] == model) & sl["validation_position"].isin(cand["sample_position"].unique())]
    held = noise[noise["band"] == "heldout"].set_index("sample_position")
    rows = []
    for _, r in sl.iterrows():
        s = int(r["validation_position"])
        sub = cand[cand["sample_position"] == s]
        rows.append({
            "sample_position": s,
            "spoof_candidates_top10_distance_median": float(1 - sub["heldout_top10_jaccard_mean"].median()),
            "spoof_candidates_top10_distance_max": float(1 - sub["heldout_top10_jaccard_mean"].min()),
            "lime_seed_noise_top10_distance_mean": float(1 - held.loc[s, "noise_top10_jaccard_mean"]),
            "lime_seed_noise_top10_distance_max": float(1 - held.loc[s, "noise_top10_jaccard_min"]),
            "shap_vs_lime_top10_distance": float(1 - r["top10_jaccard"]),
            "spoof_candidates_spearman_abs_tieaware_median": float(sub["heldout_spearman_abs_tieaware_mean"].median()),
            "shap_vs_lime_spearman_abs": float(r["spearman_abs_all"]),
            "shap_top1": r["shap_top1"], "lime_top1": r["lime_top1"],
        })
    return pd.DataFrame(rows)


def split_rules(text: Any) -> list[str]:
    """'rule A [support; more]; rule B [support]' -> ['rule A', 'rule B'] (support text can contain ';')."""
    if not isinstance(text, str) or not text.strip():
        return []
    stripped = re.sub(r"\s*\[[^\]]*\]", "", text)
    return [part.strip() for part in stripped.split(";") if part.strip()]


def split_unverified(text: Any) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    return [m.group(1) for m in re.finditer(r"(?:^|; )([a-z_]+): ", text)]


def constraint_tables(cand: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    status = (cand.groupby(["checkpoint", "search"], observed=True)["physical_status"].value_counts()
              .unstack(fill_value=0).reindex(columns=PHYSICAL_STATUS_ORDER, fill_value=0).reset_index())
    status["experimental_valid"] = cand.groupby(["checkpoint", "search"], observed=True)["experimental_valid"].sum().values
    status["n_candidates"] = cand.groupby(["checkpoint", "search"], observed=True).size().values
    teammate = (cand.groupby(["checkpoint", "search"], observed=True)["teammate_table_status"].value_counts()
                .unstack(fill_value=0).add_prefix("teammate_").reset_index())
    out["status"] = status.merge(teammate, on=["checkpoint", "search"])

    unique = cand.drop_duplicates("unique_candidate_key")
    rows = []
    for scope, frame in (("all_checkpoint_rows", cand), ("unique_candidates", unique)):
        for name, col, fn in (("violated_rule", "physical_violations", split_rules),
                              ("unverified_feature", "physical_unverified", split_unverified)):
            counts: dict[str, int] = {}
            for text in frame[col]:
                for item in set(fn(text)):
                    counts[item] = counts.get(item, 0) + 1
            rows += [{"scope": scope, "kind": name, "item": k, "candidates": v, "of": len(frame)}
                     for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
        feats: dict[str, int] = {}
        viol: dict[str, int] = {}
        for _, r in frame.iterrows():
            for f in str(r["changed_features"]).split(";"):
                if f and f != "nan":
                    feats[f] = feats.get(f, 0) + 1
                    if r["physical_status"] == "violates_checked_rule":
                        viol[f] = viol.get(f, 0) + 1
        rows += [{"scope": scope, "kind": "changed_feature", "item": k, "candidates": v, "of": len(frame),
                  "of_which_violate_checked_rule": viol.get(k, 0)} for k, v in sorted(feats.items(), key=lambda kv: -kv[1])]
    out["details"] = pd.DataFrame(rows)
    out["unique_status"] = unique["physical_status"].value_counts().reindex(PHYSICAL_STATUS_ORDER, fill_value=0) \
        .rename_axis("physical_status").reset_index(name="unique_candidates")
    return out


FUNNEL_STAGES = [
    ("1. checkpoint candidates", lambda c: pd.Series(True, index=c.index)),
    ("2. explanation changed under search seed", lambda c: c["explanation_changed_search_seed"]),
    ("3. + still attack and meets enforced constraints",
     lambda c: c["explanation_changed_search_seed"] & c["experimental_valid"].astype(bool)
     & (c["best_candidate_probability"] >= 0.5)),
    ("4. + held-out top-10 beyond seed noise", lambda c: c["exceeds_heldout_top10"].astype(bool)),
    ("5. + held-out Spearman beyond seed noise (combined test)", lambda c: c["exceeds_heldout_both"].astype(bool)),
    ("6. + no checked physical rule violated", lambda c: c["physical_status"] != "violates_checked_rule"),
    ("7. + all dependencies verified (consistent_with_checked_rules)",
     lambda c: c["physical_status"] == "consistent_with_checked_rules"),
]


def spoofing_funnel(cand: pd.DataFrame) -> pd.DataFrame:
    """Cumulative stages from 'explanation changed' to 'meets every predeclared criterion'."""
    rows = []
    for (checkpoint, group), sub in [((c, g), cand[(cand["checkpoint"] == c) & (cand["group"] == g)])
                                     for c in sorted(cand["checkpoint"].unique()) for g in ("targeted", "random")] \
            + [(("all", "all"), cand)]:
        mask = pd.Series(True, index=sub.index)
        for stage, fn in FUNNEL_STAGES:
            alone = fn(sub).astype(bool)
            mask &= alone
            rows.append({"checkpoint": checkpoint, "group": group, "stage": stage, "candidates": int(mask.sum()),
                         "requirement_alone": int(alone.sum()), "of": len(sub)})
    return pd.DataFrame(rows)


def margin_summary(cand: pd.DataFrame) -> pd.DataFrame:
    """How far each candidate is from passing each part of the combined test (negative = passes)."""
    rows = []
    for sample, sub in cand.groupby("sample_position"):
        rows.append({"sample_position": sample, "candidates": len(sub),
                     "noise_top10_min": float(sub["heldout_noise_top10_jaccard_min"].iloc[0]),
                     "noise_spearman_min": float(sub["heldout_noise_spearman_min"].iloc[0]),
                     "pass_top10": int(sub["exceeds_heldout_top10"].sum()),
                     "pass_spearman": int(sub["exceeds_heldout_spearman"].sum()),
                     "pass_both": int(sub["exceeds_heldout_both"].sum()),
                     "lowest_candidate_spearman_mean": float(sub["heldout_spearman_mean"].min()),
                     "smallest_spearman_margin": float(sub["spearman_margin_to_noise"].min()),
                     "median_spearman_margin": float(sub["spearman_margin_to_noise"].median()),
                     "lowest_candidate_top10_mean": float(sub["heldout_top10_jaccard_mean"].min()),
                     "smallest_top10_margin": float(sub["top10_margin_to_noise"].min())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Claims and key findings (numbers computed here; interpretation stays in the report)
# ---------------------------------------------------------------------------------------------
def verify_claims(data: dict[str, Any], cand: pd.DataFrame, tvr: pd.DataFrame, cons: dict[str, pd.DataFrame]) -> pd.DataFrame:
    status = data["status"]
    at50 = tvr[(tvr["checkpoint"] == 50) & (tvr["targeted_method"] != "all_targeted")]
    pooled50 = tvr[(tvr["checkpoint"] == 50) & (tvr["targeted_method"] == "all_targeted")].set_index("metric")
    positive = at50[at50["diff"] > 0]
    significant = at50[at50["ci_excludes_zero"]]
    held = pooled50.loc["heldout_top10_shift"]
    obj = pooled50.loc["search_seed_objective"]
    unique = cons["unique_status"].set_index("physical_status")["unique_candidates"]
    rows = [
        {"claim": "All 123 experiments completed successfully.",
         "verdict": "Supported" if status["completed"] == status["planned"] == 123 and not status["failed"] else "Not supported",
         "evidence": f"status.json completed={status['completed']}/{status['planned']}, failed={len(status['failed'])}; "
                     f"results.csv has {data['results']['task_id'].nunique()} unique tasks, all stopped at the explanation budget, "
                     f"all still classified as attack and within the enforced constraints."},
        {"claim": "Targeted search generally performed better than random search at 50 evaluations.",
         "verdict": ("Not supported" if significant.empty else "Partly supported"),
         "evidence": f"{len(positive)} of {len(at50)} method x metric comparisons have a higher targeted mean, "
                     f"{len(significant)} have a 95% stratified bootstrap interval that excludes 0. Pooled targeted - random at 50 calls: "
                     f"search-seed objective {obj['diff']:+.4f} [{obj['ci_low']:+.4f}, {obj['ci_high']:+.4f}], "
                     f"held-out top-10 shift {held['diff']:+.4f} [{held['ci_low']:+.4f}, {held['ci_high']:+.4f}]."},
        {"claim": "No candidates passed the combined LIME stability test.",
         "verdict": "Supported" if int(cand["exceeds_heldout_both"].sum()) == 0 else "Not supported",
         "evidence": f"exceeds_heldout_both (top-10 AND Spearman beyond held-out seed noise) = {int(cand['exceeds_heldout_both'].sum())} "
                     f"of {len(cand)}; top-10 alone = {int(cand['exceeds_heldout_top10'].sum())}; Spearman alone = "
                     f"{int(cand['exceeds_heldout_spearman'].sum())}; Part B combined (seeds 11-55) = {int(cand['partb_exceeds_seed_noise'].sum())}; "
                     f"held-out top-1 rule = {int(cand['exceeds_heldout_top1_rule'].sum())}."},
        {"claim": "Some modified network traffic records had physical-consistency problems.",
         "verdict": "Supported (understated)" if int(unique.get("consistent_with_checked_rules", 0)) == 0 else "Supported",
         "evidence": f"Checkpoint rows: {int((cand['physical_status'] == 'violates_checked_rule').sum())} of {len(cand)} violate a checked rule, "
                     f"{int((cand['physical_status'] == 'unverified_dependencies').sum())} have unverified dependencies, "
                     f"{int((cand['physical_status'] == 'consistent_with_checked_rules').sum())} are consistent with all checked rules. "
                     f"Unique candidates: {int(unique.get('violates_checked_rule', 0))} / {int(unique.get('unverified_dependencies', 0))} / "
                     f"{int(unique.get('consistent_with_checked_rules', 0))} of {int(unique.sum())}."},
    ]
    return pd.DataFrame(rows)


def key_findings(data: dict[str, Any], cand: pd.DataFrame, tvr: pd.DataFrame, budget: pd.DataFrame,
                 margins: pd.DataFrame, funnel: pd.DataFrame, cons: dict[str, pd.DataFrame]) -> pd.DataFrame:
    def pooled(checkpoint: int, metric: str) -> pd.Series:
        return tvr[(tvr["checkpoint"] == checkpoint) & (tvr["targeted_method"] == "all_targeted")
                   & (tvr["metric"] == metric)].iloc[0]

    rows = [{"finding": "Experiments completed", "value": f"{data['status']['completed']} / {data['status']['planned']} tasks; "
             f"{len(cand)} checkpoint candidates (25 and 50 calls)", "source": "status.json, results.csv, heldout_summary.csv"}]
    for c in (25, 50):
        for metric in ("search_seed_objective", "heldout_top10_shift"):
            p = pooled(c, metric)
            rows.append({"finding": f"Targeted - random, {METRICS[metric]}, {c} calls",
                         "value": f"{p['diff']:+.4f} (95% CI {p['ci_low']:+.4f} to {p['ci_high']:+.4f}); "
                                  f"targeted mean {p['targeted_mean']:.4f} (n={p['n_a']}), random mean {p['random_mean']:.4f} (n={p['n_b']}); "
                                  f"P(targeted > random) = {p['p_superiority']:.2f}",
                         "source": "step6_targeted_vs_random.csv"})
    for method in SEARCH_ORDER:
        b = budget[(budget["search"] == method) & (budget["metric"] == "heldout_top10_shift")]
        if not b.empty:
            b = b.iloc[0]
            rows.append({"finding": f"25 -> 50 calls, held-out top-10 shift, {method}",
                         "value": f"{b['mean_at_25']:.4f} -> {b['mean_at_50']:.4f} (up {b['n_increased']}, same {b['n_unchanged']}, "
                                  f"down {b['n_decreased']} of {b['n_pairs']} pairs)", "source": "step6_budget_effect.csv"})
    rows.append({"finding": "Combined held-out seed-noise test passed", "value":
                 f"{int(cand['exceeds_heldout_both'].sum())} of {len(cand)} (top-10 alone: {int(cand['exceeds_heldout_top10'].sum())}; "
                 f"Spearman alone: {int(cand['exceeds_heldout_spearman'].sum())})", "source": "heldout_summary.csv"})
    for _, m in margins.iterrows():
        rows.append({"finding": f"Sample {int(m['sample_position'])}: closest candidate to the Spearman noise threshold",
                     "value": f"lowest candidate mean rho {m['lowest_candidate_spearman_mean']:.3f} vs threshold "
                              f"(min seed-pair rho) {m['noise_spearman_min']:.3f}", "source": "step6_noise_margins.csv"})
    final = funnel[(funnel["checkpoint"] == "all")].iloc[-3:]
    rows.append({"finding": "Candidates meeting every predeclared criterion", "value":
                 "; ".join(f"{r['stage']}: {r['candidates']}" for _, r in final.iterrows()), "source": "step6_spoofing_funnel.csv"})
    unique = cons["unique_status"].set_index("physical_status")["unique_candidates"]
    rows.append({"finding": "Physical consistency (unique candidates)", "value":
                 f"violates checked rule {int(unique['violates_checked_rule'])}, unverified dependencies "
                 f"{int(unique['unverified_dependencies'])}, consistent {int(unique['consistent_with_checked_rules'])} "
                 f"(of {int(unique.sum())})", "source": "step6_constraint_details.csv"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------
def _style(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _method_color(method: str) -> str:
    return COLOR_RANDOM if method == "random" else COLOR_TARGETED


def _group_legend(include_samples: bool = True) -> list:
    handles = [Patch(color=COLOR_TARGETED, label="Targeted search"), Patch(color=COLOR_RANDOM, label="Random control")]
    if include_samples:
        handles += [Line2D([], [], marker=m, linestyle="", color=INK_MUTED, markerfacecolor="none", markersize=7,
                           label=f"sample {s}") for s, m in SAMPLE_MARKERS.items()]
    handles.append(Line2D([], [], color=INK_PRIMARY, linewidth=2, label="mean"))
    return handles


def plot_method_comparison(cand: pd.DataFrame, checkpoint: int, path: Path) -> None:
    at = cand[cand["checkpoint"] == checkpoint]
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(13.5, 4.6), facecolor=SURFACE)
    for ax, (metric, label) in zip(axes, METRICS.items()):
        _style(ax)
        for i, method in enumerate(SEARCH_ORDER):
            sub = at[at["search"] == method]
            for sample, grp in sub.groupby("sample_position"):
                x = i + rng.uniform(-0.18, 0.18, len(grp))
                ax.scatter(x, grp[metric], s=34, marker=SAMPLE_MARKERS.get(sample, "o"), facecolors="none",
                           edgecolors=_method_color(method), linewidths=1.3, alpha=0.9)
            ax.hlines(sub[metric].mean(), i - 0.3, i + 0.3, color=INK_PRIMARY, linewidth=2)
        ax.set_xticks(range(len(SEARCH_ORDER)))
        ax.set_xticklabels([f"{SEARCH_LABELS[m].replace(' ', chr(10), 1)}\nn={int((at['search'] == m).sum())}"
                            for m in SEARCH_ORDER], fontsize=8.5)
        ax.set_title(label, fontsize=10.5, color=INK_PRIMARY, loc="left")
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    axes[0].set_ylabel("larger = explanation changed more", fontsize=9)
    fig.suptitle(f"Targeted search vs random control at {checkpoint} explanation calls (Random Forest, Budget A)",
                 fontsize=12.5, color=INK_PRIMARY, x=0.01, ha="left", y=1.03)
    fig.text(0.01, 0.975, "Each mark = best candidate of one search run; marker shape = sample. Canonical greedy is deterministic "
             "(one run per sample). Larger values are explanation changes, not spoofing success.",
             fontsize=8.5, color=INK_SECONDARY, ha="left")
    fig.legend(handles=_group_legend(), loc="lower center", ncol=6, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout()
    _save(fig, path)


def plot_change_distributions(cand: pd.DataFrame, path: Path) -> None:
    """Held-out change per method and budget: top-10 shift is discrete (share per level), Spearman is a box."""
    rng = np.random.default_rng(11)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.15]})

    ax = axes[0]
    _style(ax, grid_axis="x")
    levels = sorted(cand["heldout_top10_shift"].round(6).unique())
    ramp = ["#86b6ef", "#2a78d6", "#104281", "#0d366b"]  # sequential blue, ordinal steps 250/450/650/700
    labels = []
    for y, (method, checkpoint) in enumerate((m, c) for m in SEARCH_ORDER for c in (25, 50)):
        vals = cand[(cand["search"] == method) & (cand["checkpoint"] == checkpoint)]["heldout_top10_shift"].round(6)
        left = 0.0
        for k, level in enumerate(levels):
            share = float((vals == level).mean())
            if share:
                ax.barh(y, share, left=left, color=ramp[k], edgecolor=SURFACE, linewidth=2, height=0.72)
                n = int((vals == level).sum())
                if share >= 0.07:
                    ax.text(left + share / 2, y, str(n), ha="center", va="center", fontsize=8,
                            color=INK_PRIMARY if k == 0 else "white")
            left += share
        labels.append(f"{SEARCH_LABELS[method]}, {checkpoint} calls (n={len(vals)})")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of candidates (bar labels = counts)", fontsize=9)
    ax.set_title(METRICS["heldout_top10_shift"], fontsize=10.5, color=INK_PRIMARY, loc="left")
    ax.legend(handles=[Patch(color=ramp[k], label=f"{lv:.3f}" + (" (same top-10 under every seed)" if lv == 0 else ""))
                       for k, lv in enumerate(levels)], title="held-out top-10 shift", title_fontsize=8.5,
              loc="lower center", bbox_to_anchor=(0.5, -0.36), ncol=len(levels), frameon=False, fontsize=8.5)

    for ax, (metric, label) in zip(axes[1:], [("heldout_spearman_shift", METRICS["heldout_spearman_shift"])]):
        _style(ax)
        ticks, ticklabels = [], []
        for i, method in enumerate(SEARCH_ORDER):
            for j, checkpoint in enumerate((25, 50)):
                vals = cand[(cand["search"] == method) & (cand["checkpoint"] == checkpoint)][metric].to_numpy()
                x = i * 2.6 + j * 1.0
                ax.boxplot(vals, positions=[x], widths=0.6, showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor="none", edgecolor=_method_color(method), linewidth=1.2),
                           medianprops=dict(color=INK_PRIMARY, linewidth=2),
                           whiskerprops=dict(color=_method_color(method), linewidth=1),
                           capprops=dict(color=_method_color(method), linewidth=1))
                ax.scatter(x + rng.uniform(-0.16, 0.16, len(vals)), vals, s=12, color=_method_color(method),
                           alpha=0.55 if checkpoint == 25 else 0.9, linewidths=0)
                ticks.append(x)
                ticklabels.append(f"{checkpoint} calls\nn={len(vals)}")
            ax.text(i * 2.6 + 0.5, -0.2, SEARCH_LABELS[method], transform=ax.get_xaxis_transform(), ha="center",
                    fontsize=9, color=INK_PRIMARY)
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, fontsize=8)
        ax.set_title(label, fontsize=10.5, color=INK_PRIMARY, loc="left")
        ax.legend(handles=_group_legend(include_samples=False)[:2]
                  + [Line2D([], [], color=INK_PRIMARY, linewidth=2, label="median")],
                  loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.36))
    fig.suptitle("Distribution of held-out explanation change by search method and budget (LIME seeds 22/33/44/55)",
                 fontsize=12.5, color=INK_PRIMARY, x=0.01, ha="left", y=1.02)
    plt.tight_layout()
    _save(fig, path)


def plot_budget(curves: pd.DataFrame, cand: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    _style(ax)
    styles = {"canonical": ":", "shuffled": "-", "random": "-"}
    for method in ("canonical", "shuffled", "random"):
        sub = curves[curves["search"] == method]
        ax.plot(sub["explanation_calls"], sub["mean"], color=_method_color(method), linewidth=2, linestyle=styles[method])
        if method != "canonical":
            ax.fill_between(sub["explanation_calls"], sub["q25"], sub["q75"], color=_method_color(method), alpha=0.12, linewidth=0)
        last = sub.iloc[-1]
        ax.annotate(f"{SEARCH_LABELS[method]} (n={int(last['n_tasks'])})", (last["explanation_calls"], last["mean"]),
                    xytext=(6, 0), textcoords="offset points", fontsize=8.5, color=INK_PRIMARY, va="center")
    for c in (25, 50):
        ax.axvline(c, color=AXIS, linewidth=1, linestyle="--")
    ax.set_xlim(1, 66)
    ax.set_xlabel("LIME explanation calls used (baseline included)", fontsize=9)
    ax.set_ylabel("best search-seed objective so far", fontsize=9)
    ax.set_title("A. Search progress under the search LIME seed (mean; band = IQR)", fontsize=10.5, color=INK_PRIMARY, loc="left")
    ax.text(0.01, -0.2, "Multi-start is excluded: its 25- and 50-call results are separate runs with different "
            "restart budgets, shown in panel B.", transform=ax.transAxes, fontsize=8, color=INK_SECONDARY)

    ax = axes[1]
    _style(ax)
    ends = {}
    for method in SEARCH_ORDER:
        means = [cand[(cand["search"] == method) & (cand["checkpoint"] == c)]["heldout_top10_shift"].mean() for c in (25, 50)]
        ax.plot([0, 1], means, color=_method_color(method), linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, linestyle=":" if method == "canonical" else "-")
        ends[method] = means[1]
    # Direct labels at the 50-call end, nudged apart so close values stay readable.
    gap, placed = 0.0028, []
    for method, value in sorted(ends.items(), key=lambda kv: kv[1]):
        y = max(value, placed[-1] + gap) if placed else value
        placed.append(y)
        ax.annotate(f"{SEARCH_LABELS[method]} ({value:.3f})", (1, value), xytext=(1.06, y), textcoords="data",
                    fontsize=8.5, color=INK_PRIMARY, va="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["25 calls", "50 calls"])
    ax.set_xlim(-0.2, 2.1)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("mean held-out top-10 shift", fontsize=9)
    ax.set_title("B. Held-out change, 25 vs 50 calls (mean)", fontsize=10.5, color=INK_PRIMARY, loc="left")
    fig.suptitle("Does a larger explanation budget help?", fontsize=12.5, color=INK_PRIMARY, x=0.01, ha="left", y=1.02)
    fig.legend(handles=_group_legend(include_samples=False)[:2], loc="lower center", ncol=2, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout()
    _save(fig, path)


def plot_lime_stability(cand: pd.DataFrame, pairs: pd.DataFrame, data: dict[str, Any], path: Path) -> None:
    """Per sample: the original's seed-to-seed variation vs the candidates' held-out mean; threshold = noise min."""
    samples = sorted(cand["sample_position"].unique())
    bands = data["noise_bands"].query("band == 'heldout'").set_index("sample_position")
    rows = [("spearman_rho", "heldout_spearman_mean", "spearman_rho_min", "Spearman rho (all 42 features)"),
            ("top10_jaccard", "heldout_top10_jaccard_mean", "top10_jaccard_min", "Top-10 Jaccard")]
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(2, len(samples), figsize=(13, 7.2), facecolor=SURFACE, sharey="row")
    for r, (pair_col, cand_col, band_col, label) in enumerate(rows):
        for c, sample in enumerate(samples):
            ax = axes[r, c]
            _style(ax)
            threshold = float(bands.loc[sample, band_col])
            if not pairs.empty:
                noise_vals = pairs[pairs["validation_position"] == sample][pair_col].to_numpy()
            else:  # fall back to the stored top-10 values (no Spearman pairs are stored in noise_bands.csv)
                noise_vals = np.array(json.loads(bands.loc[sample, "top10_jaccard_pair_values"])) if pair_col == "top10_jaccard" \
                    else np.array([bands.loc[sample, "spearman_rho_min"], bands.loc[sample, "spearman_rho_max"]])
            ax.scatter(rng.uniform(-0.15, 0.15, len(noise_vals)), noise_vals, s=40, color=COLOR_NOISE, zorder=3)
            sub = cand[cand["sample_position"] == sample]
            for k, (group, color) in enumerate((("targeted", COLOR_TARGETED), ("random", COLOR_RANDOM))):
                vals = sub[sub["group"] == group][cand_col].to_numpy()
                ax.scatter(1 + k + rng.uniform(-0.2, 0.2, len(vals)), vals, s=16, color=color, alpha=0.75, linewidths=0, zorder=3)
            ax.axhline(threshold, color=INK_PRIMARY, linewidth=1.2, linestyle="--")
            note = f"threshold {threshold:.3f}" + ("\n(no seed noise: any\nchange passes)" if threshold >= 1.0 else "")
            ax.text(3.15, threshold, note, fontsize=7.5, color=INK_PRIMARY, ha="right",
                    va="top" if threshold >= 1.0 else "bottom")
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels([f"original vs original\n(seed pairs, n={len(noise_vals)})",
                                f"targeted\n(n={int((sub['group'] == 'targeted').sum())})",
                                f"random\n(n={int((sub['group'] == 'random').sum())})"], fontsize=7.8)
            ax.set_xlim(-0.5, 3.2)
            if r == 0:
                ax.set_title(f"Sample {sample}", fontsize=10.5, color=INK_PRIMARY, loc="left")
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.9))
    fig.suptitle("Why no candidate passed: LIME's own seed-to-seed variation is larger than the spoofing change",
                 fontsize=12.5, color=INK_PRIMARY, x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.925, "Gray = original sample explained under two different held-out LIME seeds (22/33/44/55).  "
             "Colored = candidate vs original under the same seed, averaged over the 4 seeds.\nA candidate passes a metric "
             "only if it falls strictly below the dashed line (the lowest gray value); the combined test needs both rows.",
             fontsize=8.5, color=INK_SECONDARY, ha="left")
    _save(fig, path)


def plot_constraints(cand: pd.DataFrame, cons: dict[str, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    _style(ax, grid_axis="x")
    labels, y = [], 0
    for checkpoint in (25, 50):
        for method in SEARCH_ORDER:
            sub = cand[(cand["checkpoint"] == checkpoint) & (cand["search"] == method)]
            left = 0
            for status in PHYSICAL_STATUS_ORDER:
                n = int((sub["physical_status"] == status).sum())
                if n:
                    ax.barh(y, n, left=left, color=STATUS_COLORS[status], edgecolor=SURFACE, linewidth=2, height=0.7)
                    ax.text(left + n / 2, y, str(n), ha="center", va="center", fontsize=8,
                            color="white" if status == "violates_checked_rule" else INK_PRIMARY)
                left += n
            labels.append(f"{SEARCH_LABELS[method]}, {checkpoint} calls (n={len(sub)})")
            y += 1
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("checkpoint candidates", fontsize=9)
    ax.set_title("A. Physical-consistency status (all passed the enforced constraints)", fontsize=10.5,
                 color=INK_PRIMARY, loc="left")
    ax.legend(handles=[Patch(color=STATUS_COLORS[s], label=s.replace("_", " ")) for s in PHYSICAL_STATUS_ORDER],
              loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=False, fontsize=8.5)

    ax = axes[1]
    _style(ax, grid_axis="x")
    det = cons["details"]
    det = det[(det["scope"] == "unique_candidates") & (det["kind"] == "changed_feature")].head(10)
    ypos = np.arange(len(det))
    ax.barh(ypos, det["candidates"], color=COLOR_NOISE, height=0.7, edgecolor=SURFACE, linewidth=2,
            label="changed in candidate")
    ax.barh(ypos, det["of_which_violate_checked_rule"], color=STATUS_COLORS["violates_checked_rule"], height=0.7,
            edgecolor=SURFACE, linewidth=2, label="... and candidate violates a checked rule")
    for yy, (n, v) in enumerate(zip(det["candidates"], det["of_which_violate_checked_rule"])):
        ax.text(n + 0.5, yy, f"{int(v)}/{int(n)}", va="center", fontsize=8, color=INK_SECONDARY)
    ax.set_yticks(ypos)
    ax.set_yticklabels(det["item"], fontsize=9)
    ax.invert_yaxis()
    n_unique = int(det["of"].iloc[0]) if len(det) else 0
    ax.set_xlabel(f"unique candidates (of {n_unique})", fontsize=9)
    ax.set_title("B. Most-edited features and physical-rule violations", fontsize=10.5, color=INK_PRIMARY, loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False, fontsize=8.5)
    fig.suptitle("Constraint validity and physical consistency of the candidates", fontsize=12.5, color=INK_PRIMARY,
                 x=0.01, ha="left", y=1.02)
    plt.tight_layout()
    _save(fig, path)


def plot_funnel(funnel: pd.DataFrame, path: Path) -> None:
    sub = funnel[funnel["checkpoint"] == "all"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 4.2), facecolor=SURFACE)
    _style(ax, grid_axis="x")
    y = np.arange(len(sub))
    ax.barh(y, sub["candidates"], color=COLOR_TARGETED, height=0.65, edgecolor=SURFACE, linewidth=2)
    for yy, n, alone in zip(y, sub["candidates"], sub["requirement_alone"]):
        extra = f"   (this requirement alone: {alone})" if yy >= 3 else ""
        ax.text(n + 2, yy, f"{n}{extra}", va="center", fontsize=9.5, color=INK_PRIMARY)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["stage"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, sub["candidates"].max() * 1.12)
    ax.set_xlabel(f"checkpoint candidates (all methods, 25 and 50 calls; of {int(sub['of'].iloc[0])})", fontsize=9)
    ax.set_title("From explanation change to a defensible spoofing candidate (each stage adds one requirement)",
                 fontsize=12, color=INK_PRIMARY, loc="left")
    plt.tight_layout()
    _save(fig, path)


def plot_shap_lime_context(context: pd.DataFrame, path: Path) -> None:
    if context.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURFACE)
    _style(ax, grid_axis="x")
    series = [("spoof_candidates_top10_distance_max", "Spoofing candidate vs original (largest of 62, held-out mean)", COLOR_TARGETED),
              ("lime_seed_noise_top10_distance_max", "Original vs original, two LIME seeds (largest of 6 pairs)", COLOR_NOISE),
              ("shap_vs_lime_top10_distance", "SHAP vs LIME on the original sample", COLOR_RANDOM)]
    h = 0.26
    for k, (col, label, color) in enumerate(series):
        ypos = np.arange(len(context)) + (k - 1) * h
        ax.barh(ypos, context[col], height=h, color=color, edgecolor=SURFACE, linewidth=1.5, label=label)
        for yy, v in zip(ypos, context[col]):
            ax.text(v + 0.004, yy, f"{v:.2f}", va="center", fontsize=7.5, color=INK_SECONDARY)
    ax.set_yticks(np.arange(len(context)))
    ax.set_yticklabels([f"Sample {s}" for s in context["sample_position"]], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("top-10 Jaccard distance (1 - Jaccard; 0 = same top-10 features)", fontsize=9)
    ax.set_title("Size of the spoofing change compared with other sources of explanation disagreement (Random Forest)",
                 fontsize=11, color=INK_PRIMARY, loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=1, frameon=False, fontsize=8.5)
    plt.tight_layout()
    _save(fig, path)


# ---------------------------------------------------------------------------------------------
def run_analysis(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    data = load_run(run_dir)
    check_required_columns(data)
    checks = validate_run(data)
    cand = build_candidates(data)
    curves = best_so_far_curves(data)
    extra = [{"check": "best_so_far is non-decreasing within every task", "passed": best_so_far_is_monotone(data), "detail": ""}]
    noise, pairs = lime_stability_tables(data, cand)
    ok, gap = noise_pairs_match_bands(pairs, data)
    extra.append({"check": "results/lime_stability seed pairs reproduce the held-out noise bands", "passed": ok,
                  "detail": "file missing; plotted from noise_bands.csv" if pairs.empty else f"max gap {gap:.2e}"})
    checks = pd.concat([checks, pd.DataFrame(extra)], ignore_index=True)

    tvr = targeted_vs_random(cand)
    budget = budget_effect(cand)
    cons = constraint_tables(cand)
    funnel = spoofing_funnel(cand)
    margins = margin_summary(cand)
    context = shap_lime_context(cand, noise)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    tables = {
        "step6_validation_checks.csv": checks,
        "step6_missing_values.csv": missing_value_report(data),
        "step6_candidates.csv": cand.drop(columns=["candidate_json"]),
        "step6_method_summary.csv": method_summary(cand, ["checkpoint", "search"]),
        "step6_method_summary_by_sample.csv": method_summary(cand, ["checkpoint", "search", "sample_position"]),
        "step6_seed_summary.csv": seed_summary(cand),
        "step6_targeted_vs_random.csv": tvr,
        "step6_targeted_vs_random_by_sample.csv": targeted_vs_random_by_sample(cand),
        "step6_budget_effect.csv": budget,
        "step6_best_so_far_curves.csv": curves,
        "step6_lime_stability.csv": noise,
        "step6_noise_margins.csv": margins,
        "step6_shap_lime_context.csv": context,
        "step6_constraint_status.csv": cons["status"],
        "step6_constraint_details.csv": cons["details"],
        "step6_spoofing_funnel.csv": funnel,
        "step6_claims_check.csv": verify_claims(data, cand, tvr, cons),
        "step6_key_findings.csv": key_findings(data, cand, tvr, budget, margins, funnel, cons),
    }
    for name, frame in tables.items():
        frame.to_csv(output_dir / name, index=False)

    plot_method_comparison(cand, 25, fig_dir / "fig1_targeted_vs_random_25_calls.png")
    plot_method_comparison(cand, 50, fig_dir / "fig2_targeted_vs_random_50_calls.png")
    plot_change_distributions(cand, fig_dir / "fig3_explanation_change_distributions.png")
    plot_budget(curves, cand, fig_dir / "fig4_budget_25_vs_50.png")
    plot_lime_stability(cand, pairs, data, fig_dir / "fig5_lime_seed_stability.png")
    plot_constraints(cand, cons, fig_dir / "fig6_constraint_physical_consistency.png")
    plot_funnel(funnel, fig_dir / "fig7_spoofing_criteria_funnel.png")
    plot_shap_lime_context(context, fig_dir / "fig8_change_vs_shap_lime_disagreement.png")

    metadata = {
        "source_run_dir": str(Path(run_dir).resolve().relative_to(ROOT)).replace("\\", "/"),
        "protocol_version": data["run_config"]["protocol_version"],
        "selection": data["run_config"]["selection"],
        "heldout_run_id": data["heldout_metadata"]["run_id"],
        "candidates": len(cand),
        "checks_passed": int(checks["passed"].sum()),
        "checks_total": len(checks),
        "bootstrap_reps": BOOTSTRAP_REPS,
        "permutation_reps": PERMUTATION_REPS,
        "stat_seed": STAT_SEED,
        "context_files": {"lime_stability_pairwise": LIME_STABILITY_PAIRWISE.exists(),
                          "shap_lime_per_sample": SHAP_LIME_PER_SAMPLE.exists()},
        "tables": sorted(tables),
        "figures": sorted(p.name for p in fig_dir.glob("*.png")),
    }
    (output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return {"checks": checks, "tables": tables, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse a packaged Step 6 run (read-only).")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--output-dir", default=None,
                        help="default: results/step6_shared/analysis_<run id> next to the run directory")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir.parent / f"analysis_{run_dir.name}"
    if output_dir.resolve() == run_dir.resolve() or run_dir.resolve() in output_dir.resolve().parents:
        raise SystemExit("output directory must be outside the run directory (the packaged run is never modified)")
    result = run_analysis(run_dir, output_dir)
    checks = result["checks"]
    for _, row in checks.iterrows():
        print(f"[{'PASS' if row['passed'] else 'FAIL'}] {row['check']}" + (f" - {row['detail']}" if row["detail"] else ""))
    print(f"\n{int(checks['passed'].sum())}/{len(checks)} checks passed. Outputs written to {output_dir}")
    print("\nKey findings:")
    for _, row in result["tables"]["step6_key_findings.csv"].iterrows():
        print(f"- {row['finding']}: {row['value']}")
    if not checks["passed"].all():
        raise SystemExit("ERROR: some validation checks failed; see step6_validation_checks.csv")


if __name__ == "__main__":
    main()
