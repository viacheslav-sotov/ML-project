"""Held-out LIME evaluation of saved targeted/random spoofing candidates (Step 6, Stage B).

Search runs optimise against ONE LIME seed (11). This script re-explains the saved checkpoint
candidates under HELD-OUT LIME seeds (22, 33, 44, 55) and asks whether the candidate's change from
the original explanation is larger than the original sample's own seed-to-seed LIME variation.

Metric definitions are Part B's, reused verbatim: experiments.unsw_spoofing_explanations.pair_metrics
(checked by Part B to reproduce unsw_lime_stability.compute_pairwise_metrics exactly).

Primary stability measure
    heldout_top10_jaccard_mean = mean over the 4 held-out seeds s of
        top10_jaccard(original explanation under s, candidate explanation under s);
    heldout_top10_shift = 1 - heldout_top10_jaccard_mean;
    exceeds_heldout_top10 = heldout_top10_jaccard_mean < MIN of the 6 held-out seed-pair values of the
        ORIGINAL sample (same input, different seeds), i.e. Part B's per-metric rule.
Secondary measures
    Spearman, top-1, top-5, sign (direction) agreement on held-out seeds, their noise flags, and the
    original Part B criterion on seeds 11-55 (10 seed pairs). The latter includes the search seed 11
    and is therefore NOT a held-out test; it is reported only for comparability with Part B.

A candidate that differs from the original under one seed is never reported as spoofing. This
script only reports stability measures and noise flags; it does not decide spoofing success.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.unsw_spoofing_explanations import pair_metrics  # noqa: E402  (Part B definitions)
from experiments import unsw_targeted_spoofing as ts  # noqa: E402
from experiments.unsw_constraint_reporting import constraint_report, load_teammate_table, policy_versions  # noqa: E402

SEARCH_LIME_SEED = 11
HELDOUT_SEEDS = (22, 33, 44, 55)
PART_B_SEEDS = (11, 22, 33, 44, 55)
TOP1_MAJORITY = 3  # top-1 counts as changed if it differs in >= 3 paired seeds (Part B: 3 of 5; held-out: 3 of 4)
BAND_METRICS = ("top1_agreement", "top5_jaccard", "top10_jaccard", "spearman_rho", "direction_agreement_top10")
NOISE_RULE_METRICS = ("top5_jaccard", "top10_jaccard", "spearman_rho")
LIME_REPRO_TOL = 1e-6
SEARCH_SCORE_TOL = 1e-9  # same code path and seed: the seed-11 objective must reproduce essentially exactly
FORBIDDEN_SPLIT_KEYS = ("X_test", "y_test", "official_test_df")
PASSTHROUGH_COLUMNS = ("task_id", "search", "explanation_budget", "lime_seed", "restarts", "prefix_shortcut")
OUTPUT_ROOT = ROOT / "results" / "spoofing_targeted_heldout"

RULES = {
    "metric_definitions": "experiments.unsw_spoofing_explanations.pair_metrics (Part B), reused unchanged",
    "primary": "heldout_top10_jaccard_mean over held-out seeds 22,33,44,55; shift = 1 - mean; "
               "exceeds_heldout_top10 if mean < min of the original's 6 held-out seed-pair top10_jaccard values",
    "per_metric_rule": "for top5_jaccard, top10_jaccard, spearman_rho: exceeds if the candidate's mean (over seeds) is "
                       "strictly lower than the minimum of the original sample's seed-pair values on the same seeds",
    "heldout_both_flag": "exceeds_heldout_top10 AND exceeds_heldout_spearman (Part B's primary flag applied to held-out seeds)",
    "top1_rule": "original's top-1 agrees in all seed pairs AND candidate's top-1 differs from the original's in >= 3 paired seeds",
    "part_b_criterion": "the same rules on seeds 11,22,33,44,55 (10 pairs); includes the search seed 11, so NOT held-out",
    "no_spoofing_claims": "this script reports stability only; it never labels a candidate as successful spoofing",
    "search_score_check": "objective_score(original, candidate) recomputed under the search LIME seed must equal the saved "
                          f"search score within {SEARCH_SCORE_TOL}; any mismatch is reported and makes the script exit with an error",
    "checkpoint_validity": "checkpoint rows with checkpoint_valid=False (multi-start below its run budget) are skipped",
}


# ---------------------------------------------------------------------------------------------
# Pure metric logic (unit-tested with fake explanations)
# ---------------------------------------------------------------------------------------------
def validate_seeds(heldout_seeds, search_seed: int = SEARCH_LIME_SEED) -> tuple[int, ...]:
    seeds = tuple(int(s) for s in heldout_seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError(f"need at least two distinct held-out seeds, got {seeds}")
    if int(search_seed) in seeds:
        raise ValueError(f"the search LIME seed {search_seed} must not be a held-out seed")
    return seeds


def noise_band(weights_by_seed: dict[int, dict[str, float]], seeds) -> dict[str, Any]:
    """Seed-to-seed variation of the ORIGINAL explanation: pair_metrics over all seed pairs."""
    pairs = [pair_metrics(weights_by_seed[a], weights_by_seed[b]) for a, b in itertools.combinations(sorted(seeds), 2)]
    band: dict[str, Any] = {"n_seeds": len(seeds), "n_pairs": len(pairs), "seeds": list(sorted(seeds))}
    for metric in BAND_METRICS:
        values = [float(p[metric]) for p in pairs]
        band[f"{metric}_min"] = min(values)
        band[f"{metric}_mean"] = float(np.mean(values))
        band[f"{metric}_max"] = max(values)
    band["top1_stable_all_pairs"] = bool(all(float(p["top1_agreement"]) == 1.0 for p in pairs))
    band["top10_jaccard_pair_values"] = [round(float(p["top10_jaccard"]), 6) for p in pairs]
    return band


def candidate_shift(original_by_seed, candidate_by_seed, seeds) -> tuple[list[dict[str, Any]], dict[str, float], int]:
    """Same seed, different input: pair_metrics(original under s, candidate under s) for each seed s."""
    per_seed = []
    for seed in sorted(seeds):
        metrics = pair_metrics(original_by_seed[seed], candidate_by_seed[seed])
        per_seed.append({"seed": int(seed), **metrics})
    means = {metric: float(np.mean([row[metric] for row in per_seed])) for metric in BAND_METRICS}
    top1_changed = int(sum(1 for row in per_seed if float(row["top1_agreement"]) < 1.0))
    return per_seed, means, top1_changed


def noise_flags(means: dict[str, float], top1_changed: int, band: dict[str, Any], majority: int = TOP1_MAJORITY) -> dict[str, bool]:
    flags = {f"exceeds_{metric}": bool(means[metric] < band[f"{metric}_min"]) for metric in NOISE_RULE_METRICS}
    flags["exceeds_both_top10_and_spearman"] = bool(flags["exceeds_top10_jaccard"] and flags["exceeds_spearman_rho"])
    flags["exceeds_top1_rule"] = bool(band["top1_stable_all_pairs"] and top1_changed >= majority)
    return flags


def evaluate_candidates(
    candidates: list[dict[str, Any]],
    explain_fn: Callable[[str, int, pd.Series, int], dict[str, float]],
    heldout_seeds=HELDOUT_SEEDS,
    part_b_seeds=PART_B_SEEDS,
    search_seed: int = SEARCH_LIME_SEED,
    key_fn: Callable[[pd.Series], Any] = ts._row_key,
) -> dict[str, Any]:
    """Evaluate candidates; each dict needs model_key, sample_position, original_row, candidate_row, candidate_uid.

    Explanations are cached per (model, sample, candidate values, seed), so identical candidates and
    the original are explained once per seed.
    """
    heldout = validate_seeds(heldout_seeds, search_seed)
    part_b = tuple(int(s) for s in part_b_seeds)
    all_seeds = sorted(set(heldout) | set(part_b))
    cache: dict[tuple, dict[str, float]] = {}
    calls = {"explanations": 0}

    def explain(model_key, sample, row, seed):
        key = (model_key, int(sample), key_fn(row), int(seed))
        if key not in cache:
            cache[key] = explain_fn(model_key, int(sample), row, int(seed))
            calls["explanations"] += 1
        return cache[key]

    bands, per_seed_rows, summary_rows = [], [], []
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for candidate in candidates:
        groups.setdefault((candidate["model_key"], int(candidate["sample_position"])), []).append(candidate)

    for (model_key, sample), members in groups.items():
        original_row = members[0]["original_row"]
        for member in members:
            if key_fn(member["original_row"]) != key_fn(original_row):
                raise ValueError(f"inconsistent original rows for {model_key}/{sample}")
        original = {seed: explain(model_key, sample, original_row, seed) for seed in all_seeds}
        band_h = noise_band(original, heldout)
        band_b = noise_band(original, part_b)
        for label, band in (("heldout", band_h), ("part_b", band_b)):
            bands.append({"model_key": model_key, "sample_position": sample, "band": label, **band})
        for member in members:
            cand = {seed: explain(model_key, sample, member["candidate_row"], seed) for seed in all_seeds}
            recomputed = None
            if member.get("search_seed_objective") is not None and search_seed in all_seeds:
                recomputed = float(ts.objective_score(original[search_seed], cand[search_seed]))
            seeds_h, means_h, top1_h = candidate_shift(original, cand, heldout)
            seeds_b, means_b, top1_b = candidate_shift(original, cand, part_b)
            flags_h = noise_flags(means_h, top1_h, band_h)
            flags_b = noise_flags(means_b, top1_b, band_b)
            for row in seeds_h:
                per_seed_rows.append({"candidate_uid": member["candidate_uid"], "model_key": model_key,
                                      "sample_position": sample, "seed_set": "heldout", **row})
            if search_seed in part_b:
                for row in seeds_b:
                    if row["seed"] == search_seed:
                        per_seed_rows.append({"candidate_uid": member["candidate_uid"], "model_key": model_key,
                                              "sample_position": sample, "seed_set": "search_seed", **row})
            summary_rows.append({
                "candidate_uid": member["candidate_uid"],
                "model_key": model_key,
                "sample_position": sample,
                **{k: v for k, v in member.items() if k not in {"original_row", "candidate_row", "candidate_uid", "model_key", "sample_position"}},
                # primary
                "heldout_top10_jaccard_mean": means_h["top10_jaccard"],
                "heldout_top10_shift": 1.0 - means_h["top10_jaccard"],
                "heldout_noise_top10_jaccard_min": band_h["top10_jaccard_min"],
                "exceeds_heldout_top10": flags_h["exceeds_top10_jaccard"],
                # secondary (held-out)
                "heldout_spearman_mean": means_h["spearman_rho"],
                "heldout_top5_jaccard_mean": means_h["top5_jaccard"],
                "heldout_top1_agreement_mean": means_h["top1_agreement"],
                "heldout_sign_agreement_top10_mean": means_h["direction_agreement_top10"],
                "heldout_top1_changed_seeds": top1_h,
                "exceeds_heldout_spearman": flags_h["exceeds_spearman_rho"],
                "exceeds_heldout_top5": flags_h["exceeds_top5_jaccard"],
                "exceeds_heldout_both": flags_h["exceeds_both_top10_and_spearman"],
                "exceeds_heldout_top1_rule": flags_h["exceeds_top1_rule"],
                # Part B criterion (includes search seed 11; not held-out)
                "partb_top10_jaccard_mean": means_b["top10_jaccard"],
                "partb_spearman_mean": means_b["spearman_rho"],
                "partb_exceeds_seed_noise": flags_b["exceeds_both_top10_and_spearman"],
                "partb_exceeds_top1_rule": flags_b["exceeds_top1_rule"],
                "partb_includes_search_seed": bool(search_seed in part_b),
                "search_seed_objective_recomputed": recomputed,
                "search_score_gap": (None if recomputed is None else abs(recomputed - float(member["search_seed_objective"]))),
                "search_score_reproduced": (None if recomputed is None
                                            else bool(abs(recomputed - float(member["search_seed_objective"])) <= SEARCH_SCORE_TOL)),
            })
    return {
        "per_seed": per_seed_rows,
        "summary": summary_rows,
        "bands": bands,
        "explanations_computed": calls["explanations"],
        "heldout_seeds": list(heldout),
        "part_b_seeds": list(part_b),
    }


# ---------------------------------------------------------------------------------------------
# Inputs and the real LIME explainer (used by main(); not exercised by unit tests)
# ---------------------------------------------------------------------------------------------
def load_checkpoint_candidates(run_dirs: list[Path], X_val: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    table = load_teammate_table()
    candidates, sources = [], []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        sources.append({"run_dir": str(run_dir), "run_id": meta["run_id"], "model_sha256": meta["model_sha256"],
                        "lime_seed": meta.get("lime_seed", meta.get("search_seed"))})
        if int(sources[-1]["lime_seed"]) in HELDOUT_SEEDS:
            raise ValueError(f"{run_dir} was searched with a held-out LIME seed")
        frame = pd.read_csv(run_dir / "checkpoints.csv")
        if "checkpoint_valid" in frame.columns:
            valid = frame["checkpoint_valid"].astype(bool)
        else:  # older files: apply the rule directly
            valid = pd.Series([
                ts.checkpoint_is_valid(str(r["strategy"]), int(r["checkpoint"]), int(r["max_explanation_evaluations"]))
                for _, r in frame.iterrows()
            ], index=frame.index)
        sources[-1]["checkpoint_rows"] = int(len(frame))
        sources[-1]["skipped_invalid_checkpoints"] = int((~valid).sum())
        frame = frame[valid]
        for index, rec in frame.iterrows():
            original_row = X_val.iloc[[int(rec["sample_position"])]].iloc[0].copy()
            candidate_row = ts.row_from_json(rec["candidate_json"], original_row)
            budget_ok, budget_reason = ts._candidate_matches_budget(original_row, candidate_row, str(rec["budget"]))
            report = constraint_report(original_row.to_dict(), candidate_row.to_dict(), budget_ok, budget_reason, table)
            candidates.append({
                "candidate_uid": f"{meta['run_id']}:{index}",
                "model_key": str(rec["model_key"]),
                "sample_position": int(rec["sample_position"]),
                "original_row": original_row,
                "candidate_row": candidate_row,
                "source_run_id": meta["run_id"],
                "method": rec["method"],
                "strategy": rec["strategy"],
                "budget": rec["budget"],
                "order_seed": rec.get("order_seed"),
                "control_seed": rec.get("control_seed"),
                "checkpoint": int(rec["checkpoint"]),
                "search_seed_objective": float(rec["best_score"]),
                "candidate_is_original": bool(rec["candidate_is_original"]),
                **{col: rec[col] for col in PASSTHROUGH_COLUMNS if col in frame.columns},
                **report,
            })
    return candidates, sources


def make_lime_explain_fn(split: dict[str, Any], expected_model_sha: dict[str, str]):
    """Real explanation function: same LIME call as the search (ts._call_explanation)."""
    codec = ts.CategoricalCodec(split["X_subtrain"])
    explainer = ts.build_explainer(codec.encode(split["X_subtrain"]), codec)
    models: dict[str, Any] = {}

    def explain(model_key: str, sample: int, row: pd.Series, seed: int) -> dict[str, float]:
        if model_key not in models:
            path = ts.MODEL_DIR / f"{model_key}_validation_pipeline.joblib"
            actual = ts.file_sha256(path)
            if expected_model_sha.get(model_key) != actual:
                raise ValueError(f"{model_key} model hash {actual} does not match the search run ({expected_model_sha.get(model_key)})")
            meter = ts.CountingModel(ts.load_xai_model(model_key, split))
            models[model_key] = (meter, ts.make_predict_fn(meter, codec))
        meter, predict_fn = models[model_key]
        return ts._call_explanation(meter, row, codec, explainer, predict_fn, seed)

    explain.models = models  # exposes the meters for accounting
    return explain


def check_original_reproduces_saved(explain_fn, pairs: list[tuple[str, int, pd.Series]], seeds) -> float:
    """Original explanations must reproduce results/lime_stability (as Part B checked)."""
    saved = pd.read_csv(ROOT / "results" / "lime_stability" / "lime_stability_runs.csv")
    gap = 0.0
    for model_key, sample, row in pairs:
        for seed in seeds:
            ref = saved[(saved["model"] == model_key) & (saved["validation_position"] == sample) & (saved["seed"] == seed)]
            if ref.empty:
                continue
            ref_w = ref.set_index("feature")["signed_weight"].to_dict()
            mine = explain_fn(model_key, sample, row, seed)
            gap = max(gap, max(abs(float(mine[f]) - float(ref_w[f])) for f in ref_w))
    if gap > LIME_REPRO_TOL:
        raise ValueError(f"recomputed original LIME differs from results/lime_stability by {gap:.2e}")
    return gap


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out LIME evaluation of saved spoofing candidates.")
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Search run directories containing checkpoints.csv.")
    parser.add_argument("--heldout-seeds", nargs="+", type=int, default=list(HELDOUT_SEEDS))
    parser.add_argument("--part-b-seeds", nargs="+", type=int, default=list(PART_B_SEEDS))
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    args = parser.parse_args()

    start = time.perf_counter()
    split = {k: v for k, v in ts.get_xai_data().items() if k not in FORBIDDEN_SPLIT_KEYS}  # official test set never reachable
    candidates, sources = load_checkpoint_candidates([Path(p) for p in args.run_dirs], split["X_val"])
    expected = {}
    for source in sources:
        for key, value in source["model_sha256"].items():
            if expected.setdefault(key, value) != value:
                raise ValueError(f"run directories disagree on the {key} model hash")
    explain_fn = make_lime_explain_fn(split, expected)
    originals = {(c["model_key"], c["sample_position"]): c["original_row"] for c in candidates}
    repro_gap = check_original_reproduces_saved(explain_fn, [(m, s, r) for (m, s), r in originals.items()], args.part_b_seeds)
    result = evaluate_candidates(candidates, lambda m, s, r, seed: explain_fn(m, s, r, seed), args.heldout_seeds, args.part_b_seeds)

    run_dir = ts._make_run_dir(args.output_dir)
    pd.DataFrame(result["per_seed"]).to_csv(run_dir / "heldout_per_seed.csv", index=False)
    pd.DataFrame(result["summary"]).to_csv(run_dir / "heldout_summary.csv", index=False)
    pd.DataFrame(result["bands"]).to_csv(run_dir / "noise_bands.csv", index=False)
    meters = {k: v[0] for k, v in explain_fn.models.items()}
    metadata = {
        "run_id": run_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_runs": sources,
        "search_lime_seed": SEARCH_LIME_SEED,
        "heldout_seeds": result["heldout_seeds"],
        "part_b_seeds": result["part_b_seeds"],
        "rules": RULES,
        "top1_majority": TOP1_MAJORITY,
        "lime_reproduction_max_gap": repro_gap,
        "explanations_computed": result["explanations_computed"],
        "model_usage": {k: {"lime_batches": m.lime_batches, "lime_rows": m.lime_rows,
                            "direct_calls": m.direct_calls, "direct_rows": m.direct_rows} for k, m in meters.items()},
        "constraint_policy": policy_versions(),
        "model_sha256": expected,
        "elapsed_seconds": round(time.perf_counter() - start, 3),
    }
    mismatches = [r for r in result["summary"] if r["search_score_reproduced"] is False]
    metadata["search_score_check"] = {
        "checked": sum(r["search_score_reproduced"] is not None for r in result["summary"]),
        "mismatches": len(mismatches),
        "max_gap": max((r["search_score_gap"] for r in result["summary"] if r["search_score_gap"] is not None), default=None),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"Held-out evaluation written to {run_dir}")
    if mismatches:
        ids = ", ".join(str(r["candidate_uid"]) for r in mismatches[:10])
        raise SystemExit(f"ERROR: {len(mismatches)} candidate(s) do not reproduce their saved search score under seed "
                         f"{SEARCH_LIME_SEED} (tolerance {SEARCH_SCORE_TOL}): {ids}")


if __name__ == "__main__":
    main()
