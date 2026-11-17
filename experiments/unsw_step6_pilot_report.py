"""Step 6 v2: summary tables for a finished exhaustive-evaluation run (pilot or smoke test).

Reads the run folder written by experiments/unsw_step6_exhaustive.py and, for the failure analysis only,
the sub-training/validation split (the official test set is removed first). No LIME and no models are
used. Writes report tables into <run dir>/report/:

    run_overview.json            explanations, runtime per model, checks
    candidates_by_flow.csv       model x flow x budget x physical status group: lattice, valid, evaluated
    success_by_group.csv         model x physical status group: initial and confirmed LIME successes
    success_intervals.csv        flow-level success proportions with Clopper-Pearson (protocol v2 P1) and Wilson intervals
    explanation_change.csv       model x flow x status group: top-1 / top-10 / Spearman changes vs the noise band
    attack_preservation.csv      model x flow: P(attack) of original and candidates
    budget_effect.csv            model x smallest budget x status group: largest held-out changes
    closest_to_band.csv          model x flow: margin of the closest candidate to the noise band
    feature_rank.csv             model x flow: rank of the edited/repaired features in the original LIME explanation
    bin_crossings.csv            flow x status group: candidates whose edited/repaired features cross a LIME quartile edge
    report_tables.md             the same tables in Markdown

Pilot results are never pooled with the final large-scale experiment.

    python experiments/unsw_step6_pilot_report.py <run dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, norm

ROOT = Path(__file__).resolve().parents[1]
STATUS_GROUPS = ("consistent_with_checked_rules", "unverified_dependencies")
CHANGED_FEATURES = ("sbytes", "spkts", "smean", "sload", "rate")  # edited (sbytes, spkts) and repaired by physical_v1
HELDOUT = (22, 33, 44, 55)


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def wilson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = float(norm.ppf(1 - (1 - level) / 2))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def load(run_dir: Path) -> dict:
    read = lambda name: pd.read_csv(run_dir / name)
    data = {
        "status": json.loads((run_dir / "status.json").read_text(encoding="utf-8")),
        "config": json.loads((run_dir / "run_config.json").read_text(encoding="utf-8")),
        "manifest": json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")),
        "summary": read("heldout_summary.csv"),
        "identities": read("identities.csv"),
        "bands": read("noise_bands.csv"),
        "per_seed": read("heldout_per_seed.csv"),
    }
    conf_bands = run_dir / "confirmation_noise_bands.csv"
    data["confirm_bands"] = pd.read_csv(conf_bands) if conf_bands.stat().st_size > 2 else pd.DataFrame()
    data["unique"] = read("unique_candidates.csv")
    counts, lines, bad = Counter(), 0, 0
    data["original_weights"] = {}
    original_keys = set()
    with (run_dir / "explanations_cache.jsonl").open(encoding="utf-8") as fh:
        for raw in fh:
            lines += 1
            try:
                item = json.loads(raw)
            except ValueError:
                bad += 1
                continue
            if item.get("type") == "explanation":
                counts[(item["model_key"], int(item["seed"]))] += 1
                if int(item["seed"]) in HELDOUT:
                    data["original_weights"].setdefault((item["model_key"], int(item["sample_position"]), item["key"]), {})[int(item["seed"])] = item["weights"]
    data["cache_counts"], data["cache_bad_lines"] = counts, bad
    return data


def overview(d: dict) -> dict:
    status, counts = d["status"], d["cache_counts"]
    per_model = {}
    for model in sorted({m for m, _ in counts}):
        n = sum(v for (m, s), v in counts.items() if m == model)
        n_conf = sum(v for (m, s), v in counts.items() if m == model and s >= 400)
        sec = status["seconds_per_new_explanation"].get(model)
        per_model[model] = {"explanations": n, "of_which_confirmation": n_conf, "seconds_per_explanation": sec,
                            "lime_time_minutes": round(n * sec / 60, 1) if sec else None}
    return {
        "run_id": d["config"]["run_id"], "run_role": status.get("run_role"), "state": status["state"],
        "git": d["manifest"]["git"], "protocol_sha256": d["manifest"]["protocol_sha256"],
        "model_sha256_at_start": {k: v["sha256"] for k, v in d["manifest"]["models"].items()},
        "model_sha256_after_run": status.get("model_sha256_after_run"),
        "official_test_used": d["config"]["official_test_used"],
        "explanations_computed": status["computed_now"], "explanations_reused": status["reused"],
        "cached_explanations": status["cached_explanations"], "cache_bad_lines": d["cache_bad_lines"],
        "planned_lime_calls": status["planned_lime_calls"],
        "confirmation_candidates": status["confirmation_candidates"],
        "confirmation_explanations_computed": status["confirmation_explanations_computed"],
        "confirmation_lime_calls_max": status["confirmation_lime_calls_max"],
        "elapsed_minutes": round(status["elapsed_seconds"] / 60, 1),
        "per_model": per_model,
    }


def candidates_by_flow(d: dict) -> pd.DataFrame:
    """Rejected lattice points carry no model (they are never predicted); they are counted for every model."""
    ids = d["identities"]
    rejected = ids[~ids["valid"].astype(bool)]
    models = sorted(ids["model_key"].dropna().unique())
    rows = []
    for (model, sample, budget), g in ids[ids["valid"].astype(bool)].groupby(["model_key", "sample_position", "budget"]):
        rej = rejected[(rejected["sample_position"] == sample) & (rejected["budget"] == budget)]
        assert model in models
        row = {"model_key": model, "sample_position": int(sample), "budget": budget,
               "lattice_points": int(len(g) + len(rej)), "valid": int(len(g)),
               "rejected": int(len(rej)),
               "rejection_reason": "; ".join(sorted(set(rej["status"].astype(str)))) if len(rej) else ""}
        for status in STATUS_GROUPS:
            v = g[g["valid"] & (g["status"] == status)]
            row[f"valid_{status}"] = int(len(v))
            row[f"still_attack_{status}"] = int(v["still_attack"].astype(bool).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def success_by_group(d: dict) -> pd.DataFrame:
    s = d["summary"]
    rows = []
    for (model, status), g in s.groupby(["model_key", "physical_status"]):
        rows.append({
            "model_key": model, "physical_status_group": status, "candidates": int(len(g)),
            "flows": int(g["sample_position"].nunique()),
            "exceeds_heldout_top10": int(g["exceeds_heldout_top10"].sum()),
            "exceeds_heldout_spearman": int(g["exceeds_heldout_spearman"].sum()),
            "exceeds_heldout_both": int(g["exceeds_heldout_both"].sum()),
            "lime_success": int(g["lime_success"].sum()),
            "confirmation_evaluated": int(g["confirmation_evaluated"].sum()),
            "confirmation_passed": int(g["confirmation_passed"].sum()),
            "confirmed_lime_success": int(g["confirmed_lime_success"].sum()),
            "flows_with_lime_success": int(g.loc[g["lime_success"], "sample_position"].nunique()),
            "flows_with_confirmed_success": int(g.loc[g["confirmed_lime_success"], "sample_position"].nunique()),
            "partb_exceeds_seed_noise": int(g["partb_exceeds_seed_noise"].sum()),
        })
    return pd.DataFrame(rows)


def success_intervals(d: dict) -> pd.DataFrame:
    """Flow-level proportions per model (protocol v2 P1: flows meeting D8; exact intervals are the v2 definition)."""
    s = d["summary"]
    rows = []
    for model, g in s.groupby("model_key"):
        flows = sorted(g["sample_position"].unique())
        for label, column in (("lime_success (D8)", "lime_success"), ("confirmed_lime_success", "confirmed_lime_success")):
            k = int(g.loc[g[column], "sample_position"].nunique())
            n = len(flows)
            cp, wi = clopper_pearson(k, n), wilson(k, n)
            rows.append({"model_key": model, "measure": label, "flows_with_success": k, "flows": n,
                         "clopper_pearson_95_low": round(cp[0], 4), "clopper_pearson_95_high": round(cp[1], 4),
                         "wilson_95_low": round(wi[0], 4), "wilson_95_high": round(wi[1], 4)})
    return pd.DataFrame(rows)


def explanation_change(d: dict) -> pd.DataFrame:
    s, bands = d["summary"], d["bands"]
    hb = bands[bands["band"] == "heldout"].set_index(["model_key", "sample_position"])
    rows = []
    for (model, sample, status), g in s.groupby(["model_key", "sample_position", "physical_status"]):
        b = hb.loc[(model, sample)]
        rows.append({
            "model_key": model, "sample_position": int(sample), "physical_status_group": status, "candidates": int(len(g)),
            "band_top10_jaccard_min": round(float(b["top10_jaccard_min"]), 4),
            "band_spearman_min": round(float(b["spearman_rho_min"]), 4),
            "band_top1_stable": bool(b["top1_stable_all_pairs"]),
            "min_heldout_top10_jaccard": round(float(g["heldout_top10_jaccard_mean"].min()), 4),
            "median_heldout_top10_jaccard": round(float(g["heldout_top10_jaccard_mean"].median()), 4),
            "min_heldout_spearman": round(float(g["heldout_spearman_mean"].min()), 4),
            "median_heldout_spearman": round(float(g["heldout_spearman_mean"].median()), 4),
            "top1_changed_any_seed": int((g["heldout_top1_changed_seeds"] > 0).sum()),
            "top1_changed_3plus_seeds": int((g["heldout_top1_changed_seeds"] >= 3).sum()),
            "exceeds_top1_rule": int(g["exceeds_heldout_top1_rule"].sum()),
            "exceeds_top10": int(g["exceeds_heldout_top10"].sum()),
            "exceeds_spearman": int(g["exceeds_heldout_spearman"].sum()),
            "exceeds_both": int(g["exceeds_heldout_both"].sum()),
            "max_search_seed_objective": round(float(g["search_seed_objective"].max()), 4),
        })
    return pd.DataFrame(rows)


def attack_preservation(d: dict) -> pd.DataFrame:
    s = d["summary"]
    ids = d["identities"]
    rows = []
    for (model, sample), g in s.groupby(["model_key", "sample_position"]):
        valid = ids[(ids["model_key"] == model) & (ids["sample_position"] == sample) & ids["valid"]]
        rows.append({
            "model_key": model, "sample_position": int(sample),
            "valid_unique": int(valid["candidate_uid"].nunique()), "evaluated": int(len(g)),
            "still_attack": int((g["p_attack_candidate"] >= 0.5).sum()),
            "p_attack_original": round(float(g["p_attack_original"].iloc[0]), 4),
            "min_p_attack_candidate": round(float(g["p_attack_candidate"].min()), 4),
            "max_abs_delta_p": round(float(g["delta_p"].abs().max()), 4),
            "mean_delta_p": round(float(g["delta_p"].mean()), 4),
        })
    return pd.DataFrame(rows)


def budget_effect(d: dict) -> pd.DataFrame:
    s = d["summary"].merge(d["unique"][["candidate_uid", "smallest_budget"]].rename(columns={"candidate_uid": "unique_candidate_uid"}),
                           on="unique_candidate_uid", how="left")
    rows = []
    for (model, budget, status), g in s.groupby(["model_key", "smallest_budget", "physical_status"]):
        rows.append({"model_key": model, "smallest_budget": budget, "physical_status_group": status, "candidates": int(len(g)),
                     "max_heldout_top10_shift": round(float(g["heldout_top10_shift"].max()), 4),
                     "min_heldout_spearman": round(float(g["heldout_spearman_mean"].min()), 4),
                     "max_relative_change": round(float(g["max_relative_change"].max()), 4),
                     "exceeds_both": int(g["exceeds_heldout_both"].sum())})
    return pd.DataFrame(rows)


def closest_to_band(d: dict) -> pd.DataFrame:
    """Margin = candidate mean minus band minimum; a candidate exceeds the band only when the margin is negative."""
    s, bands = d["summary"], d["bands"]
    hb = bands[bands["band"] == "heldout"].set_index(["model_key", "sample_position"])
    rows = []
    for (model, sample), g in s.groupby(["model_key", "sample_position"]):
        b = hb.loc[(model, sample)]
        m_j = g["heldout_top10_jaccard_mean"] - float(b["top10_jaccard_min"])
        m_r = g["heldout_spearman_mean"] - float(b["spearman_rho_min"])
        joint = np.maximum(m_j, m_r)  # both must be negative for exceeds_both
        best = g.loc[joint.idxmin()]
        rows.append({"model_key": model, "sample_position": int(sample),
                     "smallest_top10_margin": round(float(m_j.min()), 4), "smallest_spearman_margin": round(float(m_r.min()), 4),
                     "closest_candidate": best["unique_candidate_uid"], "closest_status": best["physical_status"],
                     "closest_top10_jaccard": round(float(best["heldout_top10_jaccard_mean"]), 4),
                     "closest_spearman": round(float(best["heldout_spearman_mean"]), 4),
                     "band_top10_min": round(float(b["top10_jaccard_min"]), 4), "band_spearman_min": round(float(b["spearman_rho_min"]), 4)})
    return pd.DataFrame(rows)


def _original_key(original_row: pd.Series) -> str:
    sys.path.insert(0, str(ROOT))
    from experiments import unsw_step6_exhaustive as ex  # noqa: E402
    return ex.key_sha(ex.candidate_key(original_row))


def feature_rank(d: dict, X_val: pd.DataFrame) -> pd.DataFrame:
    """Rank (1 = largest |weight|) of each changed feature in the ORIGINAL explanation, mean over held-out seeds."""
    rows = []
    for model in sorted(d["summary"]["model_key"].unique()):
        for sample in sorted(d["summary"]["sample_position"].unique()):
            weights = d["original_weights"].get((model, int(sample), _original_key(X_val.iloc[[int(sample)]].iloc[0])))
            if not weights:
                continue
            ranks = {f: [] for f in CHANGED_FEATURES}
            for seed, w in weights.items():
                order = sorted(w, key=lambda f: -abs(float(w[f])))
                for f in CHANGED_FEATURES:
                    ranks[f].append(order.index(f) + 1)
            row = {"model_key": model, "sample_position": int(sample), "seeds": len(weights)}
            row.update({f"rank_{f}": round(float(np.mean(v)), 1) for f, v in ranks.items()})
            row["any_changed_feature_in_top10"] = bool(min(min(v) for v in ranks.values()) <= 10)
            rows.append(row)
    return pd.DataFrame(rows)


def bin_crossings(d: dict, X_subtrain: pd.DataFrame, X_val: pd.DataFrame) -> pd.DataFrame:
    """LIME's quartile discretizer: bin = searchsorted(25/50/75th percentiles of the sub-training data, value)."""
    edges = {f: np.percentile(X_subtrain[f].astype(float).to_numpy(), [25, 50, 75]) for f in CHANGED_FEATURES}
    unique = d["unique"]
    rows = []
    for (sample, status), g in unique.groupby(["sample_position", "physical_status"]):
        original = X_val.iloc[[int(sample)]].iloc[0]
        crossed = Counter()
        n_cross = 0
        for raw in g["candidate_json"]:
            cand = json.loads(raw)
            hit = [f for f in CHANGED_FEATURES
                   if int(np.searchsorted(edges[f], float(original[f]))) != int(np.searchsorted(edges[f], float(cand[f])))]
            crossed.update(hit)
            n_cross += bool(hit)
        rows.append({"sample_position": int(sample), "physical_status_group": status, "candidates": int(len(g)),
                     "candidates_crossing_any_edge": n_cross,
                     **{f"crossing_{f}": int(crossed[f]) for f in CHANGED_FEATURES}})
    return pd.DataFrame(rows)


def md(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_(none)_\n"
    cols = list(frame.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in frame.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    d = load(run_dir)
    if d["status"]["state"] != "complete":
        raise SystemExit(f"run is not complete: {d['status']['state']}")
    out = run_dir / "report"
    out.mkdir(exist_ok=True)
    ov = overview(d)
    sys.path.insert(0, str(ROOT))
    from experiments import unsw_step6_runner as runner  # noqa: E402
    from src.unsw.xai_setup import get_xai_data  # noqa: E402
    split = runner.guarded_split(get_xai_data())  # official test set removed
    tables = {
        "candidates_by_flow": candidates_by_flow(d),
        "success_by_group": success_by_group(d),
        "success_intervals": success_intervals(d),
        "explanation_change": explanation_change(d),
        "attack_preservation": attack_preservation(d),
        "budget_effect": budget_effect(d),
        "closest_to_band": closest_to_band(d),
        "feature_rank": feature_rank(d, split["X_val"]),
        "bin_crossings": bin_crossings(d, split["X_subtrain"], split["X_val"]),
    }
    (out / "run_overview.json").write_text(json.dumps(ov, indent=2, default=str), encoding="utf-8", newline="\n")
    parts = [f"# Report tables: {ov['run_id']} ({ov['run_role']})\n",
             "Pilot results are never pooled with the final large-scale experiment.\n",
             "```json\n" + json.dumps(ov, indent=2, default=str) + "\n```\n"]
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False, lineterminator="\n")
        parts.append(f"## {name}\n\n" + md(frame))
    (out / "report_tables.md").write_text("\n".join(parts), encoding="utf-8", newline="\n")
    print(f"report written to {out}")


if __name__ == "__main__":
    main()
