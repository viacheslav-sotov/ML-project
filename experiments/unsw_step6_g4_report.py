"""Step 6 G4: report tables for a finished G4 run (diagnostic only; reads the run folder, no LIME, no models).

Writes into <run dir>/report/:
    run_overview.json         tasks, failures, runtime, explanation counts, checks
    targeted_vs_random.csv    model x flow x checkpoint: search objective and held-out shift, shuffled greedy vs random control
    paired_tests.csv          model x checkpoint: Wilcoxon signed-rank over flows (EXPLORATORY; not predeclared for G4)
    heldout_by_group.csv      model x method x physical status: held-out LIME, confirmation, TreeSHAP D11 (diagnostic)
    heldout_by_flow.csv       model x flow: band, closest candidate, changes, D11
    features.csv              model x feature: how often edited in checkpoint candidates, and among the most-moving ones
    ttl_effect.csv            model x (TTL edited or not): held-out changes
    perturbation_size.csv     model x method: number of edited features and relative change
    report_tables.md          all tables in Markdown

Every success here is DIAGNOSTIC ONLY and never evidence of a physically realizable network attack.

    python experiments/unsw_step6_g4_report.py <run dir>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

TTL_FEATURES = ("sttl", "dttl", "ct_state_ttl")
STATUSES = ("consistent_with_checked_rules", "unverified_dependencies", "violates_checked_rule")


def load(run_dir: Path) -> dict:
    d = {"status": json.loads((run_dir / "status.json").read_text(encoding="utf-8")),
         "config": json.loads((run_dir / "run_config.json").read_text(encoding="utf-8")),
         "manifest": json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")),
         "checkpoints": pd.read_csv(run_dir / "checkpoints.csv"),
         "summary": pd.read_csv(run_dir / "heldout_summary.csv"),
         "bands": pd.read_csv(run_dir / "noise_bands.csv")}
    d["tasks"] = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((run_dir / "tasks").glob("*.json"))]
    return d


def overview(d: dict) -> dict:
    s, tasks = d["status"], d["tasks"]
    planned = d["config"]["plan"]["tasks"]
    stopped = Counter(t["stopped_reason"] for t in tasks)
    return {
        "run_id": Path(d["config"].get("run_id", "")).name or None, "run_role": s["run_role"], "state": s["state"],
        "git": d["manifest"]["git"], "g4_protocol_sha256": d["config"]["header"]["g4_protocol_sha256"],
        "model_sha256": d["config"]["header"]["model_sha256"], "model_sha256_after_run": s.get("model_sha256_after_run"),
        "official_test_used": s["official_test_used"],
        "tasks_planned": planned, "tasks_completed": len(tasks), "tasks_failed": planned - len(tasks),
        "stopped_reasons": dict(stopped),
        "search_lime_calls": s["search_lime_calls"], "search_lime_calls_max": d["config"]["plan"]["search_lime_calls_max"],
        "new_explanations_by_model": s["new_explanations"], "seconds_per_new_explanation": s["seconds_per_new_explanation"],
        "lime_minutes_by_model": {k: round(s["new_explanations"][k] * s["seconds_per_new_explanation"][k] / 60, 1)
                                  for k in s["new_explanations"]},
        "elapsed_minutes": round(s["elapsed_seconds"] / 60, 1),
        "checkpoint_rows": s["checkpoint_candidates"], "unique_candidates_evaluated": s["unique_candidates_evaluated"],
        "physical_status_counts": s["physical_status_counts"], "search_score_mismatches": s["search_score_mismatches"],
        "confirmation_candidates": s["confirmation_candidates"], "shap_evaluated": s["shap_evaluated"],
    }


def joined(d: dict) -> pd.DataFrame:
    ck = d["checkpoints"].copy()
    s = d["summary"]
    keep = ["candidate_uid", "physical_status", "n_changed", "max_relative_change", "changed_features", "p_attack_candidate",
            "heldout_top10_jaccard_mean", "heldout_spearman_mean", "heldout_top5_jaccard_mean", "heldout_top1_changed_seeds",
            "exceeds_heldout_top10", "exceeds_heldout_spearman", "exceeds_heldout_both", "lime_diagnostic_success",
            "confirmation_evaluated", "confirmation_passed", "confirmed_lime_diagnostic_success",
            "shap_top1_agreement", "shap_top5_jaccard", "shap_top10_jaccard", "shap_spearman_abs_tieaware",
            "shap_j_ref", "shap_rho_ref", "shap_d11_meets_both"]
    ck = ck.merge(s[[c for c in keep if c in s]], on="candidate_uid", how="left")
    # an unchanged original has no shift: Jaccard 1, Spearman 1
    orig = ck["candidate_is_original"].astype(bool)
    for c in ("heldout_top10_jaccard_mean", "heldout_spearman_mean", "heldout_top5_jaccard_mean"):
        ck.loc[orig, c] = 1.0
    ck["heldout_spearman_shift"] = 1.0 - ck["heldout_spearman_mean"]
    ck["heldout_top10_shift"] = 1.0 - ck["heldout_top10_jaccard_mean"]
    return ck


def targeted_vs_random(ck: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, sample, cp), g in ck.groupby(["model_key", "sample_position", "checkpoint"]):
        row = {"model_key": model, "sample_position": int(sample), "checkpoint": int(cp)}
        for method, tag in (("shuffled_greedy", "greedy"), ("random_control", "random")):
            m = g[g["method"] == method]
            row.update({f"{tag}_runs": int(len(m)),
                        f"{tag}_objective_mean": round(float(m["best_score"].mean()), 4),
                        f"{tag}_objective_max": round(float(m["best_score"].max()), 4),
                        f"{tag}_heldout_spearman_shift_mean": round(float(m["heldout_spearman_shift"].mean()), 4),
                        f"{tag}_heldout_top10_shift_mean": round(float(m["heldout_top10_shift"].mean()), 4),
                        f"{tag}_exceeds_both": int(m["exceeds_heldout_both"].fillna(False).astype(bool).sum())})
        rows.append(row)
    return pd.DataFrame(rows)


def paired_tests(tvr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, cp), g in tvr.groupby(["model_key", "checkpoint"]):
        for metric in ("objective_mean", "heldout_spearman_shift_mean"):
            diff = g[f"greedy_{metric}"] - g[f"random_{metric}"]
            nz = diff[diff != 0]
            p = float(wilcoxon(nz).pvalue) if len(nz) >= 2 else None
            rows.append({"model_key": model, "checkpoint": int(cp), "metric": metric.replace("_mean", ""),
                         "flows": int(len(g)), "greedy_better": int((diff > 0).sum()), "random_better": int((diff < 0).sum()),
                         "ties": int((diff == 0).sum()), "median_difference": round(float(diff.median()), 4),
                         "wilcoxon_p_two_sided": None if p is None else round(p, 4), "label": "exploratory"})
    return pd.DataFrame(rows)


def heldout_by_group(d: dict) -> pd.DataFrame:
    s = d["summary"]
    rows = []
    for (model, method, status), g in s.groupby(["model_key", "found_by_method", "physical_status"]):
        rows.append({
            "model_key": model, "found_by_method": method, "physical_status_group": status, "candidates": int(len(g)),
            "flows": int(g["sample_position"].nunique()),
            "exceeds_top10": int(g["exceeds_heldout_top10"].sum()), "exceeds_spearman": int(g["exceeds_heldout_spearman"].sum()),
            "exceeds_both": int(g["exceeds_heldout_both"].sum()),
            "lime_diagnostic_success": int(g["lime_diagnostic_success"].sum()),
            "confirmation_evaluated": int(g["confirmation_evaluated"].sum()), "confirmation_passed": int(g["confirmation_passed"].sum()),
            "confirmed_diagnostic": int(g["confirmed_lime_diagnostic_success"].sum()),
            "flows_with_confirmed": int(g.loc[g["confirmed_lime_diagnostic_success"].astype(bool), "sample_position"].nunique()),
            "lime_top1_changed_any_seed": int((g["heldout_top1_changed_seeds"] > 0).sum()),
            "lime_top1_rule": int(g["exceeds_heldout_top1_rule"].sum()),
            "min_lime_top5_jaccard": round(float(g["heldout_top5_jaccard_mean"].min()), 4),
            "min_lime_top10_jaccard": round(float(g["heldout_top10_jaccard_mean"].min()), 4),
            "min_lime_spearman": round(float(g["heldout_spearman_mean"].min()), 4),
            "shap_top1_changed": int((g["shap_top1_agreement"] < 1).sum()) if "shap_top1_agreement" in g else None,
            "min_shap_top5_jaccard": round(float(g["shap_top5_jaccard"].min()), 4) if "shap_top5_jaccard" in g else None,
            "min_shap_top10_jaccard": round(float(g["shap_top10_jaccard"].min()), 4) if "shap_top10_jaccard" in g else None,
            "min_shap_spearman": round(float(g["shap_spearman_abs_tieaware"].min()), 4) if "shap_spearman_abs_tieaware" in g else None,
            "shap_d11_meets_both": int(g["shap_d11_meets_both"].sum()) if "shap_d11_meets_both" in g else None,
            "min_p_attack": round(float(g["p_attack_candidate"].min()), 4),
        })
    return pd.DataFrame(rows)


def heldout_by_flow(d: dict) -> pd.DataFrame:
    s, bands = d["summary"], d["bands"]
    hb = bands[bands["band"] == "heldout"].set_index(["model_key", "sample_position"])
    rows = []
    for (model, sample), g in s.groupby(["model_key", "sample_position"]):
        b = hb.loc[(model, sample)]
        margin = np.maximum(g["heldout_top10_jaccard_mean"] - float(b["top10_jaccard_min"]),
                            g["heldout_spearman_mean"] - float(b["spearman_rho_min"]))
        best = g.loc[margin.idxmin()]
        rows.append({
            "model_key": model, "sample_position": int(sample), "candidates": int(len(g)),
            "band_top10_min": round(float(b["top10_jaccard_min"]), 4), "band_spearman_min": round(float(b["spearman_rho_min"]), 4),
            "min_top10_jaccard": round(float(g["heldout_top10_jaccard_mean"].min()), 4),
            "min_spearman": round(float(g["heldout_spearman_mean"].min()), 4),
            "exceeds_both": int(g["exceeds_heldout_both"].sum()), "confirmed": int(g["confirmed_lime_diagnostic_success"].sum()),
            "closest_joint_margin": round(float(margin.min()), 4), "closest_changed_features": best["changed_features"],
            "closest_method": best["found_by_method"], "closest_status": best["physical_status"],
            "shap_j_ref": round(float(g["shap_j_ref"].iloc[0]), 4) if "shap_j_ref" in g else None,
            "shap_rho_ref": round(float(g["shap_rho_ref"].iloc[0]), 4) if "shap_rho_ref" in g else None,
            "min_shap_top10_jaccard": round(float(g["shap_top10_jaccard"].min()), 4) if "shap_top10_jaccard" in g else None,
            "min_shap_spearman": round(float(g["shap_spearman_abs_tieaware"].min()), 4) if "shap_spearman_abs_tieaware" in g else None,
            "shap_d11_meets_both": int(g["shap_d11_meets_both"].sum()) if "shap_d11_meets_both" in g else None,
            "min_p_attack": round(float(g["p_attack_candidate"].min()), 4),
        })
    return pd.DataFrame(rows)


def features(d: dict) -> pd.DataFrame:
    s = d["summary"]
    rows = []
    for model, g in s.groupby("model_key"):
        top = g[g["exceeds_heldout_both"].astype(bool)] if g["exceeds_heldout_both"].any() else g.nsmallest(max(1, len(g) // 10), "heldout_spearman_mean")
        all_c = Counter(f for x in g["changed_features"].dropna() for f in str(x).split(";") if f)
        top_c = Counter(f for x in top["changed_features"].dropna() for f in str(x).split(";") if f)
        for f, n in all_c.most_common():
            rows.append({"model_key": model, "feature": f, "is_ttl": f in TTL_FEATURES, "candidates_editing": n,
                         "share_of_candidates": round(n / len(g), 4), "in_most_moving": top_c.get(f, 0),
                         "share_of_most_moving": round(top_c.get(f, 0) / len(top), 4), "most_moving_definition":
                         "exceeds held-out band on both" if g["exceeds_heldout_both"].any() else "lowest 10% held-out Spearman"})
    return pd.DataFrame(rows)


def ttl_effect(d: dict) -> pd.DataFrame:
    s = d["summary"].copy()
    s["edits_ttl"] = s["changed_features"].fillna("").map(lambda x: any(f in TTL_FEATURES for f in x.split(";")))
    rows = []
    for (model, ttl), g in s.groupby(["model_key", "edits_ttl"]):
        rows.append({"model_key": model, "edits_ttl_feature": bool(ttl), "candidates": int(len(g)),
                     "median_heldout_spearman": round(float(g["heldout_spearman_mean"].median()), 4),
                     "min_heldout_spearman": round(float(g["heldout_spearman_mean"].min()), 4),
                     "median_heldout_top10_jaccard": round(float(g["heldout_top10_jaccard_mean"].median()), 4),
                     "exceeds_both": int(g["exceeds_heldout_both"].sum()),
                     "confirmed": int(g["confirmed_lime_diagnostic_success"].sum()),
                     "shap_d11_meets_both": int(g["shap_d11_meets_both"].sum()) if "shap_d11_meets_both" in g else None})
    return pd.DataFrame(rows)


def perturbation_size(d: dict) -> pd.DataFrame:
    s = d["summary"]
    rows = []
    for (model, method), g in s.groupby(["model_key", "found_by_method"]):
        rows.append({"model_key": model, "found_by_method": method, "candidates": int(len(g)),
                     "edited_features_median": float(g["n_changed"].median()), "edited_features_max": int(g["n_changed"].max()),
                     "max_relative_change_median": round(float(g["max_relative_change"].median()), 4),
                     "max_relative_change_max": round(float(g["max_relative_change"].max()), 4)})
    return pd.DataFrame(rows)


def physical(d: dict) -> pd.DataFrame:
    s = d["summary"]
    rows = []
    for (model, status), g in s.groupby(["model_key", "physical_status"]):
        viol = Counter(v.split(":")[0].strip() if not v.startswith("checked rule") else v.strip()
                       for x in g["physical_violations"].fillna("") for v in str(x).split("; ") if v)
        rows.append({"model_key": model, "physical_status": status, "candidates": int(len(g)),
                     "exceeds_both": int(g["exceeds_heldout_both"].sum()),
                     "top_violations": "; ".join(f"{k} ({n})" for k, n in viol.most_common(4))})
    return pd.DataFrame(rows)


def md(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_(none)_\n"
    cols = list(frame.columns)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                     + ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in frame.iterrows()]) + "\n"


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
    ov["run_id"] = run_dir.name
    ck = joined(d)
    tvr = targeted_vs_random(ck)
    tables = {"targeted_vs_random": tvr, "paired_tests": paired_tests(tvr), "heldout_by_group": heldout_by_group(d),
              "heldout_by_flow": heldout_by_flow(d), "physical": physical(d), "perturbation_size": perturbation_size(d),
              "ttl_effect": ttl_effect(d), "features": features(d)}
    (out / "run_overview.json").write_text(json.dumps(ov, indent=2, default=str), encoding="utf-8", newline="\n")
    parts = [f"# G4 report tables: {run_dir.name} (DIAGNOSTIC ONLY)\n",
             "G4 successes are never evidence of a physically realizable network attack.\n",
             "```json\n" + json.dumps(ov, indent=2, default=str) + "\n```\n"]
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False, lineterminator="\n")
        parts.append(f"## {name}\n\n" + md(frame))
    (out / "report_tables.md").write_text("\n".join(parts), encoding="utf-8", newline="\n")
    print(f"report written to {out}")


if __name__ == "__main__":
    main()
