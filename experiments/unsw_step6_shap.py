"""Step 6 v2: TreeSHAP evaluation of the physical_v1 lattice (SHAP pilot, decision D11).

Every unique physical_v1 candidate of the 10 pilot flows (the same 1,108 candidates as the exhaustive
LIME pilot; same enumeration and candidate ids) is explained with TreeSHAP for RF and XGBoost and
compared with its original flow.

Explainer (Part B / experiments/unsw_shap_explain.py, reused unchanged through
experiments/unsw_spoofing_explanations.py): TreeExplainer with 100 background rows of the preprocessed
validation split (numpy default_rng(42)); RF on the probability scale, XGBoost (booster,
tree_path_dependent, raw) on the logit scale; one-hot columns summed back to the 42 features.

Measures per candidate (|SHAP| over 42 features, original x vs candidate x'): top-1 change, top-5 and
top-10 Jaccard, Spearman (rank and tie-aware), sign agreement in the shared top-10.

D11 (decisions doc, unchanged): reference = the normal SHAP-vs-LIME disagreement on the same original:
    J_ref   = mean over held-out LIME seeds 22/33/44/55 of top10_jaccard(SHAP(x), LIME_s(x))
    rho_ref = the same for the tie-aware Spearman
    shap_success = still attack AND valid physical_v1 candidate AND consistent_with_checked_rules
                   AND J(x, x') <= J_ref AND rho_tieaware(x, x') <= rho_ref
The original LIME explanations are computed fresh (80 calls) and checked against results/lime_stability.

Pilot: 10 flows, the current pinned RF/XGBoost models. Never pooled with the final experiment.
The official test set is removed before any code runs.

    python experiments/unsw_step6_shap.py --lime-pilot results/step6_v2_runs/exhaustive_pilot_20261117_180620
"""

from __future__ import annotations

import argparse
import json
import os
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

from experiments import unsw_step6_exhaustive as ex  # noqa: E402
from experiments import unsw_step6_manifest as mf  # noqa: E402
from experiments import unsw_step6_runner as runner  # noqa: E402
from experiments import unsw_targeted_heldout as ho  # noqa: E402
from experiments import unsw_targeted_spoofing as ts  # noqa: E402
from experiments.unsw_spoofing_explanations import (  # noqa: E402  (Part B, unchanged)
    SHAP_BACKGROUND_SEED, SHAP_BACKGROUND_SIZE, build_shap_explainer, check_shap_reproduces_saved, pair_metrics,
    shap_column_to_feature,
)
from experiments.unsw_shap_explain import validate_shap_for_model  # noqa: E402
from src.unsw.constraints import FEATURE_ORDER  # noqa: E402

SHAP_EVAL_VERSION = 1
SPEC_PATH = ROOT / "experiments" / "step6_shap_pilot.json"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "step6_v2_runs"
STATUS_GROUPS = ex.STATUS_GROUPS
METRICS = ("top1_agreement", "top5_jaccard", "top10_jaccard", "spearman_rho", "spearman_abs_tieaware",
           "direction_agreement_top10")


class ShapEvalError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------------------------
def load_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_spec(spec: dict[str, Any], protocol: dict[str, Any]) -> list[str]:
    problems = []
    if spec.get("spec_version") != "step6-v2-shap-pilot-1":
        problems.append("unknown spec_version")
    if spec.get("base_protocol", {}).get("protocol_version") != protocol.get("protocol_version"):
        problems.append("base protocol version differs")
    if sorted(spec.get("models", [])) != sorted(protocol["exhaustive_evaluation"]["models"]):
        problems.append("models must be the exhaustive-evaluation models")
    if spec.get("reference_lime_seeds") != protocol["heldout_lime_seeds"]:
        problems.append("reference_lime_seeds must be the held-out LIME seeds")
    if spec.get("candidates") != "exhaustive_evaluation":
        problems.append("candidates must be the exhaustive-evaluation lattice")
    if spec.get("background", {}).get("size") != SHAP_BACKGROUND_SIZE or spec.get("background", {}).get("seed") != SHAP_BACKGROUND_SEED:
        problems.append("background must match Part B (100 rows, default_rng(42))")
    if spec.get("role") != "pilot":
        problems.append("role must be pilot")
    return problems


# ---------------------------------------------------------------------------------------------
# Pure logic (unit-tested)
# ---------------------------------------------------------------------------------------------
def shap_pair(original: dict[str, float], candidate: dict[str, float]) -> dict[str, Any]:
    """Part B pair_metrics on SHAP attributions, plus the top-1 features."""
    m = pair_metrics(original, candidate)
    return {"orig_top1": m["orig_top1"], "cand_top1": m["mod_top1"], **{k: float(m[k]) for k in METRICS}}


def d11_reference(shap_original: dict[str, float], lime_by_seed: dict[int, dict[str, float]], seeds) -> dict[str, Any]:
    """Normal SHAP-vs-LIME disagreement on the original flow (D11)."""
    rows = [pair_metrics(shap_original, lime_by_seed[int(s)]) for s in sorted(seeds)]
    return {
        "j_ref": float(np.mean([r["top10_jaccard"] for r in rows])),
        "rho_ref": float(np.mean([r["spearman_abs_tieaware"] for r in rows])),
        "rho_ref_rank": float(np.mean([r["spearman_rho"] for r in rows])),
        "j_ref_per_seed": [round(float(r["top10_jaccard"]), 6) for r in rows],
        "rho_ref_per_seed": [round(float(r["spearman_abs_tieaware"]), 6) for r in rows],
    }


def d11_flags(metrics: dict[str, Any], ref: dict[str, Any], still_attack: bool, physical_status: str) -> dict[str, bool]:
    meets_j = bool(metrics["top10_jaccard"] <= ref["j_ref"])
    meets_rho = bool(metrics["spearman_abs_tieaware"] <= ref["rho_ref"])
    both = meets_j and meets_rho
    return {
        "d11_meets_jaccard": meets_j,
        "d11_meets_spearman": meets_rho,
        "d11_meets_both": both,
        # success needs a valid physical_v1 candidate (all evaluated candidates are) consistent with checked rules
        "shap_success": bool(both and still_attack and physical_status == "consistent_with_checked_rules"),
    }


# ---------------------------------------------------------------------------------------------
# Real SHAP (Part B configuration)
# ---------------------------------------------------------------------------------------------
def make_shap_fn(models: dict[str, Any], X_val: pd.DataFrame):
    """shap_fn(model_key, frame) -> (grouped 42-feature frame, output_kind, max additivity gap)."""
    explainers = {k: build_shap_explainer(k, m, X_val) for k, m in models.items()}

    def shap_fn(model_key: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, str, float]:
        pipeline = models[model_key]
        preprocessor, estimator = pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]
        X_pre = preprocessor.transform(frame)
        columns = [str(c) for c in preprocessor.get_feature_names_out()]
        expected = estimator.predict_proba(X_pre)[:, 1]
        values, _base, reconstruction, kind, _label = validate_shap_for_model(
            model_key, estimator, X_pre, columns, explainers[model_key](X_pre), expected)
        reference = expected if kind == "probability" else np.log(np.clip(expected, 1e-9, 1 - 1e-9) / (1 - np.clip(expected, 1e-9, 1 - 1e-9)))
        gap = float(np.max(np.abs(reconstruction - reference))) if len(frame) else 0.0
        raw = pd.DataFrame(values, columns=columns)
        grouped = raw.T.groupby({c: shap_column_to_feature(c) for c in columns}).sum().T
        return grouped[FEATURE_ORDER].reset_index(drop=True), kind, gap

    shap_fn.explainers = explainers
    return shap_fn


# ---------------------------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------------------------
def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, path)


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _new_run_dir(root: Path, role: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path, n = root / f"shap_{role}_{stamp}", 1
    while True:
        try:
            path.mkdir()
            return path
        except FileExistsError:
            path, n = root / f"shap_{role}_{stamp}_{n:02d}", n + 1


def compare_with_lime(summary: pd.DataFrame, lime_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join with the LIME pilot on (model, unique candidate); the candidate sets and P(attack) must agree."""
    lime = pd.read_csv(Path(lime_dir) / "heldout_summary.csv")
    keep = ["model_key", "unique_candidate_uid", "p_attack_candidate", "heldout_top10_jaccard_mean", "heldout_spearman_mean",
            "heldout_top1_changed_seeds", "exceeds_heldout_both", "lime_success", "confirmed_lime_success"]
    lime = lime[keep].rename(columns={c: f"lime_{c}" for c in keep[2:]})
    merged = summary.merge(lime, left_on=["model_key", "candidate_uid"], right_on=["model_key", "unique_candidate_uid"],
                           how="outer", indicator=True)
    checks = {
        "candidates_only_in_shap": int((merged["_merge"] == "left_only").sum()),
        "candidates_only_in_lime": int((merged["_merge"] == "right_only").sum()),
    }
    both = merged[merged["_merge"] == "both"].drop(columns=["_merge", "unique_candidate_uid"])
    checks["matched"] = int(len(both))
    checks["max_p_attack_gap"] = float((both["p_attack_candidate"] - both["lime_p_attack_candidate"]).abs().max()) if len(both) else None
    rows = []
    for model, g in both.groupby("model_key"):
        shap_shift = 1.0 - g["top10_jaccard"]
        lime_shift = 1.0 - g["lime_heldout_top10_jaccard_mean"]
        rows.append({
            "model_key": model, "candidates": int(len(g)),
            "shap_top10_changed": int((g["top10_jaccard"] < 1).sum()),
            "lime_top10_changed": int((g["lime_heldout_top10_jaccard_mean"] < 1).sum()),
            "both_top10_changed": int(((g["top10_jaccard"] < 1) & (g["lime_heldout_top10_jaccard_mean"] < 1)).sum()),
            "shap_top1_changed": int((g["top1_agreement"] < 1).sum()),
            "lime_top1_changed_any_seed": int((g["lime_heldout_top1_changed_seeds"] > 0).sum()),
            "spearman_corr_top10_shift": (float(shap_shift.corr(lime_shift, method="spearman"))
                                          if shap_shift.nunique() > 1 and lime_shift.nunique() > 1 else None),
            "spearman_corr_rho": (float(g["spearman_abs_tieaware"].corr(g["lime_heldout_spearman_mean"], method="spearman"))
                                  if g["spearman_abs_tieaware"].nunique() > 1 and g["lime_heldout_spearman_mean"].nunique() > 1 else None),
            "shap_d11_both": int(g["d11_meets_both"].sum()), "lime_exceeds_both": int(g["lime_exceeds_heldout_both"].sum()),
            "shap_success": int(g["shap_success"].sum()), "lime_success": int(g["lime_lime_success"].sum()),
        })
    return both, {"checks": checks, "by_model": rows}


def summarize(summary: pd.DataFrame, refs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_group, by_flow = [], []
    for (model, status), g in summary.groupby(["model_key", "physical_status"]):
        by_group.append({
            "model_key": model, "physical_status_group": status, "candidates": int(len(g)), "flows": int(g["sample_position"].nunique()),
            "still_attack": int(g["still_attack"].sum()),
            "top1_changed": int((g["top1_agreement"] < 1).sum()),
            "top5_changed": int((g["top5_jaccard"] < 1).sum()),
            "top10_changed": int((g["top10_jaccard"] < 1).sum()),
            "min_top5_jaccard": round(float(g["top5_jaccard"].min()), 4),
            "min_top10_jaccard": round(float(g["top10_jaccard"].min()), 4),
            "min_spearman_tieaware": round(float(g["spearman_abs_tieaware"].min()), 4),
            "min_spearman_rank": round(float(g["spearman_rho"].min()), 4),
            "min_sign_agreement_top10": round(float(g["direction_agreement_top10"].min()), 4),
            "d11_meets_jaccard": int(g["d11_meets_jaccard"].sum()), "d11_meets_spearman": int(g["d11_meets_spearman"].sum()),
            "d11_meets_both": int(g["d11_meets_both"].sum()),
            "shap_success": int(g["shap_success"].sum()),
            "flows_with_shap_success": int(g.loc[g["shap_success"], "sample_position"].nunique()),
        })
    r = refs.set_index(["model_key", "sample_position"])
    for (model, sample, status), g in summary.groupby(["model_key", "sample_position", "physical_status"]):
        ref = r.loc[(model, sample)]
        by_flow.append({
            "model_key": model, "sample_position": int(sample), "physical_status_group": status, "candidates": int(len(g)),
            "j_ref": round(float(ref["j_ref"]), 4), "rho_ref": round(float(ref["rho_ref"]), 4),
            "min_top10_jaccard": round(float(g["top10_jaccard"].min()), 4),
            "min_spearman_tieaware": round(float(g["spearman_abs_tieaware"].min()), 4),
            "top1_changed": int((g["top1_agreement"] < 1).sum()), "top10_changed": int((g["top10_jaccard"] < 1).sum()),
            "min_sign_agreement_top10": round(float(g["direction_agreement_top10"].min()), 4),
            "jaccard_margin": round(float(g["top10_jaccard"].min() - ref["j_ref"]), 4),
            "spearman_margin": round(float(g["spearman_abs_tieaware"].min() - ref["rho_ref"]), 4),
            "d11_meets_both": int(g["d11_meets_both"].sum()), "shap_success": int(g["shap_success"].sum()),
            "min_p_attack_candidate": round(float(g["p_attack_candidate"].min()), 4),
        })
    return pd.DataFrame(by_group), pd.DataFrame(by_flow)


def run_shap_evaluation(
    protocol: dict[str, Any] | None = None,
    spec: dict[str, Any] | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    lime_pilot_dir: str | Path | None = None,
    split: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
    models: dict[str, Any] | None = None,
    shap_fn: Callable | None = None,
    lime_fn: Callable | None = None,
    samples=None,
    check_saved: bool = True,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    protocol = mf.load_protocol(mf.PROTOCOL_V2_PATH) if protocol is None else protocol
    spec = load_spec() if spec is None else spec
    problems = mf.validate_protocol(protocol) + validate_spec(spec, protocol)
    if problems or not mf.is_v2(protocol):
        raise ShapEvalError(f"invalid protocol or SHAP spec: {problems}")
    manifest = mf.build_manifest(protocol) if manifest is None else manifest
    split = runner.guarded_split(ts.get_xai_data() if split is None else split)
    X_val = split["X_val"]
    selected = [int(s) for s in samples] if samples is not None else list(protocol["sample_positions"])
    if not selected or any(s not in protocol["sample_positions"] for s in selected):
        raise ShapEvalError("samples must come from the protocol")
    ex.check_sample_identity(protocol, split, selected)
    model_paths = None
    if models is None:
        models, shas, model_paths = ex.load_verified_models(protocol, manifest, split)
    else:
        shas = {k: manifest["models"][k]["sha256"] for k in spec["models"]}
        ex.check_pinned_hashes(protocol, shas)
    role = ex.run_role(protocol, selected)
    shap_fn = shap_fn or make_shap_fn({k: models[k] for k in spec["models"]}, X_val)
    lime_explain = lime_fn or ho.make_lime_explain_fn(split, shas)
    ref_seeds = tuple(spec["reference_lime_seeds"])

    run_dir = _new_run_dir(Path(output_root), role)
    start = time.perf_counter()
    _write_json(run_dir / "manifest.json", {**manifest, "shap_spec": spec,
                                            "shap_spec_sha256": mf.file_sha256(SPEC_PATH) if SPEC_PATH.exists() else None,
                                            "shap_module_sha256": mf.file_sha256(Path(__file__))})
    _write_json(run_dir / "status.json", {"state": "running", "run_role": role})

    # Same enumeration, deduplication and candidate ids as the exhaustive LIME pilot.
    planned = ex.plan(protocol, X_val, models, selected)
    unique, originals, attack = planned["_unique"], planned["_originals"], planned["_attack"]
    attack_idx = attack.set_index(["model_key", "candidate_uid"])
    uids = list(unique)
    cand_frame = (pd.DataFrame([unique[u]["candidate_row"] for u in uids], columns=X_val.columns)
                  .astype(X_val.dtypes.to_dict()))
    orig_frame = pd.DataFrame([originals[s] for s in selected], columns=X_val.columns).astype(X_val.dtypes.to_dict())

    saved_gap, lime_repro_gap = {}, None
    rows, value_rows, ref_rows, timing = [], [], [], {}
    lime_orig: dict[tuple[str, int], dict[int, dict[str, float]]] = {}
    for model_key in spec["models"]:
        tick = time.perf_counter()
        shap_orig, kind, gap_o = shap_fn(model_key, orig_frame)
        shap_cand, _, gap_c = shap_fn(model_key, cand_frame)
        timing[model_key] = {"shap_seconds": round(time.perf_counter() - tick, 2), "rows": int(len(cand_frame) + len(orig_frame)),
                             "output_kind": kind, "max_additivity_gap": max(gap_o, gap_c)}
        if check_saved and models is not None and model_key in getattr(shap_fn, "explainers", {}):
            saved_rows = pd.read_csv(ROOT / "results" / "metrics" / "shap" / f"{model_key}_selected_shap_values.csv")
            positions = [s for s in selected if s in set(saved_rows["validation_position"])]
            saved_gap[model_key] = {"positions_checked": len(positions),
                                    "max_gap": check_shap_reproduces_saved(model_key, models[model_key], shap_fn.explainers[model_key],
                                                                           X_val, positions) if positions else None}
        shap_o = {s: shap_orig.iloc[i].to_dict() for i, s in enumerate(selected)}
        for i, s in enumerate(selected):
            value_rows.append({"model_key": model_key, "kind": "original", "candidate_uid": f"s{s}_original", "sample_position": s,
                               **{f: float(v) for f, v in shap_o[s].items()}})
        for s in selected:
            lime_orig[(model_key, s)] = {seed: lime_explain(model_key, s, originals[s], seed) for seed in ref_seeds}
            ref = d11_reference(shap_o[s], lime_orig[(model_key, s)], ref_seeds)
            ref_rows.append({"model_key": model_key, "sample_position": s, **ref})
        refs_m = {r["sample_position"]: r for r in ref_rows if r["model_key"] == model_key}
        for i, uid in enumerate(uids):
            info = unique[uid]
            s = info["sample_position"]
            cand = shap_cand.iloc[i].to_dict()
            value_rows.append({"model_key": model_key, "kind": "candidate", "candidate_uid": uid, "sample_position": s,
                               **{f: float(v) for f, v in cand.items()}})
            att = attack_idx.loc[(model_key, uid)]
            metrics = shap_pair(shap_o[s], cand)
            still = bool(att["still_attack"])
            rows.append({
                "model_key": model_key, "candidate_uid": uid, "sample_position": s,
                "physical_status": info["physical_status"], "physical_unverified": info["physical_unverified"],
                "primary_changes": info["primary_changes"], "derived_changes": info["derived_changes"],
                "smallest_budget": info["smallest_budget"], "max_relative_change": info["max_relative_change"],
                "p_attack_original": float(att["p_attack_original"]), "p_attack_candidate": float(att["p_attack_candidate"]),
                "delta_p": float(att["delta_p"]), "still_attack": still,
                **metrics, "j_ref": refs_m[s]["j_ref"], "rho_ref": refs_m[s]["rho_ref"],
                **d11_flags(metrics, refs_m[s], still, info["physical_status"]),
                "shap_output_kind": kind, "physically_valid_claimed": False,
            })
        log(f"{model_key}: SHAP for {len(cand_frame) + len(orig_frame)} rows in {timing[model_key]['shap_seconds']} s")

    if check_saved and lime_fn is None:
        pairs = [(m, s, originals[s]) for (m, s) in lime_orig]
        lime_repro_gap = ho.check_original_reproduces_saved(lime_explain, pairs, ref_seeds)

    summary = pd.DataFrame(rows)
    refs = pd.DataFrame(ref_rows)
    by_group, by_flow = summarize(summary, refs)
    comparison, comparison_meta = (compare_with_lime(summary, Path(lime_pilot_dir)) if lime_pilot_dir else (pd.DataFrame(), None))
    if comparison_meta and (comparison_meta["checks"]["candidates_only_in_shap"] or comparison_meta["checks"]["candidates_only_in_lime"]):
        raise ShapEvalError(f"candidate sets differ from the LIME pilot: {comparison_meta['checks']}")

    _write_csv(run_dir / "shap_summary.csv", summary)
    _write_csv(run_dir / "shap_values.csv", pd.DataFrame(value_rows))
    _write_csv(run_dir / "d11_references.csv", refs.assign(j_ref_per_seed=refs["j_ref_per_seed"].map(json.dumps),
                                                           rho_ref_per_seed=refs["rho_ref_per_seed"].map(json.dumps)))
    _write_csv(run_dir / "summary_by_status_group.csv", by_group)
    _write_csv(run_dir / "summary_by_flow.csv", by_flow)
    if len(comparison):
        _write_csv(run_dir / "comparison_with_lime.csv", comparison)
    models_after = ex.assert_models_unchanged(model_paths, shas) if model_paths else None
    status = {
        "state": "complete", "run_role": role, "samples": selected, "models": spec["models"],
        "unique_candidates": len(uids), "candidate_evaluations": int(len(summary)),
        "timing": timing, "saved_shap_reproduction": saved_gap, "original_lime_vs_lime_stability_max_gap": lime_repro_gap,
        "reference_lime_calls": len(lime_orig) * len(ref_seeds),
        "comparison_with_lime_pilot": comparison_meta,
        "model_sha256": shas, "model_sha256_after_run": models_after, "official_test_used": False,
        "elapsed_seconds": round(time.perf_counter() - start, 1), "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(run_dir / "status.json", status)
    (run_dir / "README.md").write_text(README.format(role=role), encoding="utf-8", newline="\n")
    return {"run_dir": str(run_dir), "status": status, "summary": summary, "refs": refs, "by_group": by_group,
            "by_flow": by_flow, "comparison": comparison}


README = """# Step 6 v2 TreeSHAP evaluation of the physical_v1 lattice ({role})

Same 10 flows and the same unique physical_v1 candidates as the exhaustive LIME pilot, explained with
TreeSHAP (Part B configuration) for RF and XGBoost. Pilot only: never pooled with the final experiment.
No candidate is called physically valid.

| File | One row per |
|---|---|
| `shap_summary.csv` | model x unique candidate: SHAP change measures, D11 reference and flags, P(attack) |
| `shap_values.csv` | model x original/candidate: SHAP values summed to the 42 features |
| `d11_references.csv` | model x flow: J_ref and rho_ref (SHAP vs LIME on the original, held-out seeds 22-55) |
| `summary_by_status_group.csv` | model x physical status group |
| `summary_by_flow.csv` | model x flow x physical status group |
| `comparison_with_lime.csv` | model x candidate: SHAP measures next to the LIME pilot's held-out measures |

`shap_success` (D11) = still attack AND consistent_with_checked_rules AND top-10 Jaccard <= J_ref AND
tie-aware Spearman <= rho_ref. Candidates with unverified dependencies are reported but never succeed.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lime-pilot", default=None, help="exhaustive LIME pilot run folder to compare with")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--samples", nargs="+", type=int, default=None)
    args = parser.parse_args()
    out = run_shap_evaluation(output_root=args.output_root, lime_pilot_dir=args.lime_pilot, samples=args.samples)
    print(json.dumps({k: v for k, v in out["status"].items() if k != "comparison_with_lime_pilot"}, indent=2, default=str))
    print(f"Run directory: {out['run_dir']}")


if __name__ == "__main__":
    main()
