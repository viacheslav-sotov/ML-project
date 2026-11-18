"""Protocol v3 experiment E0: exhaustive evaluation of the physical_v1 lattice on the frozen 263-flow sample,
for the final models (RF, XGBoost, MLP primary; LR secondary) and both feature sets.

Success criteria are the v2 definitions, reused unchanged:
  LIME (D8)   experiments.unsw_targeted_heldout.evaluate_candidates (held-out band from seed pairs 22/33/44/55,
              candidate shift = mean over the same seeds; Part B set 11/22/33/44/55 also reported);
              lime_success = exceeds both (top-10 Jaccard AND Spearman) AND status consistent_with_checked_rules
  D14         candidates exceeding the held-out band are re-evaluated on seeds 401-404 (same function)
  TreeSHAP    experiments.unsw_step6_shap.shap_pair / d11_reference (LIME seeds 22/33/44/55) / d11_flags (D11)
  RQ5         src.unsw.v3_explain.rq5_matched (descriptive)

Rule D (frozen): a flow whose ORIGINAL a model variant does not detect (P < 0.5) is reported as not applicable for
that variant (id listed, no explanation computed); the flow is never replaced. Candidates the variant no longer
detects are counted and not explained. Paired full42-vs-no_ttl39 comparisons use flows detected by both variants.

Checkpoint/resume: every explanation is appended (and flushed) to explanations_cache.jsonl under an identity key
(explainer, model SHA-256, feature set, settings hash, seed, vector hash). --resume <run_dir> refuses a different
header, reuses every cached explanation and computes only what is missing; a line cut by an interruption is dropped
and recomputed. Outputs are rewritten from the cache, so a resumed run gives the same files as an uninterrupted one.

    python experiments/run_guarded.py experiments/unsw_v3_e0.py --budget A --plan-only
    python experiments/run_guarded.py experiments/unsw_v3_e0.py --budget A
    python experiments/run_guarded.py experiments/unsw_v3_e0.py --budget A --resume results/v3_runs_e0/<run>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import unsw_nested_cv as ncv  # noqa: E402
from experiments import unsw_step6_shap as sh  # noqa: E402  (D11, unchanged)
from experiments import unsw_targeted_heldout as ho  # noqa: E402  (D8/D14, unchanged)
from src.unsw import physical_repair as pr  # noqa: E402
from src.unsw import v3_data as vd  # noqa: E402
from src.unsw import v3_explain as vx  # noqa: E402
from src.unsw import v3_final_models as fm  # noqa: E402
from src.unsw import v3_sample as vs  # noqa: E402

E0_VERSION = 1
PROTOCOL_V3 = ROOT / "experiments" / "step6_protocol_v3.json"
OUTPUT_ROOT = ROOT / "results" / "v3_runs_e0"
CACHE = "explanations_cache.jsonl"
SEARCH_SEED, HELDOUT, PART_B, CONFIRM = 11, (22, 33, 44, 55), (11, 22, 33, 44, 55), (401, 402, 403, 404)
PRIMARY = [(m, fs) for fs in ("full42", "no_ttl39") for m in ("random_forest", "xgboost", "mlp")]
SECONDARY = [("logistic_regression", "full42"), ("logistic_regression", "no_ttl39")]
TREES = {"random_forest", "xgboost"}
# measured in results/v3_prep/readiness/readiness.json (seconds per explanation, single process)
SECONDS = {"lime": {"random_forest": 0.247, "xgboost": 0.131, "mlp": 0.066, "logistic_regression": 0.056},
           "treeshap": {"random_forest": 0.689, "xgboost": 0.024}}


class E0Error(RuntimeError):
    pass


class Interrupted(RuntimeError):
    pass


def candidate_key(row: pd.Series) -> str:
    return vx.vector_key(row, vd.feature_order())


def load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_V3.read_text(encoding="utf-8"))


def enumerate_lattice(sample: pd.DataFrame, X: pd.DataFrame, budget: str) -> dict[int, list[dict[str, Any]]]:
    out = {}
    for fid in sample["id"]:
        orig = X.loc[fid]
        cands = []
        for assignment, cand, verdict in pr.valid_lattice_candidates(orig, budget):
            cands.append({"candidate_uid": f"{fid}:{candidate_key(cand)[:16]}", "flow_id": int(fid), "candidate_row": cand,
                          "assignment": json.dumps(assignment, sort_keys=True), "n_edited_features": len(assignment),
                          "physical_status": verdict["status"], "physical_unverified": json.dumps(verdict.get("unverified", [])),
                          "primary_changes": json.dumps(verdict.get("primary_changes", {}), sort_keys=True),
                          "derived_changes": json.dumps(verdict.get("derived_changes", {}), sort_keys=True)})
        out[int(fid)] = cands
    return out


def detection(models: dict, X: pd.DataFrame, lattice: dict[int, list[dict]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Per variant: P(attack) of every original and candidate; Rule D bookkeeping."""
    rows = [c["candidate_row"] for cands in lattice.values() for c in cands]
    C = pd.DataFrame(rows)[vd.feature_order()].infer_objects() if rows else pd.DataFrame(columns=vd.feature_order())
    uids = [c["candidate_uid"] for cands in lattice.values() for c in cands]
    out = {}
    for key, m in models.items():
        p_orig = dict(zip(X.index.astype(int), m.predict_attack_proba(X)))
        p_cand = dict(zip(uids, m.predict_attack_proba(C))) if rows else {}
        det_flows = [f for f in lattice if p_orig[f] >= fm.DECISION_THRESHOLD]
        out[key] = {"p_orig": p_orig, "p_cand": p_cand,
                    "flows_not_detected": sorted(int(f) for f in lattice if p_orig[f] < fm.DECISION_THRESHOLD),
                    "detected_flows": det_flows,
                    "candidates_evaluated": [c for f in det_flows for c in lattice[f] if p_cand[c["candidate_uid"]] >= fm.DECISION_THRESHOLD],
                    "candidates_not_detected": sum(1 for f in det_flows for c in lattice[f] if p_cand[c["candidate_uid"]] < fm.DECISION_THRESHOLD),
                    "candidates_of_undetected_flows": sum(len(lattice[f]) for f in lattice if p_orig[f] < fm.DECISION_THRESHOLD)}
    return out


def plan_counts(det: dict, variants) -> dict[str, Any]:
    per, tot_s = {}, 0.0
    for key in variants:
        d = det[key]
        n_f, n_c = len(d["detected_flows"]), len(d["candidates_evaluated"])
        lime = (n_f + n_c) * len(PART_B)
        shap = (n_f + n_c) if key[0] in TREES else 0
        conf_max = 4 * (n_c + n_f)
        secs = lime * SECONDS["lime"][key[0]] + shap * SECONDS["treeshap"].get(key[0], 0.0)
        tot_s += secs
        per[f"{key[0]}_{key[1]}"] = {"flows_detected": n_f, "flows_not_detected": d["flows_not_detected"],
                                      "candidates_evaluated": n_c, "candidates_not_detected": d["candidates_not_detected"],
                                      "candidates_of_undetected_flows": d["candidates_of_undetected_flows"],
                                      "lime_calls": lime, "treeshap_evaluations": shap,
                                      "confirmation_lime_calls_worst_case": conf_max, "estimated_hours": round(secs / 3600, 2)}
    return {"per_variant": per, "lime_calls_total": sum(v["lime_calls"] for v in per.values()),
            "treeshap_total": sum(v["treeshap_evaluations"] for v in per.values()),
            "candidates_total_unique": None, "estimated_hours_total": round(tot_s / 3600, 2)}


def header_for(protocol, sample, models, budget, flows, variants, split_digest, bg) -> dict[str, Any]:
    return {"e0_version": E0_VERSION, "protocol_version": protocol["protocol_version"],
            "protocol_sha256": hashlib.sha256(PROTOCOL_V3.read_bytes()).hexdigest(),
            "code_sha256": fm.file_sha256(Path(__file__)), "v3_explain_sha256": fm.file_sha256(ROOT / "src" / "unsw" / "v3_explain.py"),
            "physical_repair_sha256": fm.file_sha256(ROOT / "src" / "unsw" / "physical_repair.py"),
            "sample_sha256": vs.sample_digest(sample), "split_digest": split_digest, "budget": budget,
            "flows": None if flows is None else sorted(int(f) for f in flows), "variants": [f"{m}_{fs}" for m, fs in variants],
            "model_sha256": {f"{m}_{fs}": models[(m, fs)].sha256 for m, fs in variants},
            "seeds": {"search": SEARCH_SEED, "heldout": list(HELDOUT), "part_b": list(PART_B), "confirmation": list(CONFIRM)},
            "lime_settings_sha256": {fs: vx.settings_hash(vx.lime_settings(n)) for fs, n in (("full42", 42), ("no_ttl39", 39))},
            "treeshap_background": bg}


def _write(path: Path, payload) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if isinstance(payload, pd.DataFrame):
        payload.to_csv(tmp, index=False, lineterminator="\n")
    else:
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8", newline="\n")
    tmp.replace(path)


def run(budget: str, variants, flows=None, resume: str | None = None, plan_only: bool = False,
        max_new: int | None = None, output_root: Path = OUTPUT_ROOT, log=print) -> dict[str, Any]:
    protocol = load_protocol()
    e0 = protocol["experiments"]["E0_exhaustive_physical_v1"]
    if not plan_only and not (protocol["status"].startswith("APPROVED") and budget in e0["approved_budgets"]):
        raise E0Error(f"protocol v3 does not approve E0 Budget {budget} (status: {protocol['status'][:60]})")
    sample = vs.load_frozen_sample(protocol)
    if flows is not None:
        missing = set(flows) - set(sample["id"])
        if missing:
            raise E0Error(f"flows not in the frozen sample: {sorted(missing)}")
        sample = sample[sample["id"].isin(flows)]
    df = vd.load_official_train()
    X = df.set_index("id").loc[sample["id"], vd.feature_order()]
    train = ncv.load_v3_train(ncv.load_protocol())
    if train["split_digest"] != protocol["data"]["split_digest"]:
        raise E0Error("v3 split digest differs from the protocol")
    models = {k: fm.load_final_model(*k) for k in variants}
    pins = protocol["explanations"]["treeshap_background"]
    pos = vx.background_positions(len(train["X"]))
    bg = {"official_ids_sha256": hashlib.sha256(",".join(map(str, train["ids"][pos])).encode()).hexdigest()}
    if bg["official_ids_sha256"] != pins["official_ids_sha256"]:
        raise E0Error("TreeSHAP background rows differ from the protocol pin")
    for key in variants:
        if key[0] in TREES:
            h = vx.array_sha256(vx.shap_background(train["X"], models[key].pipeline))
            if h != pins["matrix_sha256"][f"{key[0]}_{key[1]}"]:
                raise E0Error(f"TreeSHAP background matrix of {key} differs from the protocol pin")
            bg[f"{key[0]}_{key[1]}"] = h
    t0 = time.perf_counter()
    lattice = enumerate_lattice(sample, X, budget)
    det = detection(models, X, lattice)
    plan = plan_counts(det, variants)
    plan["candidates_total_unique"] = sum(len(v) for v in lattice.values())
    plan["flows"] = int(len(sample))
    plan["enumeration_seconds"] = round(time.perf_counter() - t0, 1)
    if plan_only:
        return {"plan": plan}

    header = header_for(protocol, sample, models, budget, flows, variants, train["split_digest"], bg)
    if resume:
        run_dir = Path(resume)
        stored = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))["header"]
        diff = sorted(k for k in header if json.dumps(stored.get(k), sort_keys=True, default=str) != json.dumps(header[k], sort_keys=True, default=str))
        if diff:
            raise E0Error(f"cannot resume {run_dir}: header differs on {diff}")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = Path(output_root) / f"e0_budget{budget}{'_smoke' if flows is not None else ''}_{stamp}"
        run_dir.mkdir(parents=True)
        _write(run_dir / "run_config.json", {"header": header, "plan": plan, "official_test_used": False,
                                             "rule_d": e0["rule_d"], "success_rule": protocol["success_criteria"]})
    cache = vx.ExplanationCache(run_dir / CACHE)
    counters = {"computed": 0}
    _write(run_dir / "status.json", {"state": "running", "plan": plan, "cached": len(cache.entries), "dropped_lines": cache.dropped_lines,
                                     "updated_at_utc": datetime.now(timezone.utc).isoformat()})
    train_X = train["X"]
    results = {}
    try:
        for key in variants:
            m, d = models[key], det[key]
            name = f"{key[0]}_{key[1]}"
            codec = vx.Codec(train_X, m.features)
            lime_ex = vx.build_lime_explainer(train_X, codec)
            st = vx.settings_hash(vx.lime_settings(len(m.features)))

            def explain_fn(model_key, flow, row, seed, m=m, codec=codec, lime_ex=lime_ex, st=st):
                k = cache.key("lime", m.sha256, m.feature_set, st, int(seed), vx.vector_key(row, m.features))
                if k not in cache.entries:
                    if max_new is not None and counters["computed"] >= max_new:
                        raise Interrupted(f"stopped after {counters['computed']} new explanations (test limit)")
                    counters["computed"] += 1
                return cache.get_or_compute(k, lambda: vx.lime_weights(lime_ex, codec, m.pipeline, row, int(seed)),
                                            {"explainer": "lime", "variant": name, "flow": int(flow), "seed": int(seed)})

            cands = [{"candidate_uid": c["candidate_uid"], "model_key": name, "sample_position": c["flow_id"],
                      "original_row": X.loc[c["flow_id"]], "candidate_row": c["candidate_row"],
                      **{k2: c[k2] for k2 in ("assignment", "n_edited_features", "physical_status", "physical_unverified", "primary_changes", "derived_changes")},
                      "p_attack_original": float(d["p_orig"][c["flow_id"]]), "p_attack_candidate": float(d["p_cand"][c["candidate_uid"]]),
                      "physically_valid_claimed": False} for c in d["candidates_evaluated"]]
            ev = ho.evaluate_candidates(cands, explain_fn, heldout_seeds=HELDOUT, part_b_seeds=PART_B, search_seed=SEARCH_SEED, key_fn=candidate_key)
            summ = pd.DataFrame(ev["summary"])
            if len(summ):
                summ["lime_success"] = summ["exceeds_heldout_both"].astype(bool) & (summ["physical_status"] == "consistent_with_checked_rules")
                flagged = [c for c in cands if c["candidate_uid"] in set(summ.loc[summ["exceeds_heldout_both"].astype(bool), "candidate_uid"])]
                conf = pd.DataFrame(ho.evaluate_candidates(flagged, explain_fn, heldout_seeds=CONFIRM, part_b_seeds=CONFIRM,
                                                           search_seed=SEARCH_SEED, key_fn=candidate_key)["summary"])
                summ["confirmation_evaluated"] = summ["candidate_uid"].isin({c["candidate_uid"] for c in flagged})
                passed = set(conf.loc[conf["exceeds_heldout_both"].astype(bool), "candidate_uid"]) if len(conf) else set()
                summ["confirmation_passed"] = summ["candidate_uid"].isin(passed)
                summ["confirmed_lime_success"] = summ["lime_success"] & summ["confirmation_passed"]
            if key[0] in TREES and len(summ):
                tex, kind = vx.build_tree_explainer(key[0], m.pipeline, vx.shap_background(train_X, m.pipeline))

                def shap_w(row, flow, m=m, tex=tex, kind=kind):
                    k = cache.key("treeshap", m.sha256, m.feature_set, kind, None, vx.vector_key(row, m.features))
                    if k not in cache.entries:
                        if max_new is not None and counters["computed"] >= max_new:
                            raise Interrupted(f"stopped after {counters['computed']} new explanations (test limit)")
                        counters["computed"] += 1
                    return cache.get_or_compute(k, lambda: vx.shap_weights(key[0], m.pipeline, tex, kind, row.to_frame().T.infer_objects(), m.features)[0],
                                                {"explainer": "treeshap", "variant": name, "flow": int(flow)})

                extra = []
                for c in cands:
                    f = c["sample_position"]
                    so, sc = shap_w(c["original_row"], f), shap_w(c["candidate_row"], f)
                    lo = {s: explain_fn(name, f, c["original_row"], s) for s in HELDOUT}
                    lc = {s: explain_fn(name, f, c["candidate_row"], s) for s in HELDOUT}
                    met = sh.shap_pair(so, sc)
                    ref = sh.d11_reference(so, lo, HELDOUT)
                    flags = sh.d11_flags(met, ref, True, c["physical_status"])
                    rq5 = vx.rq5_matched(lo, lc, so, sc)
                    extra.append({"candidate_uid": c["candidate_uid"], **{f"shap_{k2}": v for k2, v in met.items()},
                                  "d11_j_ref": ref["j_ref"], "d11_rho_ref": ref["rho_ref"], **flags,
                                  **{f"rq5_{k2}": v for k2, v in rq5.items() if k2 != "lime_seeds"}})
                summ = summ.merge(pd.DataFrame(extra), on="candidate_uid", how="left")
            meta = sample.set_index("id")[["attack_cat", "is_tcp", "short_flow", "tier"]]
            if len(summ):
                summ = summ.join(meta, on="sample_position")
            _write(run_dir / f"summary_{name}.csv", summ)
            _write(run_dir / f"noise_bands_{name}.csv", pd.DataFrame(ev["bands"]))
            _write(run_dir / f"per_seed_{name}.csv", pd.DataFrame(ev["per_seed"]))
            results[name] = {
                "flows_detected": len(d["detected_flows"]), "flows_not_detected_rule_d": d["flows_not_detected"],
                "candidates_evaluated": int(len(summ)), "candidates_not_detected": d["candidates_not_detected"],
                "by_status": {s: int((summ["physical_status"] == s).sum()) if len(summ) else 0 for s in ("consistent_with_checked_rules", "unverified_dependencies")},
                "exceeds_heldout_both": int(summ["exceeds_heldout_both"].sum()) if len(summ) else 0,
                "lime_success": int(summ["lime_success"].sum()) if len(summ) else 0,
                "confirmed_lime_success": int(summ["confirmed_lime_success"].sum()) if len(summ) else 0,
                "flows_with_confirmed_success": int(summ.loc[summ["confirmed_lime_success"], "sample_position"].nunique()) if len(summ) else 0,
                "d11_shap_success": int(summ["shap_success"].sum()) if "shap_success" in summ else None,
            }
            _write(run_dir / "status.json", {"state": "running", "variants_done": list(results), "computed_now": counters["computed"],
                                             "cached": len(cache.entries), "updated_at_utc": datetime.now(timezone.utc).isoformat()})
            log(f"{name}: {results[name]}")
    except Interrupted as exc:
        _write(run_dir / "status.json", {"state": "interrupted", "reason": str(exc), "variants_done": list(results),
                                         "computed_now": counters["computed"], "cached": len(cache.entries),
                                         "updated_at_utc": datetime.now(timezone.utc).isoformat()})
        raise
    after = {f"{k[0]}_{k[1]}": fm.file_sha256(fm.model_path(*k)) for k in variants}
    if after != header["model_sha256"]:
        raise E0Error("a model file changed during the run")
    status = {"state": "complete", "results": results, "computed_now": counters["computed"], "cached": len(cache.entries),
              "dropped_lines": cache.dropped_lines, "elapsed_seconds": round(time.perf_counter() - t0, 1),
              "model_sha256_after_run": after, "official_test_used": False, "updated_at_utc": datetime.now(timezone.utc).isoformat()}
    _write(run_dir / "status.json", status)
    return {"run_dir": str(run_dir), "status": status, "plan": plan}


def parse_variants(names) -> list[tuple[str, str]]:
    allv = PRIMARY + SECONDARY
    if not names:
        return allv
    lookup = {f"{m}_{fs}": (m, fs) for m, fs in allv}
    bad = [n for n in names if n not in lookup]
    if bad:
        raise SystemExit(f"unknown variants {bad}; choose from {sorted(lookup)}")
    return [lookup[n] for n in names]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", required=True, choices=["A", "B"])
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--flows", nargs="+", type=int, default=None, help="subset of frozen-sample ids (smoke tests only)")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--max-new-explanations", type=int, default=None, help="test limit: stop (resumably) after N new explanations")
    a = ap.parse_args()
    out = run(a.budget, parse_variants(a.variants), a.flows, a.resume, a.plan_only, a.max_new_explanations)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
