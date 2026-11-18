"""Protocol v3 readiness: explainability compatibility of the eight final models and a minimal smoke test.

No attack search is run. Checks (results/v3_prep/compat_smoke/):
  models       all 8 load only with pinned + recorded SHA-256, recorded best_params, protocol preprocessing/order
  predictions  recomputed P(attack) on every frozen sample flow equals the value pinned in the sample manifest
  TTL          no_ttl39 predictions are bit-identical when sttl/dttl/ct_state_ttl are replaced (shuffled, 0, 255);
               full42 predictions are reported for the same perturbation (sensitivity check of the test)
  LIME         every model: weights over exactly its own features (42 or 39, no TTL in no_ttl39), the prediction
               function reproduces the model's probability exactly, same seed -> identical weights
  TreeSHAP     RF and XGBoost, both sets: additivity per flow, features = model features, exact-zero counts
  smoke        one frozen sample flow per tree model/set: the first valid physical_v1 Budget-A lattice candidate
               (lattice order) that the model still detects; LIME seeds 22/33 + TreeSHAP for original and candidate;
               RQ5 matched metrics; physical report
  resume       the smoke is computed twice through the identity-keyed cache; the second pass must be all hits
               with identical results

    python experiments/run_guarded.py experiments/unsw_v3_compat_smoke.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import unsw_nested_cv as ncv  # noqa: E402
from src.unsw import physical_repair as pr  # noqa: E402
from src.unsw import v3_data as vd  # noqa: E402
from src.unsw import v3_explain as vx  # noqa: E402
from src.unsw import v3_final_models as fm  # noqa: E402

OUT = ROOT / "results" / "v3_prep" / "compat_smoke"
SAMPLE = ROOT / "results" / "v3_prep" / "attack_sample" / "sample_manifest.csv"
ALL = [(m, fs) for fs in ("full42", "no_ttl39") for m in ("random_forest", "xgboost", "mlp", "logistic_regression")]
TREES = [(m, fs) for fs in ("full42", "no_ttl39") for m in ("random_forest", "xgboost")]
SMOKE_SEEDS = (22, 33)
COMPAT_SEED = 11
PRED_TOL = 1e-12


def flow_rows(df: pd.DataFrame, ids) -> pd.DataFrame:
    by_id = df.set_index("id")
    X = by_id.loc[list(ids), vd.feature_order()].copy()
    X.index.name = "id"
    return X


def ttl_check(models, X: pd.DataFrame) -> dict:
    rng = np.random.default_rng(7)
    variants = {"shuffled": X.assign(**{f: rng.permutation(X[f].to_numpy()) for f in vd.TTL_FEATURES}),
                "all_zero": X.assign(**{f: 0 for f in vd.TTL_FEATURES}),
                "all_255": X.assign(**{f: 255 for f in vd.TTL_FEATURES})}
    out = {}
    for key, m in models.items():
        base = m.predict_attack_proba(X)
        diffs = {k: float(np.max(np.abs(m.predict_attack_proba(v) - base))) for k, v in variants.items()}
        out[f"{key[0]}_{key[1]}"] = {"max_abs_change": diffs, "invariant": all(d == 0.0 for d in diffs.values())}
    return out


def smoke_pass(cache: vx.ExplanationCache, models, explainers, shap_ex, flows: dict, X_rows: pd.DataFrame) -> list[dict]:
    rows = []
    for key in TREES:
        m = models[key]
        fid = flows[key]
        orig = X_rows.loc[fid]
        cands = pr.valid_lattice_candidates(orig, "A")
        chosen = None
        for assignment, cand, verdict in cands:
            p = float(m.predict_attack_proba(cand.to_frame().T)[0])
            if p >= fm.DECISION_THRESHOLD:
                chosen = (assignment, cand, verdict, p)
                break
        if chosen is None:
            rows.append({"model": key[0], "feature_set": key[1], "id": int(fid), "valid_candidates": len(cands), "note": "no detected candidate"})
            continue
        assignment, cand, verdict, p_cand = chosen
        codec, lime_ex = explainers[key]
        st = vx.settings_hash(vx.lime_settings(len(m.features)))

        def lime(row, seed):
            k = cache.key("lime", m.sha256, m.feature_set, st, seed, vx.vector_key(row, m.features))
            return cache.get_or_compute(k, lambda: vx.lime_weights(lime_ex, codec, m.pipeline, row, seed),
                                        {"explainer": "lime", "model": key[0], "feature_set": key[1], "seed": seed, "id": int(fid)})

        tex, kind = shap_ex[key]

        def shap_w(row):
            k = cache.key("treeshap", m.sha256, m.feature_set, kind, None, vx.vector_key(row, m.features))
            return cache.get_or_compute(k, lambda: vx.shap_weights(key[0], m.pipeline, tex, kind, row.to_frame().T.infer_objects(), m.features)[0],
                                        {"explainer": "treeshap", "model": key[0], "feature_set": key[1], "id": int(fid)})

        lo = {s: lime(orig, s) for s in SMOKE_SEEDS}
        lc = {s: lime(cand, s) for s in SMOKE_SEEDS}
        metrics = vx.rq5_matched(lo, lc, shap_w(orig), shap_w(cand))
        rows.append({"model": key[0], "feature_set": key[1], "id": int(fid), "valid_candidates": len(cands),
                     "edit": json.dumps(assignment), "p_orig": float(m.predict_attack_proba(orig.to_frame().T)[0]), "p_cand": p_cand,
                     "physical_status": verdict.get("physical_status") or verdict.get("status"), **metrics})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    report: dict = {"official_test_partition_opened": False, "problems": []}
    protocol = ncv.load_protocol()
    train = ncv.load_v3_train(protocol)
    X_train = train["X"]
    df = vd.load_official_train()
    sample = pd.read_csv(SAMPLE, float_precision="round_trip")   # exact float64 round trip
    manifest = json.loads((SAMPLE.parent / "sample_manifest.json").read_text(encoding="utf-8"))
    if fm.file_sha256(SAMPLE) != manifest["sample_manifest_csv_sha256"]:
        report["problems"].append("sample manifest CSV changed since it was frozen")
    report["split_digest_matches_sample"] = train["split_digest"] == manifest["split_digest"]

    # ---- models
    models = {key: fm.load_final_model(*key) for key in ALL}
    report["models"] = {f"{m}_{fs}": {"sha256": x.sha256, "n_features": len(x.features), "file": x.file} for (m, fs), x in models.items()}
    report["deterministic_inference"] = fm.DETERMINISTIC_INFERENCE

    # ---- prediction preservation
    # (a) exact: re-scoring the whole pool in the same batch as the manifest must reproduce every stored probability
    #     bit for bit (determinism of loading + inference);
    # (b) across batch sizes (the 263 sample flows alone): MLP/LR go through BLAS matrix products whose summation
    #     order depends on the batch, so a difference of a few ulp is allowed (PRED_TOL) and the attack decision
    #     must be identical for every flow.
    pool = pd.read_csv(SAMPLE.parent / "pool.csv", float_precision="round_trip")
    Xpool = flow_rows(df, pool["id"])
    Xs = flow_rows(df, sample["id"])
    pred = {}
    for (m, fs), x in models.items():
        col = f"p_{m}_{fs}"
        gap_pool = float(np.max(np.abs(x.predict_attack_proba(Xpool) - pool[col].to_numpy(dtype=np.float64))))
        p = x.predict_attack_proba(Xs)
        gap = float(np.max(np.abs(p - sample[col].to_numpy(dtype=np.float64))))
        same_decision = bool(np.array_equal(p >= fm.DECISION_THRESHOLD, sample[f"attack_{m}_{fs}"].to_numpy(dtype=bool)))
        pred[f"{m}_{fs}"] = {"same_batch_max_abs_gap": gap_pool, "sample_batch_max_abs_gap": gap,
                             "same_decision_all_sample_flows": same_decision, "detected": int((p >= 0.5).sum()), "of": int(len(p))}
        if gap_pool != 0.0:
            report["problems"].append(f"{m}_{fs}: same-batch predictions differ from the pool manifest by {gap_pool}")
        if gap > PRED_TOL or not same_decision:
            report["problems"].append(f"{m}_{fs}: sample predictions differ from the manifest by {gap} (decision same: {same_decision})")
    report["prediction_preservation"] = {"tolerance_across_batch_sizes": PRED_TOL, "per_model": pred}

    # ---- TTL invariance (sample flows + 500 other validation rows)
    rng = np.random.default_rng(3)
    s = protocol["data"]["split"]
    split = vd.disjoint_split(df, s["validation_fraction"], s["seed"], label_conflicts=protocol["data"]["label_conflict_rule"])
    extra = df.iloc[rng.choice(np.sort(split["validation"]), 500, replace=False)][vd.feature_order()]
    report["ttl"] = ttl_check(models, pd.concat([Xs.reset_index(drop=True), extra.reset_index(drop=True)]))
    for k, v in report["ttl"].items():
        if k.endswith("no_ttl39") and not v["invariant"]:
            report["problems"].append(f"{k}: TTL features change predictions")

    # ---- LIME compatibility (all 8)
    flow = sample[sample["is_tcp"]].iloc[0]["id"] if sample["is_tcp"].any() else sample.iloc[0]["id"]
    row = Xs.loc[flow]
    explainers, lime_rep = {}, {}
    for key, x in models.items():
        codec = vx.Codec(X_train, x.features)
        ex = vx.build_lime_explainer(X_train, codec)
        explainers[key] = (codec, ex)
        p_direct = x.pipeline.predict_proba(row.to_frame().T.infer_objects())[0, 1]
        p_fn = vx.make_predict_fn(x.pipeline, codec, row)(codec.encode(row.to_frame().T))[0, 1]
        w1 = vx.lime_weights(ex, codec, x.pipeline, row, COMPAT_SEED)
        w2 = vx.lime_weights(ex, codec, x.pipeline, row, COMPAT_SEED)
        r = {"n_weights": len(w1), "ttl_in_explanation": sorted(set(w1) & set(vd.TTL_FEATURES)),
             "predict_fn_gap": float(abs(p_direct - p_fn)), "same_seed_identical": w1 == w2,
             "exact_zero_weights": vx.exact_zero_count(w1), "top5": vx.ranked(w1)[:5]}
        lime_rep[f"{key[0]}_{key[1]}"] = r
        if r["n_weights"] != len(x.features) or (key[1] == "no_ttl39" and r["ttl_in_explanation"]) or r["predict_fn_gap"] != 0.0 or not r["same_seed_identical"]:
            report["problems"].append(f"LIME {key}: {r}")
    report["lime"] = {"flow_id": int(flow), "seed": COMPAT_SEED, "settings": {fs: vx.lime_settings(n) for fs, n in (("full42", 42), ("no_ttl39", 39))}, "per_model": lime_rep}

    # ---- TreeSHAP compatibility (RF, XGB x 2 sets) on 10 sample flows
    shap_ex, shap_rep = {}, {}
    X10 = Xs.iloc[:10]
    for key in TREES:
        x = models[key]
        tex, kind = vx.build_tree_explainer(key[0], x.pipeline, vx.shap_background(X_train, x.pipeline))
        shap_ex[key] = (tex, kind)
        ws = vx.shap_weights(key[0], x.pipeline, tex, kind, X10, x.features)
        shap_rep[f"{key[0]}_{key[1]}"] = {"output": kind, "flows": len(ws), "n_features": len(ws[0]),
                                          "ttl_in_explanation": sorted(set(ws[0]) & set(vd.TTL_FEATURES)),
                                          "exact_zero_features_max": max(vx.exact_zero_count(w) for w in ws),
                                          "additivity": f"checked per flow (tolerance {vx.ADDITIVITY_TOL})"}
    report["treeshap"] = {"background": f"{vx.SHAP_BACKGROUND_SIZE} v3 training rows, default_rng({vx.SHAP_BACKGROUND_SEED})", "per_model": shap_rep}

    # ---- smoke + resume
    cache_path = OUT / "smoke_cache.jsonl"
    if cache_path.exists():
        cache_path.unlink()
    flows = {key: int(flow) for key in TREES}
    c1 = vx.ExplanationCache(cache_path)
    first = smoke_pass(c1, models, explainers, shap_ex, flows, Xs)
    c2 = vx.ExplanationCache(cache_path)
    second = smoke_pass(c2, models, explainers, shap_ex, flows, Xs)
    same = pd.DataFrame(first).fillna("").equals(pd.DataFrame(second).fillna(""))
    report["resume"] = {"first_pass": {"hits": c1.hits, "misses": c1.misses}, "second_pass": {"hits": c2.hits, "misses": c2.misses},
                        "identical_results": bool(same)}
    if c2.misses or not same:
        report["problems"].append("cache resume did not reproduce the smoke results from the cache alone")
    pd.DataFrame(first).to_csv(OUT / "rq5_smoke.csv", index=False, lineterminator="\n")
    report["smoke"] = first
    report["seconds"] = round(time.perf_counter() - t0, 1)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8", newline="\n")
    print(json.dumps({"problems": report["problems"], "seconds": report["seconds"], "resume": report["resume"]}, indent=1))
    sys.exit(1 if report["problems"] else 0)


if __name__ == "__main__":
    main()
