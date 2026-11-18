"""Protocol v3 readiness numbers (no attack search, no explanation of any candidate):

  lattice census  for every frozen sample flow and Budget A/B: valid physical_v1 lattice candidates, their physical
                  status, and how many each primary model (both feature sets) still detects as attack
  timing          measured seconds per LIME explanation (5,000 samples) for all 8 models and per TreeSHAP
                  explanation for RF/XGBoost, on 3 sample flows x 2 seeds (originals only)

Outputs results/v3_prep/readiness/{lattice_census.csv, readiness.json}.

    python experiments/run_guarded.py experiments/unsw_v3_readiness.py
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

OUT = ROOT / "results" / "v3_prep" / "readiness"
SAMPLE = ROOT / "results" / "v3_prep" / "attack_sample" / "sample_manifest.csv"
PRIMARY = [(m, fs) for fs in ("full42", "no_ttl39") for m in ("random_forest", "xgboost", "mlp")]
ALL = PRIMARY + [("logistic_regression", "full42"), ("logistic_regression", "no_ttl39")]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = vd.load_official_train()
    sample = pd.read_csv(SAMPLE, float_precision="round_trip")
    X = df.set_index("id").loc[sample["id"], vd.feature_order()]
    models = {k: fm.load_final_model(*k) for k in ALL}
    rows = []
    t0 = time.perf_counter()
    for (fid, cat, tcp), (_, orig) in zip(sample[["id", "attack_cat", "is_tcp"]].itertuples(index=False), X.iterrows()):
        for budget in ("A", "B"):
            cands = pr.valid_lattice_candidates(orig, budget)
            rec = {"id": int(fid), "attack_cat": cat, "is_tcp": bool(tcp), "budget": budget, "valid_candidates": len(cands)}
            status = pd.Series([v["status"] for _, _, v in cands], dtype=object)
            for s in ("consistent_with_checked_rules", "unverified_dependencies", "violates_checked_rule"):
                rec[s] = int((status == s).sum())
            if cands:
                C = pd.DataFrame([c for _, c, _ in cands])[vd.feature_order()].infer_objects()
                for (m, fs) in PRIMARY:
                    rec[f"detected_{m}_{fs}"] = int((models[(m, fs)].predict_attack_proba(C) >= fm.DECISION_THRESHOLD).sum())
            else:
                for (m, fs) in PRIMARY:
                    rec[f"detected_{m}_{fs}"] = 0
            rows.append(rec)
    census = pd.DataFrame(rows)
    census.to_csv(OUT / "lattice_census.csv", index=False, lineterminator="\n")
    census_seconds = time.perf_counter() - t0

    train = ncv.load_v3_train(ncv.load_protocol())["X"]
    flows = X.iloc[:3]
    timing = {}
    for key, x in models.items():
        codec = vx.Codec(train, x.features)
        ex = vx.build_lime_explainer(train, codec)
        secs = []
        for _, row in flows.iterrows():
            for seed in (22, 33):
                t = time.perf_counter()
                vx.lime_weights(ex, codec, x.pipeline, row, seed)
                secs.append(time.perf_counter() - t)
        timing[f"lime_{key[0]}_{key[1]}"] = {"median_s": round(float(np.median(secs)), 3), "max_s": round(float(np.max(secs)), 3), "n": len(secs)}
        if key[0] in ("random_forest", "xgboost"):
            tex, kind = vx.build_tree_explainer(key[0], x.pipeline, vx.shap_background(train, x.pipeline))
            secs = []
            for i in range(len(flows)):
                t = time.perf_counter()
                vx.shap_weights(key[0], x.pipeline, tex, kind, flows.iloc[[i]], x.features)
                secs.append(time.perf_counter() - t)
            timing[f"treeshap_{key[0]}_{key[1]}"] = {"median_s": round(float(np.median(secs)), 3), "n": len(secs)}

    summ = {}
    for budget in ("A", "B"):
        c = census[census["budget"] == budget]
        summ[budget] = {"flows": int(len(c)), "flows_with_candidates": int((c["valid_candidates"] > 0).sum()),
                        "valid_candidates_total": int(c["valid_candidates"].sum()),
                        "valid_candidates_median_per_flow": float(c["valid_candidates"].median()),
                        "valid_candidates_max_per_flow": int(c["valid_candidates"].max()),
                        "consistent_with_checked_rules": int(c["consistent_with_checked_rules"].sum()),
                        "unverified_dependencies": int(c["unverified_dependencies"].sum()),
                        **{f"detected_{m}_{fs}": int(c[f"detected_{m}_{fs}"].sum()) for m, fs in PRIMARY}}
    out = {"official_test_partition_opened": False, "census_seconds": round(census_seconds, 1), "lattice": summ, "timing": timing,
           "timing_note": "single process, laptop on AC power, RF inference n_jobs=1 (deterministic), other work idle"}
    (OUT / "readiness.json").write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
