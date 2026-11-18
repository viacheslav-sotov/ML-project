"""Pre-flight audit before the first protocol-v3 experiment (E0). Read-only; run under the guard:

    python experiments/run_guarded.py experiments/unsw_v3_preflight.py

  A  sample identity: the 10 v2 pilot flows are excluded by official id (not row position); the discarded draw is
     rejected; digest and CSV hash equal the protocol pins
  B  sample soundness: unique by feature vector/group/id; no vector on the v3 training side; categories, TCP and
     short-flow counts; vector hashes and all 8 model predictions recomputed from the data
  F  isolation guard active with identity lock; 8 model hashes; protocol hashes; git: branch is a fast-forward of the
     remote branch, clean tree, no model files, raw data, local-only folders or large files in the outgoing commits
  pins  TreeSHAP background rows (official ids) and transformed background matrices, for the protocol

Writes results/v3_prep/preflight/preflight.json; exit code 1 if any check fails.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import unsw_nested_cv as ncv  # noqa: E402
from experiments import unsw_v3_attack_sample as smp  # noqa: E402
from src.unsw import official_test_lock as lock  # noqa: E402
from src.unsw import v3_data as vd  # noqa: E402
from src.unsw import v3_explain as vx  # noqa: E402
from src.unsw import v3_final_models as fm  # noqa: E402
from src.unsw import v3_sample as vs  # noqa: E402

OUT = ROOT / "results" / "v3_prep" / "preflight"
SAMPLE_DIR = ROOT / "results" / "v3_prep" / "attack_sample"
REMOTE = "origin/feature/step6-v2"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checks: dict[str, bool] = {}
    info: dict = {"official_test_partition_opened": False}
    protocol = json.loads((ROOT / "experiments" / "step6_protocol_v3.json").read_text(encoding="utf-8"))
    pins = protocol["attack_sample"]

    # ---------------- A
    df = vd.load_official_train()
    csv = SAMPLE_DIR / "sample_manifest.csv"
    sample = pd.read_csv(csv, float_precision="round_trip")
    digest = vs.sample_digest(sample)
    ref = pd.read_csv(ROOT / "results" / "reference" / "selected_flows.csv")
    pilot_rows = ref[ref["validation_position"].isin(smp.PILOT_POSITIONS_V2)]["official_train_row_index"].astype(int).to_numpy()
    pilot_ids = sorted(int(i) for i in df["id"].to_numpy()[pilot_rows])
    manifest = json.loads((SAMPLE_DIR / "sample_manifest.json").read_text(encoding="utf-8"))
    checks["A1_pilot_ids_are_official_ids_of_the_10_pilot_rows"] = manifest["excluded_v2_pilot_ids"] == pilot_ids and len(pilot_ids) == 10
    checks["A2_positions_differ_from_ids"] = sorted(pilot_rows.tolist()) != pilot_ids
    checks["A3_no_pilot_flow_in_sample"] = not (set(sample["id"]) & set(pilot_ids))
    checks["A4_not_a_discarded_draw"] = digest not in vs.DISCARDED_SAMPLE_DIGESTS
    checks["A5_digest_and_csv_hash_equal_protocol_pins"] = (digest == pins["sample_sha256"]
                                                           and fm.file_sha256(csv) == pins["sample_manifest_csv_sha256"])
    info["A"] = {"pilot_official_ids": pilot_ids, "pilot_row_positions": sorted(int(r) for r in pilot_rows),
                 "sample_sha256": digest, "csv_sha256": fm.file_sha256(csv)}

    # ---------------- B
    p = ncv.load_protocol()
    split = vd.disjoint_split(df, p["data"]["split"]["validation_fraction"], p["data"]["split"]["seed"], label_conflicts=p["data"]["label_conflict_rule"])
    info_split = split["info"].set_index("id")
    train_hashes = set(split["info"]["hash"].to_numpy()[split["train"]])
    val_ids = set(split["info"]["id"].to_numpy()[split["validation"]])
    rows = df.set_index("id").loc[sample["id"]]
    vec = [smp.vector_sha256(r) for _, r in rows.iterrows()]
    checks["B1_unique_by_id_group_vector"] = bool(sample["id"].is_unique and sample["group_id"].is_unique and len(set(vec)) == len(vec))
    checks["B2_vector_hashes_recomputed_equal"] = vec == sample["vector_sha256"].tolist()
    checks["B3_all_on_validation_side"] = set(sample["id"]) <= val_ids
    checks["B4_no_vector_on_training_side"] = not (set(info_split.loc[sample["id"], "hash"]) & train_hashes)
    checks["B5_all_singletons_uniform_category"] = bool((info_split.loc[sample["id"], "group_size"] == 1).all()
                                                       and (info_split.loc[sample["id"], "stratum"].to_numpy() == sample["attack_cat"].to_numpy()).all())
    recomputed_tcp = (rows["proto"] == "tcp").to_numpy()
    recomputed_short = ((rows["spkts"] + rows["dpkts"]) <= 2).to_numpy()
    checks["B6_flow_flags_recomputed_equal"] = bool((recomputed_tcp == sample["is_tcp"].to_numpy()).all() and (recomputed_short == sample["short_flow"].to_numpy()).all())
    pred = {}
    ok_pred = True
    X = rows[vd.feature_order()]
    for (m, fs) in fm.STAGE_RUNS:
        model = fm.load_final_model(m, fs)
        prob = model.predict_attack_proba(X)
        gap = float(np.max(np.abs(prob - sample[f"p_{m}_{fs}"].to_numpy(dtype=np.float64))))
        same = bool(np.array_equal(prob >= 0.5, sample[f"attack_{m}_{fs}"].to_numpy(dtype=bool)))
        pred[f"{m}_{fs}"] = {"max_abs_gap": gap, "same_decisions": same, "detected": int((prob >= 0.5).sum()),
                             "not_detected_ids": sorted(int(i) for i in sample.loc[prob < 0.5, "id"])}
        ok_pred &= gap <= 1e-12 and same
    checks["B7_predictions_recomputed_equal"] = ok_pred
    checks["B8_eligible_for_all_primary_full42"] = all(pred[f"{m}_full42"]["detected"] == len(sample) for m in ("random_forest", "xgboost", "mlp"))
    info["B"] = {"categories": sample["attack_cat"].value_counts().sort_index().to_dict(),
                 "tcp": int(sample["is_tcp"].sum()), "other": int((~sample["is_tcp"]).sum()), "short_flow": int(sample["short_flow"].sum()),
                 "no_reply": int(sample["no_reply"].sum()), "tiers": sample["tier"].value_counts().to_dict(), "predictions": pred}

    # ---------------- F
    checks["F1_guard_active_with_identity_lock"] = lock.is_installed() and lock.identity_locked()
    checks["F2_model_hashes"] = all(fm.file_sha256(fm.model_path(*k)) == h for k, h in fm.PINNED_SHA256.items())
    base = protocol["base"]
    checks["F3_protocol_hashes"] = (fm.file_sha256(ROOT / base["training_protocol"]["file"]) == base["training_protocol"]["sha256"]
                                   and all(fm.file_sha256(ROOT / f) == h for f, h in base["prior_protocols_unchanged"].items()))
    head, remote = git("rev-parse", "HEAD"), git("rev-parse", REMOTE)
    ff = subprocess.run(["git", "merge-base", "--is-ancestor", remote, "HEAD"], cwd=ROOT).returncode == 0
    checks["F4_fast_forward_of_remote_branch"] = ff
    checks["F5_clean_working_tree"] = git("status", "--porcelain", "--untracked-files=all") == ""
    outgoing = [ln for ln in git("diff", "--name-only", f"{remote}..HEAD").splitlines() if ln]
    bad_names = [f for f in outgoing if f.endswith((".joblib", ".pkl", ".pt")) or f.startswith(("data/raw/", "Claude outputs/"))
                 or "PROJECT_HANDOFF" in f or f.endswith("explanations_cache.jsonl") and "v3_prep" not in f or f.endswith("heldout_cache.jsonl")]
    sizes = {f: (ROOT / f).stat().st_size for f in outgoing if (ROOT / f).exists()}
    large = {f: s for f, s in sizes.items() if s > 20 * 1024 * 1024}
    tracked_models = [f for f in git("ls-files").splitlines() if f.endswith(".joblib") or (f.startswith("data/raw/") and f.endswith(".csv"))]
    pre_existing = [f for f in tracked_models
                    if subprocess.run(["git", "cat-file", "-e", f"{REMOTE}:{f}"], cwd=ROOT).returncode == 0]
    new_tracked = sorted(set(tracked_models) - set(pre_existing))
    checks["F6_no_models_raw_data_or_local_files_outgoing"] = not bad_names and not large and not new_tracked
    info["F"] = {"head": head, "remote": remote, "outgoing_commits": git("log", "--format=%h %an | %s", f"{remote}..HEAD").splitlines(),
                 "outgoing_files": len(outgoing), "largest_outgoing": sorted(sizes.items(), key=lambda x: -x[1])[:5],
                 "flagged": bad_names + list(large) + new_tracked,
                 "pre_existing_tracked_model_files_on_remote": pre_existing,
                 "ai_attribution_in_outgoing_messages": [l for l in git("log", "--format=%B", f"{remote}..HEAD").splitlines()
                                                         if any(w in l for w in ("Co-Authored", "Claude", "Generated with", "AI-assisted"))]}
    checks["F7_no_ai_attribution_in_outgoing_commits"] = not info["F"]["ai_attribution_in_outgoing_messages"]

    # ---------------- background pins
    train = ncv.load_v3_train(p)
    pos = vx.background_positions(len(train["X"]))
    pins_bg = {"rows": int(len(pos)), "source": "v3 training side, sorted official order, numpy default_rng(42) choice without replacement",
               "official_ids_sha256": hashlib.sha256(",".join(map(str, train["ids"][pos])).encode()).hexdigest(),
               "official_ids": [int(i) for i in train["ids"][pos]], "matrix_sha256": {}}
    for m in ("random_forest", "xgboost"):
        for fs in ("full42", "no_ttl39"):
            pins_bg["matrix_sha256"][f"{m}_{fs}"] = vx.array_sha256(vx.shap_background(train["X"], fm.load_final_model(m, fs).pipeline))
    info["treeshap_background_pins"] = pins_bg

    out = {"checks": checks, "all_passed": all(checks.values()), **info}
    (OUT / "preflight.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8", newline="\n")
    print(json.dumps({"checks": checks, "all_passed": out["all_passed"]}, indent=1))
    sys.exit(0 if out["all_passed"] else 1)


if __name__ == "__main__":
    main()
