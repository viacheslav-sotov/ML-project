"""Protocol v3 final attack-flow sample (WS3): pool, eligibility, tiered stratified draw, frozen manifest.

Rule (protocol step6-v3-draft-1, attack_sample; approved Nov 17), implemented exactly as below, fixed before any run:
  pool         v3 VALIDATION side only; label 1; not in a label-conflict group; uniform category (stratum = attack_cat,
               i.e. not MixedAttack); the 10 v2 pilot flows excluded by official id
  eligibility  predicted attack (P >= 0.5) by all three PRIMARY full42 final models (RF, XGBoost, MLP);
               no_ttl39 and LR predictions are recorded, never used for selection
  tier 1       singleton groups (feature vector occurs once in the official training partition)
  tier 2       only for a category with fewer than 30 eligible tier-1 flows: one representative per non-singleton
               uniform group = the row with the lowest official id (all rows of a group share the vector and
               therefore every prediction); used only to fill up to 30
  draw         numpy.random.default_rng(2026), ONE generator for the whole sample; categories in sorted order;
               within a category: tier 1 first (all of it if <= 30, otherwise 30 drawn), then tier 2 fill;
               each draw stratified by flow type (tcp = proto 'tcp', other) with the allocation
               k_s = largest-remainder rounding of n * eligible_s / eligible, strata in the order (tcp, other);
               rng.choice(sorted ids of the stratum, size=k_s, replace=False)
  shortfall    recorded per category; never filled from another category or from the official test partition

Outputs (results/v3_prep/attack_sample/):
  pool.csv               every pool flow with tier, flags, all 8 model probabilities and eligibility
  sample_manifest.csv    the frozen sample (one row per flow)
  sample_manifest.json   rule, implementation details, model hashes, split digest, per-category counts, digest of the sample
  audit.json             pool and eligibility counts per category and tier, exclusions, checks

    python experiments/run_guarded.py experiments/unsw_v3_attack_sample.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import unsw_nested_cv as ncv  # noqa: E402
from src.unsw import v3_data as vd  # noqa: E402
from src.unsw import v3_final_models as fm  # noqa: E402

OUT = ROOT / "results" / "v3_prep" / "attack_sample"
SEED = 2026
PER_CATEGORY = 30
PRIMARY = ("random_forest", "xgboost", "mlp")
ALL_MODELS = [(m, fs) for fs in ("full42", "no_ttl39") for m in ("random_forest", "xgboost", "mlp", "logistic_regression")]
STRATA = ("tcp", "other")
SAMPLE_VERSION = "v3-sample-1"


PILOT_POSITIONS_V2 = [2941, 3040, 3206, 27094, 15137, 4407, 30107, 34186, 15741, 25740]


def pilot_official_ids(df: pd.DataFrame) -> list[int]:
    """Official ids (the 'id' column) of the 10 v2 pilot flows. The reference table stores 0-based ROW POSITIONS
    in the official training file (official_train_row_index), so they are looked up, not used as ids."""
    ref = pd.read_csv(ROOT / "results" / "reference" / "selected_flows.csv")
    rows = ref[ref["validation_position"].isin(PILOT_POSITIONS_V2)]["official_train_row_index"].astype(int)
    if len(rows) != len(PILOT_POSITIONS_V2):
        raise RuntimeError("reference table does not list all 10 v2 pilot flows")
    return sorted(int(i) for i in df["id"].to_numpy()[rows.to_numpy()])


def vector_sha256(row: pd.Series) -> str:
    """Exact identity of the 42 feature values (repr of floats, strings as is), independent of pandas hashing."""
    vals = [str(row[f]) if isinstance(row[f], str) else repr(float(row[f])) for f in vd.feature_order()]
    return hashlib.sha256("|".join(vals).encode("utf-8")).hexdigest()


def allocate(n: int, counts: dict[str, int]) -> dict[str, int]:
    """Largest-remainder allocation of n over strata in proportion to counts (ties: STRATA order)."""
    total = sum(counts.values())
    if n >= total:
        return dict(counts)
    exact = {s: n * counts[s] / total for s in STRATA}
    k = {s: int(np.floor(exact[s])) for s in STRATA}
    left = n - sum(k.values())
    for s in sorted(STRATA, key=lambda s: (-(exact[s] - k[s]), STRATA.index(s)))[:left]:
        k[s] += 1
    return {s: min(k[s], counts[s]) for s in STRATA}


def stratified_draw(rng: np.random.Generator, frame: pd.DataFrame, n: int) -> list[int]:
    """ids drawn from frame (columns id, stratum); all of them if n >= len(frame)."""
    counts = {s: int((frame["flow_stratum"] == s).sum()) for s in STRATA}
    alloc = allocate(n, counts)
    chosen = []
    for s in STRATA:
        ids = np.sort(frame.loc[frame["flow_stratum"] == s, "id"].to_numpy())
        if alloc[s] >= len(ids):
            chosen += ids.tolist()
        elif alloc[s] > 0:
            chosen += sorted(rng.choice(ids, size=alloc[s], replace=False).tolist())
    return chosen


def build_pool(df: pd.DataFrame, split: dict, pilot_ids: set[int]) -> tuple[pd.DataFrame, dict]:
    info = split["info"]
    va = np.sort(split["validation"])
    v = info.iloc[va].copy()
    excl = {"validation_rows": int(len(v)), "normal": int((v["label"] == 0).sum())}
    v = v[v["label"] == 1]
    excl["label_conflict"] = int(v["label_conflict"].sum())
    v = v[~v["label_conflict"]]
    excl["mixed_attack_category"] = int((v["stratum"] == "MixedAttack").sum())
    v = v[v["stratum"] != "MixedAttack"]
    if not (v["stratum"] == v["attack_cat"]).all():
        raise RuntimeError("uniform-category pool rows must have stratum == attack_cat")
    excl["v2_pilot_flows"] = int(v["id"].isin(pilot_ids).sum())
    v = v[~v["id"].isin(pilot_ids)]
    rows = v["row"].to_numpy()
    X = df.iloc[rows]
    pool = pd.DataFrame({
        "id": v["id"].astype(int).to_numpy(), "row": rows, "attack_cat": v["attack_cat"].to_numpy(),
        "group_id": v["group_id"].to_numpy(), "group_size": v["group_size"].astype(int).to_numpy(),
        "feature_hash64": v["hash"].astype(np.uint64).astype(str).to_numpy(),
        "proto": X["proto"].to_numpy(), "service": X["service"].to_numpy(), "state": X["state"].to_numpy(),
        "spkts": X["spkts"].to_numpy(), "dpkts": X["dpkts"].to_numpy(), "dur": X["dur"].to_numpy(),
    })
    pool["is_tcp"] = pool["proto"] == "tcp"
    pool["flow_stratum"] = np.where(pool["is_tcp"], "tcp", "other")
    pool["short_flow"] = (pool["spkts"] + pool["dpkts"]) <= 2
    pool["no_reply"] = pool["dpkts"] == 0
    pool["zero_duration"] = pool["dur"] == 0
    pool["duplicate_status"] = np.where(pool["group_size"] == 1, "singleton", "duplicate_group")
    pool["vector_sha256"] = [vector_sha256(r) for _, r in X.iterrows()]
    # every flow in a validation-side group is on the validation side (duplicate-disjoint split)
    return pool.reset_index(drop=True), excl


def add_predictions(pool: pd.DataFrame, df: pd.DataFrame, models: dict) -> pd.DataFrame:
    X = df.iloc[pool["row"].to_numpy()][vd.feature_order()]
    for (m, fs), model in models.items():
        p = model.predict_attack_proba(X)          # float64, full precision (exact CSV round trip)
        pool[f"p_{m}_{fs}"] = p
        pool[f"attack_{m}_{fs}"] = p >= fm.DECISION_THRESHOLD
    pool["eligible"] = np.logical_and.reduce([pool[f"attack_{m}_full42"] for m in PRIMARY])
    return pool


def assign_tiers(pool: pd.DataFrame) -> pd.DataFrame:
    pool["tier"] = np.where(pool["group_size"] == 1, 1, 2)
    rep = pool[pool["tier"] == 2].groupby("group_id")["id"].transform("min")
    pool["tier2_representative"] = False
    pool.loc[rep.index, "tier2_representative"] = pool.loc[rep.index, "id"] == rep
    return pool


def draw_sample(pool: pd.DataFrame, seed: int = SEED, per_category: int = PER_CATEGORY) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    chosen, per_cat = [], {}
    elig = pool[pool["eligible"]]
    for cat in sorted(pool["attack_cat"].unique()):
        t1 = elig[(elig["attack_cat"] == cat) & (elig["tier"] == 1)]
        t2 = elig[(elig["attack_cat"] == cat) & (elig["tier"] == 2) & elig["tier2_representative"]]
        ids1 = stratified_draw(rng, t1, per_category)
        ids2 = stratified_draw(rng, t2, per_category - len(ids1)) if len(ids1) < per_category and len(t2) else []
        chosen += [(i, 1) for i in ids1] + [(i, 2) for i in ids2]
        n = len(ids1) + len(ids2)
        per_cat[cat] = {"pool": int((pool["attack_cat"] == cat).sum()),
                        "eligible_tier1": int(len(t1)), "eligible_tier1_tcp": int(t1["is_tcp"].sum()),
                        "eligible_tier2_groups": int(len(t2)), "selected_tier1": len(ids1), "selected_tier2": len(ids2),
                        "selected": n, "shortfall": per_category - n}
    sel = pool.set_index("id").loc[[i for i, _ in chosen]].reset_index()
    sel.insert(0, "sample_index", np.arange(len(sel)))
    return sel, per_cat


def sample_digest(sample: pd.DataFrame) -> str:
    payload = "\n".join(f"{r.id}|{r.vector_sha256}|{r.attack_cat}|{r.tier}" for r in sample.itertuples())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "sample_manifest.json").exists():
        raise SystemExit(f"{OUT / 'sample_manifest.json'} exists: the sample is frozen; refusing to redraw")
    protocol = ncv.load_protocol()
    df = vd.load_official_train()
    s = protocol["data"]["split"]
    split = vd.disjoint_split(df, s["validation_fraction"], s["seed"], label_conflicts=protocol["data"]["label_conflict_rule"])
    audit = vd.audit(split, df)
    if audit["shared_feature_vectors"] or audit["overlapping_rows"]:
        raise RuntimeError("split is not duplicate-disjoint")
    digest = vd.split_digest(split)
    models = {key: fm.load_final_model(*key) for key in ALL_MODELS}
    pilot_ids = pilot_official_ids(df)
    pool, excl = build_pool(df, split, set(pilot_ids))
    pool = assign_tiers(add_predictions(pool, df, models))
    # checks: pool groups are entirely on the validation side, no pool vector on the train side
    info = split["info"]
    train_hashes = set(info["hash"].to_numpy()[split["train"]])
    assert not (set(pool["feature_hash64"].astype(np.uint64)) & train_hashes)
    sample, per_cat = draw_sample(pool)
    sample_cols = ["sample_index", "id", "attack_cat", "tier", "duplicate_status", "group_id", "group_size",
                   "feature_hash64", "vector_sha256", "proto", "service", "state", "is_tcp", "flow_stratum",
                   "short_flow", "no_reply", "zero_duration", "spkts", "dpkts", "dur"] + \
                  [c for c in pool.columns if c.startswith(("p_", "attack_"))]
    sample = sample[sample_cols]
    pool_cols = ["id", "attack_cat", "tier", "tier2_representative", "duplicate_status", "group_id", "group_size",
                 "feature_hash64", "vector_sha256", "proto", "is_tcp", "flow_stratum", "short_flow", "no_reply",
                 "eligible"] + [c for c in pool.columns if c.startswith(("p_", "attack_"))]
    pool[pool_cols].sort_values("id").to_csv(OUT / "pool.csv", index=False, lineterminator="\n")
    sample.to_csv(OUT / "sample_manifest.csv", index=False, lineterminator="\n")
    by_model = {f"{m}_{fs}": {"pool_detected": int(pool[f"attack_{m}_{fs}"].sum()), "sample_detected": int(sample[f"attack_{m}_{fs}"].sum())}
                for m, fs in ALL_MODELS}
    manifest = {
        "sample_version": SAMPLE_VERSION, "protocol_version": protocol["protocol_version"], "status": "frozen candidate sample (needs approval)",
        "rule": protocol["attack_sample"], "implementation": __doc__.split("Outputs")[0].strip(),
        "seed": SEED, "per_category": PER_CATEGORY, "decision_threshold": fm.DECISION_THRESHOLD,
        "inference": {"probabilities": "float64, full precision", "deterministic_settings": fm.DETERMINISTIC_INFERENCE},
        "split_digest": digest, "official_test_partition_opened": False,
        "models": [{"model": m.model_key, "feature_set": m.feature_set, "sha256": m.sha256, "file": m.file,
                    "used_for_eligibility": m.feature_set == "full42" and m.model_key in PRIMARY} for m in models.values()],
        "excluded_v2_pilot_ids": pilot_ids,
        "n_selected": int(len(sample)), "per_category": per_cat,
        "sample_sha256": sample_digest(sample), "sample_manifest_csv_sha256": fm.file_sha256(OUT / "sample_manifest.csv"),
        "code_sha256": fm.file_sha256(Path(__file__)),
    }
    (OUT / "sample_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")
    audit_out = {"split_audit": {k: audit[k] for k in ("rows_train", "rows_validation", "rows_excluded", "shared_feature_vectors", "overlapping_rows")},
                 "pool_exclusions": excl, "pool_size": int(len(pool)), "eligible": int(pool["eligible"].sum()),
                 "by_model_detection": by_model, "per_category": per_cat,
                 "sample_by_tier": {int(k): int(v) for k, v in sample["tier"].value_counts().items()},
                 "sample_by_flow": {"tcp": int(sample["is_tcp"].sum()), "other": int((~sample["is_tcp"]).sum()),
                                    "short_flow": int(sample["short_flow"].sum()), "no_reply": int(sample["no_reply"].sum())},
                 "checks": {"pool_vectors_on_train_side": 0, "sample_ids_unique": bool(sample["id"].is_unique),
                            "sample_groups_unique": bool(sample["group_id"].is_unique),
                            "sample_all_eligible": bool(np.logical_and.reduce([sample[f"attack_{m}_full42"] for m in PRIMARY]).all()),
                            "pilot_ids_in_sample": int(sample["id"].isin(pilot_ids).sum())}}
    (OUT / "audit.json").write_text(json.dumps(audit_out, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({"n_selected": len(sample), "per_category": per_cat, "sample_sha256": manifest["sample_sha256"]}, indent=1))


if __name__ == "__main__":
    main()
