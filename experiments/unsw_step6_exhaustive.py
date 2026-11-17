"""Step 6 v2: exhaustive LIME evaluation of the physical_v1 lattice (protocol step6-v2, "exhaustive_evaluation").

Every valid physical_v1 lattice candidate of every protocol sample and perturbation budget that a model
still predicts as attack is explained with LIME under the search seed (11) and the held-out seeds
(22/33/44/55), and compared with the original under the same seed (held-out rule of
experiments/unsw_targeted_heldout.py, reused unchanged).

Scope: definitive only for the enumerated physical_v1 lattice, not for every possible network modification.

Deduplication
    A candidate is identified by (sample, values of the attackable features); derived features are a
    deterministic repair of these. Budget A's lattice is part of budget B's, so the same candidate can
    appear under both budgets: it is explained once per (model, seed). Every identity
    (model, sample, budget, policy, candidate) is kept in identities.csv.

Resume / checkpoint
    Every LIME explanation is appended to explanations_cache.jsonl (flushed and fsynced) the moment it is
    computed. A re-run with --resume <run dir> loads the cache and only computes what is missing; a
    truncated last line (crash during a write) is dropped. The cache header records the protocol, model,
    LIME-configuration and code hashes; a mismatch refuses the resume.

Reporting
    Candidates consistent_with_checked_rules are reported separately from candidates with
    unverified_dependencies (packet-timing dependencies of spkts). No candidate is called physically valid.

The official test set is removed before any code runs (runner.guarded_split).

    python experiments/unsw_step6_exhaustive.py --plan-only          # exact counts, no LIME
    python experiments/unsw_step6_exhaustive.py                      # run (new folder under results/step6_v2_runs)
    python experiments/unsw_step6_exhaustive.py --resume <run dir>   # continue an interrupted run
"""

from __future__ import annotations

import argparse
import hashlib
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

from experiments import unsw_step6_manifest as mf  # noqa: E402
from experiments import unsw_step6_runner as runner  # noqa: E402
from experiments import unsw_targeted_heldout as ho  # noqa: E402
from experiments import unsw_targeted_spoofing as ts  # noqa: E402
from src.unsw import physical_repair as pr  # noqa: E402

EXHAUSTIVE_VERSION = 1
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "step6_v2_runs"
CACHE_NAME = "explanations_cache.jsonl"
STATUS_GROUPS = ("consistent_with_checked_rules", "unverified_dependencies")


class ExhaustiveError(RuntimeError):
    pass


class Interrupted(RuntimeError):
    """Raised by the explanation limit used in tests to simulate a crash."""


def candidate_key(row: pd.Series) -> tuple[float, ...]:
    return ts._row_key(row)


def key_sha(key: tuple[float, ...]) -> str:
    return hashlib.sha256(json.dumps(list(key)).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------
# Enumeration and deduplication
# ---------------------------------------------------------------------------------------------
def enumerate_physical(X_val: pd.DataFrame, samples, budgets) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[int, pd.Series]]:
    """identities (one row per sample x budget x lattice candidate), unique valid candidates, originals."""
    identities, unique, originals = [], {}, {}
    for sample in samples:
        original = X_val.iloc[[int(sample)]].iloc[0].copy()
        originals[int(sample)] = original
        for budget in budgets:
            valid = {candidate_key(c): (a, c, v) for a, c, v in pr.valid_lattice_candidates(original, budget)}
            for assignment, candidate, error in pr.enumerate_lattice(original, budget):
                row = {"sample_position": int(sample), "budget": budget, "policy": pr.POLICY_PHYSICAL,
                       "assignment": json.dumps(assignment, sort_keys=True)}
                if candidate is None:
                    identities.append({**row, "candidate_uid": None, "valid": False, "status": "repair_failed", "violations": error})
                    continue
                key = candidate_key(candidate)
                uid = f"s{int(sample)}_{key_sha(key)}"
                if key not in valid:
                    verdict = pr.check_candidate(original, candidate, budget)
                    identities.append({**row, "candidate_uid": uid, "valid": False, "status": verdict["status"],
                                       "violations": "; ".join(verdict["violations"]) or "validator"})
                    continue
                _, _, verdict = valid[key]
                identities.append({**row, "candidate_uid": uid, "valid": True, "status": verdict["status"], "violations": ""})
                if uid not in unique:
                    unique[uid] = {
                        "candidate_uid": uid, "sample_position": int(sample), "candidate_row": candidate,
                        "assignment": json.dumps(assignment, sort_keys=True),
                        "physical_status": verdict["status"], "physical_unverified": "; ".join(verdict["unverified"]),
                        "primary_changes": json.dumps(verdict["primary_changes"], sort_keys=True),
                        "derived_changes": json.dumps(verdict["derived_changes"], sort_keys=True),
                        "n_edited_features": len(verdict["primary_changes"]),
                        "max_relative_change": max(verdict["primary_changes"].values()),
                        "smallest_budget": budget,
                    }
    return pd.DataFrame(identities), unique, originals


def attack_check(models: dict[str, Any], unique: dict[str, dict[str, Any]], originals: dict[int, pd.Series],
                 X_val: pd.DataFrame) -> pd.DataFrame:
    """P(attack) of every unique candidate and original per model (one batched prediction per model)."""
    rows = []
    uids = list(unique)
    frame = pd.DataFrame([unique[u]["candidate_row"] for u in uids], columns=X_val.columns).astype(X_val.dtypes.to_dict()) if uids else None
    orig_frame = pd.DataFrame([originals[s] for s in originals], columns=X_val.columns).astype(X_val.dtypes.to_dict())
    for key, model in models.items():
        p_orig = dict(zip(originals, model.predict_proba(orig_frame)[:, 1]))
        probs = model.predict_proba(frame)[:, 1] if uids else []
        for uid, p in zip(uids, probs):
            sample = unique[uid]["sample_position"]
            rows.append({"model_key": key, "candidate_uid": uid, "sample_position": sample,
                         "p_attack_original": float(p_orig[sample]), "p_attack_candidate": float(p),
                         "delta_p": float(p) - float(p_orig[sample]), "still_attack": bool(p >= 0.5),
                         "original_eligible": bool(p_orig[sample] >= 0.5)})
    return pd.DataFrame(rows)


def plan(protocol: dict[str, Any], X_val: pd.DataFrame, models: dict[str, Any], samples=None) -> dict[str, Any]:
    spec = protocol["exhaustive_evaluation"]
    samples = list(samples or protocol["sample_positions"])
    seeds = sorted(spec["lime_seeds"])
    identities, unique, originals = enumerate_physical(X_val, samples, spec["perturbation_budgets"])
    attack = attack_check({k: models[k] for k in spec["models"]}, unique, originals, X_val)
    per_model = {}
    for key in spec["models"]:
        sub = attack[(attack["model_key"] == key)] if len(attack) else attack
        eligible_samples = sorted({int(s) for s in samples if len(sub) and bool(sub[sub["sample_position"] == s]["original_eligible"].any())})
        evaluated = sub[sub["still_attack"] & sub["original_eligible"]] if len(sub) else sub
        per_model[key] = {"unique_candidates_evaluated": int(len(evaluated)),
                          "originals": len(eligible_samples),
                          "lime_calls": int((len(evaluated) + len(eligible_samples)) * len(seeds))}
    valid = identities[identities["valid"]]
    statuses = pd.Series({u: unique[u]["physical_status"] for u in unique}, dtype=object)
    return {
        "samples": samples, "budgets": spec["perturbation_budgets"], "models": spec["models"], "lime_seeds": seeds,
        "identities_total": int(len(identities)),
        "identities_valid": int(len(valid)),
        "identities_valid_by_status": valid["status"].value_counts().to_dict(),
        "unique_valid_candidates": int(len(unique)),
        "unique_by_status": statuses.value_counts().to_dict() if len(statuses) else {},
        "per_model": per_model,
        "lime_calls_total": int(sum(m["lime_calls"] for m in per_model.values())),
        "_identities": identities, "_unique": unique, "_originals": originals, "_attack": attack,
    }


# ---------------------------------------------------------------------------------------------
# Persistent explanation cache (resume / checkpoint)
# ---------------------------------------------------------------------------------------------
class ExplanationCache:
    def __init__(self, path: Path, header: dict[str, Any]):
        self.path = Path(path)
        self.header = header
        self.entries: dict[tuple[str, int, str, int], dict[str, float]] = {}
        self.dropped_lines = 0
        if self.path.exists():
            self._load()
        else:
            self._append({"type": "header", **header})

    def _load(self) -> None:
        lines = self.path.read_bytes().split(b"\n")
        good: list[bytes] = []
        for index, raw in enumerate(lines):
            if not raw.strip():
                continue
            try:
                item = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                if index >= len(lines) - 2:  # a truncated final line from an interrupted write
                    self.dropped_lines += 1
                    continue
                raise ExhaustiveError(f"corrupt cache line {index + 1} in {self.path}")
            good.append(raw)
            if item.get("type") == "header":
                stored = {k: v for k, v in item.items() if k != "type"}
                if stored != self.header:
                    changed = sorted(k for k in set(stored) | set(self.header) if stored.get(k) != self.header.get(k))
                    raise ExhaustiveError(f"cannot resume: cache header differs on {changed}")
            elif item.get("type") == "explanation":
                self.entries[(item["model_key"], int(item["sample_position"]), item["key"], int(item["seed"]))] = item["weights"]
        if not good:
            raise ExhaustiveError(f"empty cache file {self.path}")
        if self.dropped_lines:  # rewrite without the truncated line (atomic)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_bytes(b"\n".join(good) + b"\n")
            os.replace(tmp, self.path)

    def _append(self, item: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(item, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def get(self, model_key: str, sample: int, key: str, seed: int):
        return self.entries.get((model_key, int(sample), key, int(seed)))

    def put(self, model_key: str, sample: int, key: str, seed: int, weights: dict[str, float]) -> None:
        self.entries[(model_key, int(sample), key, int(seed))] = weights
        self._append({"type": "explanation", "model_key": model_key, "sample_position": int(sample),
                      "key": key, "seed": int(seed), "weights": weights})


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


def summarize(summary: pd.DataFrame, identities: pd.DataFrame) -> pd.DataFrame:
    """Per model x sample x budget x physical status group: counts and the strongest held-out changes."""
    rows = []
    valid = identities[identities["valid"]]
    for model_key in sorted(summary["model_key"].unique()):
        msum = summary[summary["model_key"] == model_key].set_index("candidate_uid")
        for (sample, budget), group in valid.groupby(["sample_position", "budget"]):
            for status in STATUS_GROUPS:
                uids = group[group["status"] == status]["candidate_uid"]
                sub = msum.loc[msum.index.intersection(uids)]
                n_identities = int(len(uids))
                rows.append({
                    "model_key": model_key, "sample_position": int(sample), "budget": budget, "policy": pr.POLICY_PHYSICAL,
                    "physical_status_group": status,
                    "valid_lattice_candidates": n_identities,
                    "evaluated_still_attack": int(len(sub)),
                    "exceeds_heldout_top10": int(sub["exceeds_heldout_top10"].sum()) if len(sub) else 0,
                    "exceeds_heldout_spearman": int(sub["exceeds_heldout_spearman"].sum()) if len(sub) else 0,
                    "exceeds_heldout_both": int(sub["exceeds_heldout_both"].sum()) if len(sub) else 0,
                    "lime_success_count": int(sub["lime_success"].sum()) if len(sub) else 0,
                    "max_heldout_top10_shift": float(sub["heldout_top10_shift"].max()) if len(sub) else None,
                    "min_heldout_spearman_mean": float(sub["heldout_spearman_mean"].min()) if len(sub) else None,
                    "max_search_seed_objective": float(sub["search_seed_objective"].max()) if len(sub) else None,
                    "min_delta_p": float(sub["delta_p"].min()) if len(sub) else None,
                })
    return pd.DataFrame(rows)


def run_exhaustive(
    protocol: dict[str, Any] | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    resume_dir: str | Path | None = None,
    split: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
    models: dict[str, Any] | None = None,
    explain_fn: Callable[[str, int, pd.Series, int], dict[str, float]] | None = None,
    samples=None,
    max_new_explanations: int | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    protocol = mf.load_protocol(mf.PROTOCOL_V2_PATH) if protocol is None else protocol
    problems = mf.validate_protocol(protocol)
    if problems or not mf.is_v2(protocol):
        raise ExhaustiveError(f"a valid step6-v2 protocol is required: {problems}")
    spec = protocol["exhaustive_evaluation"]
    seeds = sorted(spec["lime_seeds"])
    heldout = tuple(protocol["heldout_lime_seeds"])
    search_seed = int(protocol["lime_search_seed"])
    manifest = mf.build_manifest(protocol) if manifest is None else manifest
    split = runner.guarded_split(ts.get_xai_data() if split is None else split)
    X_val = split["X_val"]
    if models is None:
        shas = runner.verify_model_files(runner.make_selection(protocol, models=spec["models"]), manifest)
        models = {k: ts.load_xai_model(k, split) for k in spec["models"]}
    else:
        shas = {k: manifest["models"][k]["sha256"] for k in spec["models"]}
    if explain_fn is None:
        explain_fn = ho.make_lime_explain_fn(split, shas)

    selected_samples = list(samples) if samples is not None else list(protocol["sample_positions"])
    if any(s not in protocol["sample_positions"] for s in selected_samples):
        raise ExhaustiveError("samples must come from the protocol")
    header = {
        "exhaustive_version": EXHAUSTIVE_VERSION,
        "protocol_sha256": manifest["protocol_sha256"],
        "model_sha256": dict(sorted(shas.items())),
        "lime_configuration": json.loads(json.dumps(manifest["lime_configuration"], default=str)),
        "physical_repair_sha256": mf.file_sha256(ROOT / "src" / "unsw" / "physical_repair.py"),
        "objective_source_sha256": mf.source_sha256(ts.objective_score),
        "samples": selected_samples,
        "lime_seeds": seeds,
    }

    if resume_dir is None:
        run_dir = _new_run_dir(Path(output_root))
        with (run_dir / "manifest.json").open("x", encoding="utf-8", newline="\n") as fh:
            fh.write(runner.dumps(manifest))
    else:
        run_dir = Path(resume_dir)
        if not (run_dir / CACHE_NAME).exists():
            raise ExhaustiveError(f"{run_dir} has no {CACHE_NAME}; not an exhaustive run folder")
    cache = ExplanationCache(run_dir / CACHE_NAME, header)
    if cache.dropped_lines:
        log(f"dropped {cache.dropped_lines} truncated cache line(s)")

    planned = plan(protocol, X_val, models, selected_samples)
    identities, unique, originals, attack = planned["_identities"], planned["_unique"], planned["_originals"], planned["_attack"]
    public_plan = {k: v for k, v in planned.items() if not k.startswith("_")}
    _write_json(run_dir / "run_config.json", {"header": header, "plan": public_plan, "run_id": run_dir.name,
                                              "scope": spec["scope"], "official_test_used": False})

    counters = {"computed": 0, "reused": 0}

    def cached_explain(model_key: str, sample: int, row: pd.Series, seed: int) -> dict[str, float]:
        key = key_sha(candidate_key(row))
        hit = cache.get(model_key, sample, key, seed)
        if hit is not None:
            counters["reused"] += 1
            return hit
        if max_new_explanations is not None and counters["computed"] >= max_new_explanations:
            raise Interrupted(f"stopped after {counters['computed']} new explanations (test limit)")
        weights = {f: float(w) for f, w in explain_fn(model_key, sample, row, seed).items()}
        cache.put(model_key, sample, key, seed, weights)
        counters["computed"] += 1
        return weights

    _write_json(run_dir / "status.json", {"state": "running", "planned_lime_calls": public_plan["lime_calls_total"],
                                          "cached_explanations": len(cache.entries),
                                          "updated_at_utc": datetime.now(timezone.utc).isoformat()})
    start = time.perf_counter()
    candidates = []
    attack_idx = attack.set_index(["model_key", "candidate_uid"]) if len(attack) else attack
    for model_key in spec["models"]:
        for uid, info in unique.items():
            row = attack_idx.loc[(model_key, uid)]
            if not (row["still_attack"] and row["original_eligible"]):
                continue
            candidates.append({
                "candidate_uid": f"{model_key}:{uid}", "unique_candidate_uid": uid, "model_key": model_key,
                "sample_position": info["sample_position"], "original_row": originals[info["sample_position"]],
                "candidate_row": info["candidate_row"], "physical_status": info["physical_status"],
                "physical_unverified": info["physical_unverified"], "primary_changes": info["primary_changes"],
                "derived_changes": info["derived_changes"], "n_edited_features": info["n_edited_features"],
                "max_relative_change": info["max_relative_change"], "assignment": info["assignment"],
                "p_attack_original": row["p_attack_original"], "p_attack_candidate": row["p_attack_candidate"],
                "delta_p": row["delta_p"], "physically_valid_claimed": False,
            })
    try:
        evaluation = ho.evaluate_candidates(candidates, cached_explain, heldout_seeds=heldout,
                                            part_b_seeds=tuple(protocol["part_b_lime_seeds"]), search_seed=search_seed,
                                            key_fn=candidate_key)
    except Interrupted:
        _write_json(run_dir / "status.json", {"state": "interrupted", "computed_now": counters["computed"],
                                              "reused": counters["reused"], "cached_explanations": len(cache.entries),
                                              "planned_lime_calls": public_plan["lime_calls_total"],
                                              "updated_at_utc": datetime.now(timezone.utc).isoformat()})
        raise

    summary = pd.DataFrame(evaluation["summary"])
    if len(summary):
        objective = []
        for cand in candidates:
            orig_w = cache.get(cand["model_key"], cand["sample_position"], key_sha(candidate_key(cand["original_row"])), search_seed)
            cand_w = cache.get(cand["model_key"], cand["sample_position"], key_sha(candidate_key(cand["candidate_row"])), search_seed)
            objective.append({"candidate_uid": cand["candidate_uid"], "search_seed_objective": float(ts.objective_score(orig_w, cand_w))})
        summary = summary.merge(pd.DataFrame(objective), on="candidate_uid", how="left")
        summary["lime_success"] = (summary["exceeds_heldout_both"].astype(bool)
                                   & (summary["physical_status"] == "consistent_with_checked_rules"))
        summary = summary.drop(columns=[c for c in ("search_seed_objective_recomputed", "search_score_gap", "search_score_reproduced") if c in summary])
    model_identities = identities.merge(attack[["model_key", "candidate_uid", "p_attack_candidate", "still_attack"]],
                                        on="candidate_uid", how="left") if len(attack) else identities
    by_group = summarize(summary, identities) if len(summary) else pd.DataFrame()
    unique_frame = pd.DataFrame([{k: v for k, v in info.items() if k != "candidate_row"} |
                                 {"candidate_json": json.dumps(info["candidate_row"].to_dict(), default=str)} for info in unique.values()])

    _write_csv(run_dir / "identities.csv", model_identities)
    _write_csv(run_dir / "unique_candidates.csv", unique_frame)
    _write_csv(run_dir / "heldout_summary.csv", summary)
    _write_csv(run_dir / "heldout_per_seed.csv", pd.DataFrame(evaluation["per_seed"]))
    _write_csv(run_dir / "noise_bands.csv", pd.DataFrame(evaluation["bands"]))
    _write_csv(run_dir / "summary_by_status_group.csv", by_group)
    status = {"state": "complete", "computed_now": counters["computed"], "reused": counters["reused"],
              "cached_explanations": len(cache.entries), "planned_lime_calls": public_plan["lime_calls_total"],
              "elapsed_seconds": round(time.perf_counter() - start, 1),
              "updated_at_utc": datetime.now(timezone.utc).isoformat()}
    _write_json(run_dir / "status.json", status)
    (run_dir / "README.md").write_text(README, encoding="utf-8", newline="\n")
    return {"run_dir": str(run_dir), "status": status, "plan": public_plan, "summary": summary, "by_group": by_group}


def _new_run_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path, suffix = root / f"exhaustive_{stamp}", 1
    while True:
        try:
            path.mkdir()
            return path
        except FileExistsError:
            path = root / f"exhaustive_{stamp}_{suffix:02d}"
            suffix += 1


README = """# Step 6 v2 exhaustive evaluation of the physical_v1 lattice

Scope: definitive only for the enumerated physical_v1 lattice (sbytes/spkts increase-only, 1% grid,
budgets A/B), not for every possible network modification. No candidate is called physically valid.

| File | One row per |
|---|---|
| `identities.csv` | model x sample x budget x lattice candidate (valid or not), with P(attack) |
| `unique_candidates.csv` | unique valid candidate (deduplicated across budgets) with physical status and changes |
| `heldout_summary.csv` | model x unique candidate: held-out LIME metrics, noise flags, search-seed objective, delta P, lime_success |
| `heldout_per_seed.csv` | model x candidate x seed |
| `noise_bands.csv` | model x sample: seed-noise bands of the original |
| `summary_by_status_group.csv` | model x sample x budget x physical status group (consistent vs unverified) |
| `explanations_cache.jsonl` | every LIME explanation (resume cache) |

`lime_success` = exceeds the held-out combined test AND physical status consistent_with_checked_rules.
Candidates with unverified packet-timing dependencies are reported in their own group and never count
as a success.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", default=str(mf.PROTOCOL_V2_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--resume", default=None)
    parser.add_argument("--plan-only", action="store_true", help="print exact candidate and LIME-call counts; no LIME")
    args = parser.parse_args()
    protocol = mf.load_protocol(Path(args.protocol))
    if args.plan_only:
        problems = mf.validate_protocol(protocol)
        if problems:
            raise SystemExit(f"protocol invalid: {problems}")
        manifest = mf.build_manifest(protocol)
        split = runner.guarded_split(ts.get_xai_data())
        spec = protocol["exhaustive_evaluation"]
        runner.verify_model_files(runner.make_selection(protocol, models=spec["models"]), manifest)
        models = {k: ts.load_xai_model(k, split) for k in spec["models"]}
        planned = plan(protocol, split["X_val"], models)
        public = {k: v for k, v in planned.items() if not k.startswith("_")}
        unique = planned["_unique"]
        public["unique_by_sample"] = pd.Series([u["sample_position"] for u in unique.values()]).value_counts().sort_index().to_dict()
        public["unique_by_sample_and_status"] = {
            f"{s}|{st}": int(n) for (s, st), n in
            pd.DataFrame([(u["sample_position"], u["physical_status"]) for u in unique.values()]).value_counts().sort_index().items()}
        print(json.dumps(public, indent=2, default=str))
        return
    out = run_exhaustive(protocol, args.output_root, args.resume)
    print(json.dumps(out["status"], indent=2))
    print(f"Run directory: {out['run_dir']}")


if __name__ == "__main__":
    main()
