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

Confirmation (stage 3B)
    Every candidate that exceeds the held-out band on both top-10 Jaccard and Spearman is explained again
    under the confirmation seeds (401-404) and tested once more against the original's band on those
    seeds. lime_success (success criterion D8) is unchanged; confirmed_lime_success additionally requires
    the confirmation to pass. Both are reported.

Role and safeguards (stage 3B)
    A run over all protocol samples is labelled "pilot" (protocol exhaustive_evaluation.role); a run over a
    subset is labelled "smoke_test". Samples must match the protocol's pinned identity (official training
    row index, attack label). Model files must exist and match the protocol's pinned SHA-256; they are
    never refitted, and the hashes are checked again after the run.

The official test set is removed before any code runs (runner.guarded_split).

    python experiments/unsw_step6_exhaustive.py --plan-only          # exact counts, no LIME
    python experiments/unsw_step6_exhaustive.py                      # run (new folder under results/step6_v2_runs)
    python experiments/unsw_step6_exhaustive.py --resume <run dir>   # continue an interrupted run
    python experiments/unsw_step6_exhaustive.py --samples 2941 25740 --output-root <dir>   # smoke test
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

EXHAUSTIVE_VERSION = 2
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
# Selection and model safeguards
# ---------------------------------------------------------------------------------------------
def run_role(protocol: dict[str, Any], samples) -> str:
    """'pilot' (or the protocol's role) for the full protocol sample set, 'smoke_test' for a subset."""
    if sorted(int(s) for s in samples) == sorted(int(s) for s in protocol["sample_positions"]):
        return str(protocol["exhaustive_evaluation"]["role"])
    return "smoke_test"


def check_sample_identity(protocol: dict[str, Any], split: dict[str, Any], samples) -> None:
    """Every sample position must still point at the pinned official-training row and be an attack."""
    identity = protocol["sample_identity"]
    X_val, y_val = split["X_val"], split["y_val"]
    problems = []
    for sample in samples:
        pin = identity.get(str(int(sample)))
        if pin is None:
            problems.append(f"{sample}: no pinned identity")
            continue
        if not 0 <= int(sample) < len(X_val):
            problems.append(f"{sample}: outside the validation split")
            continue
        row_index = int(X_val.index[int(sample)])
        if row_index != int(pin["official_train_row_index"]):
            problems.append(f"{sample}: validation row is official-training row {row_index}, pinned {pin['official_train_row_index']}")
        if int(y_val.iloc[int(sample)]) != 1:
            problems.append(f"{sample}: label is not attack")
    if problems:
        raise ExhaustiveError("sample identity check failed: " + "; ".join(problems))


def check_pinned_hashes(protocol: dict[str, Any], shas: dict[str, str]) -> None:
    pins = protocol["expected_model_sha256"]
    wrong = {k: v for k, v in shas.items() if pins.get(k) != v}
    if wrong:
        raise ExhaustiveError("model_sha256 does not match the protocol's pinned hash: "
                              + "; ".join(f"{k} is {v}, pinned {pins.get(k)}" for k, v in sorted(wrong.items())))


def load_verified_models(protocol: dict[str, Any], manifest: dict[str, Any], split: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str], dict[str, Path]]:
    """Check existence and pinned hashes BEFORE loading, so load_xai_model can never refit a missing file."""
    keys = protocol["exhaustive_evaluation"]["models"]
    try:
        shas = runner.verify_model_files(runner.make_selection(protocol, models=keys), manifest)
    except runner.RunnerError as err:
        raise ExhaustiveError(str(err)) from None
    check_pinned_hashes(protocol, shas)
    paths = {k: ROOT / manifest["models"][k]["file"] for k in keys}
    models = {k: ts.load_xai_model(k, split) for k in keys}
    assert_models_unchanged(paths, shas)
    return models, shas, paths


def assert_models_unchanged(paths: dict[str, Path], shas: dict[str, str]) -> dict[str, str]:
    now = {k: mf.file_sha256(p) if Path(p).exists() else "missing" for k, p in paths.items()}
    changed = sorted(k for k in paths if now[k] != shas[k])
    if changed:
        raise ExhaustiveError(f"model file(s) changed during the run: {changed}")
    return now


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
        n_confirm = len((spec.get("confirmation") or {}).get("lime_seeds") or [])
        per_model[key] = {"unique_candidates_evaluated": int(len(evaluated)),
                          "originals": len(eligible_samples),
                          "lime_calls": int((len(evaluated) + len(eligible_samples)) * len(seeds)),
                          "confirmation_lime_calls_max": int((len(evaluated) + len(eligible_samples)) * n_confirm)}
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
        "confirmation_lime_seeds": sorted((spec.get("confirmation") or {}).get("lime_seeds") or []),
        "confirmation_lime_calls_max": int(sum(m["confirmation_lime_calls_max"] for m in per_model.values())),
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
                    "confirmation_evaluated": int(sub["confirmation_evaluated"].sum()) if len(sub) else 0,
                    "confirmation_passed": int(sub["confirmation_passed"].sum()) if len(sub) else 0,
                    "confirmed_lime_success_count": int(sub["confirmed_lime_success"].sum()) if len(sub) else 0,
                    "max_heldout_top10_shift": float(sub["heldout_top10_shift"].max()) if len(sub) else None,
                    "min_heldout_spearman_mean": float(sub["heldout_spearman_mean"].min()) if len(sub) else None,
                    "max_search_seed_objective": float(sub["search_seed_objective"].max()) if len(sub) else None,
                    "min_delta_p": float(sub["delta_p"].min()) if len(sub) else None,
                })
    return pd.DataFrame(rows)


def overview(summary: pd.DataFrame) -> list[dict[str, Any]]:
    """Per model x physical status group: candidate and sample counts (written to pilot_summary.json)."""
    rows = []
    for model_key in sorted(summary["model_key"].unique()):
        for status in STATUS_GROUPS:
            sub = summary[(summary["model_key"] == model_key) & (summary["physical_status"] == status)]
            rows.append({
                "model_key": model_key, "physical_status_group": status,
                "candidates_evaluated": int(len(sub)),
                "samples": int(sub["sample_position"].nunique()),
                "exceeds_heldout_top10": int(sub["exceeds_heldout_top10"].sum()),
                "exceeds_heldout_both": int(sub["exceeds_heldout_both"].sum()),
                "lime_success": int(sub["lime_success"].sum()),
                "confirmation_passed": int(sub["confirmation_passed"].sum()),
                "confirmed_lime_success": int(sub["confirmed_lime_success"].sum()),
                "samples_with_lime_success": int(sub.loc[sub["lime_success"], "sample_position"].nunique()),
                "samples_with_confirmed_lime_success": int(sub.loc[sub["confirmed_lime_success"], "sample_position"].nunique()),
            })
    return rows


CONFIRM_COLUMNS = {"heldout_top10_jaccard_mean": "confirm_top10_jaccard_mean", "heldout_spearman_mean": "confirm_spearman_mean",
                   "heldout_noise_top10_jaccard_min": "confirm_noise_top10_jaccard_min",
                   "exceeds_heldout_top10": "confirm_exceeds_top10", "exceeds_heldout_spearman": "confirm_exceeds_spearman",
                   "exceeds_heldout_both": "confirmation_passed"}


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
    selected_samples = [int(s) for s in samples] if samples is not None else list(protocol["sample_positions"])
    if not selected_samples or any(s not in protocol["sample_positions"] for s in selected_samples):
        raise ExhaustiveError("samples must come from the protocol")
    check_sample_identity(protocol, split, selected_samples)
    model_paths = None
    if models is None:
        models, shas, model_paths = load_verified_models(protocol, manifest, split)
    else:
        shas = {k: manifest["models"][k]["sha256"] for k in spec["models"]}
        check_pinned_hashes(protocol, shas)
    if explain_fn is None:
        explain_fn = ho.make_lime_explain_fn(split, shas)
    role = run_role(protocol, selected_samples)
    confirm_seeds = tuple(sorted(spec["confirmation"]["lime_seeds"]))

    header = {
        "exhaustive_version": EXHAUSTIVE_VERSION,
        "protocol_sha256": manifest["protocol_sha256"],
        "model_sha256": dict(sorted(shas.items())),
        "lime_configuration": json.loads(json.dumps(manifest["lime_configuration"], default=str)),
        "physical_repair_sha256": mf.file_sha256(ROOT / "src" / "unsw" / "physical_repair.py"),
        "objective_source_sha256": mf.source_sha256(ts.objective_score),
        "samples": selected_samples,
        "lime_seeds": seeds,
        "confirmation_lime_seeds": list(confirm_seeds),
        "run_role": role,
    }

    if resume_dir is None:
        run_dir = _new_run_dir(Path(output_root), role)
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
                                              "run_role": role, "role_note": spec.get("role_note"),
                                              "confirmation": spec["confirmation"],
                                              "scope": spec["scope"], "official_test_used": False})

    counters = {"computed": 0, "reused": 0}
    timing: dict[str, list[float]] = {}

    def cached_explain(model_key: str, sample: int, row: pd.Series, seed: int) -> dict[str, float]:
        key = key_sha(candidate_key(row))
        hit = cache.get(model_key, sample, key, seed)
        if hit is not None:
            counters["reused"] += 1
            return hit
        if max_new_explanations is not None and counters["computed"] >= max_new_explanations:
            raise Interrupted(f"stopped after {counters['computed']} new explanations (test limit)")
        tick = time.perf_counter()
        weights = {f: float(w) for f, w in explain_fn(model_key, sample, row, seed).items()}
        timing.setdefault(model_key, []).append(time.perf_counter() - tick)
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
    def interrupted_status() -> None:
        _write_json(run_dir / "status.json", {"state": "interrupted", "computed_now": counters["computed"],
                                              "reused": counters["reused"], "cached_explanations": len(cache.entries),
                                              "planned_lime_calls": public_plan["lime_calls_total"],
                                              "updated_at_utc": datetime.now(timezone.utc).isoformat()})

    try:
        evaluation = ho.evaluate_candidates(candidates, cached_explain, heldout_seeds=heldout,
                                            part_b_seeds=tuple(protocol["part_b_lime_seeds"]), search_seed=search_seed,
                                            key_fn=candidate_key)
    except Interrupted:
        interrupted_status()
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

    # Confirmation on independent seeds: only candidates that exceed the held-out band on both measures.
    flagged_ids = set(summary.loc[summary["exceeds_heldout_both"].astype(bool), "candidate_uid"]) if len(summary) else set()
    flagged = [{k: c[k] for k in ("candidate_uid", "model_key", "sample_position", "original_row", "candidate_row")}
               for c in candidates if c["candidate_uid"] in flagged_ids]
    computed_before_confirmation = counters["computed"]
    try:
        confirmation = ho.evaluate_candidates(flagged, cached_explain, heldout_seeds=confirm_seeds,
                                              part_b_seeds=confirm_seeds, search_seed=search_seed, key_fn=candidate_key)
    except Interrupted:
        interrupted_status()
        raise
    confirm_bands = pd.DataFrame([{**b, "band": "confirmation"} for b in confirmation["bands"] if b["band"] == "heldout"])
    confirm_per_seed = pd.DataFrame([{**r, "seed_set": "confirmation"} for r in confirmation["per_seed"] if r["seed_set"] == "heldout"])
    if len(summary):
        conf = pd.DataFrame(confirmation["summary"])
        if len(conf):
            conf = conf[["candidate_uid", "model_key", "sample_position", *CONFIRM_COLUMNS]].rename(columns=CONFIRM_COLUMNS)
            spearman_min = confirm_bands[["model_key", "sample_position", "spearman_rho_min"]].rename(
                columns={"spearman_rho_min": "confirm_noise_spearman_min"})
            conf = conf.merge(spearman_min, on=["model_key", "sample_position"], how="left").drop(columns=["model_key", "sample_position"])
        else:
            conf = pd.DataFrame(columns=["candidate_uid", *CONFIRM_COLUMNS.values(), "confirm_noise_spearman_min"])
        summary = summary.merge(conf, on="candidate_uid", how="left")
        summary["confirmation_evaluated"] = summary["candidate_uid"].isin(flagged_ids)
        summary["confirmation_passed"] = summary["confirmation_passed"].fillna(False).astype(bool)
        summary["confirmed_lime_success"] = summary["lime_success"].astype(bool) & summary["confirmation_passed"]
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
    _write_csv(run_dir / "confirmation_noise_bands.csv", confirm_bands)
    _write_csv(run_dir / "confirmation_per_seed.csv", confirm_per_seed)
    models_after = assert_models_unchanged(model_paths, shas) if model_paths else None
    status = {"state": "complete", "run_role": role, "computed_now": counters["computed"], "reused": counters["reused"],
              "confirmation_candidates": len(flagged),
              "confirmation_explanations_computed": counters["computed"] - computed_before_confirmation,
              "cached_explanations": len(cache.entries), "planned_lime_calls": public_plan["lime_calls_total"],
              "confirmation_lime_calls_max": public_plan["confirmation_lime_calls_max"],
              "seconds_per_new_explanation": {k: round(float(np.mean(v)), 4) for k, v in sorted(timing.items())},
              "model_sha256_after_run": models_after,
              "elapsed_seconds": round(time.perf_counter() - start, 1),
              "updated_at_utc": datetime.now(timezone.utc).isoformat()}
    _write_json(run_dir / "pilot_summary.json", {"run_role": role, "role_note": spec.get("role_note"),
                                                  "by_model_and_status_group": overview(summary) if len(summary) else [],
                                                  "success_rule": protocol["success_criteria"]["lime"],
                                                  "confirmation": spec["confirmation"], "scope": spec["scope"]})
    _write_json(run_dir / "status.json", status)
    (run_dir / "README.md").write_text(README.format(role=role), encoding="utf-8", newline="\n")
    return {"run_dir": str(run_dir), "status": status, "plan": public_plan, "summary": summary, "by_group": by_group}


def _new_run_dir(root: Path, role: str = "pilot") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path, suffix = root / f"exhaustive_{role}_{stamp}", 1
    while True:
        try:
            path.mkdir()
            return path
        except FileExistsError:
            path = root / f"exhaustive_{role}_{stamp}_{suffix:02d}"
            suffix += 1


README = """# Step 6 v2 exhaustive evaluation of the physical_v1 lattice ({role})

Run role: **{role}**. A pilot covers the 10 Step 6 v2 flows and the current RF/XGBoost validation models.
It is not the final experimental design and is never pooled with final results. A smoke test covers a
subset of the flows and only checks the pipeline and runtime.

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
| `confirmation_noise_bands.csv` | model x sample with a flagged candidate: band of the original on seeds 401-404 |
| `confirmation_per_seed.csv` | model x flagged candidate x confirmation seed |
| `pilot_summary.json` | model x physical status group: candidate and sample counts |
| `explanations_cache.jsonl` | every LIME explanation (resume cache) |

`lime_success` = exceeds the held-out combined test AND physical status consistent_with_checked_rules
(success criterion D8, unchanged). `confirmed_lime_success` = `lime_success` AND the same combined test
passes again on the independent confirmation seeds 401-404. Candidates with unverified packet-timing
dependencies are reported in their own group and never count as a success.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", default=str(mf.PROTOCOL_V2_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--resume", default=None)
    parser.add_argument("--plan-only", action="store_true", help="print exact candidate and LIME-call counts; no LIME")
    parser.add_argument("--samples", nargs="+", type=int, default=None,
                        help="subset of protocol samples (labelled smoke_test); default: all protocol samples (pilot)")
    args = parser.parse_args()
    protocol = mf.load_protocol(Path(args.protocol))
    if args.plan_only:
        problems = mf.validate_protocol(protocol)
        if problems:
            raise SystemExit(f"protocol invalid: {problems}")
        manifest = mf.build_manifest(protocol)
        split = runner.guarded_split(ts.get_xai_data())
        samples = args.samples or protocol["sample_positions"]
        if any(s not in protocol["sample_positions"] for s in samples):
            raise SystemExit("samples must come from the protocol")
        check_sample_identity(protocol, split, samples)
        models, _, _ = load_verified_models(protocol, manifest, split)
        planned = plan(protocol, split["X_val"], models, samples)
        public = {"run_role": run_role(protocol, samples), **{k: v for k, v in planned.items() if not k.startswith("_")}}
        unique = planned["_unique"]
        public["unique_by_sample"] = pd.Series([u["sample_position"] for u in unique.values()]).value_counts().sort_index().to_dict()
        public["unique_by_sample_and_status"] = {
            f"{s}|{st}": int(n) for (s, st), n in
            pd.DataFrame([(u["sample_position"], u["physical_status"]) for u in unique.values()]).value_counts().sort_index().items()}
        print(json.dumps(public, indent=2, default=str))
        return
    out = run_exhaustive(protocol, args.output_root, args.resume, samples=args.samples)
    print(json.dumps(out["status"], indent=2))
    print(f"Run directory: {out['run_dir']}")


if __name__ == "__main__":
    main()
