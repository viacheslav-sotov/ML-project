"""Step 6 G4: diagnostic unconstrained upper bound (protocol step6-g4-diagnostic-1, decision D7).

G4 asks only whether LIME (and TreeSHAP) on these models can be moved beyond seed noise AT ALL with
large edits. It is separate from protocol step6-v2 and from the current_v1 and physical_v1 policies,
whose code is not used or changed here. A G4 success is NEVER evidence of a physically realizable
network attack: G4 edits derived and dependent features independently and does not repair them.

Policy unconstrained_g4_v1: any of 35 non-frozen numeric features, up to 10 features, each up to 50% of
max(1, |original|), integer features rounded, values clamped to the observed training range, the
attacked model must still predict attack. Methods: shuffled greedy (order seeds 101-110) and a matched
random control (control seeds 201-210), LIME search seed 11, 100 LIME calls, checkpoints 25/50/100.

After the searches, every checkpoint candidate is
- labelled with physical_report_v2 (consistent / unverified / violates checked rule, reported separately);
- re-explained with LIME on held-out seeds 22/33/44/55 (D8 combined test, DIAGNOSTIC ONLY), with
  confirmation on seeds 401-404 for candidates beyond the band;
- optionally explained with TreeSHAP and compared with the D11 reference (DIAGNOSTIC ONLY).

The official test set is removed before any code runs.

    python experiments/unsw_step6_g4.py --plan-only
    python experiments/unsw_step6_g4.py --smoke --output-root <dir>        # smoke test (protocol smoke_test settings)
    python experiments/unsw_step6_g4.py                                    # full G4 run (needs approval)
    python experiments/unsw_step6_g4.py --resume <run dir>
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
from src.unsw import physical_repair as pr  # noqa: E402
from src.unsw.constraints import FEATURE_ORDER, INTEGER_FEATURES, NUMERIC_MIN_MAX  # noqa: E402

G4_VERSION = "step6-g4-diagnostic-1"
G4_PATH = ROOT / "experiments" / "step6_protocol_g4.json"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "step6_g4_runs"
G4_RUNNER_VERSION = 1
SNAP_TOLERANCE = 1e-9
PHYSICAL_STATUSES = ("consistent_with_checked_rules", "unverified_dependencies", "violates_checked_rule")
METHODS = ("shuffled_greedy", "random_control")


class G4Error(RuntimeError):
    pass


class Interrupted(RuntimeError):
    """Raised by the task limit used in tests to simulate a crash."""


# ---------------------------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------------------------
def load_g4(path: Path = G4_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_g4(p: dict[str, Any]) -> list[str]:
    problems = []
    if p.get("protocol_version") != G4_VERSION:
        return [f"protocol_version must be {G4_VERSION}"]
    pol = p["policy"]
    editable, frozen = pol["editable_features"], pol["frozen_features"]
    if set(editable) & set(frozen):
        problems.append("editable and frozen features overlap")
    if sorted(editable + frozen) != sorted(FEATURE_ORDER):
        problems.append("editable + frozen features must be exactly the 42 features")
    if any(f not in NUMERIC_MIN_MAX for f in editable):
        problems.append("every editable feature must be numeric")
    if pol.get("name") != "unconstrained_g4_v1" or pol["name"] in pr.POLICIES:
        problems.append("policy name must be unconstrained_g4_v1 (separate from current_v1/physical_v1)")
    if pol.get("max_changed_features") != 10 or pol.get("max_relative_change") != 0.5:
        problems.append("D7 fixes up to 10 features and 50% per feature")
    if sorted(pol.get("directions", [])) != [-1, 1] or not pol.get("greedy_step_fractions"):
        problems.append("directions must be [-1, 1] with at least one step fraction")
    if any(not 0 < s <= pol.get("max_relative_change", 0) for s in pol.get("greedy_step_fractions", [])):
        problems.append("greedy step fractions must be in (0, max_relative_change]")
    seeds = {
        "search": {p["objective"]["lime_search_seed"]}, "order": set(p["methods"]["shuffled_greedy"]["order_seeds"]),
        "control": set(p["methods"]["random_control"]["control_seeds"]), "heldout": set(p["heldout_lime_seeds"]),
        "confirmation": set(p["confirmation_lime_seeds"]),
    }
    names = list(seeds)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if seeds[a] & seeds[b]:
                problems.append(f"{a} and {b} seeds overlap")
    if set(p["part_b_lime_seeds"]) != seeds["search"] | seeds["heldout"]:
        problems.append("part_b_lime_seeds must be the search seed plus the held-out seeds")
    cps = p["checkpoints"]
    if cps != sorted(cps) or cps[-1] != p["explanation_budget"]:
        problems.append("checkpoints must be ascending and end at the explanation budget")
    for model in p["models"]:
        value = str(p["expected_model_sha256"].get(model, ""))
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            problems.append(f"expected_model_sha256 missing or malformed for {model}")
    if sorted(p["sample_identity"]) != sorted(str(s) for s in p["sample_positions"]):
        problems.append("sample_identity must cover exactly the sample_positions")
    smoke = p["smoke_test"]
    if (any(s not in p["sample_positions"] for s in smoke["samples"])
            or any(s not in seeds["order"] for s in smoke["order_seeds"])
            or any(s not in seeds["control"] for s in smoke["control_seeds"])
            or smoke["explanation_budget"] > p["explanation_budget"]):
        problems.append("smoke_test settings must be a subset of the protocol")
    if "never evidence of a physically realizable" not in p.get("interpretation", ""):
        problems.append("interpretation statement missing")
    return problems


# ---------------------------------------------------------------------------------------------
# Policy unconstrained_g4_v1
# ---------------------------------------------------------------------------------------------
class G4Policy:
    def __init__(self, protocol: dict[str, Any]):
        pol = protocol["policy"]
        self.name = pol["name"]
        self.features = list(pol["editable_features"])
        self.frozen = list(pol["frozen_features"])
        self.max_changed = int(pol["max_changed_features"])
        self.max_relative = float(pol["max_relative_change"])
        self.steps = [float(s) for s in pol["greedy_step_fractions"]]

    def key(self, row: pd.Series) -> tuple[float, ...]:
        return tuple(float(row[f]) for f in self.features)

    def changed(self, original: pd.Series, row: pd.Series) -> list[str]:
        return [f for f in self.features if float(row[f]) != float(original[f])]

    @staticmethod
    def relative(original_value: float, value: float) -> float:
        return abs(float(value) - float(original_value)) / max(1.0, abs(float(original_value)))

    def mutate(self, current: pd.Series, original: pd.Series, feature: str, sign: int, fraction: float) -> pd.Series:
        proposal = current.copy()
        unit = max(1.0, abs(float(original[feature])))
        value = float(current[feature]) + int(sign) * float(fraction) * unit
        if feature in INTEGER_FEATURES:
            value = float(round(value))
        lo, hi = NUMERIC_MIN_MAX[feature]
        value = float(min(hi, max(lo, value)))
        proposal[feature] = original[feature] if abs(value - float(original[feature])) <= SNAP_TOLERANCE * unit else value
        return proposal

    def check(self, original: pd.Series, row: pd.Series) -> tuple[bool, str | None]:
        frozen_changed = [f for f in self.frozen if row[f] != original[f]]
        if frozen_changed:
            return False, f"frozen feature changed: {frozen_changed}"
        changed = self.changed(original, row)
        if not changed:
            return False, "no change"
        if len(changed) > self.max_changed:
            return False, f"more than {self.max_changed} features changed"
        worst = max(self.relative(original[f], row[f]) for f in changed)
        if worst > self.max_relative + 1e-12:
            return False, f"relative change {worst:.3f} exceeds {self.max_relative}"
        return True, None

    def summary(self, original: pd.Series, row: pd.Series) -> dict[str, Any]:
        changed = self.changed(original, row)
        rel = {f: round(self.relative(original[f], row[f]), 6) for f in changed}
        return {"changed_features": changed, "n_changed": len(changed), "relative_changes": rel,
                "max_relative_change": max(rel.values()) if rel else 0.0,
                "changes": {f: float(row[f]) for f in changed}}


def apply_changes(original: pd.Series, changes: dict[str, float]) -> pd.Series:
    row = original.copy()
    for f, v in changes.items():
        row[f] = v
    return row


def _model_tag(model_key: str) -> int:
    return int(sum(ord(c) for c in model_key))


# ---------------------------------------------------------------------------------------------
# Searches (search seed 11, budget counted in LIME explanation calls, baseline included)
# ---------------------------------------------------------------------------------------------
def _checkpoints(history: list[dict[str, Any]], checkpoints, budget: int) -> dict[int, dict[str, Any]]:
    out = {}
    for cp in checkpoints:
        if cp > budget:
            continue
        within = [h for h in history if h["calls"] <= cp]
        best = max(within, key=lambda h: (h["score"], -h["calls"]), default=None)
        out[int(cp)] = ({"best_score": 0.0, "candidate_is_original": True, "changes": {}, "p_attack": None, "at_call": 1}
                        if best is None or best["score"] <= 0.0 else
                        {"best_score": best["score"], "candidate_is_original": False, "changes": best["changes"],
                         "p_attack": best["p_attack"], "at_call": best["calls"]})
    return out


def shuffled_greedy(policy: G4Policy, model_key: str, sample: int, original: pd.Series, prob_fn, explain_fn,
                    order_seed: int, budget: int, search_seed: int) -> dict[str, Any]:
    rng = np.random.default_rng([int(order_seed), int(sample), _model_tag(model_key)])
    order = [str(f) for f in rng.permutation(np.array(policy.features, dtype=object))]
    directions = {f: [int(d) for d in rng.permutation(np.array([-1, 1]))] for f in order}
    baseline = explain_fn(model_key, sample, original, search_seed)
    calls, best_score, current = 1, 0.0, original.copy()
    seen = {policy.key(original)}
    history, counts = [], {"proposals": 0, "no_change": 0, "duplicate": 0, "out_of_policy": 0, "benign": 0}
    stopped = "explanation_budget_exhausted"
    while calls < budget:
        improved = False
        for f in order:
            for d in directions[f]:
                for step in policy.steps:
                    if calls >= budget:
                        break
                    proposal = policy.mutate(current, original, f, d, step)
                    counts["proposals"] += 1
                    key = policy.key(proposal)
                    if key == policy.key(current):
                        counts["no_change"] += 1
                        continue
                    if key in seen:
                        counts["duplicate"] += 1
                        continue
                    seen.add(key)
                    ok, _ = policy.check(original, proposal)
                    if not ok:
                        counts["out_of_policy"] += 1
                        continue
                    p = float(prob_fn(model_key, proposal))
                    if p < 0.5:
                        counts["benign"] += 1
                        continue
                    score = float(ts.objective_score(baseline, explain_fn(model_key, sample, proposal, search_seed)))
                    calls += 1
                    history.append({"calls": calls, "score": score, "p_attack": p, "changes": policy.summary(original, proposal)["changes"]})
                    if score > best_score:
                        best_score, current, improved = score, proposal.copy(), True
                        break
                if improved or calls >= budget:
                    break
            if improved or calls >= budget:
                break
        if not improved and calls < budget:
            stopped = "local_optimum_full_sweep_without_improvement"
            break
    return {"calls": calls, "history": history, "counts": counts, "stopped_reason": stopped, "feature_order": order}


def random_control(policy: G4Policy, model_key: str, sample: int, original: pd.Series, prob_fn, explain_fn,
                   control_seed: int, budget: int, search_seed: int, attempts_per_call: int) -> dict[str, Any]:
    rng = np.random.default_rng([int(control_seed), int(sample), _model_tag(model_key), 4])
    baseline = explain_fn(model_key, sample, original, search_seed)
    calls, attempts, max_attempts = 1, 0, attempts_per_call * budget
    seen = {policy.key(original)}
    history, counts = [], {"proposals": 0, "no_change": 0, "duplicate": 0, "out_of_policy": 0, "benign": 0}
    stopped = "explanation_budget_exhausted"
    while calls < budget:
        if attempts >= max_attempts:
            stopped = "attempt_limit_reached"
            break
        attempts += 1
        counts["proposals"] += 1
        k = int(rng.integers(1, policy.max_changed + 1))
        chosen = [str(f) for f in rng.choice(np.array(policy.features, dtype=object), size=k, replace=False)]
        proposal = original.copy()
        for f in chosen:
            sign = int(rng.choice([-1, 1]))
            proposal = policy.mutate(proposal, original, f, sign, float(rng.uniform(0.0, policy.max_relative)))
        key = policy.key(proposal)
        if key == policy.key(original):
            counts["no_change"] += 1
            continue
        if key in seen:
            counts["duplicate"] += 1
            continue
        seen.add(key)
        ok, _ = policy.check(original, proposal)
        if not ok:
            counts["out_of_policy"] += 1
            continue
        p = float(prob_fn(model_key, proposal))
        if p < 0.5:
            counts["benign"] += 1
            continue
        score = float(ts.objective_score(baseline, explain_fn(model_key, sample, proposal, search_seed)))
        calls += 1
        history.append({"calls": calls, "score": score, "p_attack": p, "changes": policy.summary(original, proposal)["changes"]})
    return {"calls": calls, "history": history, "counts": counts, "stopped_reason": stopped, "attempts": attempts}


# ---------------------------------------------------------------------------------------------
# Planning, I/O
# ---------------------------------------------------------------------------------------------
def plan_tasks(protocol: dict[str, Any], samples, models, order_seeds, control_seeds) -> list[dict[str, Any]]:
    tasks = []
    for model in models:
        for sample in samples:
            for seed in order_seeds:
                tasks.append({"task_id": f"g4_{model}_s{sample}_shuffled_o{seed}", "model_key": model, "sample_position": int(sample),
                              "method": "shuffled_greedy", "seed": int(seed)})
            for seed in control_seeds:
                tasks.append({"task_id": f"g4_{model}_s{sample}_random_c{seed}", "model_key": model, "sample_position": int(sample),
                              "method": "random_control", "seed": int(seed)})
    return tasks


def selection_from(protocol: dict[str, Any], smoke: bool, samples=None, order_seeds=None, control_seeds=None,
                   budget=None, models=None) -> dict[str, Any]:
    base = protocol["smoke_test"] if smoke else {}
    sel = {
        "samples": [int(s) for s in (samples or base.get("samples") or protocol["sample_positions"])],
        "models": list(models or protocol["models"]),
        "order_seeds": [int(s) for s in (order_seeds or base.get("order_seeds") or protocol["methods"]["shuffled_greedy"]["order_seeds"])],
        "control_seeds": [int(s) for s in (control_seeds or base.get("control_seeds") or protocol["methods"]["random_control"]["control_seeds"])],
        "explanation_budget": int(budget or base.get("explanation_budget") or protocol["explanation_budget"]),
    }
    if (any(s not in protocol["sample_positions"] for s in sel["samples"]) or any(m not in protocol["models"] for m in sel["models"])
            or any(s not in protocol["methods"]["shuffled_greedy"]["order_seeds"] for s in sel["order_seeds"])
            or any(s not in protocol["methods"]["random_control"]["control_seeds"] for s in sel["control_seeds"])
            or not 1 < sel["explanation_budget"] <= protocol["explanation_budget"]):
        raise G4Error("the selection must be a subset of the G4 protocol")
    full = (sorted(sel["samples"]) == sorted(protocol["sample_positions"]) and sorted(sel["models"]) == sorted(protocol["models"])
            and sel["order_seeds"] == protocol["methods"]["shuffled_greedy"]["order_seeds"]
            and sel["control_seeds"] == protocol["methods"]["random_control"]["control_seeds"]
            and sel["explanation_budget"] == protocol["explanation_budget"])
    sel["run_role"] = "diagnostic_pilot" if full else "smoke_test"
    sel["checkpoints"] = [c for c in protocol["checkpoints"] if c <= sel["explanation_budget"]]
    if sel["explanation_budget"] not in sel["checkpoints"]:
        sel["checkpoints"].append(sel["explanation_budget"])
    return sel


def plan_summary(protocol: dict[str, Any], sel: dict[str, Any]) -> dict[str, Any]:
    tasks = plan_tasks(protocol, sel["samples"], sel["models"], sel["order_seeds"], sel["control_seeds"])
    n_ck = len(sel["checkpoints"])
    heldout_seeds = len(set(protocol["heldout_lime_seeds"]) | set(protocol["part_b_lime_seeds"]))
    max_unique = len(tasks) * n_ck
    originals = len(sel["samples"]) * len(sel["models"])
    return {"run_role": sel["run_role"], "tasks": len(tasks), "search_lime_calls_max": len(tasks) * sel["explanation_budget"],
            "checkpoint_candidates_max": max_unique,
            "heldout_lime_calls_max": (max_unique + originals) * heldout_seeds,
            "confirmation_lime_calls_max": (max_unique + originals) * len(protocol["confirmation_lime_seeds"])}


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, path)


def _new_run_dir(root: Path, role: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path, n = root / f"g4_{role}_{stamp}", 1
    while True:
        try:
            path.mkdir()
            return path
        except FileExistsError:
            path, n = root / f"g4_{role}_{stamp}_{n:02d}", n + 1


# ---------------------------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------------------------
def run_g4(
    protocol: dict[str, Any] | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    resume_dir: str | Path | None = None,
    smoke: bool = False,
    selection: dict[str, Any] | None = None,
    split: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
    models: dict[str, Any] | None = None,
    explain_fn: Callable | None = None,
    shap_fn: Callable | None = None,
    with_shap: bool = True,
    max_tasks: int | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    protocol = load_g4() if protocol is None else protocol
    problems = validate_g4(protocol)
    if problems:
        raise G4Error(f"invalid G4 protocol: {problems}")
    base_protocol = mf.load_protocol(mf.PROTOCOL_V2_PATH)
    manifest = mf.build_manifest(base_protocol) if manifest is None else manifest
    split = runner.guarded_split(ts.get_xai_data() if split is None else split)
    X_val = split["X_val"]
    sel = selection or selection_from(protocol, smoke)
    ex.check_sample_identity(protocol, split, sel["samples"])
    model_paths = None
    if models is None:
        models, shas, model_paths = ex.load_verified_models(base_protocol, manifest, split)
        ex.check_pinned_hashes(protocol, {k: shas[k] for k in sel["models"]})
    else:
        shas = {k: manifest["models"][k]["sha256"] for k in sel["models"]}
        ex.check_pinned_hashes(protocol, shas)
    explain_fn = explain_fn or ho.make_lime_explain_fn(split, shas)
    policy = G4Policy(protocol)
    search_seed = int(protocol["objective"]["lime_search_seed"])

    header = {"g4_runner_version": G4_RUNNER_VERSION, "protocol_version": protocol["protocol_version"],
              "g4_protocol_sha256": mf.file_sha256(G4_PATH) if G4_PATH.exists() else None,
              "g4_module_sha256": mf.file_sha256(Path(__file__)),
              "model_sha256": dict(sorted(shas.items())), "selection": sel,
              "objective_source_sha256": mf.source_sha256(ts.objective_score),
              "lime_configuration": json.loads(json.dumps(manifest["lime_configuration"], default=str))}
    if resume_dir is None:
        run_dir = _new_run_dir(Path(output_root), sel["run_role"])
        _write_json(run_dir / "manifest.json", {**manifest, "g4_protocol": protocol, "g4_header": header})
        _write_json(run_dir / "run_config.json", {"header": header, "plan": plan_summary(protocol, sel),
                                                  "interpretation": protocol["interpretation"], "official_test_used": False})
    else:
        run_dir = Path(resume_dir)
        stored = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))["header"]
        if stored != json.loads(json.dumps(header, default=str)):
            changed = sorted(k for k in set(stored) | set(header) if stored.get(k) != json.loads(json.dumps(header, default=str)).get(k))
            raise G4Error(f"cannot resume: header differs on {changed}")
    (run_dir / "tasks").mkdir(exist_ok=True)

    def prob_fn(model_key: str, row: pd.Series) -> float:
        frame = pd.DataFrame([row], columns=X_val.columns).astype(X_val.dtypes.to_dict())
        return float(models[model_key].predict_proba(frame)[:, 1][0])

    timing: dict[str, list[float]] = {}

    def timed_explain(model_key, sample, row, seed):
        tick = time.perf_counter()
        weights = explain_fn(model_key, sample, row, seed)
        timing.setdefault(model_key, []).append(time.perf_counter() - tick)
        return weights

    start = time.perf_counter()
    tasks = plan_tasks(protocol, sel["samples"], sel["models"], sel["order_seeds"], sel["control_seeds"])
    done_now = skipped = 0
    for task in tasks:
        path = run_dir / "tasks" / f"{task['task_id']}.json"
        if path.exists():
            skipped += 1
            continue
        if max_tasks is not None and done_now >= max_tasks:
            _write_json(run_dir / "status.json", {"state": "interrupted", "tasks_done": skipped + done_now, "tasks": len(tasks)})
            raise Interrupted(f"stopped after {done_now} tasks (test limit)")
        original = X_val.iloc[[task["sample_position"]]].iloc[0].copy()
        if prob_fn(task["model_key"], original) < 0.5:
            raise G4Error(f"original {task['sample_position']} is not predicted as attack by {task['model_key']}")
        if task["method"] == "shuffled_greedy":
            out = shuffled_greedy(policy, task["model_key"], task["sample_position"], original, prob_fn, timed_explain,
                                  task["seed"], sel["explanation_budget"], search_seed)
        else:
            out = random_control(policy, task["model_key"], task["sample_position"], original, prob_fn, timed_explain,
                                 task["seed"], sel["explanation_budget"], search_seed,
                                 int(protocol["methods"]["random_control"]["max_attempts_per_lime_call"]))
        record = {**task, "policy": policy.name, "explanation_budget": sel["explanation_budget"], **out,
                  "checkpoints": _checkpoints(out["history"], sel["checkpoints"], sel["explanation_budget"])}
        _write_json(path, record)
        done_now += 1
        log(f"{task['task_id']}: {out['calls']} calls, best {max([h['score'] for h in out['history']], default=0.0):.4f}")

    records = [json.loads((run_dir / "tasks" / f"{t['task_id']}.json").read_text(encoding="utf-8")) for t in tasks]
    post = evaluate_checkpoints(protocol, sel, records, X_val, policy, run_dir, header, timed_explain, models, shap_fn, with_shap)
    models_after = ex.assert_models_unchanged(model_paths, shas) if model_paths else None
    status = {"state": "complete", "run_role": sel["run_role"], "tasks": len(tasks), "tasks_run_now": done_now, "tasks_reused": skipped,
              "search_lime_calls": int(sum(r["calls"] for r in records)),
              "seconds_per_new_explanation": {k: round(float(np.mean(v)), 4) for k, v in sorted(timing.items())},
              "new_explanations": {k: len(v) for k, v in sorted(timing.items())},
              **post["status"], "model_sha256_after_run": models_after, "official_test_used": False,
              "elapsed_seconds": round(time.perf_counter() - start, 1), "updated_at_utc": datetime.now(timezone.utc).isoformat()}
    _write_json(run_dir / "status.json", status)
    (run_dir / "README.md").write_text(README.format(role=sel["run_role"]), encoding="utf-8", newline="\n")
    return {"run_dir": str(run_dir), "records": records, **{k: v for k, v in post.items() if k != "status"}, "status": status}


def evaluate_checkpoints(protocol, sel, records, X_val, policy, run_dir, header, explain_fn, models, shap_fn, with_shap):
    """Physical report, held-out LIME (D8, diagnostic), confirmation and optional TreeSHAP (D11, diagnostic)."""
    cache = ex.ExplanationCache(run_dir / "heldout_cache.jsonl", {k: header[k] for k in ("g4_protocol_sha256", "model_sha256", "lime_configuration")})

    def cached(model_key, sample, row, seed):
        key = ex.key_sha(policy.key(row))
        hit = cache.get(model_key, sample, key, seed)
        if hit is None:
            hit = {f: float(w) for f, w in explain_fn(model_key, sample, row, seed).items()}
            cache.put(model_key, sample, key, seed, hit)
        return hit

    ck_rows, unique = [], {}
    for rec in records:
        original = X_val.iloc[[rec["sample_position"]]].iloc[0].copy()
        for cp, best in rec["checkpoints"].items():
            row = {"task_id": rec["task_id"], "model_key": rec["model_key"], "sample_position": rec["sample_position"],
                   "method": rec["method"], "seed": rec["seed"], "checkpoint": int(cp), "best_score": best["best_score"],
                   "candidate_is_original": best["candidate_is_original"], "stopped_reason": rec["stopped_reason"],
                   "calls_used": rec["calls"]}
            if not best["candidate_is_original"]:
                cand = apply_changes(original, best["changes"])
                uid = f"{rec['model_key']}:s{rec['sample_position']}_{ex.key_sha(policy.key(cand))}"
                row["candidate_uid"] = uid
                if uid not in unique:
                    phys = pr.physical_report(original, cand)
                    summ = policy.summary(original, cand)
                    unique[uid] = {"candidate_uid": uid, "model_key": rec["model_key"], "sample_position": rec["sample_position"],
                                   "original_row": original, "candidate_row": cand, "search_seed_objective": best["best_score"],
                                   "p_attack_candidate": best["p_attack"], "physical_status": phys["status"],
                                   "physical_violations": "; ".join(phys["violations"]), "physical_unverified": "; ".join(phys["unverified"]),
                                   "n_changed": summ["n_changed"], "max_relative_change": summ["max_relative_change"],
                                   "changed_features": ";".join(summ["changed_features"]),
                                   "changes_json": json.dumps(summ["changes"], sort_keys=True), "physically_valid_claimed": False,
                                   "found_by_method": rec["method"], "found_by_seed": rec["seed"], "first_checkpoint": int(cp)}
            ck_rows.append(row)
    cands = list(unique.values())
    heldout = tuple(protocol["heldout_lime_seeds"])
    ev = ho.evaluate_candidates(cands, cached, heldout_seeds=heldout, part_b_seeds=tuple(protocol["part_b_lime_seeds"]),
                                search_seed=int(protocol["objective"]["lime_search_seed"]), key_fn=policy.key)
    summary = pd.DataFrame(ev["summary"])
    mismatches = 0
    if len(summary):
        mismatches = int((summary["search_score_reproduced"] == False).sum())  # noqa: E712
        summary["lime_diagnostic_success"] = summary["exceeds_heldout_both"].astype(bool)  # in policy and still attack by construction
    flagged_ids = set(summary.loc[summary["exceeds_heldout_both"].astype(bool), "candidate_uid"]) if len(summary) else set()
    flagged = [{k: c[k] for k in ("candidate_uid", "model_key", "sample_position", "original_row", "candidate_row")} for c in cands
               if c["candidate_uid"] in flagged_ids]
    conf_seeds = tuple(protocol["confirmation_lime_seeds"])
    conf = ho.evaluate_candidates(flagged, cached, heldout_seeds=conf_seeds, part_b_seeds=conf_seeds,
                                  search_seed=int(protocol["objective"]["lime_search_seed"]), key_fn=policy.key)
    if len(summary):
        c = pd.DataFrame(conf["summary"])
        c = (c[["candidate_uid", "heldout_top10_jaccard_mean", "heldout_spearman_mean", "exceeds_heldout_both"]]
             .rename(columns={"heldout_top10_jaccard_mean": "confirm_top10_jaccard_mean", "heldout_spearman_mean": "confirm_spearman_mean",
                              "exceeds_heldout_both": "confirmation_passed"})
             if len(c) else pd.DataFrame(columns=["candidate_uid", "confirm_top10_jaccard_mean", "confirm_spearman_mean", "confirmation_passed"]))
        summary = summary.merge(c, on="candidate_uid", how="left")
        summary["confirmation_evaluated"] = summary["candidate_uid"].isin(flagged_ids)
        summary["confirmation_passed"] = summary["confirmation_passed"].fillna(False).astype(bool)
        summary["confirmed_lime_diagnostic_success"] = summary["lime_diagnostic_success"] & summary["confirmation_passed"]

    shap_frame = pd.DataFrame()
    if with_shap and len(cands):
        from experiments import unsw_step6_shap as sh
        shap_fn = shap_fn or sh.make_shap_fn({k: models[k] for k in sel["models"]}, X_val)
        rows = []
        for model_key in sel["models"]:
            mc = [c for c in cands if c["model_key"] == model_key]
            if not mc:
                continue
            samples = sorted({c["sample_position"] for c in mc})
            frame = lambda rs: pd.DataFrame(rs, columns=X_val.columns).astype(X_val.dtypes.to_dict())  # noqa: E731
            so, _, _ = shap_fn(model_key, frame([X_val.iloc[s] for s in samples]))
            sc, kind, _ = shap_fn(model_key, frame([c["candidate_row"] for c in mc]))
            shap_o = {s: so.iloc[i].to_dict() for i, s in enumerate(samples)}
            refs = {s: sh.d11_reference(shap_o[s], {seed: cached(model_key, s, X_val.iloc[s], seed) for seed in heldout}, heldout)
                    for s in samples}
            for i, cnd in enumerate(mc):
                m = sh.shap_pair(shap_o[cnd["sample_position"]], sc.iloc[i].to_dict())
                ref = refs[cnd["sample_position"]]
                flags = sh.d11_flags(m, ref, True, cnd["physical_status"])
                rows.append({"candidate_uid": cnd["candidate_uid"], "shap_" + "output_kind": kind,
                             **{f"shap_{k}": v for k, v in m.items()}, "shap_j_ref": ref["j_ref"], "shap_rho_ref": ref["rho_ref"],
                             "shap_d11_meets_both": flags["d11_meets_both"]})
        shap_frame = pd.DataFrame(rows)
        if len(summary) and len(shap_frame):
            summary = summary.merge(shap_frame, on="candidate_uid", how="left")

    checkpoints = pd.DataFrame(ck_rows)
    by_method = summarize_g4(checkpoints, summary)
    _write_csv(run_dir / "checkpoints.csv", checkpoints)
    _write_csv(run_dir / "heldout_summary.csv", summary)
    _write_csv(run_dir / "heldout_per_seed.csv", pd.DataFrame(ev["per_seed"]))
    _write_csv(run_dir / "noise_bands.csv", pd.DataFrame(ev["bands"]))
    _write_csv(run_dir / "confirmation_per_seed.csv", pd.DataFrame(conf["per_seed"]))
    _write_csv(run_dir / "summary_by_method.csv", by_method)
    return {"summary": summary, "checkpoints": checkpoints, "by_method": by_method,
            "status": {"checkpoint_candidates": int(len(checkpoints)), "unique_candidates_evaluated": len(cands),
                       "physical_status_counts": pd.Series([c["physical_status"] for c in cands], dtype=object).value_counts().to_dict(),
                       "search_score_mismatches": mismatches, "confirmation_candidates": len(flagged),
                       "heldout_cache_entries": len(cache.entries), "shap_evaluated": int(len(shap_frame))}}


def summarize_g4(checkpoints: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    if not len(checkpoints):
        return pd.DataFrame()
    ck = checkpoints.merge(summary, on="candidate_uid", how="left", suffixes=("", "_c")) if len(summary) else checkpoints.copy()
    if "physical_status" not in ck:
        ck["physical_status"] = None
    rows = []
    for (model, method, cp), g in ck.groupby(["model_key", "method", "checkpoint"]):
        for status in PHYSICAL_STATUSES + ("original",):
            sub = g[g["candidate_is_original"].astype(bool)] if status == "original" else g[(~g["candidate_is_original"].astype(bool)) & (g["physical_status"] == status)]
            if status == "original" and not len(sub):
                continue
            row = {"model_key": model, "method": method, "checkpoint": int(cp), "physical_status_group": status, "runs": int(len(sub)),
                   "max_search_objective": round(float(sub["best_score"].max()), 4) if len(sub) else None}
            if status != "original" and len(sub):
                row.update({
                    "exceeds_heldout_top10": int(sub["exceeds_heldout_top10"].sum()),
                    "exceeds_heldout_both": int(sub["exceeds_heldout_both"].sum()),
                    "confirmed": int(sub["confirmed_lime_diagnostic_success"].sum()),
                    "min_heldout_top10_jaccard": round(float(sub["heldout_top10_jaccard_mean"].min()), 4),
                    "min_heldout_spearman": round(float(sub["heldout_spearman_mean"].min()), 4),
                    "shap_d11_meets_both": int(sub["shap_d11_meets_both"].sum()) if "shap_d11_meets_both" in sub else None,
                })
            rows.append(row)
    return pd.DataFrame(rows)


README = """# Step 6 G4: diagnostic unconstrained upper bound ({role})

Protocol step6-g4-diagnostic-1 (decision D7). G4 edits up to 10 of 35 numeric features by up to 50%,
derived features included and NOT repaired. It asks only whether LIME/TreeSHAP can be moved beyond seed
noise at all. **A G4 success is never evidence of a physically realizable network attack.** Every
candidate carries physical_report_v2; consistent, unverified and rule-violating candidates are counted
separately, and none is called physically valid.

| File | One row per |
|---|---|
| `tasks/*.json` | search run (method, seed, model, flow): history, accounting, checkpoints |
| `checkpoints.csv` | run x checkpoint: best candidate (or the original if nothing improved) |
| `heldout_summary.csv` | unique checkpoint candidate: physical status, held-out LIME (D8, diagnostic), confirmation, TreeSHAP (D11, diagnostic) |
| `heldout_per_seed.csv`, `noise_bands.csv`, `confirmation_per_seed.csv` | held-out details |
| `summary_by_method.csv` | model x method x checkpoint x physical status group |
| `heldout_cache.jsonl` | held-out and confirmation LIME explanations (resume cache) |
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="use the protocol's smoke_test settings")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no-shap", action="store_true")
    args = parser.parse_args()
    protocol = load_g4()
    problems = validate_g4(protocol)
    if problems:
        raise SystemExit(f"G4 protocol invalid: {problems}")
    sel = selection_from(protocol, args.smoke)
    if args.plan_only:
        print(json.dumps({"selection": sel, **plan_summary(protocol, sel)}, indent=2))
        return
    if args.resume:
        stored = json.loads((Path(args.resume) / "run_config.json").read_text(encoding="utf-8"))["header"]["selection"]
        sel = stored
    out = run_g4(protocol, args.output_root, args.resume, selection=sel, with_shap=not args.no_shap)
    print(json.dumps(out["status"], indent=2, default=str))
    print(f"Run directory: {out['run_dir']}")


if __name__ == "__main__":
    main()
