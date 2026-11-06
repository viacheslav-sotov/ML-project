"""Step 6 experiment runner: executes the predeclared protocol (experiments/step6_protocol.json).

Work is split into independent, deterministic TASKS. Each task is one search run:

    canonical   greedy_baseline_ref_v1            one task per configuration (deterministic, never repeated),
                                                  run at the largest requested budget; smaller budgets are
                                                  read from checkpoints (prefix-equivalent, tested)
    shuffled    greedy_shuffled_v1                one task per order seed, largest budget + checkpoints
    multistart  multistart_greedy_v1 (K, fair)    one task per order seed AND per budget (fair-share caps
                                                  depend on the budget, so no prefix shortcut)
    random      random control (v1 path)          one task per control seed, largest budget + checkpoints

Order seeds and control seeds come from separate, disjoint protocol lists. Every task runs at LIME
seed protocol["lime_search_seed"], must stay within its explanation budget (baseline included), and
must keep its candidates classified as attacks. Each finished task is written atomically to
tasks/<task_id>.json (+ tasks/<task_id>.history.csv.gz); a re-run with --resume skips finished tasks,
re-runs failed or missing ones, and refuses to continue if the manifest (data, models, code) changed.
Failures are recorded in tasks/<task_id>.failed.json and failures.json and make the runner exit with
an error; nothing is skipped silently.

Outputs (run directory): manifest.json, run_config.json, run_metadata.json (links the manifest by
SHA-256), results.csv, checkpoints.csv, best_so_far.csv, restarts.csv, status.json, README.md.
checkpoints.csv is directly readable by experiments/unsw_targeted_heldout.py.

The official UNSW-NB15 test set is removed from the data split before any search code sees it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import unsw_constraint_reporting as cr  # noqa: E402
from experiments import unsw_step6_manifest as mf  # noqa: E402
from experiments import unsw_targeted_spoofing as ts  # noqa: E402

DEFAULT_OUTPUT_ROOT = ROOT / "results" / "step6_runs"
METHODS = ("canonical", "shuffled", "multistart", "random")
METHOD_STRATEGY = {
    "canonical": ts.STRATEGY_V1,
    "shuffled": ts.STRATEGY_SHUFFLED,
    "multistart": ts.STRATEGY_MULTISTART,
    "random": ts.STRATEGY_V1,  # the random control runs on the v1 mutation path (unchanged)
}
FORBIDDEN_SPLIT_KEYS = ("X_test", "y_test", "official_test_df")
RUNNER_VERSION = 1
# Manifest fields that must be identical to resume a run (timestamps and git state may differ).
RESUME_KEYS = ("protocol_sha256", "models", "datasets", "split", "samples", "seeds", "lime_configuration",
               "implementation_hashes", "constraint_policy", "software")


class RunnerError(RuntimeError):
    """Raised when tasks fail or a run cannot be (re)started safely."""


class GuardedSplit(dict):
    """The data split without the official test set; any attempt to read it raises."""

    def __getitem__(self, key):
        if key in FORBIDDEN_SPLIT_KEYS:
            raise KeyError(f"the official test set ({key}) is not available to Step 6 code")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in FORBIDDEN_SPLIT_KEYS:
            raise KeyError(f"the official test set ({key}) is not available to Step 6 code")
        return super().get(key, default)


def guarded_split(split: dict[str, Any]) -> GuardedSplit:
    return GuardedSplit({k: v for k, v in split.items() if k not in FORBIDDEN_SPLIT_KEYS})


# ---------------------------------------------------------------------------------------------
# Selection (a subset of the protocol) and task planning
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Selection:
    models: tuple[str, ...]
    samples: tuple[int, ...]
    budgets: tuple[str, ...]
    explanation_budgets: tuple[int, ...]
    order_seeds: tuple[int, ...]
    control_seeds: tuple[int, ...]
    methods: tuple[str, ...] = METHODS

    def to_json(self) -> dict[str, Any]:
        return {k: list(v) for k, v in asdict(self).items()}


def make_selection(protocol: dict[str, Any], models=None, samples=None, budgets=None, explanation_budgets=None,
                   order_seeds=None, control_seeds=None, methods=None) -> Selection:
    """Default = the full protocol; every requested value must be part of the protocol."""

    def pick(name: str, requested, allowed, cast):
        values = list(allowed) if requested is None else [cast(v) for v in requested]
        if not values:
            raise ValueError(f"{name}: empty selection")
        if len(set(values)) != len(values):
            raise ValueError(f"{name}: duplicate values {values}")
        outside = [v for v in values if v not in allowed]
        if outside:
            raise ValueError(f"{name}: {outside} are not in the predeclared protocol {list(allowed)}")
        return tuple(v for v in allowed if v in values)  # protocol order

    return Selection(
        models=pick("models", models, protocol["models"], str),
        samples=pick("samples", samples, protocol["sample_positions"], int),
        budgets=pick("budgets", budgets, protocol["perturbation_budgets"], str),
        explanation_budgets=pick("explanation_budgets", explanation_budgets, protocol["explanation_budgets"], int),
        order_seeds=pick("order_seeds", order_seeds, protocol["order_seeds"], int),
        control_seeds=pick("control_seeds", control_seeds, protocol["control_seeds"], int),
        methods=pick("methods", methods, list(METHODS), str),
    )


@dataclass(frozen=True)
class Task:
    method: str
    model_key: str
    sample_position: int
    budget_key: str
    explanation_budget: int
    checkpoints: tuple[int, ...]
    order_seed: int | None = None
    control_seed: int | None = None

    @property
    def strategy(self) -> str:
        return METHOD_STRATEGY[self.method]

    @property
    def task_id(self) -> str:
        seed = f"_o{self.order_seed}" if self.order_seed is not None else ""
        seed += f"_c{self.control_seed}" if self.control_seed is not None else ""
        return f"{self.model_key}_s{self.sample_position}_{self.budget_key}_{self.method}_b{self.explanation_budget}{seed}"

    def spec(self) -> dict[str, Any]:
        return {**asdict(self), "checkpoints": list(self.checkpoints), "strategy": self.strategy, "task_id": self.task_id}


def plan_tasks(selection: Selection) -> list[Task]:
    tasks: list[Task] = []
    largest = max(selection.explanation_budgets)
    all_checkpoints = tuple(selection.explanation_budgets)
    for model in selection.models:
        for sample in selection.samples:
            for budget in selection.budgets:
                base = dict(model_key=model, sample_position=sample, budget_key=budget)
                if "canonical" in selection.methods:
                    tasks.append(Task("canonical", explanation_budget=largest, checkpoints=all_checkpoints, **base))
                if "shuffled" in selection.methods:
                    for order_seed in selection.order_seeds:
                        tasks.append(Task("shuffled", explanation_budget=largest, checkpoints=all_checkpoints, order_seed=order_seed, **base))
                if "multistart" in selection.methods:
                    for order_seed in selection.order_seeds:
                        for eb in selection.explanation_budgets:
                            tasks.append(Task("multistart", explanation_budget=eb, checkpoints=(eb,), order_seed=order_seed, **base))
                if "random" in selection.methods:
                    for control_seed in selection.control_seeds:
                        tasks.append(Task("random", explanation_budget=largest, checkpoints=all_checkpoints, control_seed=control_seed, **base))
    ids = [t.task_id for t in tasks]
    if len(ids) != len(set(ids)):
        raise RunnerError("duplicate task ids in the plan")
    return tasks


# ---------------------------------------------------------------------------------------------
# Helpers: JSON, atomic writes, hashing
# ---------------------------------------------------------------------------------------------
def _default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=_default)


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_csv_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------------------------
# One task
# ---------------------------------------------------------------------------------------------
@dataclass
class Context:
    protocol: dict[str, Any]
    split: GuardedSplit
    models: dict[str, Any]
    model_sha: dict[str, str]
    teammate_table: dict[str, dict[str, str]]
    original_json: dict[int, str] = field(default_factory=dict)


def execute_task(task: Task, ctx: Context) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = ctx.protocol
    lime_seed = int(protocol["lime_search_seed"])
    meter = ts.CountingModel(ctx.models[task.model_key])
    start = time.perf_counter()
    if task.method == "random":
        result = ts.random_control_search(
            task.model_key, task.sample_position, task.budget_key, ctx.split, task.explanation_budget, lime_seed,
            strategy=task.strategy, max_attempts=int(protocol["max_random_attempts"]), model=meter,
            control_seed=task.control_seed,
        )
        history = result["history"]
        best_probability = result["best_probability"]
    else:
        multi = protocol["strategies"]["multistart_greedy_v1"]
        result = ts.run_targeted_search(
            task.model_key, task.sample_position, task.budget_key, ctx.split, task.explanation_budget, lime_seed,
            strategy=task.strategy, model=meter,
            order_seed=task.order_seed if task.order_seed is not None else ts.DEFAULT_ORDER_SEED,
            restarts=int(multi["restarts"]), fair_share=bool(multi["fair_share"]),
            include_canonical_first=bool(multi["include_canonical_first"]),
            model_fingerprint=f"sha256:{ctx.model_sha[task.model_key]}",
        )
        history = result["search_history"]
        best_probability = result["best_candidate_probability"]
    runtime = time.perf_counter() - start
    counts = result["evaluation_counts"]
    accounting = ts.prediction_accounting(counts, meter)

    # Hard checks: fail loudly instead of saving a questionable result.
    if counts["explanation_calls"] > task.explanation_budget:
        raise RunnerError(f"{task.task_id}: used {counts['explanation_calls']} explanations > budget {task.explanation_budget}")
    if not accounting["accounting_consistent"]:
        raise RunnerError(f"{task.task_id}: inconsistent prediction accounting {accounting}")
    if float(best_probability) < 0.5:
        raise RunnerError(f"{task.task_id}: best candidate is no longer classified as an attack (p={best_probability})")

    original_row = ctx.split["X_val"].iloc[[task.sample_position]].iloc[0]
    original_json = json.dumps(original_row.to_dict(), default=str)
    identity = {
        "task_id": task.task_id, "search": task.method, "method": "random_control" if task.method == "random" else "targeted",
        "strategy": task.strategy, "model_key": task.model_key, "sample_position": task.sample_position,
        "budget": task.budget_key, "explanation_budget": task.explanation_budget, "lime_seed": lime_seed,
        "order_seed": task.order_seed, "control_seed": task.control_seed,
        "restarts": int(protocol["strategies"]["multistart_greedy_v1"]["restarts"]) if task.method == "multistart" else None,
    }
    checkpoints = []
    for record in ts.best_so_far_checkpoints(history, list(task.checkpoints), counts["explanation_calls"], original_json):
        if not ts.checkpoint_is_valid(task.strategy, record["checkpoint"], task.explanation_budget):
            raise RunnerError(f"{task.task_id}: planned an invalid checkpoint {record['checkpoint']}")
        candidate = ts.row_from_json(record["candidate_json"], original_row)
        budget_ok, budget_reason = ts._candidate_matches_budget(original_row, candidate, task.budget_key)
        report = cr.constraint_report(original_row.to_dict(), candidate.to_dict(), budget_ok, budget_reason, ctx.teammate_table)
        probability = record["best_candidate_probability"]
        if probability is not None and float(probability) < 0.5:
            raise RunnerError(f"{task.task_id}: checkpoint {record['checkpoint']} candidate is not classified as an attack")
        checkpoints.append({
            **identity,
            "max_explanation_evaluations": task.explanation_budget,
            "checkpoint_valid": True,
            "prefix_shortcut": bool(record["checkpoint"] < task.explanation_budget),
            **record,
            **{f"constraint_{k}": v for k, v in report.items() if k != "changed_features"},
        })
    if task.method == "multistart":
        curve = [{**identity, **row} for row in result["best_so_far_curve"]]
    else:
        curve = [{**identity, **row} for row in ts._curve_from_history(history, identity["method"], task.strategy)]
    restarts = [{**identity, **row} for row in result.get("restarts", [])]
    summary = {
        **identity,
        "objective_score": float(result["best_score"]),
        "original_probability": float(result["original_probability"]),
        "best_probability": float(best_probability),
        "still_attack": bool(float(best_probability) >= 0.5),
        "constraint_valid": bool(result["constraint_valid"]),
        "best_changed_features": ";".join(result["best_changed_features"]),
        "stopped_reason": result["stopped_reason"],
        "attempts": result.get("attempts"),
        "unused_explanation_calls": int(task.explanation_budget - counts["explanation_calls"]),
        **{k: int(counts[k]) for k in ts.ACCOUNTING_KEYS},
        **{k: int(counts[k]) for k in ts.MULTISTART_EXTRA_KEYS if k in counts},
        **accounting,
        "runtime_seconds": round(runtime, 3),
    }
    payload = {"status": "ok", "runner_version": RUNNER_VERSION, "spec": task.spec(), "summary": summary,
               "checkpoints": checkpoints, "best_so_far": curve, "restarts": restarts}
    return payload, history


# ---------------------------------------------------------------------------------------------
# Run directory: create / resume, manifest, consolidation
# ---------------------------------------------------------------------------------------------
def _manifest_signature(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: manifest.get(key) for key in RESUME_KEYS}


def _runner_sha() -> str:
    return mf.file_sha256(Path(__file__))


def verify_model_files(selection: Selection, manifest: dict[str, Any]) -> dict[str, str]:
    """Every selected model file must exist and match the manifest hash (checked before any run directory is made)."""
    shas = {}
    for key in selection.models:
        entry = manifest["models"][key]
        path = ROOT / entry["file"]
        if not path.exists():
            raise RunnerError(f"model file missing: {path} (the runner never retrains)")
        actual = mf.file_sha256(path)
        if actual != entry["sha256"]:
            raise RunnerError(f"model {key} hash {actual} does not match the manifest {entry['sha256']}")
        shas[key] = actual
    return shas


def run_step6(
    selection: Selection | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    resume_dir: str | Path | None = None,
    split: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
    protocol: dict[str, Any] | None = None,
    model_loader: Callable | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    protocol = mf.load_protocol() if protocol is None else protocol
    problems = mf.validate_protocol(protocol)
    if problems:
        raise RunnerError(f"protocol invalid: {problems}")
    model_loader = model_loader or ts.load_xai_model
    fresh_manifest = mf.build_manifest(protocol) if manifest is None else manifest
    problems = mf.validate_manifest(fresh_manifest, check_files=False)
    if problems:
        raise RunnerError(f"manifest invalid: {problems}")

    if resume_dir is None:
        if selection is None:
            selection = make_selection(protocol)
        shas = verify_model_files(selection, fresh_manifest)
        run_dir = ts._make_run_dir(output_root)
        manifest_text = dumps(fresh_manifest)
        with (run_dir / "manifest.json").open("x", encoding="utf-8") as fh:  # never overwrite
            fh.write(manifest_text)
        run_config = {
            "runner_version": RUNNER_VERSION,
            "runner_sha256": _runner_sha(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection": selection.to_json(),
            "manifest_sha256": sha256_text(manifest_text),
            "protocol_version": protocol["protocol_version"],
        }
        with (run_dir / "run_config.json").open("x", encoding="utf-8") as fh:
            fh.write(dumps(run_config))
        resumed = False
    else:
        run_dir = Path(resume_dir)
        if not (run_dir / "run_config.json").exists():
            raise RunnerError(f"{run_dir} is not a runner directory")
        run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        saved_selection = make_selection(protocol, **run_config["selection"])
        if selection is not None and selection != saved_selection:
            raise RunnerError("the requested selection differs from the saved run; start a new run instead")
        selection = saved_selection
        manifest_text = (run_dir / "manifest.json").read_text(encoding="utf-8")
        if sha256_text(manifest_text) != run_config["manifest_sha256"]:
            raise RunnerError("manifest.json was modified after the run started")
        saved_manifest = json.loads(manifest_text)
        if _manifest_signature(saved_manifest) != _manifest_signature(json.loads(dumps(fresh_manifest))):
            changed = [k for k in RESUME_KEYS if saved_manifest.get(k) != json.loads(dumps(fresh_manifest)).get(k)]
            raise RunnerError(f"cannot resume: data/models/code changed since the run started ({changed})")
        if run_config["runner_sha256"] != _runner_sha():
            raise RunnerError("cannot resume: the runner code changed since the run started")
        shas = verify_model_files(selection, saved_manifest)
        resumed = True

    run_manifest = json.loads(manifest_text)
    split = guarded_split(ts.get_xai_data() if split is None else split)
    models = {key: model_loader(key, split) for key in selection.models}
    ctx = Context(protocol=protocol, split=split, models=models, model_sha=shas, teammate_table=cr.load_teammate_table())

    tasks_dir = run_dir / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    tasks = plan_tasks(selection)
    done, ran, failed = [], [], []
    for index, task in enumerate(tasks, start=1):
        result_path = tasks_dir / f"{task.task_id}.json"
        failed_path = tasks_dir / f"{task.task_id}.failed.json"
        if result_path.exists():
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            if saved.get("status") == "ok" and saved.get("spec") == json.loads(dumps(task.spec())):
                done.append(task.task_id)
                continue
            raise RunnerError(f"{result_path} exists but does not match the planned task")
        log(f"[{index}/{len(tasks)}] {task.task_id}")
        try:
            payload, history = execute_task(task, ctx)
        except Exception as exc:  # noqa: BLE001 - every failure is recorded and reported
            failed.append(task.task_id)
            atomic_write_text(failed_path, dumps({"status": "failed", "spec": task.spec(), "error": repr(exc),
                                                  "traceback": traceback.format_exc(),
                                                  "at_utc": datetime.now(timezone.utc).isoformat()}))
            log(f"    FAILED: {exc!r}")
            continue
        atomic_write_csv_gz(tasks_dir / f"{task.task_id}.history.csv.gz", history)
        atomic_write_text(result_path, dumps(payload))  # written last: marks the task complete
        if failed_path.exists():
            failed_path.unlink()
        ran.append(task.task_id)

    consolidated = consolidate(run_dir, tasks)
    status = {
        "planned": len(tasks), "completed": len(done) + len(ran), "skipped_already_done": len(done),
        "ran_now": len(ran), "failed": failed, "resumed": resumed,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_text(run_dir / "status.json", dumps(status))
    write_metadata(run_dir, run_config, run_manifest, selection, protocol, tasks, consolidated, status)
    if failed:
        atomic_write_text(run_dir / "failures.json", dumps({"failed_tasks": failed}))
        raise RunnerError(f"{len(failed)} task(s) failed: {failed}. Details in {run_dir / 'tasks'}; re-run with --resume {run_dir}")
    if (run_dir / "failures.json").exists():
        (run_dir / "failures.json").unlink()
    return {"run_dir": str(run_dir), "status": status, **consolidated}


def consolidate(run_dir: Path, tasks: list[Task]) -> dict[str, Any]:
    summaries, checkpoints, curves, restarts = [], [], [], []
    for task in sorted(tasks, key=lambda t: t.task_id):
        path = run_dir / "tasks" / f"{task.task_id}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        summaries.append(payload["summary"])
        checkpoints.extend(payload["checkpoints"])
        curves.extend(payload["best_so_far"])
        restarts.extend(payload["restarts"])
    frames = {"results.csv": summaries, "checkpoints.csv": checkpoints, "best_so_far.csv": curves, "restarts.csv": restarts}
    for name, rows in frames.items():
        tmp = run_dir / f"{name}.tmp"
        (pd.DataFrame(rows) if rows else pd.DataFrame(columns=["task_id"])).to_csv(tmp, index=False)
        os.replace(tmp, run_dir / name)
    return {"n_results": len(summaries), "n_checkpoints": len(checkpoints)}


def write_metadata(run_dir, run_config, manifest, selection, protocol, tasks, consolidated, status) -> None:
    metadata = {
        "run_id": run_dir.name,
        "runner_version": RUNNER_VERSION,
        "runner_sha256": run_config["runner_sha256"],
        "manifest": {"path": "manifest.json", "sha256": run_config["manifest_sha256"],
                     "protocol_version": manifest["protocol_version"], "git_commit": manifest["git"]["commit"],
                     "git_dirty": manifest["git"]["dirty"]},
        "selection": selection.to_json(),
        "lime_seed": int(protocol["lime_search_seed"]),
        "search_seed": int(protocol["lime_search_seed"]),
        "heldout_lime_seeds": protocol["heldout_lime_seeds"],
        "restarts": int(protocol["strategies"]["multistart_greedy_v1"]["restarts"]),
        "max_random_attempts": int(protocol["max_random_attempts"]),
        "model_sha256": {k: v["sha256"] for k, v in manifest["models"].items()},
        "official_test_used": False,
        "tasks": [t.spec() for t in tasks],
        "status": status,
        "outputs": consolidated,
        "rules": {
            "budgets": "explanation_calls <= explanation_budget for every task, baseline explanation included",
            "canonical": "deterministic: one task per configuration at the largest budget",
            "checkpoints": "smaller budgets read from the largest-budget run (prefix equivalence, tested) for canonical/shuffled/random; multi-start runs separately per budget",
            "seeds": "order seeds drive shuffled/multi-start only; control seeds drive the random control only; LIME seed fixed",
            "attack": "every saved candidate must keep P(attack) >= 0.5",
            "constraints": "constraint_experimental_valid = current validator + budget; constraint_physical_status never claims physical validity",
        },
    }
    atomic_write_text(run_dir / "run_metadata.json", dumps(metadata))
    atomic_write_text(run_dir / "README.md", README)


README = """# Step 6 runner output

| File | One row per | Key columns |
|---|---|---|
| `results.csv` | task (search run) | task_id, search (canonical/shuffled/multistart/random), model_key, sample_position, budget (A/B), explanation_budget, order_seed, control_seed, objective_score, best_probability, still_attack, stopped_reason, explanation_calls, scored_proposals, accounting columns, runtime_seconds |
| `checkpoints.csv` | task x checkpoint | task_id, search, method, strategy, checkpoint, checkpoint_valid, prefix_shortcut, best_score, explanation_calls_at_best, best_candidate_probability, candidate_json, candidate_is_original, constraint_* |
| `best_so_far.csv` | fresh explanation | task_id, search, explanation_calls, score, best_so_far |
| `restarts.csv` | multi-start restart | task_id, restart_id, restart_seed, feature_order, direction_order, fresh_call_cap, explanation_calls_used, stopped_reason, best_score |
| `tasks/<task_id>.json` | task | full payload (spec, summary, checkpoints, curve, restarts) |
| `tasks/<task_id>.history.csv.gz` | proposal | complete candidate history of the task |
| `manifest.json`, `run_config.json`, `run_metadata.json`, `status.json` | run | provenance, selection, manifest hash, completion status |

Comparisons: use `checkpoints.csv` rows with `checkpoint == B` to compare methods at budget B
(explanation calls, baseline included). Accounting: `explanation_evaluations` are LIME explanations
(the budgeted quantity); `lime_internal_prediction_rows` are the model rows LIME evaluated
internally; `top_level_prediction_calls` are the search's own single-row checks.
`constraint_physical_status` is never a claim of physical realism.
Held-out evaluation: `python experiments/unsw_targeted_heldout.py --run-dirs <this directory>`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--samples", nargs="+", type=int)
    parser.add_argument("--budgets", nargs="+", help="perturbation budgets (A/B)")
    parser.add_argument("--explanation-budgets", nargs="+", type=int)
    parser.add_argument("--order-seeds", nargs="+", type=int)
    parser.add_argument("--control-seeds", nargs="+", type=int)
    parser.add_argument("--methods", nargs="+", choices=list(METHODS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--resume", default=None, help="existing run directory to resume")
    args = parser.parse_args()
    protocol = mf.load_protocol()
    selection = None
    if args.resume is None or any(v is not None for v in (args.models, args.samples, args.budgets, args.explanation_budgets,
                                                          args.order_seeds, args.control_seeds, args.methods)):
        selection = make_selection(protocol, args.models, args.samples, args.budgets, args.explanation_budgets,
                                   args.order_seeds, args.control_seeds, args.methods)
    start = time.perf_counter()
    try:
        out = run_step6(selection, args.output_root, args.resume, protocol=protocol)
    except RunnerError as exc:
        raise SystemExit(f"ERROR: {exc}")
    print(f"Run directory: {out['run_dir']}")
    print(f"Elapsed seconds: {time.perf_counter() - start:.1f}")
    print(json.dumps(out["status"], indent=2))


if __name__ == "__main__":
    main()
