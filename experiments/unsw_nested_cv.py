"""Protocol v3 (draft): resumable nested cross-validation on the v3 duplicate-disjoint TRAIN side.

Folds are random, grouped (identical 42-feature vectors never cross a fold) and stratified by group stratum.
This is grouped stratified CV, NOT time-based validation (the chronological S2 study is separate).

Units (each written atomically to units/<model>__<feature_set>__<unit>.json; --resume skips finished units):
    outer0..outer4  inner RandomizedSearchCV (3 grouped folds) on the outer-train part, refit, score the outer fold
    final           the same search on the whole v3 train side; refit; the model is saved with its SHA-256

Nothing here reads the official test partition (src.unsw.v3_data.load_official_train refuses it).

    python experiments/unsw_nested_cv.py --plan-only
    python experiments/unsw_nested_cv.py --benchmark            # timing benchmark (outer0, 2 trials) -- only after approval
    python experiments/unsw_nested_cv.py --models xgboost --feature-sets full42     # real run -- only after approval
    python experiments/unsw_nested_cv.py --resume <run dir>
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
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.unsw import v3_data as vd  # noqa: E402
from src.unsw import v3_pipelines as vp  # noqa: E402

NCV_VERSION = 1
PROTOCOL_V3_PATH = ROOT / "experiments" / "step6_protocol_v3_draft.json"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "v3_runs"


class NestedCVError(RuntimeError):
    pass


class Interrupted(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_protocol(path: Path = PROTOCOL_V3_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_protocol(p: dict[str, Any]) -> list[str]:
    problems = []
    ncv = p.get("nested_cv", {})
    for key in ("outer", "inner", "search_spaces", "selection_metric"):
        if key not in ncv:
            problems.append(f"nested_cv.{key} missing")
    if problems:
        return problems
    if ncv["outer"]["splitter"] != "StratifiedGroupKFold" or ncv["inner"]["splitter"] != "StratifiedGroupKFold":
        problems.append("outer and inner splitters must be StratifiedGroupKFold (grouped)")
    if ncv["outer"]["random_state"] == ncv["inner"]["random_state"]:
        problems.append("outer and inner seeds must differ")
    for m in vp.MODELS:
        spec = ncv["search_spaces"].get(m)
        if not spec or int(spec.get("trials", 0)) < 1 or not spec.get("space"):
            problems.append(f"search space missing for {m}")
    if "NOT time-based" not in ncv.get("applies_to", ""):
        problems.append("nested_cv.applies_to must state that the folds are not time-based")
    if p.get("data", {}).get("label_conflict_rule") not in ("exclude", "keep"):
        problems.append("data.label_conflict_rule must be exclude or keep")
    if sorted(p.get("models", {}).get("feature_sets", {})) != sorted(vp.FEATURE_SETS):
        problems.append("feature sets must be full42 and no_ttl39")
    return problems


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------
def load_v3_train(protocol: dict[str, Any], df: pd.DataFrame | None = None) -> dict[str, Any]:
    """The v3 TRAIN side (duplicate-disjoint split); refuses a split that shares any feature vector."""
    df = vd.load_official_train() if df is None else df
    s = protocol["data"]["split"]
    rule = protocol["data"]["label_conflict_rule"]
    split = vd.disjoint_split(df, s["validation_fraction"], s["seed"], label_conflicts=rule)
    audit = vd.audit(split, df)
    if audit["shared_feature_vectors"] != 0 or audit["overlapping_rows"] != 0:
        raise NestedCVError(f"split is not duplicate-disjoint: {audit['shared_feature_vectors']} shared vectors")
    info = split["info"]
    tr = np.sort(split["train"])
    return {"X": df.iloc[tr][vd.feature_order()].reset_index(drop=True), "y": df["label"].astype(int).to_numpy()[tr],
            "strata": info["stratum"].to_numpy()[tr], "groups": info["group_id"].to_numpy()[tr],
            "categories": info["attack_cat"].to_numpy()[tr], "ids": info["id"].to_numpy()[tr],
            "split_digest": vd.split_digest(split), "audit": audit}


def data_digest(data: dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(data["X"], index=False).to_numpy().tobytes())
    h.update(np.asarray(data["y"]).tobytes())
    h.update(np.asarray(data["groups"]).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------
# Folds, metrics, units
# ---------------------------------------------------------------------------------------------
def make_folds(strata, groups, cfg: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    sgkf = StratifiedGroupKFold(n_splits=int(cfg["n_splits"]), shuffle=bool(cfg["shuffle"]), random_state=int(cfg["random_state"]))
    return [(np.asarray(a), np.asarray(b)) for a, b in sgkf.split(np.zeros(len(strata)), strata, groups)]


def check_disjoint(groups, folds) -> None:
    for a, b in folds:
        if set(np.asarray(groups)[a]) & set(np.asarray(groups)[b]):
            raise NestedCVError("a group appears on both sides of a fold")


def metrics(y_true, prob, categories) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    pred = (np.asarray(prob) >= 0.5).astype(int)
    normal = y_true == 0
    out = {
        "f1": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "false_positive_rate": float(pred[normal].mean()) if normal.any() else None,
        "roc_auc": float(roc_auc_score(y_true, prob)) if len(set(y_true)) == 2 else None,
        "pr_auc": float(average_precision_score(y_true, prob)) if len(set(y_true)) == 2 else None,
        "rows": int(len(y_true)),
    }
    cats = np.asarray(categories)
    out["recall_by_category"] = {str(c): round(float(pred[(cats == c) & (y_true == 1)].mean()), 6)
                                 for c in sorted(set(cats[y_true == 1]))}
    out["rows_by_category"] = {str(c): int((cats == c).sum()) for c in sorted(set(cats))}
    return out


def plan_units(protocol: dict[str, Any], models, feature_sets, benchmark: bool = False) -> list[dict[str, Any]]:
    n_outer = int(protocol["nested_cv"]["outer"]["n_splits"])
    units = []
    for m in models:
        for fs in feature_sets:
            names = ["outer0"] if benchmark else [f"outer{k}" for k in range(n_outer)] + ["final"]
            for u in names:
                units.append({"unit_id": f"{m}__{fs}__{u}", "model_key": m, "feature_set": fs, "unit": u})
    return units


def fit_count(protocol: dict[str, Any], model: str, benchmark: bool = False, n_iter: int | None = None) -> int:
    trials = int(n_iter or protocol["nested_cv"]["search_spaces"][model]["trials"])
    n_inner = int(protocol["nested_cv"]["inner"]["n_splits"])
    per_search = n_inner * trials + 1
    return per_search if benchmark else int(protocol["nested_cv"]["outer"]["n_splits"]) * per_search + per_search


def run_unit(unit: dict[str, Any], data: dict[str, Any], protocol: dict[str, Any], make_pipeline: Callable,
             space_fn: Callable, n_iter: int | None, model_dir: Path | None) -> dict[str, Any]:
    ncv = protocol["nested_cv"]
    X, y, strata, groups, cats = data["X"], data["y"], data["strata"], data["groups"], data["categories"]
    if unit["unit"] == "final":
        train_idx, test_idx = np.arange(len(y)), None
    else:
        folds = make_folds(strata, groups, ncv["outer"])
        check_disjoint(groups, folds)
        train_idx, test_idx = folds[int(unit["unit"].replace("outer", ""))]
    inner = make_folds(strata[train_idx], groups[train_idx], ncv["inner"])
    check_disjoint(groups[train_idx], inner)
    trials = int(n_iter or ncv["search_spaces"][unit["model_key"]]["trials"])
    search = RandomizedSearchCV(make_pipeline(unit["model_key"], unit["feature_set"]), space_fn(unit["model_key"]),
                                n_iter=trials, scoring="f1", cv=inner, refit=True, random_state=44, n_jobs=1, error_score="raise")
    tick = time.perf_counter()
    search.fit(X.iloc[train_idx], y[train_idx])
    fit_seconds = time.perf_counter() - tick
    cv = search.cv_results_
    record = {
        **unit, "trials": trials, "inner_splits": len(inner), "fits": len(inner) * trials + 1,
        "train_rows": int(len(train_idx)), "train_groups": int(len(set(groups[train_idx]))),
        "best_params": {k.replace("model__", ""): (list(v) if isinstance(v, tuple) else v) for k, v in search.best_params_.items()},
        "best_inner_f1": float(search.best_score_),
        "candidates": [{"params": {k.replace("model__", ""): (list(v) if isinstance(v, tuple) else v) for k, v in p.items()},
                        "mean_inner_f1": float(m), "std_inner_f1": float(s), "mean_fit_seconds": float(t)}
                       for p, m, s, t in zip(cv["params"], cv["mean_test_score"], cv["std_test_score"], cv["mean_fit_time"])],
        "search_seconds": round(fit_seconds, 2), "seconds_per_fit": round(fit_seconds / (len(inner) * trials + 1), 3),
    }
    if test_idx is not None:
        prob = search.best_estimator_.predict_proba(X.iloc[test_idx])[:, 1]
        record.update({"test_rows": int(len(test_idx)), "outer_metrics": metrics(y[test_idx], prob, cats[test_idx]),
                       "outer_groups_shared": len(set(groups[train_idx]) & set(groups[test_idx]))})
    elif model_dir is not None:
        import joblib
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / f"{unit['model_key']}_{unit['feature_set']}.joblib"
        tmp = path.with_name(path.name + ".tmp")
        joblib.dump({"pipeline": search.best_estimator_, "best_params": record["best_params"], "protocol_version": protocol["protocol_version"]}, tmp)
        os.replace(tmp, path)
        record.update({"model_file": str(path), "model_sha256": file_sha256(path)})
    return record


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def run(protocol: dict[str, Any] | None = None, output_root: str | Path = DEFAULT_OUTPUT_ROOT, resume_dir: str | Path | None = None,
        models=None, feature_sets=None, benchmark: bool = False, data: dict[str, Any] | None = None,
        make_pipeline: Callable | None = None, space_fn: Callable | None = None, n_iter: int | None = None,
        max_units: int | None = None, protocol_path: Path = PROTOCOL_V3_PATH, log: Callable = print) -> dict[str, Any]:
    protocol = load_protocol(protocol_path) if protocol is None else protocol
    problems = validate_protocol(protocol)
    if problems:
        raise NestedCVError(f"invalid v3 protocol: {problems}")
    models = list(models or vp.MODELS)
    feature_sets = list(feature_sets or vp.FEATURE_SETS)
    if benchmark and n_iter is None:
        n_iter = 2
    data = load_v3_train(protocol) if data is None else data
    make_pipeline = make_pipeline or (lambda m, fs: vp.build_pipeline(m, fs))
    space_fn = space_fn or (lambda m: vp.search_space(protocol, m))
    role = "benchmark" if benchmark else "nested_cv"
    header = {"ncv_version": NCV_VERSION, "protocol_version": protocol["protocol_version"],
              "protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
              "module_sha256": file_sha256(Path(__file__)), "data_digest": data_digest(data),
              "split_digest": data.get("split_digest"), "models": models, "feature_sets": feature_sets,
              "role": role, "n_iter_override": n_iter}
    if resume_dir is None:
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        run_dir = root / f"{role}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir()
        _write_json(run_dir / "run_config.json", {"header": header, "audit": data.get("audit"),
                                                  "fits_planned": {m: fit_count(protocol, m, benchmark, n_iter) * len(feature_sets) for m in models},
                                                  "official_test_used": False})
    else:
        run_dir = Path(resume_dir)
        stored = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))["header"]
        now = json.loads(json.dumps(header, default=str))
        if stored != now:
            raise NestedCVError(f"cannot resume: header differs on {sorted(k for k in now if stored.get(k) != now.get(k))}")
    (run_dir / "units").mkdir(exist_ok=True)
    units = plan_units(protocol, models, feature_sets, benchmark)
    done = skipped = 0
    for unit in units:
        path = run_dir / "units" / f"{unit['unit_id']}.json"
        if path.exists():
            skipped += 1
            continue
        if max_units is not None and done >= max_units:
            _write_json(run_dir / "status.json", {"state": "interrupted", "units_done": skipped + done, "units": len(units)})
            raise Interrupted(f"stopped after {done} units (test limit)")
        record = run_unit(unit, data, protocol, make_pipeline, space_fn, n_iter,
                          None if benchmark else run_dir / "models")
        _write_json(path, record)
        done += 1
        log(f"{unit['unit_id']}: {record['search_seconds']} s")
    records = [json.loads((run_dir / "units" / f"{u['unit_id']}.json").read_text(encoding="utf-8")) for u in units]
    summary = summarize(records)
    _write_json(run_dir / "summary.json", summary)
    status = {"state": "complete", "role": role, "units": len(units), "units_run_now": done, "units_reused": skipped,
              "official_test_used": False, "updated_at_utc": datetime.now(timezone.utc).isoformat()}
    _write_json(run_dir / "status.json", status)
    return {"run_dir": str(run_dir), "status": status, "records": records, "summary": summary}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for r in records:
        key = f"{r['model_key']}__{r['feature_set']}"
        entry = out.setdefault(key, {"outer": [], "final": None, "seconds_per_fit": []})
        entry["seconds_per_fit"].append(r["seconds_per_fit"])
        if r["unit"] == "final":
            entry["final"] = {"best_params": r["best_params"], "model_sha256": r.get("model_sha256")}
        else:
            entry["outer"].append({"unit": r["unit"], "best_params": r["best_params"], **{k: r["outer_metrics"][k] for k in ("f1", "recall", "false_positive_rate", "pr_auc")}})
    for entry in out.values():
        f1 = [o["f1"] for o in entry["outer"]]
        entry["outer_f1_mean"] = float(np.mean(f1)) if f1 else None
        entry["outer_f1_sd"] = float(np.std(f1, ddof=1)) if len(f1) > 1 else None
        entry["mean_seconds_per_fit"] = float(np.mean(entry.pop("seconds_per_fit")))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--feature-sets", nargs="+", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    protocol = load_protocol()
    if args.plan_only:
        models = args.models or list(vp.MODELS)
        fsets = args.feature_sets or list(vp.FEATURE_SETS)
        print(json.dumps({"units": len(plan_units(protocol, models, fsets, args.benchmark)),
                          "fits": {m: fit_count(protocol, m, args.benchmark, 2 if args.benchmark else None) * len(fsets) for m in models}}, indent=2))
        return
    out = run(protocol, args.output_root, args.resume, args.models, args.feature_sets, benchmark=args.benchmark)
    print(json.dumps(out["status"], indent=2))
    print(f"Run directory: {out['run_dir']}")


if __name__ == "__main__":
    main()
