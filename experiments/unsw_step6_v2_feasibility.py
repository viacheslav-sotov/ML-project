"""Step 6 v2 feasibility check for the physical_v1 policy (stage 2; no LIME, no search, no training).

For each of the 10 v2 samples and each perturbation budget (A, B):
  - which features are editable under physical_v1 and their legal lattice values;
  - every candidate in the joint legal lattice: built and repaired exactly as the search does, checked
    by src/unsw/physical_repair.check_candidate, the validator, the budget and the checked physical rules;
  - with the saved RF and XGBoost validation pipelines (hashes checked against the v1 manifest):
    P(attack) of the original and of every valid candidate (one batched prediction per sample and model);
  - for context, how many single-feature v1 moves (current_v1 policy) pass the validator and the budget.

Reads only the sub-training/validation split; the official test set is never touched.

    python experiments/unsw_step6_v2_feasibility.py [--output-dir results/step6_v2/feasibility]
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import unsw_constraint_reporting as cr  # noqa: E402
from experiments import unsw_step6_manifest as mf  # noqa: E402
from experiments import unsw_targeted_spoofing as ts  # noqa: E402
from src.unsw import physical_repair as pr  # noqa: E402
from src.unsw.constraints import validate_candidate  # noqa: E402

DEFAULT_OUTPUT_DIR = ROOT / "results" / "step6_v2" / "feasibility"
V1_MANIFEST = ROOT / "results" / "step6_minimum" / "run_20261106_203918" / "manifest.json"
V2_SAMPLES = (2941, 3040, 3206, 15137, 27094, 4407, 30107, 34186, 15741, 25740)
MODELS = ("random_forest", "xgboost")
BUDGETS = ("A", "B")


def lattice_candidates(original_row: pd.Series, budget: str) -> list[tuple[dict[str, float], pd.Series]]:
    moves = pr.legal_moves(original_row, budget)
    features = list(moves)
    out = []
    for combo in itertools.product(*[[None] + moves[f] for f in features]):
        assignment = {f: v for f, v in zip(features, combo) if v is not None}
        if not assignment:
            continue
        row = original_row.copy()
        for feature, value in assignment.items():
            row[feature] = value
        try:
            out.append((assignment, pr.repair(original_row, row)))
        except pr.PhysicalRepairError as exc:
            out.append((assignment, exc))
    return out


def current_v1_single_moves(original_row: pd.Series, budget: str) -> int:
    """Distinct single-feature v1 moves from the original that pass the validator and the budget."""
    seen, ok = set(), 0
    for feature in ts.NUMERIC_DIRECTLY_CONTROLLABLE_FEATURES:
        for direction in (-1, 1):
            for step in ts.choose_feature_steps(budget):
                proposal = ts.mutation_proposal_v1(original_row, original_row, feature, direction, step)
                key = ts._row_key(proposal)
                if key == ts._row_key(original_row) or key in seen:
                    continue
                seen.add(key)
                if (validate_candidate(original_row.to_dict(), proposal.to_dict())["valid"]
                        and ts._candidate_matches_budget(original_row, proposal, budget)[0]):
                    ok += 1
    return ok


def load_models(split) -> tuple[dict[str, Any], dict[str, str]]:
    manifest = json.loads(V1_MANIFEST.read_text(encoding="utf-8"))
    models, hashes = {}, {}
    for key in MODELS:
        entry = manifest["models"][key]
        actual = mf.file_sha256(ROOT / entry["file"])
        if actual != entry["sha256"]:
            raise SystemExit(f"model {key} hash {actual} differs from the v1 manifest {entry['sha256']}")
        models[key] = ts.load_xai_model(key, split)
        hashes[key] = actual
    return models, hashes


def run(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    split = ts.get_xai_data()
    X_val, y_val = split["X_val"], split["y_val"]
    models, hashes = load_models(split)
    summary_rows, lattice_rows, candidate_rows = [], [], []
    for position in V2_SAMPLES:
        original = X_val.iloc[[position]].iloc[0].copy()
        p_orig = {k: float(m.predict_proba(original.to_frame().T)[0, 1]) for k, m in models.items()}
        for budget in BUDGETS:
            moves = pr.legal_moves(original, budget)
            for feature in pr.PHYSICAL_V1_EDITABLE:
                values = moves.get(feature, [])
                lattice_rows.append({
                    "validation_position": position, "budget": budget, "feature": feature,
                    "original_value": float(original[feature]), "n_legal_values": len(values),
                    "min_value": min(values) if values else None, "max_value": max(values) if values else None,
                    "max_relative_change": (max(values) - float(original[feature])) / max(1.0, abs(float(original[feature]))) if values else 0.0,
                    "values": ";".join(f"{v:g}" for v in values),
                })
            built = lattice_candidates(original, budget)
            valid_rows, statuses, reasons = [], {}, {}
            derived_max: dict[str, float] = {}
            for assignment, candidate in built:
                record = {"validation_position": position, "budget": budget,
                          "assignment": json.dumps(assignment, sort_keys=True)}
                if isinstance(candidate, Exception):
                    record.update(valid=False, status="repair_failed", violations=str(candidate))
                else:
                    verdict = pr.check_candidate(original, candidate, budget)
                    validator_ok = validate_candidate(original.to_dict(), candidate.to_dict())["valid"]
                    budget_ok = ts._candidate_matches_budget(original, candidate, budget)[0]
                    reported = cr.physical_consistency(original.to_dict(), candidate.to_dict())["physical_status"]
                    valid = verdict["valid"] and validator_ok and budget_ok
                    record.update(valid=valid, status=verdict["status"], violations="; ".join(verdict["violations"]),
                                  unverified="; ".join(verdict["unverified"]), validator_ok=validator_ok, budget_ok=budget_ok,
                                  v1_reporting_status=reported,
                                  derived_changes=json.dumps(verdict["derived_changes"], sort_keys=True))
                    if valid:
                        valid_rows.append((record, candidate))
                        for feature, change in verdict["derived_changes"].items():
                            derived_max[feature] = max(derived_max.get(feature, 0.0), change)
                    else:
                        for v in verdict["violations"]:
                            key = v.split(" [")[0]
                            reasons[key] = reasons.get(key, 0) + 1
                statuses[record["status"]] = statuses.get(record["status"], 0) + 1
                candidate_rows.append(record)
            probs = {}
            if valid_rows:
                frame = pd.DataFrame([c for _, c in valid_rows], columns=X_val.columns).astype(X_val.dtypes.to_dict())
                for key, model in models.items():
                    probs[key] = model.predict_proba(frame)[:, 1]
                    for (record, _), p in zip(valid_rows, probs[key]):
                        record[f"p_attack_{key}"] = float(p)
            row = {
                "validation_position": position, "proto": original["proto"], "service": original["service"],
                "state": original["state"], "label": int(y_val.iloc[position]), "budget": budget,
                "editable_features": ";".join(moves) if moves else "(none)",
                "n_legal_values_sbytes": len(moves.get("sbytes", [])), "n_legal_values_spkts": len(moves.get("spkts", [])),
                "single_feature_moves": sum(len(v) for v in moves.values()),
                "joint_lattice_candidates": len(built),
                "valid_candidates": len(valid_rows),
                "valid_consistent_with_checked_rules": sum(1 for r, _ in valid_rows if r["status"] == "consistent_with_checked_rules"),
                "valid_unverified_dependencies": sum(1 for r, _ in valid_rows if r["status"] == "unverified_dependencies"),
                "rejected": len(built) - len(valid_rows),
                "rejection_reasons": "; ".join(f"{k} ({n})" for k, n in sorted(reasons.items())),
                "max_derived_change": json.dumps({k: round(v, 4) for k, v in sorted(derived_max.items())}),
                "current_v1_single_moves": current_v1_single_moves(original, budget),
            }
            for key in models:
                row[f"p_attack_original_{key}"] = p_orig[key]
                row[f"eligible_{key}"] = bool(p_orig[key] >= 0.5)
                if key in probs:
                    row[f"valid_still_attack_{key}"] = int((probs[key] >= 0.5).sum())
                    row[f"min_p_attack_{key}"] = float(probs[key].min())
                else:
                    row[f"valid_still_attack_{key}"] = 0
                    row[f"min_p_attack_{key}"] = None
            summary_rows.append(row)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "feasibility_summary.csv", index=False)
    pd.DataFrame(lattice_rows).to_csv(output_dir / "legal_values.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(output_dir / "lattice_candidates.csv", index=False)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    metadata = {
        "script": "experiments/unsw_step6_v2_feasibility.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "policy": pr.policy_description(),
        "samples": list(V2_SAMPLES), "budgets": list(BUDGETS), "models": list(MODELS),
        "model_sha256": hashes,
        "official_test_used": False,
        "lime_used": False,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8", newline="\n")
    return {"summary": summary, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    out = run(Path(args.output_dir))
    pd.set_option("display.width", 250)
    cols = ["validation_position", "proto", "budget", "editable_features", "single_feature_moves", "joint_lattice_candidates",
            "valid_candidates", "valid_consistent_with_checked_rules", "valid_unverified_dependencies",
            "valid_still_attack_random_forest", "valid_still_attack_xgboost", "current_v1_single_moves"]
    print(out["summary"][cols].to_string(index=False))
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
