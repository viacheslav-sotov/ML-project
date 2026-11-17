"""Export the reference tables requested for review: feature taxonomy, selected flows and seed families.

Read-only. Uses only code constants, the protocols, the shared sample manifest and the sub-train/validation
split (to describe the selected flows); loads no model and never touches the official test set.

    python experiments/export_reference_tables.py [--output-dir results/reference]

Outputs
    feature_taxonomy.csv   42 features: validator category and rules (src/unsw/constraints.py), teammate
                           table status, role under physical_v1 (editable / repaired / excluded + reason),
                           unverified dependency (physical_report_v2) and the checked physical rules using it
    selected_flows.csv     the 20 shared validation flows, which ones Step 6 v1 and v2 use, attack type,
                           flow type and key features
    seed_families.csv      every seed used by Step 6 v1 and v2 and its purpose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import unsw_constraint_reporting as cr  # noqa: E402
from experiments import unsw_step6_manifest as mf  # noqa: E402
from src.unsw import physical_repair as pr  # noqa: E402
from src.unsw.constraints import FEATURE_METADATA, NUMERIC_MIN_MAX  # noqa: E402

DEFAULT_OUTPUT_DIR = ROOT / "results" / "reference"
SHARED_SAMPLES = ROOT / "results" / "metrics" / "shap" / "shared_validation_sample_identifiers.csv"
TEAMMATE_TABLE = ROOT / "results" / "unsw_nb15" / "network_feature_constraints.csv"


def feature_taxonomy() -> pd.DataFrame:
    teammate = pd.read_csv(TEAMMATE_TABLE).set_index("Feature")
    unverified = pr.unverified_dependencies_v2()
    rows = []
    for meta in FEATURE_METADATA:
        f = meta["feature"]
        if f in pr.PHYSICAL_V1_EDITABLE:
            role, reason = "editable (increase only)", ""
        elif f in pr.REPAIRED_FEATURES:
            role, reason = "repaired", pr.REPAIRED_FEATURES[f]
        elif f in pr.PHYSICAL_V1_EXCLUDED:
            role, reason = "excluded", pr.PHYSICAL_V1_EXCLUDED[f]
        else:
            role, reason = "fixed", f"not directly controllable ({meta['constraint_category']})"
        rules = [name for name, inputs, _rule, _support in cr.CHECKED_RULES if f in inputs]
        tm = teammate.loc[f] if f in teammate.index else None
        rows.append({
            "feature": f,
            "data_type": meta["data_type"],
            "validator_category": meta["constraint_category"],
            "attacker_controllability": meta["attacker_controllability"],
            "integer_required": meta["integer_required"],
            "observed_min": NUMERIC_MIN_MAX.get(f, (None, None))[0],
            "observed_max": NUMERIC_MIN_MAX.get(f, (None, None))[1],
            "related_features": ";".join(meta.get("related_features") or []),
            "validator_dependency_rule": meta.get("dependency_rule", ""),
            "teammate_can_change": None if tm is None else tm["Can Change?"],
            "teammate_constraint": None if tm is None else tm["Constraint"],
            "physical_v1_role": role,
            "physical_v1_reason": reason,
            "unverified_dependency_v2": unverified.get(f, ""),
            "checked_physical_rules": " | ".join(rules),
        })
    return pd.DataFrame(rows)


def selected_flows(v1: dict, v2: dict) -> pd.DataFrame:
    from src.unsw.xai_setup import get_xai_data

    split = get_xai_data()
    X_val, y_val = split["X_val"], split["y_val"]
    shared = pd.read_csv(SHARED_SAMPLES)
    rows = []
    for rec in shared.itertuples():
        pos = int(rec.validation_position)
        x = X_val.iloc[pos]
        rows.append({
            "validation_position": pos,
            "official_train_row_index": int(rec.original_training_row_index),
            "selection_seed": int(rec.seed),
            "label": int(y_val.iloc[pos]),
            "proto": x["proto"], "service": x["service"], "state": x["state"],
            "spkts": int(x["spkts"]), "dpkts": int(x["dpkts"]), "sbytes": int(x["sbytes"]), "sttl": int(x["sttl"]),
            "flow_type": "tcp_session" if x["proto"] == "tcp" else ("short_no_reply" if int(x["dpkts"]) == 0 else "other"),
            "in_step6_v1": pos in v1["sample_positions"],
            "in_step6_v2": pos in v2["sample_positions"],
        })
    out = pd.DataFrame(rows)
    from src.unsw.xai_setup import load_shared_samples
    cats = load_shared_samples(split).set_index("validation_position")["attack_cat"]
    out.insert(4, "attack_cat", out["validation_position"].map(cats))
    return out


def seed_families(v1: dict, v2: dict) -> pd.DataFrame:
    rows = []

    def add(protocol_version, family, seeds, purpose):
        rows.append({"protocol": protocol_version, "family": family, "seeds": ";".join(str(s) for s in seeds), "purpose": purpose})

    for p in (v1, v2):
        version = p["protocol_version"]
        add(version, "lime_search_seed", [p["lime_search_seed"]], "LIME seed the search optimizes (single-seed objective)")
        add(version, "heldout_lime_seeds", p["heldout_lime_seeds"], "held-out LIME seeds for the success test; never seen by the search")
        add(version, "part_b_lime_seeds", p["part_b_lime_seeds"], "Part B LIME stability seeds (comparison with v1/Part B only)")
        add(version, "order_seeds", p["order_seeds"], "feature/direction order of shuffled and multi-start greedy")
        add(version, "control_seeds", p["control_seeds"], "random-control RNG")
    for name, spec in v2["search_objectives"].items():
        add(v2["protocol_version"], f"search_objective:{name}", spec["lime_seeds"], spec["aggregation"])
    add(v2["protocol_version"], "defense_lime_seeds", v2["defense_lime_seeds"], "DEF-1 stabilized (5-seed averaged) LIME")
    add("both", "split_seed", [42], "train_test_split(official_train, 0.20, stratify=label, random_state=42)")
    add("both", "sample_selection_seed", [42], "shared 20-sample validation manifest")
    add("both", "shap_background_seed", [42], "SHAP background rows (default_rng(42))")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    v1, v2 = mf.load_protocol(mf.PROTOCOL_PATH), mf.load_protocol(mf.PROTOCOL_V2_PATH)
    for name, frame in (("feature_taxonomy.csv", feature_taxonomy()),
                        ("selected_flows.csv", selected_flows(v1, v2)),
                        ("seed_families.csv", seed_families(v1, v2))):
        frame.to_csv(out / name, index=False, lineterminator="\n")
        print(f"{name}: {len(frame)} rows")
    (out / "README.md").write_text(
        "# Reference tables\n\nGenerated by `python experiments/export_reference_tables.py` (read-only; no model, "
        "no official test set).\n\n| File | Content |\n|---|---|\n"
        "| `feature_taxonomy.csv` | 42 features: validator category, teammate table, physical_v1 role, unverified dependencies, checked rules |\n"
        "| `selected_flows.csv` | 20 shared validation flows; which are used by Step 6 v1 (3) and v2 (10) |\n"
        "| `seed_families.csv` | every seed family of Step 6 v1/v2 and its purpose |\n",
        encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
