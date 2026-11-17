"""Read-only check of the physical formulas proposed for the Step 6 v2 repair policy.

For every candidate repair formula and zero-coupling rule, count how many rows of the data satisfy it.
Only formulas with support >= REPAIR_THRESHOLD (99.9%) on the sub-training split may be used to
repair derived features in v2 (docs/step6_v2_decisions.md, D2a/D2b). The script also lists, for the
ten v2 samples, which rules hold on the original flow and which edits look feasible
(TTL values seen in the data, packet-count rounding, duration/inter-packet-time coupling).

Reads only the sub-training and validation splits; the official test set is never touched.
Loads no model, computes no explanation and changes no file except its own output folder.

    python experiments/unsw_physical_formula_check.py [--output-dir results/step6_v2/physical_formula_check]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT_DIR = ROOT / "results" / "step6_v2" / "physical_formula_check"
REPAIR_THRESHOLD = 0.999   # support needed before a formula may be used for repair
REL_TOL = 0.01             # rate/load/inter-packet formulas: "within 1%" (as in unsw_constraint_reporting)
ABS_TOL = 1e-9
V2_SAMPLES = (2941, 3040, 3206, 15137, 27094, 4407, 30107, 34186, 15741, 25740)
BUDGET_MAX_RELATIVE = {"A": 0.10, "B": 0.25}
TCP_ONLY = ("swin", "dwin", "stcpb", "dtcpb", "synack", "ackdat", "tcprtt")
DPKTS_ZERO_COUPLED = ("dbytes", "dmean", "dload", "dinpkt", "djit", "dloss")
SPKTS_LE1_COUPLED = ("sinpkt", "sjit")
SPLITS_USED = ("X_subtrain", "X_val")


def _close(actual: pd.Series, expected: pd.Series) -> pd.Series:
    return (actual - expected).abs() <= np.maximum(ABS_TOL, REL_TOL * expected.abs())


def _rounds_to(value: pd.Series, x: pd.Series) -> pd.Series:
    """value == round(x); at an exact .5 either neighbour is accepted (same convention as the reporter)."""
    lower = np.floor(x)
    tie = (x - lower - 0.5).abs() < 1e-9
    return (value == np.round(x)) | (tie & ((value == lower) | (value == lower + 1)))


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.where(den != 0, np.nan)


# (rule_id, group, description, applies(df) -> mask, holds(df) -> mask, used_for)
Rule = tuple[str, str, str, Callable[[pd.DataFrame], pd.Series], Callable[[pd.DataFrame], pd.Series], str]


def build_rules() -> list[Rule]:
    rules: list[Rule] = [
        ("R01", "derived", "smean = round(sbytes / spkts)", lambda d: d["spkts"] > 0,
         lambda d: _rounds_to(d["smean"], _ratio(d["sbytes"], d["spkts"])), "repair smean"),
        ("R02", "derived", "dmean = round(dbytes / dpkts), 0 if dpkts = 0", lambda d: pd.Series(True, index=d.index),
         lambda d: np.where(d["dpkts"] == 0, d["dmean"] == 0, _rounds_to(d["dmean"], _ratio(d["dbytes"], d["dpkts"]))), "repair dmean"),
        ("R03", "derived", "rate = (spkts + dpkts - 1) / dur, within 1% (dur > 0)", lambda d: d["dur"] > 0,
         lambda d: _close(d["rate"], (d["spkts"] + d["dpkts"] - 1) / d["dur"]), "repair rate"),
        ("R03b", "derived", "rate = 0 when dur = 0", lambda d: d["dur"] == 0,
         lambda d: d["rate"] == 0, "repair rate (dur = 0)"),
        ("R04", "derived", "sload = (sbytes - smean) * 8 / dur, within 1% (dur > 0)", lambda d: d["dur"] > 0,
         lambda d: _close(d["sload"], (d["sbytes"] - d["smean"]) * 8.0 / d["dur"]), "repair sload"),
        ("R05", "derived", "dload = (dbytes - dmean) * 8 / dur, within 1% (dur > 0)", lambda d: d["dur"] > 0,
         lambda d: _close(d["dload"], (d["dbytes"] - d["dmean"]) * 8.0 / d["dur"]), "repair dload"),
        ("R06", "derived", "tcprtt = synack + ackdat", lambda d: pd.Series(True, index=d.index),
         lambda d: (d["tcprtt"] - (d["synack"] + d["ackdat"])).abs() <= ABS_TOL, "repair tcprtt"),
        ("R07", "derived", "ct_state_ttl is determined by (state, sttl, dttl)", lambda d: pd.Series(True, index=d.index),
         _ttl_deterministic, "repair ct_state_ttl by lookup"),
        ("R08", "timing", "sinpkt = 1000 * dur / (spkts - 1), within 1% (spkts >= 2)", lambda d: d["spkts"] >= 2,
         lambda d: _close(d["sinpkt"], 1000.0 * d["dur"] / (d["spkts"] - 1)), "repair sinpkt when dur changes"),
        ("R08b", "timing", "sinpkt = 1000 * dur, within 1% (spkts = 2 only)", lambda d: d["spkts"] == 2,
         lambda d: _close(d["sinpkt"], 1000.0 * d["dur"]), "repair sinpkt for 2-packet flows"),
        ("R09", "timing", "dinpkt = 1000 * dur / (dpkts - 1), within 1% (dpkts >= 2)", lambda d: d["dpkts"] >= 2,
         lambda d: _close(d["dinpkt"], 1000.0 * d["dur"] / (d["dpkts"] - 1)), "repair dinpkt when dur changes"),
        ("R12", "timing", "sjit = 0 when spkts = 2", lambda d: d["spkts"] == 2,
         lambda d: d["sjit"] == 0, "dur edit on 2-packet flows leaves sjit = 0"),
    ]
    for i, feature in enumerate(DPKTS_ZERO_COUPLED, start=1):
        rules.append((f"R10.{i}", "zero_coupling", f"{feature} = 0 when dpkts = 0", lambda d: d["dpkts"] == 0,
                      lambda d, f=feature: d[f] == 0, "zero-coupling"))
    for i, feature in enumerate(SPKTS_LE1_COUPLED, start=1):
        rules.append((f"R11.{i}", "zero_coupling", f"{feature} = 0 when spkts <= 1", lambda d: d["spkts"] <= 1,
                      lambda d, f=feature: d[f] == 0, "zero-coupling"))
    for i, feature in enumerate(TCP_ONLY, start=1):
        rules.append((f"R13.{i}", "zero_coupling", f"{feature} = 0 for non-TCP flows", lambda d: d["proto"].astype(str) != "tcp",
                      lambda d, f=feature: d[f] == 0, "zero-coupling"))
    rules += [
        ("R14", "bounds", "sloss <= spkts", lambda d: pd.Series(True, index=d.index), lambda d: d["sloss"] <= d["spkts"], "info (sloss excluded)"),
        ("R15", "bounds", "dloss <= dpkts", lambda d: pd.Series(True, index=d.index), lambda d: d["dloss"] <= d["dpkts"], "info (dloss excluded)"),
        ("R16", "bounds", "sbytes >= 28 * spkts (observed bound)", lambda d: pd.Series(True, index=d.index),
         lambda d: d["sbytes"] >= 28 * d["spkts"], "info"),
    ]
    return rules


def _ttl_deterministic(d: pd.DataFrame) -> pd.Series:
    counts = d.groupby(["state", "sttl", "dttl"], observed=True)["ct_state_ttl"].transform("nunique")
    return counts == 1


def check_rules(frame: pd.DataFrame, split_name: str, rules: list[Rule] | None = None) -> pd.DataFrame:
    rows = []
    for rule_id, group, description, applies, holds, used_for in (rules or build_rules()):
        mask = pd.Series(np.asarray(applies(frame), dtype=bool), index=frame.index)
        ok = pd.Series(np.asarray(holds(frame), dtype=bool), index=frame.index) & mask
        n = int(mask.sum())
        k = int(ok.sum())
        share = (k / n) if n else float("nan")
        rows.append({
            "rule_id": rule_id, "group": group, "rule": description, "split": split_name,
            "applicable_rows": n, "satisfied_rows": k, "violating_rows": n - k, "support": share,
            "meets_repair_threshold": bool(n and share >= REPAIR_THRESHOLD), "used_for": used_for,
        })
    return pd.DataFrame(rows)


def ttl_lookup(frame: pd.DataFrame) -> pd.DataFrame:
    table = (frame.groupby(["state", "sttl", "dttl"], observed=True)
             .agg(ct_state_ttl_values=("ct_state_ttl", lambda s: ";".join(str(v) for v in sorted(s.unique()))),
                  n_values=("ct_state_ttl", "nunique"), rows=("ct_state_ttl", "size"))
             .reset_index())
    return table.sort_values(["state", "sttl", "dttl"]).reset_index(drop=True)


def sample_feasibility(samples: pd.DataFrame, positions, lookup: pd.DataFrame, rules: list[Rule]) -> pd.DataFrame:
    """Per sample: rules failing on the original, and which proposed edits look feasible."""
    rows = []
    for pos, (_, rec) in zip(positions, samples.iterrows()):
        one = rec.to_frame().T.infer_objects()
        failing = []
        for rule_id, _group, description, applies, holds, _used in rules:
            if rule_id == "R07":
                continue  # needs the whole table
            if bool(np.asarray(applies(one), dtype=bool)[0]) and not bool(np.asarray(holds(one), dtype=bool)[0]):
                failing.append(f"{rule_id} {description}")
        sttl, dttl, state = float(rec["sttl"]), float(rec["dttl"]), rec["state"]
        same_ctx = lookup[(lookup["state"] == state) & (lookup["dttl"] == dttl) & (lookup["n_values"] == 1)]
        own = same_ctx[same_ctx["sttl"] == sttl]
        out: dict[str, Any] = {
            "validation_position": int(pos), "proto": rec["proto"], "service": rec["service"], "state": state,
            "spkts": int(rec["spkts"]), "dpkts": int(rec["dpkts"]), "sbytes": int(rec["sbytes"]), "dur": float(rec["dur"]),
            "sttl": int(sttl), "dttl": int(dttl), "ct_state_ttl": int(rec["ct_state_ttl"]),
            "original_ttl_combo_in_lookup": bool(len(own) == 1),
            "rules_failing_on_original": "; ".join(failing),
        }
        for budget, cap in BUDGET_MAX_RELATIVE.items():
            unit = max(1.0, abs(sttl))
            alt = same_ctx[(same_ctx["sttl"] != sttl) & ((same_ctx["sttl"] - sttl).abs() / unit <= cap + 1e-12)]
            out[f"sttl_alternatives_{budget}"] = "; ".join(
                f"{int(r.sttl)}->ct_state_ttl {r.ct_state_ttl_values}" for r in alt.itertuples())
            out[f"spkts_plus1_allowed_{budget}"] = bool(1.0 / max(1.0, float(rec["spkts"])) <= cap + 1e-12)
        spkts = int(rec["spkts"])
        if spkts == 2:
            out["dur_edit_repairable"] = bool(not any(f.startswith(("R08b", "R12")) for f in failing))
        elif spkts > 2:
            out["dur_edit_repairable"] = bool(not any(f.startswith("R08 ") for f in failing))
        else:
            out["dur_edit_repairable"] = True  # spkts <= 1: no source inter-packet time to keep consistent
        rows.append(out)
    return pd.DataFrame(rows)


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def run(output_dir: Path = DEFAULT_OUTPUT_DIR, split: dict[str, Any] | None = None) -> dict[str, Any]:
    if split is None:
        from src.unsw.xai_setup import get_xai_data
        split = get_xai_data()
    frames = {name: split[name] for name in SPLITS_USED}  # only these two splits are read
    del split
    rules = build_rules()
    support = pd.concat([check_rules(frames["X_subtrain"], "subtrain", rules),
                         check_rules(frames["X_val"], "validation", rules)], ignore_index=True)
    lookup = ttl_lookup(frames["X_subtrain"])
    samples = frames["X_val"].iloc[list(V2_SAMPLES)]
    feasibility = sample_feasibility(samples, V2_SAMPLES, lookup, rules)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    support.to_csv(output_dir / "formula_support.csv", index=False)
    lookup.to_csv(output_dir / "ttl_lookup_subtrain.csv", index=False)
    feasibility.to_csv(output_dir / "sample_feasibility.csv", index=False)
    sub = support[support["split"] == "subtrain"]
    metadata = {
        "script": "experiments/unsw_physical_formula_check.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "splits_read": {name: int(len(frame)) for name, frame in frames.items()},
        "official_test_used": False,
        "repair_threshold": REPAIR_THRESHOLD,
        "relative_tolerance": REL_TOL,
        "rules_meeting_threshold_subtrain": sub.loc[sub["meets_repair_threshold"], "rule_id"].tolist(),
        "rules_below_threshold_subtrain": sub.loc[~sub["meets_repair_threshold"], "rule_id"].tolist(),
        "ttl_lookup_combinations": int(len(lookup)),
        "ttl_lookup_conflicting_combinations": int((lookup["n_values"] > 1).sum()),
        "samples": list(V2_SAMPLES),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8", newline="\n")
    return {"support": support, "lookup": lookup, "feasibility": feasibility, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    out = run(Path(args.output_dir))
    pd.set_option("display.width", 200)
    sub = out["support"][out["support"]["split"] == "subtrain"]
    print(sub[["rule_id", "rule", "applicable_rows", "violating_rows", "support", "meets_repair_threshold"]].to_string(index=False))
    print(json.dumps({k: out["metadata"][k] for k in ("splits_read", "rules_below_threshold_subtrain",
                                                     "ttl_lookup_combinations", "ttl_lookup_conflicting_combinations")}, indent=2))
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
