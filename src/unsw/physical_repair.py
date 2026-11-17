"""Physical constraint policy `physical_v1` for Step 6 v2 (docs/step6_v2_decisions.md, D2).

`current_v1` (the v1 policy: src/unsw/constraints.validate_candidate + budget) is not defined or
changed here; the searches keep using it unchanged by default.

physical_v1 in short
    editable   sbytes and spkts, increase only (all other features fixed or recomputed)
    repaired   after every edit, derived features whose formula holds for >= 99.9% of sub-train rows
               (results/step6_v2/physical_formula_check/formula_support.csv): smean, dmean, rate,
               sload, dload, tcprtt
    moves      a fixed lattice of values per feature: original + k * 1% * max(1, |original|),
               k = 1 .. cap/1% (Budget A: 10, Budget B: 25), rounded for integer features, de-duplicated,
               within the per-feature cap and the observed range. Targeted searches and the random
               control use exactly the same lattice (matched legal moves).
    rejected   any candidate that edits a non-editable feature, decreases a feature, leaves a derived
               feature stale, breaks a zero-coupling rule, newly violates one of the checked physical
               rules of experiments/unsw_constraint_reporting.py, or leaves the observed value range.
    reported   dependencies that exist but cannot be verified from the 42 features (e.g. sinpkt/sjit
               after an spkts edit). They do not make a candidate invalid, but its status is then
               "unverified_dependencies", never "consistent_with_checked_rules".

A formula with 99.9% support is evidence, not proof of physical validity; "consistent_with_checked_rules"
is the strongest status any candidate can get. Nothing here claims physical validity.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import pandas as pd

from src.unsw.constraints import FEATURE_METADATA, FEATURE_ORDER, INTEGER_FEATURES, NUMERIC_MIN_MAX

POLICY_CURRENT = "current_v1"
POLICY_PHYSICAL = "physical_v1"
POLICIES = (POLICY_CURRENT, POLICY_PHYSICAL)

# Budget caps (must equal unsw_targeted_spoofing.BUDGET_POLICY; checked by a test).
MAX_CHANGED = {"A": 5, "B": 10}
MAX_RELATIVE = {"A": 0.10, "B": 0.25}
AGGREGATE_CAP = {"A": float("inf"), "B": 1.5}
GRID_STEP = 0.01  # 1% of max(1, |original value|)

PHYSICAL_V1_EDITABLE: tuple[str, ...] = ("sbytes", "spkts")
PHYSICAL_V1_DIRECTION = {"sbytes": 1, "spkts": 1}  # increase only (teammate table)
PHYSICAL_V1_EXCLUDED: dict[str, str] = {
    "dur": "excluded: sinpkt/sjit cannot be repaired (sinpkt = 1000*dur/(spkts-1) holds for 74.7% of sub-train rows)",
    "sttl": "excluded: alternative (state, sttl, dttl) combinations have too little support (1 and 8 rows)",
    "dttl": "excluded: affects ct_state_ttl; not approved",
    "sloss": "excluded until its dependencies are verified",
    "swin": "excluded until its dependencies are verified",
    "stcpb": "excluded until its dependencies are verified",
    "ackdat": "excluded until its dependencies are verified",
    "sinpkt": "excluded: no verified formula",
    "dinpkt": "excluded: no verified formula",
    "sjit": "excluded: no verified formula",
    "djit": "excluded: no verified formula",
    "trans_depth": "excluded: service-conditioned, not checkable",
    "response_body_len": "excluded: service-conditioned, not checkable",
    "synack": "excluded: teammate table 'No'",
    "dtcpb": "excluded: teammate table 'No'",
    "dwin": "excluded: teammate table 'No'",
    "dloss": "excluded: teammate table 'No'",
    "dpkts": "excluded: 'Limited' (server-controlled)",
    "dbytes": "excluded: 'Limited' (server-controlled)",
}

# Derived features that may change, and only through repair (formula support on sub-train).
REPAIRED_FEATURES: dict[str, str] = {
    "smean": "smean = round(sbytes / spkts) [100%]",
    "dmean": "dmean = round(dbytes / dpkts), 0 if dpkts = 0 [100%]",
    "rate": "rate = (spkts + dpkts - 1) / dur, dur > 0 [100%]",
    "sload": "sload = (sbytes - smean) * 8 / dur, dur > 0 [99.975%]",
    "dload": "dload = (dbytes - dmean) * 8 / dur, dur > 0 [99.989%]",
    "tcprtt": "tcprtt = synack + ackdat [100%]",
}

# Dependencies that exist for an editable feature but cannot be verified or repaired.
UNVERIFIED_FOR_EDIT: dict[str, str] = {
    "spkts": "sinpkt/sjit (mean inter-packet time and jitter) depend on spkts and dur; "
             "sinpkt = 1000*dur/(spkts-1) holds for only 74.7% of sub-train rows, so they stay unchanged",
}

DPKTS_ZERO_COUPLED = ("dbytes", "dmean", "dload", "dinpkt", "djit", "dloss")   # 100% support
TCP_ONLY = ("swin", "dwin", "stcpb", "dtcpb", "synack", "ackdat", "tcprtt")      # 100% support
_CATEGORY = {row["feature"]: row["constraint_category"] for row in FEATURE_METADATA}


class PhysicalRepairError(ValueError):
    """A candidate whose derived features cannot be recomputed (e.g. dur = 0)."""


def _f(row: Mapping[str, Any], feature: str) -> float:
    return float(row[feature])


def _rel(original: float, value: float) -> float:
    return abs(value - original) / max(1.0, abs(original))


# ---------------------------------------------------------------------------------------------
# Legal moves
# ---------------------------------------------------------------------------------------------
def legal_values(original_row: Mapping[str, Any], feature: str, budget_key: str) -> list[float]:
    """Distinct values the feature may take under physical_v1 (ascending, original excluded)."""
    if feature not in PHYSICAL_V1_EDITABLE:
        return []
    cap = MAX_RELATIVE[budget_key]
    original = _f(original_row, feature)
    unit = max(1.0, abs(original))
    hi = NUMERIC_MIN_MAX[feature][1]
    values: list[float] = []
    for k in range(1, int(round(cap / GRID_STEP)) + 1):
        value = original + PHYSICAL_V1_DIRECTION[feature] * k * GRID_STEP * unit
        if feature in INTEGER_FEATURES:
            value = float(round(value))
        if value > hi or value == original or _rel(original, value) > cap + 1e-12:
            continue
        if value not in values:
            values.append(value)
    return sorted(values)


def legal_moves(original_row: Mapping[str, Any], budget_key: str) -> dict[str, list[float]]:
    """Editable features with at least one legal value -> their legal values."""
    moves = {f: legal_values(original_row, f, budget_key) for f in PHYSICAL_V1_EDITABLE}
    return {f: v for f, v in moves.items() if v}


def moves_digest(moves: Mapping[str, list[float]]) -> str:
    return hashlib.sha256(json.dumps({k: list(v) for k, v in sorted(moves.items())}).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------------------------
def repair(original_row: pd.Series, candidate_row: pd.Series) -> pd.Series:
    """Recompute derived features whose inputs differ from the original. Raises PhysicalRepairError."""
    row = candidate_row.copy()

    def changed(*features: str) -> bool:
        return any(_f(row, f) != _f(original_row, f) for f in features)

    dur = _f(row, "dur")
    if changed("sbytes", "spkts"):
        spkts = _f(row, "spkts")
        if spkts <= 0:
            raise PhysicalRepairError("smean undefined for spkts = 0")
        row["smean"] = float(round(_f(row, "sbytes") / spkts))
    if changed("dbytes", "dpkts"):
        dpkts = _f(row, "dpkts")
        row["dmean"] = 0.0 if dpkts == 0 else float(round(_f(row, "dbytes") / dpkts))
    if changed("spkts", "dpkts", "dur"):
        if dur <= 0:
            raise PhysicalRepairError("rate cannot be repaired for dur = 0 (rule support 99.86%)")
        row["rate"] = (_f(row, "spkts") + _f(row, "dpkts") - 1.0) / dur
    if changed("sbytes", "smean", "dur"):
        if dur <= 0:
            raise PhysicalRepairError("sload cannot be repaired for dur = 0")
        row["sload"] = (_f(row, "sbytes") - _f(row, "smean")) * 8.0 / dur
    if changed("dbytes", "dmean", "dur"):
        if dur <= 0:
            raise PhysicalRepairError("dload cannot be repaired for dur = 0")
        row["dload"] = (_f(row, "dbytes") - _f(row, "dmean")) * 8.0 / dur
    if changed("synack", "ackdat"):
        row["tcprtt"] = _f(row, "synack") + _f(row, "ackdat")
    return row


def lattice_move(current_row: pd.Series, original_row: pd.Series, feature: str, offset: int,
                 values: list[float]) -> pd.Series:
    """Move `feature` `offset` lattice positions up from its current position, then repair.

    Returns the unchanged current row when the move leaves the lattice or cannot be repaired
    (callers count it as "no change").
    """
    current = _f(current_row, feature)
    position = 0 if current == _f(original_row, feature) else values.index(current) + 1
    target = position + int(offset)
    if offset < 1 or target > len(values):
        return current_row.copy()
    proposal = current_row.copy()
    proposal[feature] = values[target - 1]
    try:
        return repair(original_row, proposal)
    except PhysicalRepairError:
        return current_row.copy()


# ---------------------------------------------------------------------------------------------
# Verification of a finished candidate
# ---------------------------------------------------------------------------------------------
def _zero_coupling_violations(row: Mapping[str, Any]) -> list[str]:
    problems = []
    if _f(row, "dpkts") == 0:
        problems += [f"{f} must be 0 when dpkts = 0" for f in DPKTS_ZERO_COUPLED if _f(row, f) != 0]
    if str(row["proto"]) != "tcp":
        problems += [f"{f} must be 0 for non-TCP flows" for f in TCP_ONLY if _f(row, f) != 0]
    return problems


def check_candidate(original_row: pd.Series, candidate_row: pd.Series, budget_key: str) -> dict[str, Any]:
    """Full physical_v1 verdict for a candidate (primary and derived changes checked)."""
    from experiments.unsw_constraint_reporting import CHECKED_RULES  # same rules as the reporting

    violations: list[str] = []
    primary, derived, other = {}, {}, []
    for feature in FEATURE_ORDER:
        a, b = original_row[feature], candidate_row[feature]
        if a == b or (isinstance(a, (int, float)) and isinstance(b, (int, float)) and float(a) == float(b)):
            continue
        if feature in PHYSICAL_V1_EDITABLE:
            primary[feature] = _rel(float(a), float(b))
            if (float(b) - float(a)) * PHYSICAL_V1_DIRECTION[feature] < 0:
                violations.append(f"{feature} decreased (increase only)")
        elif feature in REPAIRED_FEATURES:
            derived[feature] = _rel(float(a), float(b))
        else:
            other.append(feature)
    violations += [f"non-editable feature changed: {f} ({_CATEGORY.get(f)})" for f in other]

    # Budget on primary features only.
    if len(primary) > MAX_CHANGED[budget_key]:
        violations.append(f"more than {MAX_CHANGED[budget_key]} edited features")
    if primary and max(primary.values()) > MAX_RELATIVE[budget_key] + 1e-12:
        violations.append(f"edit exceeds {MAX_RELATIVE[budget_key]:.0%} of the original value")
    if sum(primary.values()) > AGGREGATE_CAP[budget_key] + 1e-12:
        violations.append("aggregate relative change exceeds the budget")

    # Derived features must equal their repaired values (never stale, never edited directly).
    try:
        base = candidate_row.copy()
        for feature in REPAIRED_FEATURES:
            base[feature] = original_row[feature]  # recompute from the edited inputs only
        expected = repair(original_row, base)
        for feature in REPAIRED_FEATURES:
            if float(expected[feature]) != float(candidate_row[feature]):
                violations.append(f"{feature} is not consistent with its repair formula")
    except PhysicalRepairError as exc:
        violations.append(str(exc))

    # Observed value range.
    for feature in list(primary) + list(derived):
        lo, hi = NUMERIC_MIN_MAX[feature]
        if not lo <= float(candidate_row[feature]) <= hi:
            violations.append(f"{feature} outside the observed range [{lo}, {hi}]")

    # Zero-coupling and checked physical rules: anything that held on the original must still hold.
    original_coupling = set(_zero_coupling_violations(original_row))
    violations += [p for p in _zero_coupling_violations(candidate_row) if p not in original_coupling]
    for name, _inputs, rule, _support in CHECKED_RULES:
        if bool(rule(original_row)) and not bool(rule(candidate_row)):
            violations.append(f"checked rule violated: {name}")

    unverified = [f"{f}: {UNVERIFIED_FOR_EDIT[f]}" for f in primary if f in UNVERIFIED_FOR_EDIT]
    if violations:
        status = "violates_checked_rule"
    elif unverified:
        status = "unverified_dependencies"
    else:
        status = "consistent_with_checked_rules"
    return {
        "valid": not violations,
        "status": status,
        "violations": violations,
        "unverified": unverified,
        "primary_changes": primary,
        "derived_changes": derived,
        "physically_valid_claimed": False,
    }


def policy_description() -> dict[str, Any]:
    return {
        "policy": POLICY_PHYSICAL,
        "editable": list(PHYSICAL_V1_EDITABLE),
        "direction": {f: ("increase" if d > 0 else "decrease") for f, d in PHYSICAL_V1_DIRECTION.items()},
        "grid_step": GRID_STEP,
        "repaired": REPAIRED_FEATURES,
        "excluded": PHYSICAL_V1_EXCLUDED,
        "unverified_for_edit": UNVERIFIED_FOR_EDIT,
        "zero_coupling": {"dpkts = 0": list(DPKTS_ZERO_COUPLED), "non-TCP": list(TCP_ONLY)},
    }
