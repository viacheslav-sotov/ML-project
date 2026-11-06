"""Three-level constraint reporting for spoofing candidates (reporting only; changes no policy).

1. experimental_validity  - the CURRENT experimental policy: src/unsw/constraints.validate_candidate()
                            plus the perturbation budget. This is what the searches enforce.
2. teammate_table         - compliance with the stricter, human-readable feature-editability table
                            results/unsw_nb15/network_feature_constraints.csv (Ghiday branch, 414dda2).
                            Reported only; NOT enforced.
3. physical_consistency   - dependent network features that can actually be checked from the 42
                            features, using rules documented in the teammate table. A candidate is
                            never labelled "physically valid": the best status is
                            "consistent_with_checked_rules", and any changed feature whose
                            dependencies cannot be checked yields "unverified_dependencies".

src/unsw/constraints.py is not modified and remains the experimental baseline.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.unsw.constraints import FEATURE_METADATA, validate_candidate  # noqa: E402

TEAMMATE_TABLE_PATH = ROOT / "results" / "unsw_nb15" / "network_feature_constraints.csv"
CONSTRAINTS_PY_PATH = ROOT / "src" / "unsw" / "constraints.py"
EXPERIMENTAL_POLICY_NAME = "src/unsw/constraints.py validate_candidate + budget A/B (Step 5b policy)"

ATTACKABLE_NUMERIC_FEATURES = [
    row["feature"] for row in FEATURE_METADATA
    if row["constraint_category"] == "DIRECTLY_CONTROLLABLE" and row["feature"] not in {"proto", "service", "state"}
]

TEAMMATE_STATUS = ("compliant", "needs_review", "non_compliant")
PHYSICAL_STATUS = ("consistent_with_checked_rules", "unverified_dependencies", "violates_checked_rule")
TCP_ONLY_FEATURES = ("swin", "dwin", "stcpb", "dtcpb", "synack", "ackdat", "tcprtt")
REL_TOL = 0.01  # the teammate table states the rate/load formulas hold "within 1%"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _num(row: Mapping[str, Any], feature: str) -> float:
    return float(row[feature])


def changed_features(original: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[str]:
    return [f for f in ATTACKABLE_NUMERIC_FEATURES if _num(candidate, f) != _num(original, f)]


# ---------------------------------------------------------------------------------------------
# 1. Experimental validity (current policy)
# ---------------------------------------------------------------------------------------------
def experimental_validity(original: Mapping[str, Any], candidate: Mapping[str, Any], budget_ok: bool, budget_reason: str | None) -> dict[str, Any]:
    validation = validate_candidate(dict(original), dict(candidate))
    reasons = list(validation["violated_constraints"])
    if not budget_ok:
        reasons.append(budget_reason or "budget violation")
    return {
        "experimental_policy": EXPERIMENTAL_POLICY_NAME,
        "experimental_valid": bool(validation["valid"] and budget_ok),
        "experimental_reasons": "; ".join(reasons),
        "validator_warnings": "; ".join(validation["warnings"]),
    }


# ---------------------------------------------------------------------------------------------
# 2. Teammate feature-editability table (reported, not enforced)
# ---------------------------------------------------------------------------------------------
def load_teammate_table(path: Path = TEAMMATE_TABLE_PATH) -> dict[str, dict[str, str]]:
    if not Path(path).exists():
        raise FileNotFoundError(f"teammate constraint table not found: {path}")
    table = pd.read_csv(path)
    required = {"Feature", "Can Change?", "Constraint", "Depends On", "Notes"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"teammate table is missing columns {sorted(missing)}")
    return {str(r["Feature"]): {k: str(r[k]) for k in required} for _, r in table.iterrows()}


def teammate_feature_status(rating: str, direction: int) -> tuple[str, str]:
    """Map a 'Can Change?' rating and the edit direction (+1 increase, -1 decrease) to a status."""
    text = rating.strip().lower()
    if text == "yes" or text.startswith("yes (via"):
        return "compliant", rating
    if text.startswith("yes (increase only)"):
        if direction < 0:
            return "non_compliant", f"{rating}: decreased"
        return ("needs_review", rating) if "needs review" in text else ("compliant", rating)
    if text.startswith("needs review"):
        return "needs_review", rating
    # "No", "Limited", "Indirectly (derived)", "Indirectly (other flows)" and anything unknown.
    return "non_compliant", rating


def teammate_table_compliance(original: Mapping[str, Any], candidate: Mapping[str, Any], table: dict[str, dict[str, str]]) -> dict[str, Any]:
    details = []
    worst = "compliant"
    order = {status: i for i, status in enumerate(TEAMMATE_STATUS)}
    for feature in changed_features(original, candidate):
        if feature not in table:
            status, why = "needs_review", "feature not in teammate table"
        else:
            direction = 1 if _num(candidate, feature) > _num(original, feature) else -1
            status, why = teammate_feature_status(table[feature]["Can Change?"], direction)
        details.append(f"{feature}={status} ({why})")
        if order[status] > order[worst]:
            worst = status
    return {
        "teammate_table_status": worst,
        "teammate_table_details": "; ".join(details),
        "teammate_table_enforced": False,
    }


# ---------------------------------------------------------------------------------------------
# 3. Physical consistency of dependent network features
# ---------------------------------------------------------------------------------------------
def _close(actual: float, expected: float, rel_tol: float = REL_TOL, abs_tol: float = 1e-9) -> bool:
    return abs(actual - expected) <= max(abs_tol, rel_tol * abs(expected))


def _rate_rule(r):
    dur = _num(r, "dur")
    expected = 0.0 if dur == 0 else (_num(r, "spkts") + _num(r, "dpkts") - 1.0) / dur
    return _close(_num(r, "rate"), expected)


def _sload_rule(r):
    dur = _num(r, "dur")
    return True if dur == 0 else _close(_num(r, "sload"), (_num(r, "sbytes") - _num(r, "smean")) * 8.0 / dur)


def _dload_rule(r):
    dur = _num(r, "dur")
    return True if dur == 0 else _close(_num(r, "dload"), (_num(r, "dbytes") - _num(r, "dmean")) * 8.0 / dur)


def _rounds_to(value: float, x: float) -> bool:
    """value == round(x); at an exact .5 either neighbour is accepted (rounding convention unknown)."""
    lower = float(int(x // 1))
    if abs((x - lower) - 0.5) < 1e-9:
        return value in (lower, lower + 1.0)
    return value == float(round(x))


def _smean_rule(r):
    spkts = _num(r, "spkts")
    return spkts > 0 and _rounds_to(_num(r, "smean"), _num(r, "sbytes") / spkts)


def _dmean_rule(r):
    dpkts = _num(r, "dpkts")
    if dpkts == 0:
        return _num(r, "dmean") == 0.0
    return _rounds_to(_num(r, "dmean"), _num(r, "dbytes") / dpkts)


def _tcp_only_zero_rule(r):
    if str(r["proto"]) == "tcp":
        return True
    return all(_num(r, f) == 0.0 for f in TCP_ONLY_FEATURES)


# (name, inputs, check, support quoted from the teammate table)
CHECKED_RULES: list[tuple[str, tuple[str, ...], Callable[[Mapping[str, Any]], bool], str]] = [
    ("tcprtt = synack + ackdat", ("synack", "ackdat", "tcprtt"),
     lambda r: abs(_num(r, "tcprtt") - (_num(r, "synack") + _num(r, "ackdat"))) <= 1e-9, "100% of rows"),
    ("TCP-only fields are 0 for non-TCP flows", TCP_ONLY_FEATURES + ("proto",), _tcp_only_zero_rule, "100% of rows"),
    ("smean = round(sbytes / spkts)", ("sbytes", "spkts", "smean"), _smean_rule, "100% of rows"),
    ("dmean = round(dbytes / dpkts), 0 if dpkts = 0", ("dbytes", "dpkts", "dmean"), _dmean_rule, "100% of rows"),
    ("rate = (spkts + dpkts - 1) / dur (0 if dur = 0), within 1%", ("spkts", "dpkts", "dur", "rate"), _rate_rule,
     "100% of rows with dur > 0; 99.85% for dur = 0"),
    ("sload = (sbytes - smean) * 8 / dur, within 1%", ("sbytes", "smean", "dur", "sload"), _sload_rule, "99.98% of rows with dur > 0"),
    ("dload = (dbytes - dmean) * 8 / dur, within 1%", ("dbytes", "dmean", "dur", "dload"), _dload_rule, "99.98% of rows with dur > 0"),
    ("sloss <= spkts", ("sloss", "spkts"), lambda r: _num(r, "sloss") <= _num(r, "spkts"), "99.99% of rows"),
    ("dloss <= dpkts", ("dloss", "dpkts"), lambda r: _num(r, "dloss") <= _num(r, "dpkts"), "100% of rows"),
    ("spkts >= 1", ("spkts",), lambda r: _num(r, "spkts") >= 1, "100% of rows"),
    ("sbytes >= 28 * spkts (observed bound, not a protocol rule)", ("sbytes", "spkts"),
     lambda r: _num(r, "sbytes") >= 28.0 * _num(r, "spkts"), "observed minimum sbytes/spkts = 28"),
]

# Dependencies that exist but cannot be checked from the 42 features with a verified formula.
UNVERIFIED_DEPENDENCIES: dict[str, str] = {
    "sinpkt": "sinpkt relation to dur/spkts unverified (sinpkt*(spkts-1)/1000 ~= dur holds for only 74.7% of rows)",
    "dinpkt": "dinpkt relation to dur/dpkts unverified",
    "sjit": "sjit formula (from sinpkt, spkts) unverified",
    "djit": "djit formula (from dinpkt, dpkts) unverified",
    "sttl": "ct_state_ttl is determined by (state, sttl, dttl); mapping not checked here",
    "dttl": "ct_state_ttl is determined by (state, sttl, dttl); mapping not checked here",
    "trans_depth": "trans_depth > 0 is service-conditioned (mostly http); not checked",
    "response_body_len": "response_body_len > 0 is service-conditioned (mostly http); not checked",
    "dur": "dur also constrains sinpkt/sjit timing (unverified relation)",
}


def physical_consistency(original: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    changed = set(changed_features(original, candidate))
    violated, original_violates, checked = [], [], []
    for name, inputs, rule, support in CHECKED_RULES:
        holds_original = bool(rule(original))
        holds_candidate = bool(rule(candidate))
        checked.append(name)
        if not holds_original:
            original_violates.append(name)
            continue
        if not holds_candidate:
            violated.append(f"{name} [{support}]")
    unverified = [f"{f}: {UNVERIFIED_DEPENDENCIES[f]}" for f in sorted(changed) if f in UNVERIFIED_DEPENDENCIES]
    if violated:
        status = "violates_checked_rule"
    elif unverified:
        status = "unverified_dependencies"
    else:
        status = "consistent_with_checked_rules"
    return {
        "physical_status": status,
        "physical_violations": "; ".join(violated),
        "physical_unverified": "; ".join(unverified),
        "physical_rules_checked": len(checked),
        "physical_rules_failing_on_original": "; ".join(original_violates),
        "physically_valid_claimed": False,
    }


# ---------------------------------------------------------------------------------------------
def constraint_report(
    original: Mapping[str, Any],
    candidate: Mapping[str, Any],
    budget_ok: bool,
    budget_reason: str | None = None,
    table: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    table = load_teammate_table() if table is None else table
    report = {"changed_features": ";".join(changed_features(original, candidate))}
    report.update(experimental_validity(original, candidate, budget_ok, budget_reason))
    report.update(teammate_table_compliance(original, candidate, table))
    report.update(physical_consistency(original, candidate))
    return report


def policy_versions() -> dict[str, str]:
    return {
        "experimental_policy": EXPERIMENTAL_POLICY_NAME,
        "constraints_py_sha256": file_sha256(CONSTRAINTS_PY_PATH),
        "teammate_table_path": str(TEAMMATE_TABLE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "teammate_table_sha256": file_sha256(TEAMMATE_TABLE_PATH),
        "teammate_table_enforced": "no (reporting only)",
        "physical_rules": "; ".join(name for name, _, _, _ in CHECKED_RULES),
    }
