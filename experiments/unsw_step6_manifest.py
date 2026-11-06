"""Step 6 experimental manifest: validate the predeclared protocol and resolve it into a
machine-readable manifest with hashes, sample IDs and software versions.

- experiments/step6_protocol.json is the predeclared protocol (seeds, budgets, strategies, rules).
- build_manifest() resolves it against the repository: model hashes, dataset hashes labelled by their
  ROLE (the two UNSW files are swapped by name in this repository), split sizes, sample IDs, source
  hashes of the objective/metric implementations, constraint-policy hashes, software versions, git commit.
- validate_protocol() / validate_manifest() return a list of problems (empty = valid).
- write_manifest() writes to a new, unique file and never overwrites.

Usage:
    python experiments/unsw_step6_manifest.py --validate            # check protocol + resolve, print summary
    python experiments/unsw_step6_manifest.py --write               # also write results/step6_manifests/manifest_<ts>.json
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_PATH = ROOT / "experiments" / "step6_protocol.json"
MANIFEST_ROOT = ROOT / "results" / "step6_manifests"
KNOWN_MODELS = ("random_forest", "xgboost")
KNOWN_PERTURBATION_BUDGETS = ("A", "B")
REQUIRED_STRATEGIES = ("greedy_baseline_ref_v1", "greedy_shuffled_v1", "multistart_greedy_v1", "random_control")
SOFTWARE = ("numpy", "pandas", "scikit-learn", "scipy", "xgboost", "lime", "shap", "joblib")
OFFICIAL_TRAIN_ROWS = 175_341
OFFICIAL_TEST_ROWS = 82_332


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256(obj: Any) -> str:
    return hashlib.sha256(inspect.getsource(obj).encode("utf-8")).hexdigest()


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_protocol(protocol: dict[str, Any]) -> list[str]:
    problems: list[str] = []

    def need(key: str) -> Any:
        if key not in protocol:
            problems.append(f"missing key: {key}")
            return None
        return protocol[key]

    models = need("models") or []
    for m in models:
        if m not in KNOWN_MODELS:
            problems.append(f"unknown model {m!r}")
    for b in need("perturbation_budgets") or []:
        if b not in KNOWN_PERTURBATION_BUDGETS:
            problems.append(f"unknown perturbation budget {b!r}")
    samples = need("sample_positions") or []
    if len(set(samples)) != len(samples) or not all(isinstance(s, int) and s >= 0 for s in samples):
        problems.append("sample_positions must be distinct non-negative integers")
    budgets = need("explanation_budgets") or []
    if not budgets or budgets != sorted(set(budgets)) or not all(isinstance(b, int) and b >= 2 for b in budgets):
        problems.append("explanation_budgets must be strictly increasing integers >= 2 (baseline + at least one candidate)")
    search = need("lime_search_seed")
    heldout = need("heldout_lime_seeds") or []
    part_b = need("part_b_lime_seeds") or []
    order = need("order_seeds") or []
    control = need("control_seeds") or []
    if search in heldout:
        problems.append("lime_search_seed must not be one of the heldout_lime_seeds")
    if len(set(heldout)) != len(heldout) or len(heldout) < 2:
        problems.append("heldout_lime_seeds must contain at least two distinct seeds")
    if search is not None and search not in part_b:
        problems.append("part_b_lime_seeds should contain the search seed (Part B's original seed set)")
    for name, seeds in (("order_seeds", order), ("control_seeds", control)):
        if not seeds or len(set(seeds)) != len(seeds) or not all(isinstance(s, int) for s in seeds):
            problems.append(f"{name} must be a non-empty list of distinct integers")
    if set(order) & set(control):
        problems.append("order_seeds and control_seeds overlap")
    lime_seeds = set(heldout) | set(part_b) | ({search} if search is not None else set())
    if (set(order) | set(control)) & lime_seeds:
        problems.append("search-order/control seeds overlap LIME seeds (keep the seed families disjoint)")
    max_attempts = need("max_random_attempts")
    if not isinstance(max_attempts, int) or (budgets and max_attempts < 2 * max(budgets)):
        problems.append("max_random_attempts must be a fixed integer >= 2 * max explanation budget (needed for prefix equivalence)")
    strategies = need("strategies") or {}
    for name in REQUIRED_STRATEGIES:
        if name not in strategies:
            problems.append(f"missing strategy {name}")
    multi = strategies.get("multistart_greedy_v1", {})
    if multi:
        if multi.get("prefix_shortcut") is not False:
            problems.append("multistart_greedy_v1 must use separate runs per budget (prefix_shortcut=false): fair-share depends on the budget")
        if not isinstance(multi.get("restarts"), int) or multi.get("restarts") < 1:
            problems.append("multistart_greedy_v1.restarts must be a positive integer")
        if multi.get("fair_share") is not True:
            problems.append("multistart_greedy_v1.fair_share must be true (predeclared)")
        if multi.get("include_canonical_first") is not False:
            problems.append("multistart_greedy_v1.include_canonical_first must be false (predeclared)")
    for name, spec in strategies.items():
        seeded_by = spec.get("seeded_by")
        if seeded_by not in (None, "order_seeds", "control_seeds"):
            problems.append(f"{name}: unknown seeded_by {seeded_by!r}")
    for key in ("objective", "heldout_metrics", "constraint_policy"):
        need(key)
    return problems


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unavailable"


def software_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in SOFTWARE:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def build_manifest(protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve the protocol against the repository (reads hashes and the validation split; no training)."""
    from experiments import unsw_targeted_spoofing as ts
    from experiments.unsw_constraint_reporting import policy_versions
    from experiments.unsw_spoofing_explanations import pair_metrics
    from src.unsw.xai_setup import get_xai_data, load_shared_samples

    protocol = load_protocol() if protocol is None else protocol
    problems = validate_protocol(protocol)
    if problems:
        raise ValueError("protocol is invalid: " + "; ".join(problems))

    split = get_xai_data()
    shared = load_shared_samples(split).set_index("validation_position")
    samples = []
    for pos in protocol["sample_positions"]:
        if pos not in shared.index:
            raise ValueError(f"sample {pos} is not in the shared manifest")
        row = shared.loc[pos]
        samples.append({
            "validation_position": int(pos),
            "original_training_row_index": int(row["original_training_row_index"]),
            "official_id": int(row["official_id"]),
            "attack_cat": str(row["attack_cat"]),
            "label": int(row["label"]),
        })

    raw = ROOT / "data" / "raw"
    files = {name: raw / name for name in ("UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv")}
    rows = {name: sum(1 for _ in path.open("rb")) - 1 for name, path in files.items()}
    roles = {}
    for name, n in rows.items():
        role = "official_train" if n == OFFICIAL_TRAIN_ROWS else "official_test" if n == OFFICIAL_TEST_ROWS else "unknown"
        roles[role] = {"file": f"data/raw/{name}", "rows": n, "sha256": file_sha256(files[name])}

    models = {}
    for key in protocol["models"]:
        path = ts.MODEL_DIR / f"{key}_validation_pipeline.joblib"
        models[key] = {"file": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": file_sha256(path)}

    return {
        "manifest_version": 1,
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode("utf-8")).hexdigest(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {"commit": _git("rev-parse", "HEAD"), "branch": _git("branch", "--show-current"),
                "dirty": bool(_git("status", "--porcelain", "--untracked-files=no"))},
        "protocol": protocol,
        "models": models,
        "datasets": {
            "by_role": roles,
            "note": "role assigned by row count; the file names in data/raw are swapped relative to their roles",
            "official_test_used": False,
        },
        "split": {
            "method": "train_test_split(official_train, test_size=0.20, stratify=label, random_state=42)",
            "subtrain_rows": int(len(split["X_subtrain"])),
            "validation_rows": int(len(split["X_val"])),
            "feature_order": list(ts.FEATURE_ORDER),
        },
        "samples": samples,
        "seeds": {
            "lime_search_seed": protocol["lime_search_seed"],
            "heldout_lime_seeds": protocol["heldout_lime_seeds"],
            "part_b_lime_seeds": protocol["part_b_lime_seeds"],
            "order_seeds": protocol["order_seeds"],
            "control_seeds": protocol["control_seeds"],
        },
        "lime_configuration": ts.explanation_config(),
        "implementation_hashes": {
            "objective_score": source_sha256(ts.objective_score),
            "pair_metrics_part_b": source_sha256(pair_metrics),
            "mutation_proposal_v1": source_sha256(ts.mutation_proposal_v1),
            "targeted_search_script": file_sha256(ROOT / "experiments" / "unsw_targeted_spoofing.py"),
            "heldout_script": file_sha256(ROOT / "experiments" / "unsw_targeted_heldout.py"),
        },
        "constraint_policy": policy_versions(),
        "software": software_versions(),
    }


def validate_manifest(manifest: dict[str, Any], check_files: bool = True) -> list[str]:
    problems = validate_protocol(manifest.get("protocol", {}))
    for key in ("models", "datasets", "split", "samples", "seeds", "implementation_hashes", "constraint_policy", "software", "git"):
        if key not in manifest:
            problems.append(f"manifest missing {key}")
    if problems:
        return problems
    protocol = manifest["protocol"]
    if manifest["protocol_sha256"] != hashlib.sha256(json.dumps(protocol, sort_keys=True).encode("utf-8")).hexdigest():
        problems.append("protocol_sha256 does not match the embedded protocol")
    if [s["validation_position"] for s in manifest["samples"]] != protocol["sample_positions"]:
        problems.append("resolved samples differ from the protocol")
    if sorted(manifest["models"]) != sorted(protocol["models"]):
        problems.append("resolved models differ from the protocol")
    roles = manifest["datasets"]["by_role"]
    if set(roles) != {"official_train", "official_test"}:
        problems.append(f"dataset roles not resolved: {sorted(roles)}")
    if manifest["datasets"].get("official_test_used") is not False:
        problems.append("official_test_used must be false")
    if check_files:
        for key, entry in manifest["models"].items():
            if file_sha256(ROOT / entry["file"]) != entry["sha256"]:
                problems.append(f"model {key} hash changed")
        for role, entry in roles.items():
            if file_sha256(ROOT / entry["file"]) != entry["sha256"]:
                problems.append(f"dataset {role} hash changed")
    return problems


def write_manifest(manifest: dict[str, Any], root: Path = MANIFEST_ROOT) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path, suffix = root / f"manifest_{stamp}.json", 1
    while True:
        try:
            with path.open("x", encoding="utf-8") as fh:  # "x": fail instead of overwriting
                json.dump(manifest, fh, indent=2, sort_keys=True)
            return path
        except FileExistsError:
            path = root / f"manifest_{stamp}_{suffix:02d}.json"
            suffix += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validate", action="store_true", help="validate the protocol, resolve it and print a summary")
    parser.add_argument("--write", action="store_true", help="also write a new manifest file under results/step6_manifests")
    args = parser.parse_args()
    protocol = load_protocol()
    problems = validate_protocol(protocol)
    if problems:
        raise SystemExit("protocol invalid:\n- " + "\n- ".join(problems))
    manifest = build_manifest(protocol)
    problems = validate_manifest(manifest)
    if problems:
        raise SystemExit("manifest invalid:\n- " + "\n- ".join(problems))
    print(json.dumps({k: manifest[k] for k in ("protocol_version", "git", "models", "datasets", "samples", "seeds", "software")}, indent=2))
    if args.write:
        print(f"written: {write_manifest(manifest)}")


if __name__ == "__main__":
    main()
