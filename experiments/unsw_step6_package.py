"""Build a small, analysis-ready package from a finished Step 6 run (read-only on the source run).

The package contains the consolidated CSVs, the held-out evaluation outputs, the manifest,
configuration, metadata and READMEs, copied byte-for-byte. The per-task raw files (tasks/*.json and
tasks/*.history.csv.gz) stay in the local run directory: every number in the package is derived from
them by the runner, and the experiment itself is reproducible from the code, manifest and seeds.

Before copying, the source run is verified:
  - status.json: completed == planned and no failed tasks;
  - results.csv has one row per planned task; checkpoints.csv has every planned checkpoint and all are valid;
  - the manifest matches its recorded SHA-256 (raw bytes, or LF-normalized for runner version 1);
  - the selected model files still match the manifest hashes;
  - exactly one held-out run, covering every checkpoint, with zero search-score mismatches;
  - no packaged file contains absolute personal paths, e-mail addresses or credential-like strings.

Checksums (SHA256SUMS) are computed on LF-normalized content so they verify on any platform, even
after git converts line endings. Usage:

    python experiments/unsw_step6_package.py --run-dir results/step6_minimum/run_20261106_203918
    python experiments/unsw_step6_package.py --verify results/step6_shared/run_20261106_203918
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import unsw_step6_manifest as mf  # noqa: E402
from experiments import unsw_step6_runner as runner  # noqa: E402

DEFAULT_OUTPUT_ROOT = ROOT / "results" / "step6_shared"
RUN_FILES = ("results.csv", "checkpoints.csv", "best_so_far.csv", "restarts.csv", "manifest.json",
             "run_config.json", "run_metadata.json", "status.json", "README.md")
HELDOUT_FILES = ("heldout_summary.csv", "heldout_per_seed.csv", "noise_bands.csv", "run_metadata.json")
SENSITIVE = re.compile(
    r"[A-Za-z]:\\\\Users\\\\|[A-Za-z]:\\Users\\|/Users/|/home/[^/\s]+/|[\w.+-]+@[\w-]+\.[\w.]+|"
    r"ghp_[A-Za-z0-9]{10,}|github_pat_|api[_-]?key|password|BEGIN (RSA|OPENSSH|PRIVATE)",
    re.IGNORECASE,
)


class PackageError(RuntimeError):
    pass


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def scan_sensitive(paths: list[Path]) -> list[str]:
    hits = []
    for path in paths:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        for match in SENSITIVE.finditer(text):
            hits.append(f"{Path(path).name}: ...{text[max(0, match.start() - 30):match.end() + 30]!r}...")
    return hits


def verify_run(run_dir: Path) -> dict[str, Any]:
    """Check the finished run before anything is copied; raise PackageError on any problem."""
    run_dir = Path(run_dir)
    for name in RUN_FILES:
        if not (run_dir / name).is_file():
            raise PackageError(f"missing {name} in {run_dir}")
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("failed") or status.get("completed") != status.get("planned"):
        raise PackageError(f"run is incomplete: {status}")
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    tasks = meta["tasks"]
    results = pd.read_csv(run_dir / "results.csv")
    checkpoints = pd.read_csv(run_dir / "checkpoints.csv")
    if len(results) != len(tasks) or not results["task_id"].is_unique or set(results["task_id"]) != {t["task_id"] for t in tasks}:
        raise PackageError(f"results.csv has {len(results)} rows for {len(tasks)} planned tasks")
    expected_checkpoints = sum(len(t["checkpoints"]) for t in tasks)
    if len(checkpoints) != expected_checkpoints or not checkpoints["checkpoint_valid"].astype(bool).all():
        raise PackageError(f"checkpoints.csv has {len(checkpoints)} rows (expected {expected_checkpoints}) or invalid rows")
    hash_status = runner.manifest_hash_status(run_dir / "manifest.json", config["manifest_sha256"])
    if hash_status is None or meta["manifest"]["sha256"] != config["manifest_sha256"]:
        raise PackageError("manifest.json does not match its recorded SHA-256")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    for key in meta["selection"]["models"]:
        entry = manifest["models"][key]
        if mf.file_sha256(ROOT / entry["file"]) != entry["sha256"]:
            raise PackageError(f"model file {entry['file']} no longer matches the manifest")
    heldout_runs = sorted(p for p in (run_dir / "heldout").glob("run_*") if p.is_dir()) if (run_dir / "heldout").exists() else []
    if len(heldout_runs) != 1:
        raise PackageError(f"expected exactly one held-out run under {run_dir / 'heldout'}, found {len(heldout_runs)}")
    heldout_dir = heldout_runs[0]
    for name in HELDOUT_FILES:
        if not (heldout_dir / name).is_file():
            raise PackageError(f"missing heldout file {name}")
    heldout = pd.read_csv(heldout_dir / "heldout_summary.csv")
    heldout_meta = json.loads((heldout_dir / "run_metadata.json").read_text(encoding="utf-8"))
    keys = set(zip(heldout["task_id"], heldout["checkpoint"]))
    if len(heldout) != len(checkpoints) or keys != set(zip(checkpoints["task_id"], checkpoints["checkpoint"])):
        raise PackageError("held-out summary does not cover every checkpoint exactly once")
    check = heldout_meta.get("search_score_check", {})
    if check.get("mismatches", 1) != 0 or check.get("checked") != len(checkpoints):
        raise PackageError(f"held-out search-score check failed: {check}")
    return {
        "run_dir": run_dir, "heldout_dir": heldout_dir, "status": status, "config": config, "meta": meta,
        "manifest": manifest, "manifest_hash_status": hash_status, "n_tasks": len(results),
        "n_checkpoints": len(checkpoints), "heldout_meta": heldout_meta,
    }


def _relative(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return Path(path).name


def build_package(run_dir: str | Path, output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> Path:
    info = verify_run(Path(run_dir))
    run_dir, heldout_dir = info["run_dir"], info["heldout_dir"]
    sources = [(run_dir / name, name) for name in RUN_FILES]
    sources += [(heldout_dir / name, f"heldout/{name}") for name in HELDOUT_FILES]
    hits = scan_sensitive([src for src, _ in sources])
    if hits:
        raise PackageError("sensitive content found; nothing was packaged:\n" + "\n".join(hits[:10]))

    target = Path(output_root) / run_dir.name
    target.mkdir(parents=True, exist_ok=False)  # never overwrite an existing package
    (target / "heldout").mkdir()
    for src, rel in sources:
        shutil.copyfile(src, target / rel)  # byte-for-byte copy; the source run is never modified
        if (target / rel).read_bytes() != src.read_bytes():
            raise PackageError(f"copy of {rel} differs from the source")

    tasks_dir = run_dir / "tasks"
    task_files = sorted(tasks_dir.glob("*")) if tasks_dir.exists() else []
    meta, manifest = info["meta"], info["manifest"]
    rows = "\n".join(f"| `{rel}` | {(target / rel).stat().st_size:,} |" for _, rel in sources)
    package_md = f"""# Step 6 results package: {run_dir.name}

Analysis-ready copy of a finished Step 6 run. Files are byte-for-byte copies of the source run;
checksums in `SHA256SUMS` are computed on LF-normalized content (platform-independent).

## Provenance

| Item | Value |
|---|---|
| Source run | `{_relative(run_dir)}` |
| Held-out run | `{_relative(heldout_dir)}` |
| Code commit (from manifest) | `{manifest['git']['commit']}` (dirty: {manifest['git']['dirty']}) |
| Protocol | {manifest['protocol_version']} |
| Manifest SHA-256 (recorded) | `{info['config']['manifest_sha256']}` |
| Manifest hash verification | {info['manifest_hash_status']} {'(runner v1 wrote CRLF on Windows and hashed the LF text; content identical)' if info['manifest_hash_status'] == 'lf_normalized' else ''} |
| Model SHA-256 | {', '.join(f"{k}: `{v}`" for k, v in meta['model_sha256'].items())} |
| Official test set used | {meta['official_test_used']} |
| Tasks / checkpoint candidates | {info['n_tasks']} / {info['n_checkpoints']} (all complete, all checkpoints valid) |
| Held-out search-score check | {info['heldout_meta']['search_score_check']} |
| Selection | `{json.dumps(meta['selection'], sort_keys=True)}` |
| Packaged at (UTC) | {datetime.now(timezone.utc).isoformat()} |

## Files

| File | Bytes |
|---|---|
{rows}

`README.md` (from the run) describes the columns. Comparisons at budget B: rows of
`checkpoints.csv` with `checkpoint == B`. Join held-out results on `task_id` + `checkpoint`.

## Not included

`tasks/` ({len(task_files)} files, {sum(p.stat().st_size for p in task_files):,} bytes): per-task JSON
payloads and complete candidate histories. They remain in the local source run; all packaged CSVs are
consolidated from them. Ask for them if a candidate's full search history is needed.

## Verify

`python experiments/unsw_step6_package.py --verify <this directory>`
"""
    (target / "PACKAGE.md").write_bytes(package_md.encode("utf-8"))
    sums = ["# sha256 of LF-normalized content  path"] + [f"{lf_sha256(target / rel)}  {rel}" for _, rel in sources]
    sums.append(f"{lf_sha256(target / 'PACKAGE.md')}  PACKAGE.md")
    (target / "SHA256SUMS").write_bytes(("\n".join(sums) + "\n").encode("utf-8"))
    return target


def verify_package(package_dir: str | Path) -> list[str]:
    package_dir = Path(package_dir)
    problems = []
    lines = [l for l in (package_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines() if l and not l.startswith("#")]
    listed = set()
    for line in lines:
        digest, rel = line.split("  ", 1)
        listed.add(rel)
        path = package_dir / rel
        if not path.is_file():
            problems.append(f"missing {rel}")
        elif lf_sha256(path) != digest:
            problems.append(f"checksum mismatch {rel}")
    extra = {str(p.relative_to(package_dir)).replace("\\", "/") for p in package_dir.rglob("*") if p.is_file()} - listed - {"SHA256SUMS"}
    problems += [f"unlisted file {rel}" for rel in sorted(extra)]
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir")
    group.add_argument("--verify")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    if args.verify:
        problems = verify_package(args.verify)
        if problems:
            raise SystemExit("ERROR:\n" + "\n".join(problems))
        print(f"OK: {args.verify} matches SHA256SUMS")
        return
    try:
        target = build_package(args.run_dir, args.output_root)
    except PackageError as exc:
        raise SystemExit(f"ERROR: {exc}")
    problems = verify_package(target)
    if problems:
        raise SystemExit("ERROR: package failed verification:\n" + "\n".join(problems))
    print(f"Package written and verified: {target}")


if __name__ == "__main__":
    main()
