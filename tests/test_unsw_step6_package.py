"""Results packaging: verification before copying, byte-identical copies, no source changes,
sensitive-content scan, checksums. Uses a fake runner run (no LIME, no training)."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from experiments import unsw_step6_package as pkg
from test_unsw_step6_runner import FakeModel, PILOT, fakes, manifest, original, run, split  # noqa: F401  (fixtures)


def _write_fake_heldout(run_dir: Path, mismatches: int = 0, drop_one: bool = False) -> Path:
    cp = pd.read_csv(run_dir / "checkpoints.csv")
    rows = cp[["task_id", "checkpoint"]].copy()
    if drop_one:
        rows = rows.iloc[1:]
    rows["heldout_top10_jaccard_mean"] = 1.0
    heldout = run_dir / "heldout" / "run_20990101_000000"
    heldout.mkdir(parents=True)
    rows.to_csv(heldout / "heldout_summary.csv", index=False)
    pd.DataFrame({"candidate_uid": ["x"], "seed": [22]}).to_csv(heldout / "heldout_per_seed.csv", index=False)
    pd.DataFrame({"sample_position": [2941], "band": ["heldout"]}).to_csv(heldout / "noise_bands.csv", index=False)
    (heldout / "run_metadata.json").write_text(json.dumps(
        {"search_score_check": {"checked": len(cp), "mismatches": mismatches, "max_gap": 0.0}}), encoding="utf-8")
    return heldout


@pytest.fixture
def finished_run(tmp_path, split, manifest, original, fakes):
    out = run(tmp_path / "runs", split, manifest, original)
    run_dir = Path(out["run_dir"])
    _write_fake_heldout(run_dir)
    return run_dir


def _tree_hashes(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def test_package_is_byte_identical_complete_and_verifiable(finished_run, tmp_path):
    target = pkg.build_package(finished_run, tmp_path / "shared")
    for name in pkg.RUN_FILES:
        assert (target / name).read_bytes() == (finished_run / name).read_bytes(), name
    heldout_src = next((finished_run / "heldout").glob("run_*"))
    for name in pkg.HELDOUT_FILES:
        assert (target / "heldout" / name).read_bytes() == (heldout_src / name).read_bytes(), name
    assert not (target / "tasks").exists()
    assert pkg.verify_package(target) == []
    text = (target / "PACKAGE.md").read_text(encoding="utf-8")
    assert "Not included" in text and "tasks/" in text and "raw_bytes" in text
    listed = [l.split("  ", 1)[1] for l in (target / "SHA256SUMS").read_text(encoding="utf-8").splitlines() if l and not l.startswith("#")]
    assert set(listed) == set(pkg.RUN_FILES) | {f"heldout/{n}" for n in pkg.HELDOUT_FILES} | {"PACKAGE.md"}


def test_source_run_is_never_modified(finished_run, tmp_path):
    before = _tree_hashes(finished_run)
    pkg.build_package(finished_run, tmp_path / "shared")
    assert _tree_hashes(finished_run) == before


def test_existing_package_is_never_overwritten(finished_run, tmp_path):
    pkg.build_package(finished_run, tmp_path / "shared")
    with pytest.raises(FileExistsError):
        pkg.build_package(finished_run, tmp_path / "shared")


def test_incomplete_run_is_refused(finished_run, tmp_path):
    status = json.loads((finished_run / "status.json").read_text(encoding="utf-8"))
    status["failed"] = ["x"]
    (finished_run / "status.json").write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(pkg.PackageError, match="incomplete"):
        pkg.build_package(finished_run, tmp_path / "shared")
    assert not (tmp_path / "shared").exists()


@pytest.mark.parametrize("planted", ["C:\\Users\\someone\\secret.txt", "contact someone@example.com", "ghp_ABCDEFGHIJKLMNOP", "/home/someone/data"])
def test_sensitive_content_blocks_packaging(finished_run, tmp_path, planted):
    readme = finished_run / "README.md"
    readme.write_bytes(readme.read_bytes() + planted.encode("utf-8"))
    with pytest.raises(pkg.PackageError, match="sensitive"):
        pkg.build_package(finished_run, tmp_path / "shared")
    assert not (tmp_path / "shared" / finished_run.name).exists()


def test_heldout_must_cover_every_checkpoint_without_score_mismatches(tmp_path, split, manifest, original, fakes):
    for kwargs, message in (({"drop_one": True}, "cover every checkpoint"), ({"mismatches": 1}, "search-score check")):
        out = run(tmp_path / f"runs_{message[:5]}", split, manifest, original)
        _write_fake_heldout(Path(out["run_dir"]), **kwargs)
        with pytest.raises(pkg.PackageError, match=message):
            pkg.build_package(out["run_dir"], tmp_path / "shared")


def test_tampered_manifest_is_refused(finished_run, tmp_path):
    path = finished_run / "manifest.json"
    path.write_bytes(path.read_bytes().replace(b'"manifest_version": 1', b'"manifest_version": 9'))
    with pytest.raises(pkg.PackageError, match="manifest"):
        pkg.build_package(finished_run, tmp_path / "shared")


def test_legacy_crlf_manifest_is_accepted_and_documented(finished_run, tmp_path):
    path = finished_run / "manifest.json"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))  # as written by runner v1 on Windows
    target = pkg.build_package(finished_run, tmp_path / "shared")
    assert "lf_normalized" in (target / "PACKAGE.md").read_text(encoding="utf-8")
    assert (target / "manifest.json").read_bytes() == path.read_bytes()  # historical bytes preserved
    assert pkg.verify_package(target) == []


def test_verification_detects_changes_and_extra_files(finished_run, tmp_path):
    target = pkg.build_package(finished_run, tmp_path / "shared")
    (target / "results.csv").write_bytes((target / "results.csv").read_bytes() + b"x\n")
    (target / "notes.txt").write_text("extra", encoding="utf-8")
    problems = pkg.verify_package(target)
    assert "checksum mismatch results.csv" in problems and "unlisted file notes.txt" in problems


def test_checksums_ignore_line_endings_only(finished_run, tmp_path):
    target = pkg.build_package(finished_run, tmp_path / "shared")
    csv = target / "results.csv"
    csv.write_bytes(csv.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))  # git autocrlf checkout
    assert pkg.verify_package(target) == []
