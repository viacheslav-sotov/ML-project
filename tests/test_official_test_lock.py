"""The official UNSW-NB15 test partition is never opened by the legacy (v1/v2) code paths or the v3 code.

Synthetic tests (always run): the guard blocks every way of opening the locked file name, and the default data
loaders read only the training partition. Real-data tests (skipped when data/raw is absent) show that the
legacy split, the manifests and the v3 data are identical to what was recorded before the change.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.unsw import data
from src.unsw.official_test_lock import LOCKED_FILE_NAME, OfficialTestLocked, is_installed

ROOT = Path(__file__).resolve().parents[1]
REAL_TRAIN = ROOT / "data" / "raw" / data.OFFICIAL_TRAIN_FILE_NAME
needs_data = pytest.mark.skipif(not REAL_TRAIN.exists(), reason="raw official training partition not present")


# ---------------------------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------------------------
def test_guard_is_active_for_the_whole_suite():
    assert is_installed()
    assert data.OFFICIAL_TEST_FILE_NAME == LOCKED_FILE_NAME


@pytest.mark.parametrize("opener", [
    lambda p: open(p, "rb"),
    lambda p: Path(p).open("rb"),
    lambda p: Path(p).read_bytes(),
    lambda p: pd.read_csv(p),
    lambda p: hashlib.sha256(Path(p).read_bytes()),
    lambda p: open(p, "w"),
])
def test_guard_refuses_every_open_of_the_locked_name(tmp_path, opener):
    target = tmp_path / LOCKED_FILE_NAME          # does not exist: the guard fires before any filesystem access
    with pytest.raises(OfficialTestLocked):
        opener(target)
    assert not target.exists()


def test_guard_refuses_aliases_of_a_locked_file(tmp_path):
    from src.unsw import official_test_lock as lock
    target = tmp_path / "locked_copy.csv"
    target.write_text("x\n")
    assert lock.lock_identity_of(target)          # from now on this file is locked under any name
    link = tmp_path / "alias.csv"
    os.link(target, link)                          # hard link: another name for the same file
    for path in (target, link, tmp_path / "." / "alias.csv"):
        with pytest.raises(OfficialTestLocked):
            open(path, "rb")


def test_real_locked_file_identity_is_registered_when_present():
    from src.unsw import official_test_lock as lock
    if lock.DEFAULT_LOCKED_PATH.exists():          # metadata only (os.stat); the file is not opened
        assert lock.identity_locked()


def test_guarded_launcher_installs_the_guard_before_the_script(tmp_path):
    import subprocess, sys
    probe = tmp_path / "probe.py"
    probe.write_text("from src.unsw.official_test_lock import is_installed\nprint('installed', is_installed())\n"
                     "try:\n    open('UNSW_NB15_training-set.csv')\nexcept PermissionError:\n    print('blocked')\n")
    out = subprocess.run([sys.executable, str(ROOT / "experiments" / "run_guarded.py"), str(probe)],
                         capture_output=True, text=True, cwd=tmp_path, check=True).stdout
    assert "installed True" in out and "blocked" in out


def test_guard_ignores_other_files(tmp_path):
    other = tmp_path / data.OFFICIAL_TRAIN_FILE_NAME
    other.write_text("a\n1\n")
    assert other.read_text() == "a\n1\n"


# ---------------------------------------------------------------------------------------------
# Default loaders read only the training partition (synthetic raw directory)
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_raw(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    rng = np.random.default_rng(0)
    n = data.OFFICIAL_TRAIN_ROWS
    label = rng.integers(0, 2, n)
    pd.DataFrame({
        "id": np.arange(1, n + 1), "dur": rng.random(n).round(6), "proto": rng.choice(["tcp", "udp"], n),
        "service": rng.choice(["-", "http"], n), "state": rng.choice(["FIN", "INT"], n),
        "attack_cat": np.where(label == 1, "Generic", "Normal"), "label": label,
    }).to_csv(raw / data.OFFICIAL_TRAIN_FILE_NAME, index=False)
    return raw


def test_default_loaders_never_touch_the_test_partition(synthetic_raw, monkeypatch):
    monkeypatch.setattr(data, "RAW_DIR", synthetic_raw)
    train, test = data.load_official_data()
    assert test is None and len(train) == data.OFFICIAL_TRAIN_ROWS
    split = data.create_data_split()
    assert not {"X_test", "y_test", "official_test_df"} & set(split)
    assert (len(split["X_subtrain"]), len(split["X_val"])) == (140_272, 35_069)


def test_train_only_split_equals_the_original_recipe(synthetic_raw, monkeypatch):
    monkeypatch.setattr(data, "RAW_DIR", synthetic_raw)
    split = data.create_data_split()
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(synthetic_raw / data.OFFICIAL_TRAIN_FILE_NAME)
    X, y = data.prepare_features_target(df)
    X_sub, X_val, _, _ = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    assert split["X_subtrain"].index.equals(X_sub.index) and split["X_val"].index.equals(X_val.index)
    pd.testing.assert_frame_equal(split["X_train_full"], X)


def test_loading_the_test_partition_requires_an_explicit_opt_in_and_is_blocked_here(synthetic_raw):
    with pytest.raises(OfficialTestLocked):
        data.load_official_data(include_official_test=True)


def test_wrong_training_file_is_refused(tmp_path, monkeypatch):
    pd.DataFrame({"label": [0, 1], "attack_cat": ["Normal", "Generic"]}).to_csv(tmp_path / data.OFFICIAL_TRAIN_FILE_NAME, index=False)
    monkeypatch.setattr(data, "RAW_DIR", tmp_path)
    with pytest.raises(ValueError, match="rows"):
        data.load_official_train()


# ---------------------------------------------------------------------------------------------
# Real data: identical to what was recorded before the change
# ---------------------------------------------------------------------------------------------
RECORDED_MANIFESTS = [
    ROOT / "results" / "step6_v2" / "prerun_manifest" / "manifest_prerun_v2.json",
    ROOT / "results" / "step6_g4_runs" / "g4_diagnostic_pilot_20261117_192235" / "manifest.json",
]


@needs_data
def test_legacy_split_is_unchanged_and_has_no_test_keys():
    from experiments import unsw_step6_runner as runner
    from src.unsw.xai_setup import get_xai_data, load_shared_samples
    split = get_xai_data()
    assert not {"X_test", "y_test", "official_test_df"} & set(split)
    assert (len(split["X_subtrain"]), len(split["X_val"])) == (140_272, 35_069)
    shared = load_shared_samples(runner.guarded_split(split))
    for _, row in shared.iterrows():   # every shared flow still maps to the same official row
        assert split["X_val"].index[int(row["validation_position"])] == int(row["original_training_row_index"])


@needs_data
@pytest.mark.parametrize("recorded", RECORDED_MANIFESTS, ids=["prerun_v2", "g4"])
def test_fresh_manifest_datasets_equal_the_recorded_ones(recorded):
    from experiments import unsw_step6_manifest as mf
    saved = json.loads(recorded.read_text(encoding="utf-8"))
    fresh = mf.build_manifest(saved["protocol"])
    assert fresh["datasets"] == saved["datasets"]
    assert list(fresh["datasets"]["by_role"]) == list(saved["datasets"]["by_role"])
    assert fresh["split"] == saved["split"] and fresh["samples"] == saved["samples"]


@needs_data
def test_recorded_manifest_still_validates_with_files():
    from experiments import unsw_step6_manifest as mf
    saved = json.loads(RECORDED_MANIFESTS[0].read_text(encoding="utf-8"))
    assert mf.validate_manifest(saved, check_files=True) == []
    tampered = json.loads(json.dumps(saved))
    tampered["datasets"]["by_role"]["official_test"]["sha256"] = "0" * 64
    assert any("official_test" in p for p in mf.validate_manifest(tampered, check_files=True))


@needs_data
def test_v3_training_data_digest_is_unchanged():
    from experiments import unsw_nested_cv as ncv
    bench = ROOT / "results" / "v3_runs" / "benchmark_20261117_215046" / "run_config.json"
    recorded = json.loads(bench.read_text(encoding="utf-8"))["header"]
    v3 = ncv.load_v3_train(ncv.load_protocol())
    assert ncv.data_digest(v3) == recorded["data_digest"]
    assert v3["split_digest"] == recorded["split_digest"]
