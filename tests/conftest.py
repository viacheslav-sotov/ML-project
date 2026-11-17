import pytest

from src.unsw.data import OFFICIAL_TEST_FILE_NAME, OFFICIAL_TEST_PIN
from src.unsw.official_test_lock import install_guard

# The official test partition must never be opened by the test suite: any attempt raises before the file is opened.
install_guard()


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: uses real LIME explanations and saved model files; exclude with -m \"not slow\"",
    )


@pytest.fixture(autouse=True)
def _pinned_official_test_hash(monkeypatch):
    """unsw_targeted_spoofing.run_experiment records the SHA-256 of both raw files in its run metadata.

    That module is hash-pinned in the existing manifests, so it is not edited. In tests its file_sha256 returns
    the recorded identity for the locked file instead of reading it (same value the real hash would give).
    """
    import sys
    ts = sys.modules.get("experiments.unsw_targeted_spoofing")
    if ts is None:
        return
    original = ts.file_sha256

    def pinned(path):
        from pathlib import Path
        if Path(path).name == OFFICIAL_TEST_FILE_NAME:
            return OFFICIAL_TEST_PIN["sha256"]
        return original(path)

    monkeypatch.setattr(ts, "file_sha256", pinned)
