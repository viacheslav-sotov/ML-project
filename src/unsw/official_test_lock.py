"""Process-wide guard that refuses any attempt to open the official UNSW-NB15 test partition.

install_guard() registers a Python audit hook (PEP 578). Every file open in this process (builtins.open,
io.open, pathlib.Path.open, os.open, pandas.read_csv, hashing helpers, imports) raises an "open" audit event
*before* the file is opened. The hook refuses the open when
  - the file name is the official test partition's (UNSW_NB15_training-set.csv; the names in data/raw are
    swapped), compared case-insensitively, or
  - the target is the same file as the locked one under another name: a hard link, a symbolic link, a Windows
    8.3 short name (e.g. UNSW_N~1.CSV) or an alternate path. Identity is compared by (st_dev, st_ino) from
    os.stat, a metadata query that does not read the file's contents.

Audit hooks cannot be removed once added; the guard stays active for the rest of the process.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

LOCKED_FILE_NAME = "UNSW_NB15_training-set.csv"
DEFAULT_LOCKED_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / LOCKED_FILE_NAME
_installed = False
_locked_ids: set[tuple[int, int]] = set()


class OfficialTestLocked(PermissionError):
    pass


def _identity(path) -> tuple[int, int] | None:
    try:
        st = os.stat(path)
    except (OSError, ValueError, TypeError):
        return None
    return (st.st_dev, st.st_ino) if st.st_ino else None


def _is_locked(path) -> bool:
    if isinstance(path, int):  # an already-open file descriptor
        return False
    try:
        name = os.path.basename(os.fsdecode(path))
    except TypeError:
        return False
    if name.lower() == LOCKED_FILE_NAME.lower():
        return True
    return bool(_locked_ids) and _identity(path) in _locked_ids


def _hook(event: str, args) -> None:
    if event == "open" and args and _is_locked(args[0]):
        raise OfficialTestLocked(f"the official test partition ({LOCKED_FILE_NAME}) is locked and must not be opened")


def lock_identity_of(path) -> bool:
    """Also refuse any other name for this file (hard links, symlinks, short names). Metadata only."""
    ident = _identity(path)
    if ident is not None:
        _locked_ids.add(ident)
    return ident is not None


def install_guard(locked_path=DEFAULT_LOCKED_PATH) -> None:
    global _installed
    if not _installed:
        lock_identity_of(locked_path)
        sys.addaudithook(_hook)
        _installed = True


def is_installed() -> bool:
    return _installed


def identity_locked() -> bool:
    return bool(_locked_ids)
