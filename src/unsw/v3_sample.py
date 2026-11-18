"""Loader for the frozen protocol-v3 attack sample. The only supported way to read the sample.

It refuses the file unless the CSV SHA-256 and the sample digest equal the values pinned in
experiments/step6_protocol_v3.json, the digest is not a known discarded draw, and the identity columns are sound.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = ROOT / "results" / "v3_prep" / "attack_sample"
PROTOCOL_V3 = ROOT / "experiments" / "step6_protocol_v3.json"
# Draws that must never be used: the first draw excluded row positions instead of official ids (Nov 18).
DISCARDED_SAMPLE_DIGESTS = {"93a0f50e163f0c71d1d30a275ad26ad76322e5d4df94f8f6ac1cb00de6cd3522"}


class SampleError(RuntimeError):
    pass


def sample_digest(sample: pd.DataFrame) -> str:
    payload = "\n".join(f"{r.id}|{r.vector_sha256}|{r.attack_cat}|{r.tier}" for r in sample.itertuples())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_frozen_sample(protocol: dict | None = None, sample_dir: Path = SAMPLE_DIR) -> pd.DataFrame:
    protocol = json.loads(PROTOCOL_V3.read_text(encoding="utf-8")) if protocol is None else protocol
    pins = protocol["attack_sample"]
    csv = Path(sample_dir) / "sample_manifest.csv"
    if hashlib.sha256(csv.read_bytes()).hexdigest() != pins["sample_manifest_csv_sha256"]:
        raise SampleError(f"{csv} does not match the pinned SHA-256")
    sample = pd.read_csv(csv, float_precision="round_trip")
    digest = sample_digest(sample)
    if digest in DISCARDED_SAMPLE_DIGESTS:
        raise SampleError("this is a discarded draw and must not be used")
    if digest != pins["sample_sha256"] or len(sample) != pins["flows"]:
        raise SampleError("sample digest or size differs from the protocol pin")
    if not (sample["id"].is_unique and sample["group_id"].is_unique and sample["vector_sha256"].is_unique):
        raise SampleError("sample flows are not unique by id / group / feature vector")
    if set(sample["id"]) & set(protocol["attack_sample"]["excluded_v2_pilot_official_ids"]):
        raise SampleError("a v2 pilot flow is in the sample")
    return sample
