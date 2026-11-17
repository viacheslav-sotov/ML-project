"""v1 baseline regression guard (Step 6 v2, decision D1).

The v1 experiment (protocol step6-v1, runs under results/step6_minimum and results/step6_shared) must
stay exactly as it is while v2 is built. These tests prove it:

fast (always run)
    - the v1 protocol file still matches the protocol hash recorded in the v1 manifest;
    - the source of objective_score, mutation_proposal_v1 and pair_metrics is unchanged (manifest hashes);
    - the v1 task planner still produces the same 123 task specs, in the same order;
    - the shared v1 package still matches its SHA256SUMS.
slow (real LIME, saved model; ~1 minute)
    - four saved v1 tasks (one per method) are re-run and must reproduce their saved task file exactly
      (summary except runtime, checkpoints incl. candidate_json, best-so-far curve, restarts, history).
"""

import io
import json
from pathlib import Path

import pandas as pd
import pytest

from experiments import unsw_constraint_reporting as cr
from experiments import unsw_step6_manifest as mf
from experiments import unsw_step6_package as pkg
from experiments import unsw_step6_runner as runner
from experiments import unsw_targeted_spoofing as ts
from experiments.unsw_spoofing_explanations import pair_metrics

ROOT = Path(__file__).resolve().parents[1]
V1_RUN = ROOT / "results" / "step6_minimum" / "run_20261106_203918"
V1_PACKAGE = ROOT / "results" / "step6_shared" / "run_20261106_203918"
V1_PROTOCOL = ROOT / "experiments" / "step6_protocol.json"

pytestmark = pytest.mark.skipif(not (V1_RUN / "manifest.json").exists(), reason="v1 minimum run not present")


def _v1_manifest() -> dict:
    return json.loads((V1_RUN / "manifest.json").read_text(encoding="utf-8"))


def _norm(obj):
    return json.loads(runner.dumps(obj))


def test_v1_protocol_file_unchanged():
    protocol = json.loads(V1_PROTOCOL.read_text(encoding="utf-8"))
    manifest = _v1_manifest()
    assert protocol == manifest["protocol"]
    assert mf.validate_protocol(protocol) == []
    import hashlib
    assert hashlib.sha256(json.dumps(protocol, sort_keys=True).encode("utf-8")).hexdigest() == manifest["protocol_sha256"]


def test_v1_scoring_and_mutation_source_unchanged():
    recorded = _v1_manifest()["implementation_hashes"]
    assert mf.source_sha256(ts.objective_score) == recorded["objective_score"]
    assert mf.source_sha256(ts.mutation_proposal_v1) == recorded["mutation_proposal_v1"]
    assert mf.source_sha256(pair_metrics) == recorded["pair_metrics_part_b"]


def test_v1_task_plan_unchanged():
    meta = json.loads((V1_RUN / "run_metadata.json").read_text(encoding="utf-8"))
    protocol = json.loads(V1_PROTOCOL.read_text(encoding="utf-8"))
    selection = runner.make_selection(protocol, **meta["selection"])
    planned = [_norm(t.spec()) for t in runner.plan_tasks(selection)]
    assert len(planned) == 123
    assert planned == meta["tasks"]


def test_v1_shared_package_unchanged():
    if not (V1_PACKAGE / "SHA256SUMS").exists():
        pytest.skip("v1 shared package not present")
    assert pkg.verify_package(V1_PACKAGE) == []


# ---------------------------------------------------------------------------------------------
# Slow: re-run saved v1 tasks with the real model and LIME and compare with the saved files.
# ---------------------------------------------------------------------------------------------
REPLAY_TASKS = (
    "random_forest_s2941_A_canonical_b50",
    "random_forest_s3206_A_shuffled_b50_o101",
    "random_forest_s4407_A_multistart_b25_o101",
    "random_forest_s2941_A_random_b50_c201",
)


@pytest.fixture(scope="module")
def v1_context():
    manifest = _v1_manifest()
    entry = manifest["models"]["random_forest"]
    path = ROOT / entry["file"]
    if not path.exists():
        pytest.skip(f"model file missing: {path}")
    actual = mf.file_sha256(path)
    if actual != entry["sha256"]:
        pytest.skip(f"local random_forest model {actual[:16]} differs from the v1 manifest {entry['sha256'][:16]}; "
                    "the replay can only run with the exact v1 model file")
    split = runner.guarded_split(ts.get_xai_data())
    model = ts.load_xai_model("random_forest", split)
    return runner.Context(protocol=manifest["protocol"], split=split, models={"random_forest": model},
                          model_sha={"random_forest": actual}, teammate_table=cr.load_teammate_table())


@pytest.mark.slow
@pytest.mark.parametrize("task_id", REPLAY_TASKS)
def test_v1_task_reproduces_saved_result_exactly(v1_context, task_id):
    saved = json.loads((V1_RUN / "tasks" / f"{task_id}.json").read_text(encoding="utf-8"))
    spec = {k: v for k, v in saved["spec"].items() if k not in ("strategy", "task_id")}
    spec["checkpoints"] = tuple(spec["checkpoints"])
    task = runner.Task(**spec)
    assert task.task_id == task_id

    payload, history = runner.execute_task(task, v1_context)
    payload = _norm(payload)

    new_summary = {k: v for k, v in payload["summary"].items() if k != "runtime_seconds"}
    old_summary = {k: v for k, v in saved["summary"].items() if k != "runtime_seconds"}
    assert new_summary == old_summary
    assert payload["checkpoints"] == saved["checkpoints"]
    assert payload["best_so_far"] == saved["best_so_far"]
    assert payload["restarts"] == saved["restarts"]

    buffer = io.StringIO()
    pd.DataFrame(history).to_csv(buffer, index=False)
    new_history = pd.read_csv(io.StringIO(buffer.getvalue()))
    old_history = pd.read_csv(V1_RUN / "tasks" / f"{task_id}.history.csv.gz", compression="gzip")
    pd.testing.assert_frame_equal(new_history, old_history, check_dtype=False)
