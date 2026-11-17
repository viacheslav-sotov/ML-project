# Step 6 v2 pre-run manifest

`manifest_prerun_v2.json` was written with `experiments/unsw_step6_manifest.py` for protocol `step6-v2` **before any v2 experiment**. It records the starting state: protocol hash, data and model hashes, code hashes, split, flows, seeds, LIME configuration and software versions, at Git commit 15f4860 (clean tree).

It replaces the earlier pre-run manifest from commit 69a1cda. The protocol was amended in Stage 3B, still before any v2 run: confirmation seeds 401–404, the pilot label for the 10-flow exhaustive evaluation, pinned model hashes and pinned sample identities. The earlier version stays in the Git history.

It is **not** a result. Each v2 run writes its own manifest into its run folder.
