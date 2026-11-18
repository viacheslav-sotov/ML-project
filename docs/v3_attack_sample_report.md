# Final attack-flow sample (WS3): selection report

**Date:** Nov 18, 2026 · **Branch:** `feature/v3-preparation` (local) · **Status:** frozen candidate sample, **needs approval**
**Files:** `results/v3_prep/attack_sample/` (`sample_manifest.csv`, `sample_manifest.json`, `pool.csv`, `audit.json`), produced by `experiments/unsw_v3_attack_sample.py` under `experiments/run_guarded.py` (official test partition never opened).

## 1. Rule (approved Nov 17, unchanged)

The rule is protocol `step6-v3-draft-1`, `attack_sample`. Its implementation details were fixed in the code before the draw:

| Step | Implementation |
|---|---|
| Pool | v3 **validation** side only; label attack; not in a label-conflict group; uniform category (not `MixedAttack`); the 10 v2 pilot flows excluded by **official id** |
| Eligibility | P(attack) ≥ 0.5 for **all three primary full42 final models** (RF, XGBoost, MLP). no_ttl39 and LR predictions are recorded, never used for selection |
| Tier 1 | singleton groups (the feature vector occurs once in the official training partition) |
| Tier 2 | only for a category with < 30 eligible tier-1 flows; one representative per non-singleton uniform group (lowest official id) |
| Draw | `numpy.random.default_rng(2026)`, one generator; categories in sorted order; tier 1 first (all if ≤ 30, else 30 drawn), then tier-2 fill. Each draw is stratified by flow type (TCP = `proto == "tcp"`, other), with a largest-remainder allocation of n proportional to the eligible counts and `rng.choice` without replacement over sorted ids |
| Shortfall | recorded, never filled from another category or the official test partition |

## 2. Result

**263 flows** (sample digest `0ac83281b7e75be19742e8879a8647e14e25ebae2e3879bd0b056609bcac954f`; CSV SHA-256 `51c4d872…`).

| Category | Pool | Eligible tier 1 (TCP) | Tier-2 groups | Selected | TCP / other | Shortfall |
|---|---|---|---|---|---|---|
| Analysis | 111 | 32 (32) | 10 | 30 | 30 / 0 | 0 |
| Backdoor | 81 | 77 (53) | 1 | 30 | 21 / 9 | 0 |
| DoS | 504 | 473 (423) | 7 | 30 | 27 / 3 | 0 |
| Exploits | 4,019 | 3,296 (3,187) | 321 | 30 | 29 / 1 | 0 |
| Fuzzers | 3,221 | 1,990 (1,483) | 163 | 30 | 22 / 8 | 0 |
| Generic | 7,936 | 165 (99) | 615 | 30 | 18 / 12 | 0 |
| Reconnaissance | 1,721 | 1,169 (987) | 52 | 30 | 25 / 5 | 0 |
| Shellcode | 225 | 209 (113) | 7 | 30 | 16 / 14 | 0 |
| **Worms** | 23 | 23 (20) | 0 | **23** | 20 / 3 | **7** |
| **Total** | 17,841 | 7,434 | | **263** | **208 / 55** | 7 |

- **Tiers:** all 263 flows are tier 1. Every category except Worms had ≥ 30 eligible singletons; Worms has no tier-2 groups.
- **Flow types:** 208 TCP sessions, 55 others. All 55 others are short flows (≤ 2 packets in total) with no reply. Pinned per flow: `is_tcp`, `short_flow`, `no_reply`, `zero_duration`, `spkts`, `dpkts`, `dur`, proto, service, state.
- **Pool exclusions** (validation side, 34,881 rows): 11,121 normal; 0 label-conflict rows (removed upfront by rule C-1); 5,918 `MixedAttack` rows; 1 v2 pilot flow. Of the 10 pilot flows, 2 are on the v3 validation side; one of them sits in a `MixedAttack` group.
- **Detection** of the 263 sample flows (all eligible by construction):

  | Model | full42 | no_ttl39 |
  |---|---|---|
  | RF | 263 | 260 |
  | XGBoost | 263 | 262 |
  | MLP | 263 | 261 |
  | LR | 257 | 261 |

  A flow that a no_ttl39 or LR model does not detect is reported for that model, never replaced.

## 3. Pinned per flow (`sample_manifest.csv`)

- **Identity:** official `id`; `vector_sha256` (exact 42 values); `feature_hash64` (the pandas hash used for grouping); `group_id`, `group_size`, `duplicate_status`, `tier`.
- **Flow attributes:** `attack_cat`, TCP/short/no-reply/zero-duration flags.
- **Predictions:** P(attack) and decision for all 8 final models, as float64 at full precision.

`sample_manifest.json` records:
- the rule and the implementation text;
- the seed;
- the split digest `1d4c347d…`;
- all 8 model SHA-256s;
- the excluded pilot ids;
- per-category counts;
- the code hash;
- `official_test_partition_opened: false`.

## 4. Checks (all pass)

- `audit.json`:
  - 0 pool vectors on the train side;
  - ids unique and groups unique (one flow per feature-vector group);
  - every flow eligible;
  - 0 pilot ids in the sample.
- **Tests** (`tests/test_v3_attack_sample.py`, 10):
  - allocation, determinism, tier-2 fill, shortfall without cross-category filling, ineligible flows never drawn;
  - the frozen sample is reproduced exactly from `pool.csv`;
  - rule compliance (validation side, group-disjoint from training, no `MixedAttack` or conflicts, vector hashes recomputed from the data);
  - excluded ids are the true official ids of the 10 pilot rows;
  - redraw refused once frozen.
- **Prediction preservation** (compatibility check): re-scoring the whole pool reproduces every stored probability bit for bit. Scoring the 263 flows alone differs by ≤ 3.3e-16 for MLP/LR (batch-dependent BLAS order) and 0 for RF/XGBoost, with identical decisions.

## 5. What happened during the draw (full disclosure)

1. **First draw discarded, before any use.**
   - The reference table stores the pilot flows as 0-based row positions (`official_train_row_index`), while the pool uses official ids (= row + 1).
   - The first draw therefore excluded 3 unrelated flows instead of the pilot flows. It was noticed in the audit counts (3 instead of ≤ 2). None of the true pilot ids was drawn.
   - The rule says "by official id", so the lookup was fixed and a test added. The uncommitted draw (digest `93a0f50e…`) was set aside, and the sample was redrawn with the same rule and seed.
   - Because the generator is shared across categories, the redraw differs in 23 flows (Exploits onward). No selection setting was changed.
2. **Manifest regenerated for precision only.**
   - The first manifest stored XGBoost's float32 probabilities without casting to float64, so the CSV lost precision.
   - It was regenerated with full float64 precision (and deterministic RF inference). The sample digest is identical (`0ac83281…`), so the same 263 flows were drawn.
3. **Labelling note on an older file.** `results/v3_prep/split_audit/audit.json` (`v2_pilot_flows_in_v3`) is keyed by row positions, not official ids. The side assignment there is correct; only the key label is imprecise. The file is unchanged.

## 6. Decisions

- **Approve the frozen sample**: 263 flows, with the Worms shortfall of 7 reported and not filled.
