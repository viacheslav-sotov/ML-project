# Protocol v3 readiness report

**Date:** Nov 18, 2026 · **Branch:** `feature/v3-preparation` (local) · **Protocol file:** `experiments/step6_protocol_v3.json` (`step6-v3-rc1`, status **PROPOSED**)
**Measurements:** `experiments/unsw_v3_readiness.py` → `results/v3_prep/readiness/` (lattice census of the 263 sample flows; LIME/TreeSHAP timing on 3 original flows). No candidate was explained and no attack search was run.

## 1. What is ready

| Component | State |
|---|---|
| v3 split, 8 final models (pinned), official-test guard | Done; verified (state checker, 373 tests, v1 replays) |
| Final attack sample (263 flows) | Frozen candidate, needs approval (`docs/v3_attack_sample_report.md`) |
| LIME / TreeSHAP on new models, RQ5 metrics, resumable cache | Done; compatibility and smoke clean (`docs/v3_compatibility_smoke_report.md`) |
| `physical_v1`, `current_v1`, G4 policies | Unchanged; work on the new flows (census below) |
| Protocol v3 RC1 | Written; carries over D8, D11, D14, seeds, budgets, G1–G6, DEF-1/DEF-2, D17 statistics, D10 test lock |
| Still to build after approval | E0 runner on the v3 layer (exhaustive + confirmation), search runner adapter for the new models (G1–G4 subsample), multi-seed objective (G2), DEF-2 calibration, analysis generalization (Feven) |

Protocol v3 keeps the training protocol `step6-v3-draft-1` unchanged, because its hash is pinned in every training run header. It also pins:
- the sample digest and CSV hash;
- the 8 model hashes;
- the hashes of the v2 and G4 protocols.

`tests/test_protocol_v3.py` keeps these pins consistent.

## 2. Measured inputs

**`physical_v1` lattice on the 263 flows** (valid candidates; every flow has at least one):

| Budget | Total | Median / max per flow | Consistent | Unverified | Still detected (lowest of the 6 primary models) |
|---|---|---|---|---|---|
| A | 5,947 | 21 / 120 | 2,533 | 3,414 | 5,882 (MLP no_ttl39) |
| B | 26,597 | 77 / 675 | 6,477 | 20,120 | 26,230 (MLP no_ttl39) |

The 55 short flows have at most 10 (A) or 25 (B) candidates each. TCP sessions carry 91–95% of all candidates.

**Time per explanation** (single process, AC power, RF `n_jobs=1`):

| | RF | XGBoost | MLP | LR |
|---|---|---|---|---|
| LIME (5,000 samples) | 0.235 s | 0.128 s | 0.069 s | 0.054 s |
| TreeSHAP (1 flow) | 0.65 s | 0.024 s | — | — |

## 3. Cost of the experiments (single process; upper bounds)

| Experiment | Scope | LIME/SHAP calls | Time |
|---|---|---|---|
| **E0 exhaustive `physical_v1`, Budgets A+B** | 263 flows × 6 primary variants; LIME seeds 11 + 22/33/44/55; TreeSHAP RF/XGB; confirmation 401–404 for candidates beyond the band | ≈ 0.98 M LIME, 131 k TreeSHAP | LIME ≈ 39 h (RF 21, XGB 12, MLP 6) + TreeSHAP ≈ 12 h ⇒ **≈ 52 h** |
| E0, Budget A only | same | ≈ 0.19 M LIME, 25 k TreeSHAP | ≈ 7.5 h + TreeSHAP ≈ 2.3 h ⇒ **≈ 10 h** |
| E0 with held-out seeds only (no seed 11) | — | −20% LIME | A+B ≈ 44 h |
| G1 as in v2 at 263 flows | 7,700 calls per flow and variant | ≈ 12 M | **not feasible** |
| G1 P2 core on a 27-flow subsample | shuffled vs random, 10+10, B, 100 calls, full42 RF/XGB/MLP | ≈ 0.18 M | ≈ 7 h |
| G2 (multi-seed K=3) on the subsample | same, ×3 per candidate | ≈ 0.5 M | ≈ 19 h (≈ 10 h with 5+5 seeds) |
| G3 `current_v1` on the subsample | canonical, shuffled, random; RF/XGB full42 | ≈ 0.12 M | ≈ 6 h |
| G4 diagnostic on the subsample | 10+10 × 100, RF/XGB × full42/no_ttl39 | ≈ 0.24 M | ≈ 12 h |
| DEF-1 | 5-seed averaged LIME on the G2 outputs | small | < 1 h |
| DEF-2 calibration | 600 clean validation flows × (4 LIME seeds + TreeSHAP), RF/XGB full42 | ≈ 5 k | < 1 h |

Notes:
- A+B is an upper bound: the Budget-A lattice is (almost) contained in B, and identical candidates are evaluated once.
- Running 3–4 processes in parallel on the 8-core laptop would cut wall time roughly 3×. Under the current rules this needs a parallel-safety benchmark first: time, memory, and bit-identical results versus a single process.

## 4. How v3 maps the original plan (nothing dropped silently)

| Item | v3 RC1 |
|---|---|
| Exhaustive `physical_v1` | **E0** on all 263 flows, all 6 primary variants; LIME and TreeSHAP on identical candidates (answers RQ1 constrained, RQ3, RQ5, TTL ablation) |
| G1 | P2 core (shuffled vs matched random) on a predeclared subsample; canonical and multi-start dropped on the large sample (E0 already covers every legal candidate) — **decision** |
| G2 / DEF-1 | adaptive multi-seed attacker on the subsample, then DEF-1 — **decision** |
| G3 | `current_v1` contrast for RQ2 on the subsample — **decision** |
| G4 | diagnostic upper bound on the subsample, both feature sets (shows what TTL removal does) — **decision** |
| G5 | TreeSHAP exhaustive inside E0; TreeSHAP search only under G4 — **decision** |
| G6 | MLP is a primary model everywhere; LR reported as a secondary baseline in E0 only — **decision** |
| DEF-2 | calibrated on clean validation flows outside the sample; false-alarm rate reported — **decision** |
| D8, D11, D14, D17, D10 | unchanged |

## 5. Decisions required before the first final attack experiment (E0)

| # | Decision | Recommendation |
|---|---|---|
| D-v3-1 | Approve the frozen sample (263 flows, Worms 23 with shortfall 7) | Approve |
| D-v3-2 | E0 scope: Budgets A+B (≈ 52 h single process) or A first, B after review | **A first** (≈ 10 h), then B, same code and identities |
| D-v3-3 | E0 LIME seeds: v2 set 11 + 22/33/44/55, or held-out only | Keep the v2 set (comparability with the pilot) |
| D-v3-4 | Approve the two documented explanation changes: LIME/TreeSHAP statistics and background from the v3 training side; explainer feature space = the model's own features | Approve |
| D-v3-5 | Approve deterministic RF inference (`n_jobs=1`) for all explanation work | Approve |
| D-v3-6 | Parallel execution: run a short parallel-safety benchmark (e.g. 3 processes, one flow subset, compared bit-for-bit with single-process output) | Approve the benchmark; stay single-process until it passes |
| D-v3-7 | RQ5 analysis as written (descriptive, paired by flow, bootstrap 95% intervals, seed 2026); TTL ablation McNemar as descriptive | Approve |
| D-v3-8 | Approve protocol `step6-v3-rc1` as a whole for E0 (search grids can be approved separately) | Approve E0 now |

Needed later, before the search experiments (not before E0):

| # | Decision | Recommendation |
|---|---|---|
| D-v3-9 | Subsample for the search grids: 27 flows (3 per category, stratified by flow type, `default_rng(2026)`) | Approve, or name another size |
| D-v3-10 | Which search grids run: G1 P2 core, G2 + DEF-1, G3, G4; definition of the P2 family for Holm | G1 core + G4 first; G2/DEF-1 and G3 if time allows |
| D-v3-11 | DEF-2 calibration set and threshold rule | 300 normal + 300 attack clean validation flows; flag above the clean 95th percentile |
| D-v3-12 | Attacker-knowledge levels K1–K3 (roadmap WS7) | K1/K2 now, K3 if time allows |

## 6. Repository integration (Part A)

| Branch | Tip | Relation |
|---|---|---|
| `feature/step6-v2` | `582d8d3` | 3 commits ahead of GitHub (`b9885b2`) |
| `fix/lock-official-test` | `54ed7bc` | `feature/step6-v2` + isolation fix + nested-CV Stage 1–3 results/reports |
| `feature/v3-preparation` | this work | `fix/lock-official-test` + v3 preparation commits |

All branches form one linear history, so `feature/step6-v2` can simply be **fast-forwarded** to `feature/v3-preparation`. Nothing is rewritten and no merge commit is needed. Feven's commits are untouched.

**Proposed, after your approval:**
```
git checkout feature/step6-v2
git merge --ff-only feature/v3-preparation
git -c http.version=HTTP/1.1 push --no-tags origin feature/step6-v2
```

The local branches are kept as markers; `main` stays untouched. Note that the GitHub repository is public:
- the pushed results include the frozen sample's official ids and model probabilities;
- the 8 model files (*.joblib, ≈ 436 MB, 2 RF files ≈ 427 MB) stay outside Git, so the model-sharing decision from the roadmap remains open.
