# Part C: Master Research Roadmap

**Project:** Explainable AI for Intrusion Detection Systems (UNSW-NB15): explanation spoofing under physical constraints.
**Team:** Viacheslav Sotov, Feven Ghiday.
**Prepared:** Nov 17, 2026, after the Report 9 work, the Step 6 v2 Stages 1–3 and Part A. **Updated** the same day after Stage 3B, the 10-flow exhaustive LIME pilot, the TreeSHAP pilot, the **completed G4 diagnostic pilot**, WS1b and the v3 preparation. **Updated Nov 18** after nested CV (all 8 final models), the official-test isolation fix, the frozen final attack sample, the explainability compatibility/smoke check and the protocol v3 release candidate. **No final attack experiment has been run.**

**How to read this document**
- It is the **master checklist**. The original Step 6 v2 plan is kept in full (section 4). The professor's feedback is **added** to it (section 3), not swapped in.
- Every item has a status:
  - **Done:** completed and verified;
  - **Ready:** code exists and is tested, not run;
  - **Planned:** designed, no code yet;
  - **Proposed:** an idea that needs a team decision;
  - **Decision:** blocked on your approval.
- Nothing in this roadmap claims a result that has not been produced. **No physically realizable explanation attack has been demonstrated.**
  - The 10-flow exhaustive LIME pilot on the `physical_v1` lattice found 0 successes; the TreeSHAP pilot found 0 D11 successes (pilots only; not poolable with the final experiment).
  - The **completed G4 diagnostic pilot** moved the XGBoost LIME explanation beyond seed noise on 1 of 10 flows (flow 15741, confirmed on seeds 401–404). Every such candidate **breaks checked physical rules** and edits `sttl`. This is **diagnostic explanation manipulation**, not an attack (see the box below).

> **Diagnostic explanation manipulation vs a physically realizable attack**
> - **Diagnostic explanation manipulation** (G4): a feature-space edit, with no physical-validity requirement, that changes the explanation beyond the success criterion while the flow is still classified as an attack. It shows that the explainer *can* be moved, giving an upper bound for RQ2. It says nothing about what an attacker could do with real traffic.
> - **Physically realizable attack candidate:** passes the predeclared success criterion (D8 with D14 confirmation for LIME; D11 for TreeSHAP), stays within `physical_v1`, and is **consistent with checked rules**. Even then it is reported as "consistent with checked rules", never "physically valid", until packet-level validation (WS4b) supports it.
> - **Counts so far:** 0 realizable-attack candidates; diagnostic manipulation on 1/10 flows (XGBoost, LIME); 0/10 for RF; 0 for TreeSHAP (D11).

---

## 1. Fixed rules (apply to every item)

| # | Rule |
|---|---|
| R-1 | v1 code and results stay unchanged (guarded by `tests/test_step6_v1_baseline.py`, 4 exact replays) |
| R-2 | Every new experiment gets a predeclared protocol (or a protocol amendment) **before** it runs, and its own run folders and manifests |
| R-3 | The official UNSW-NB15 test set stays **locked**. It is used once, at the very end, in a single predeclared final evaluation (D10) |
| R-4 | Success criteria, thresholds and the primary analysis are fixed before the data they apply to exists. Pilot results never change them |
| R-5 | No candidate is ever called "physically valid". Physical status is reported as consistent with checked rules, unverified dependencies, or violating a checked rule |
| R-6 | No push, no merge into `main`, no training and no expensive run without approval. Commits: Viacheslav's identity, short natural messages, no AI attribution; Feven's commits are not rewritten without her approval |
| R-7 | Model files are pinned by SHA-256 and never refitted silently |

---

## 2. Where we are (verified, Nov 17–18, 2026)

| Area | Status | Evidence |
|---|---|---|
| Baseline models, SHAP, LIME, LIME stability on 20 shared flows | Done | `results/metrics`, `results/lime_stability`, earlier reports |
| Step 6 v1 (RF, flows 2941/3206/4407, Budget A, 25/50 calls) | Done | 123/123 tasks, 186 candidates, **0/186 passed** the combined held-out test; 0 physically consistent; 75 violated a checked rule |
| Feven's v1 analysis | Done | 18 tables, 8 figures, 9 tests, 22 validation checks |
| Stage 1: formula check on 140,272 sub-train rows | Done | `results/step6_v2/physical_formula_check/` |
| Stage 2: `physical_v1` policy, 1% lattice, matched random control | Done | 33 tests |
| Stage 2: feasibility of the 10 v2 flows | Done | 1,366 valid candidates (333 consistent, 1,033 unverified), all still attacks |
| Stage 3: protocol v2, runner v3, exhaustive evaluator | Ready | 22 tests |
| Stage 3B: confirmation seeds, pilot label, model-hash and sample-identity safeguards | Done (Nov 17) | commits `15f4860`, `13154e1`, `691e612`, pushed |
| **10-flow exhaustive pilot** (RF, XGBoost, `physical_v1` lattice) | **Done (Nov 17)** | 11,180/11,180 explanations, 23.6 min; **0 LIME successes, 0 confirmed**, 0 candidates beyond the band on any single measure; all candidates still attacks. See `STEP6_V2_EXHAUSTIVE_PILOT_REPORT.md` |
| **TreeSHAP pilot** (same 1,108 candidates, RF/XGBoost, D11) | **Done (Nov 17)** | **0 D11 successes**; SHAP changes the top-10 more often than LIME but its ranking stays far above the SHAP-vs-LIME reference. See `STEP6_V2_SHAP_G4_REPORT.md` |
| **G4 diagnostic pilot** (protocol `step6-g4-diagnostic-1`) | **Done (Nov 17)**; results pushed at `644c041` | 400/400 tasks, 129 min. **DIAGNOSTIC ONLY, not an attack:** XGBoost LIME moved beyond seed noise and confirmed on seeds 401–404 on **1 of 10 flows** (15741; 3 candidates, all breaking checked physical rules, all editing `sttl` 254→255). RF 0/10; TreeSHAP D11 0 for both. TTL edits drive most of the sensitivity (XGBoost median held-out ρ 0.648 with TTL edits vs 0.960 without). Random vs greedy: random was better (**exploratory** Wilcoxon only, not confirmatory). See `STEP6_G4_DIAGNOSTIC_REPORT.md` |
| **Nested-CV timing benchmark** (outer fold 0, 2 trials, 56 fits) | **Done (Nov 17)** | 17.5 min; peak memory < 0.9 GB; `NESTED_CV_BENCHMARK_REPORT.md` |
| **Nested CV, all 8 final models** (RF, XGBoost, MLP, LR × full42 / no_ttl39) | **Done (Nov 17–18)** | Stages 1–3, ≈ 99 + 95 + 296 min; all outer folds and final refits complete; 0 shared groups; outer F1 full42 / no_ttl39: XGBoost 0.9699 / 0.9693, RF 0.9686 / 0.9684, MLP 0.9620 / 0.9620, LR 0.9559 / 0.9478 (descriptive). `docs/nested_cv_stage{1,2,3}_report.md` |
| **Official-test isolation fix** | **Done (Nov 17)**, local commit | default loaders read only the official training file; audit-hook guard with identity lock in tests and `experiments/run_guarded.py`; earlier accesses recorded in `docs/official_test_access_record.md` (the file was parsed and hashed by legacy code, never used for any fit, selection or score) |
| **Final attack sample (WS3)** | **Frozen candidate, needs approval (Nov 18)** | 263 flows: 30 per category, Worms 23 (shortfall 7); all tier 1; 208 TCP / 55 short no-reply; eligible for RF, XGBoost and MLP full42; `docs/v3_attack_sample_report.md` |
| **Explainability compatibility + smoke (WS5/WS10)** | **Done (Nov 18)** | all 8 models: pinned hashes, preprocessing, bit-reproducible predictions; no_ttl39 provably TTL-blind; LIME on each model's own features; TreeSHAP additivity; RQ5 matched metrics; cache resume; `docs/v3_compatibility_smoke_report.md` |
| **Protocol v3 release candidate** (`step6-v3-rc1`) | **Proposed, needs approval** | `experiments/step6_protocol_v3.json`; measured costs and open scope decisions in `docs/protocol_v3_readiness_report.md` |
| **WS1b time-split investigation** | **Done (Nov 17)** | Two capture days (Jan 22, Feb 18 2015); official training partition is almost entirely Feb 18; a valid time split must be built from raw records (recommended: Feb 18 first 80% vs last 20%, as a robustness check). **46.6% of current validation rows have an exact copy in sub-train.** See `WS1B_TIME_SPLIT_REPORT.md` |
| **Nested-CV and TTL-ablation plan** | **Approved in direction (Nov 17)** | grouped category-stratified nested CV; F1; LR secondary; TTL option A; see `NESTED_CV_TTL_PLAN.md` |
| **v3 preparation** (no training) | **Done (Nov 17)** | protocol v3 draft; **duplicate-disjoint v3 split** (0 shared vectors; 940 label-conflict rows excluded; categories within 0.8 pp of 20%); v3 pipelines (42/39 features, MLP/LR scaled); resumable nested-CV framework; 31 tests. See `NEXT_STAGE_DATA_AND_MODELS_REPORT.md` |
| Part A: GitHub package and reproducibility README | Done | branch `feature/step6-v2`, pushed at `d04a97c` |
| Part B: Nov 18 presentation and script | Done | Slides artifact and `PRESENTATION_NOV18_SCRIPT.md` |

**Already available but not yet in the attack pipeline:**
- an MLP (`experiments/unsw_mlp.py`, `results/metrics/unsw_mlp_cv_summary.csv`);
- XGBoost cross-validation (`unsw_xgboost_cv_summary.csv`);
- a small randomized tuning comparison (`unsw_small_tuning_comparison.csv`).

None of these is **nested** cross-validation, and the RF used in Step 6 has fixed settings (200 trees).

---

## 3. Professor's requirements: traceability

Each line of the feedback is mapped to a workstream (section 5) and given a status.

| # | Requirement (from the feedback) | Workstream | Status |
|---|---|---|---|
| P1 | Nov 18 progress presentation covering work since Report 9 | — | **Done** (deck and script) |
| P2 | GitHub: runner, analysis, 42-feature taxonomy, validator, selected flows and seeds, manifests, summary tables, 8 figures, 9 tests, reproducibility README | WS13 | **Done** (pushed). RF model file not in Git; sharing not set up yet |
| P3 | Fixed, larger sample per attack family (about 30 per category) | WS3 | **Frozen candidate (Nov 18), needs approval:** 263 flows, Worms shortfall 7 |
| P4 | RF, XGBoost and MLP with defensible tuning and nested CV | WS2 | **Done (Nov 18):** grouped, category-stratified 5×3 nested CV on the v3 training side; all 8 final models pinned |
| P5 | Attack both LIME and TreeSHAP | WS5 (G1–G3 LIME, G5 TreeSHAP); RQ5 | LIME exhaustive pilot done; TreeSHAP exhaustive pilot done (D11); G4 diagnostic done (LIME + TreeSHAP check); SHAP-targeted search G5 planned; matched descriptive LIME-vs-TreeSHAP comparison proposed for RQ5 (WS6) |
| P6 | Formalize the objective range, an example and success thresholds | WS6 | Mostly done (formula, range, example, D8, D11, D14); paper write-up planned |
| P7 | Time-based split from the raw Stime/Ltime timestamps | WS1 | **Investigated (WS1b)**: feasible only as a raw-built dataset; recommended as a robustness check |
| P8 | Remove `sttl`, `dttl`, `ct_state_ttl` (TTL ablation) | WS1 + WS2 | Models **done** (option A); no_ttl39 models verified TTL-blind; explanation comparison **Ready** (E0, needs protocol approval) |
| P9 | Recompute derived features; validate from packets with Argus/Zeek | WS4 | Recomputation done (verified formulas); packet-level validation planned |
| P10 | Explain the concrete harm to an analyst from misleading explanations | WS11 | Planned |
| P11 | Recent related work, constrained-attack literature, three literature clusters | WS11 | Planned |
| P12 | At least one explanation-consistency defense | WS8 (DEF-1, DEF-2) | Planned |
| P13 | Ablations: constrained vs unconstrained, budget, attacker knowledge | WS7 | Unconstrained side: **G4 diagnostic pilot done**; G1/G3 and budgets ready; attacker-knowledge levels proposed |
| P14 | Wilson intervals, targeted vs random, failure analysis, threats to validity, ethics, reproducibility | WS9, WS11 | Planned. **Decided (D17):** Wilson primary, Clopper–Pearson secondary in the final protocol. A failure analysis exists for the pilots. The G4 random-vs-greedy comparison is **exploratory only**; the confirmatory targeted-vs-random test is P2 in the final protocol |
| P15 | Preserve the original Step 6 v2 plan, exhaustive evaluation, G1–G6 and both defenses | Section 4 | **Preserved** |

---

## 4. Original Step 6 v2 plan (preserved in full)

| Item | What | Status |
|---|---|---|
| D1 | v1 unchanged; v2 separate and resumable | Done |
| D2 | `physical_v1`: sbytes/spkts increase-only, 1% lattice, repair of smean/rate/sload (dmean/dload/tcprtt), zero-coupling rules | Done |
| D2c-1 | `sttl` only with ≥ 30 rows of support (effectively excluded) | Decision (recommended: approve) |
| D2c-2 | Optional looser, separately named policy (ratio-preserving timing repair) | Proposed |
| D3 | 10 flows (pilot set) | Done; now the **pilot** set (D15) |
| D4 | Multi-seed objective K=3 (11/66/77), budget in LIME calls | Planned (refused by the runner until implemented) |
| D5 | Multi-start in G1 with matched random controls | Ready |
| D6 | Current-policy comparison (G3) | Ready |
| D7 | G4 diagnostic unconstrained upper bound | **Done (pilot, Nov 17):** separate protocol `step6-g4-diagnostic-1`; diagnostic only (see G4 row) |
| D8 | LIME success criterion (combined held-out test + physical status) | Fixed |
| D9 | Always-reported measures (ΔP, perturbation size, repaired changes, statuses) | Ready |
| D10 | Official test set: one final evaluation | Decision (content), near the end |
| D11 | SHAP-targeted success criterion (cross-explainer bar: change ≥ the SHAP-vs-LIME disagreement on the original) | Fixed before any SHAP run; **kept unchanged for all existing results**. Not directly comparable with D8 (see WS6). RQ5 uses matched descriptive metrics instead; no success criterion is changed retrospectively |
| D12 | LR and MLP pipelines | Decision (training); now part of WS2 |
| D13 | Primary analysis P1/P2; fallback **not adopted** | Fixed |
| D14 | Confirmation seeds 401–404 | Done (Stage 3B); not exercised in the pilot (nothing flagged) |
| D15 | 10-flow exhaustive evaluation labelled as pilot | Done |
| D16 | Pinned model hashes and sample identity | Done (Stage 3B) |
| D17 | Final protocol: Wilson primary, Clopper–Pearson secondary; v2 pilot P1 unchanged | Recorded (Nov 17), before any final result |
| Exhaustive | Every legal `physical_v1` candidate, RF/XGBoost, seeds 11–55 (+401–404) | **Pilot done** (0 successes); final version on the large sample (WS5) |
| G1 | `physical_v1`, single seed, canonical/shuffled/multi-start/random: 2,040 tasks, ≤ 154,000 LIME calls (10 flows) | Ready; blocked on downstream modules (WS10) |
| G2 | `physical_v1`, multi-seed K=3, shuffled + random: 800 tasks, ≤ 80,000 calls | Planned (needs D4) |
| G3 | `current_v1`, canonical/shuffled/random: 840 tasks, ≤ 84,000 calls | Ready; blocked on WS10 |
| G4 | Unconstrained diagnostic: any non-frozen numeric feature, ≤ 10 features, ±50%, shuffled + random, 100 calls | **Pilot done** (10 flows, current models): XGBoost 1/10 flows confirmed (diagnostic only), RF 0/10, TreeSHAP 0; to be repeated on the final flows/models |
| G5 | SHAP-targeted search with D11, plus the held-out LIME transfer check | Planned |
| G6 | LR/MLP grid: `physical_v1`, single seed, shuffled + random, A/B, 100 calls | Planned (needs WS2) |
| DEF-1 | 5-seed averaged LIME (seeds 301–305) against the G2 adaptive attacker | Planned |
| DEF-2 | Consistency checks, with the false-alarm rate on clean validation flows | Planned |
| Stages | stage0 implementation → stage1 pilot → stage2 RF → stage3 XGB → stage4 held-out and analysis | Stage 0 done; stage 1: exhaustive pilot done, search pilot pending WS10 |
| RQ1–RQ4 | From `results/research/unsw_threat_model.md` | See section 7 (RQ5 added Nov 17) |

**Grid sizes above are for the 10 pilot flows.** With the larger sample they grow roughly in proportion (section 8).

---

## 5. Workstreams

### WS1. Data, splits and feature ablations

**WS1a. Current split (Done).** Official training file → 80/20 stratified split (seed 42): 140,272 sub-train and 35,069 validation rows. All v1/v2 pilot flows and the current models come from these. **For the final design it is superseded by the approved v3 duplicate-disjoint split** (139,520 train / 34,881 validation, 0 shared feature vectors, digest `1d4c347d…`); the v1/v2 split and its results stay unchanged.

**WS1b. Time-based split investigation (DONE Nov 17; P7).** Findings in `WS1B_TIME_SPLIT_REPORT.md`:
- two capture periods, Jan 22 (attacks only in the first ~2.3 h) and Feb 18 (attacks every hour);
- the official training partition is almost entirely Feb 18 and only 43% of its rows can be timed (biased against Generic), so it can't be split by time as is;
- a raw-built split **S2 (Feb 18, first 80% vs last 20% of time)** is feasible with all 9 categories on both sides; recommended as a predeclared robustness check;
- the raw files need category cleaning and 0-filling of blank `ct_flw_http_mthd`/`is_ftp_login`/`ct_ftp_cmd`; `rate` is rebuilt;
- the raw data includes the official test records, so the time-split study is self-contained and never touches or replaces D10;
- **46.6% of current validation rows (64.1% of attacks) have an exact copy in the sub-training split**; all 5 short pilot flows do, none of the 5 TCP sessions.

Original plan text, kept for reference:
- **Verified:** the raw files `UNSW-NB15_1..4.csv` are on the laptop. They have no header row; the column names are in `NUSW-NB15_features.csv` and include `Stime` and `Ltime`.
- **Verified:** the raw files have **no `rate` column** (the partitioned files do), and use different names (`smeansz`, `Sintpkt`, `res_bdy_len`, …). `rate` can be rebuilt with the formula verified in Stage 1 (`rate = (spkts+dpkts−1)/dur`, 100% support on the partitioned data). This must be re-checked on the raw data.
- **Not known yet:** there is no row identifier linking the partitioned train/test files to the raw records. A time split therefore means **building a new dataset from the raw records**, not re-sorting the existing one.
- **Steps:**
  1. map raw columns to the 42 model features (rename, rebuild `rate`), and check the mapping on a sample of rows;
  2. measure duplicates, label and `attack_cat` quality, and the time span of each file;
  3. propose a cut (for example, train on the earlier capture period and validate on the later one), and report the category mix on each side, since some attack families may appear only in one period;
  4. decide whether the time split becomes the **main** split or a **robustness check**. Recommendation: a robustness check, because the official partition is what the literature reports.
- **Output:** `docs/time_split_investigation.md` and a script; no training until approved.

**WS1c. TTL ablation (Planned, prioritized; P8).**
- Remove `sttl`, `dttl` and `ct_state_ttl` (39 features), retrain the final models (WS2), and re-run the same attack design on the same flows.
- **Approved: TTL option A**, i.e. the 39-feature models are tuned **independently** in their own nested CV (not the 42-feature hyperparameters reused).
- **Why it is now a priority:** in the G4 diagnostic pilot, the only explanation changes beyond seed noise all edited `sttl`, and TTL edits lowered the XGBoost held-out ρ far more than other edits (median 0.648 vs 0.960). Results on 42-feature models may therefore mostly measure a TTL artefact.
- **Why it matters:** TTL features are known to separate attack and normal traffic in UNSW-NB15 for artefactual reasons, and `ct_state_ttl` is one of the strongest features here. If the explanation is dominated by TTL, both detection and spoofing results may depend on an artefact.
- **Reported:** detection metrics with and without TTL; the explanation top-10 with and without TTL; spoofing results on both models.
- The `physical_v1` policy does not change (TTL features are already excluded from editing).

### WS2. Models with defensible tuning (protocol approved Nov 17; P4)

See `NESTED_CV_TTL_PLAN.md` and `NEXT_STAGE_DATA_AND_MODELS_REPORT.md`:
- nested CV on the v3 training side, outer 5 × inner 3, seeds 42/43/44;
- category-stratified, **duplicate-grouped** folds (because of the WS1b leakage finding);
- F1 selection; predeclared search spaces (RF 15, XGBoost 25, MLP 12, LR 4 trials; MLP and LR with scaling);
- final configuration by plain CV on the whole v3 training side; new pinned model files;
- two feature sets, `full42` and `no_ttl39` (TTL option A);
- estimated cost ≈ 4.5–7.5 h per feature set, to be replaced by the measured timing benchmark.

**Execution priority (Nov 17), within the approved protocol:** the **independently tuned `no_ttl39` models are trained first**, then `full42`. Nothing else changes: same folds, seeds, search spaces, trial counts and selection metric for both feature sets. The order only decides which models are available first if time runs short. Both sets remain required for the TTL ablation (WS1c). Status: framework ready and tested; **timing benchmark not run; no training started.**

Original plan text, kept for reference:

- **Models:** RF, XGBoost, MLP (required). LR is kept as an optional fourth model for RQ3 (decision D12).
- **Nested CV:** outer 5-fold stratified CV on the sub-train split for an unbiased performance estimate; inner 3-fold randomized search for tuning. The search space and the number of trials are written into the protocol amendment **before** tuning.
- **Final models:** the configuration chosen by the inner search, refit on the whole sub-train split. They are saved with their hashes pinned in the protocol (as D16 does now), and evaluated on the validation split.
- **Selection metric:** F1 on the attack class (or macro-F1; decide once).
- **Cost:** must be measured on a small inner search first. RF with 200 trees on 140k rows is the slowest part. Run detached and in stages.
- **Outputs:** `results/models_v3/*` (outside ordinary Git, like now), nested-CV fold metrics, chosen hyperparameters, validation metrics.
- **Consequence:** the pilot uses the **current** RF/XGBoost files, which are not the nested-CV models. Pilot results are never pooled with final results.

### WS3. Larger fixed sample (frozen candidate, needs approval; P3)

**Update (Nov 18):** drawn with the approved rule (seed 2026) and frozen: `results/v3_prep/attack_sample/` (sample digest `0ac83281…`). 263 flows, 30 per category, Worms 23 (shortfall 7, not filled); all tier 1; eligibility by RF/XGBoost/MLP full42. An implementation error in the pilot-flow exclusion (row positions used as ids) was caught before use and corrected; the uncommitted first draw was discarded and is recorded in the sample report. The text below is the original proposal, kept for reference.

**Update (Nov 17):** the **tiered attack-sample rule on the v3 validation side** is approved (tier 1: flows whose feature vector is unique; tier 2: one representative per duplicate group; label-conflict and MixedAttack groups excluded; v2 pilot flows excluded). The rule and pool sizes below were written for the old split and are kept for reference; the v3 pool sizes are in `results/v3_prep/split_audit/attack_pool_before_models.csv`.

**Proposed selection rule** (to be written into the protocol amendment before any final run):
1. **Pool:** validation-split rows labelled attack, per `attack_cat`.
2. **Eligibility:** predicted as attack by **every** final model (RF, XGBoost, MLP; and LR if included), so that every model is attacked on the same flows.
3. **Exclusion:** the 10 pilot flows (recommended, so the final sample is independent of everything seen in the pilot).
4. **Draw:** 30 per category with a fixed seed (for example `numpy.random.default_rng(2026)`), stratified by flow type (TCP session vs short flow) in proportion to the eligible pool.
5. **Small categories:** if fewer than 30 are eligible, take all of them and report the number.
6. **Pinned:** each selected flow is pinned by its official-training row index (as D16 does now).

**Verified pool sizes** (validation split, attacks before the eligibility check):

| Category | Validation attacks | of which TCP |
|---|---|---|
| Generic | 8,005 | 91 |
| Exploits | 6,583 | 3,880 |
| Fuzzers | 3,674 | 2,415 |
| DoS | 2,483 | 484 |
| Reconnaissance | 2,135 | 1,003 |
| Analysis | 422 | 120 |
| Backdoor | 314 | 57 |
| Shellcode | 225 | 111 |
| **Worms** | **28** | 27 |

- **Worms cannot reach 30** in the validation split: at most 28, fewer after the eligibility check. This must be reported, not fixed by using the test set.
- Total: about **270 flows** (9 categories × 30, Worms ≤ 28).

### WS4. Physical validity

**WS4a. Recomputation of derived features (Done).**
- Verified formulas (≥ 99.9% support) are used for repair; unverified ones are excluded or reported.

**WS4b. Packet-level validation with Argus/Zeek (Planned; P9).**
- **Goal:** check, on real packets, that a `physical_v1` edit (more source bytes, more source packets) produces the feature vector our repair predicts. This tests the assumptions behind "consistent with checked rules".
- **Steps:**
  1. obtain a small subset of the UNSW-NB15 pcap files (not on the laptop; availability and size must be checked);
  2. reproduce the published features for a few flows with Argus (and Zeek for the content features), and measure how closely they match the CSV;
  3. for those flows, create modified pcaps (padding payloads, adding packets) and regenerate the features;
  4. compare the regenerated features with our repaired candidates: `smean`, `rate`, `sload` and the unverified timing features (`sinpkt`, `sjit`).
- **Outcome:** either evidence that the repair matches packet-level reality, or a list of mismatches that changes the policy (as a new, separately named policy).
- **Risk:** UNSW-NB15 used its own Argus/Bro configuration and 12 extra derived algorithms; exact reproduction may not be possible. Report the match rate honestly.

### WS5. Attacks on LIME and TreeSHAP

| Item | Explainer | Status | Depends on |
|---|---|---|---|
| Exhaustive evaluation, **pilot** (10 flows, current RF/XGBoost) | LIME | **Done**: 0 successes | — |
| Exhaustive evaluation, **final** (E0: 263 flows, 6 primary model variants, LIME + TreeSHAP) | LIME, TreeSHAP | **Ready** (code, sample, models); needs protocol v3 approval | measured cost: ≈ 39 h LIME + 12 h TreeSHAP single-process for Budgets A+B (upper bound), ≈ 10 h for Budget A only |
| G1 / G3 (search, matched random) | LIME | Ready | WS10 downstream modules |
| G2 multi-seed (adaptive attacker) | LIME | Planned | D4 implementation |
| G4 unconstrained diagnostic | LIME (+SHAP check) | **Pilot done (Nov 17); diagnostic only**: XGBoost 1/10 flows, RF 0/10, TreeSHAP 0 | Final version on the v3 flows and models |
| **G5 TreeSHAP-targeted search** | TreeSHAP (RF, XGBoost) | Planned; success judged by D11 (unchanged); matched RQ5 metrics reported descriptively (WS6) | SHAP evaluation module (done) |
| TreeSHAP exhaustive evaluation of the `physical_v1` lattice (pilot) | TreeSHAP | **Done**: 0 D11 successes | — |
| G6 MLP (and LR) | LIME; SHAP for MLP only as a small KernelSHAP check | Planned | WS2 |

- **TreeSHAP exhaustive:** TreeSHAP is deterministic and fast for tree models, so an exhaustive TreeSHAP evaluation of the whole `physical_v1` lattice is cheap. Recommended as part of G5.
- **MLP and SHAP:** TreeSHAP does not apply to an MLP. A small KernelSHAP check is planned; MLP SHAP results will be labelled as approximate.

### WS6. Objective and success thresholds (P6)

- **Objective (fixed):** `score = 0.7·(1 − top-10 Jaccard) + 0.3·(1 − Spearman ρ)/2`, range 0 (same explanation) to 1 (no shared top-10 feature and reversed ranking).
  - Example: one top-10 feature swapped (Jaccard 9/11 ≈ 0.818) and ρ = 0.90 gives 0.7·0.182 + 0.3·0.05 ≈ **0.14**.
- **Success is not a score threshold.**
  - LIME (D8): still attack, within policy, held-out top-10 Jaccard **and** Spearman both below the flow's own held-out seed-pair minimum, and consistent with checked rules. Confirmed success (D14) also requires the same test on seeds 401–404.
  - TreeSHAP (D11): still attack, within policy, physical status OK, and the change at least as large as the normal SHAP-vs-LIME disagreement on that flow.
- **To do:** write this as a formal definition in the paper (notation, the band, the two criteria, a worked example).

**D8 and D11 are not directly comparable (RQ5).**
- D8 asks whether LIME moved more than **LIME's own seed noise** (a within-explainer bar from 6 held-out seed pairs).
- D11 asks whether TreeSHAP moved at least as much as **SHAP and LIME disagree on the original flow** (a cross-explainer bar). That bar depends on LIME's behaviour and is usually far lower in Jaccard/ρ than any within-SHAP noise, so D11 is a much stricter test.
- Therefore "0 D11 successes vs n D8 successes" **cannot be read as "TreeSHAP is more robust than LIME"**. D11 stays unchanged and is reported for all existing results (TreeSHAP pilot, G4); it is not used for the RQ5 comparison.

**No retrospective changes.** D8, D11 and D14 stay exactly as predeclared for every experiment already run (v1, the exhaustive pilot, the TreeSHAP pilot, G4). The RQ5 analysis below adds **descriptive** measures for future runs only; it does not redefine success for any existing or already-specified result.

**Proposed RQ5 analysis** (to be written into protocol v3 before any final run; not implemented):
1. **Primary, descriptive: matched ranking-change metrics on identical candidates.**
   - Same flows, models, feature sets and candidates (the exhaustive `physical_v1` lattice and the final G4 candidates), explained by both LIME and TreeSHAP.
   - For each candidate and each explainer, compare the explanation of the candidate with **that explainer's own explanation of the original flow**, using the same metrics: top-10 Jaccard, Spearman ρ (tie-aware), top-1 change, and the objective score.
   - LIME is evaluated on the fixed held-out seeds (22/33/44/55, averaged), with its seed-noise band reported alongside as context. TreeSHAP uses the configuration of the existing pilot.
   - Analysis at flow level (e.g. the median and the largest change per flow): paired differences LIME − TreeSHAP, summarised with medians and bootstrap intervals over flows. This is reported as description, not as a test of a success criterion.
2. **Separate sensitivity analysis: TreeSHAP background resampling.**
   - Recompute TreeSHAP under several background draws to show how much TreeSHAP itself varies with the background, and whether the RQ5 picture changes.
   - It is **not** treated as a noise threshold equivalent to LIME's seed band, and it is not used to declare TreeSHAP successes.
   - **Design note (verified in the code):** the current XGBoost TreeSHAP uses `tree_path_dependent` (no background), while RF uses an interventional background. The sensitivity analysis therefore needs interventional TreeSHAP for XGBoost as an extra, clearly labelled configuration.
3. **Success counts** are reported under each explainer's own predeclared criterion (D8/D14 for LIME, D11 for TreeSHAP), with Wilson intervals (D17), next to the RQ5 metrics but **not** compared directly with each other.
4. **Reported by** model, feature set (`full42`/`no_ttl39`) and constrained vs diagnostic candidates, with diagnostic candidates never presented as attacks.

### WS7. Ablations (P13)

| Ablation | Comparison | Status |
|---|---|---|
| Constrained vs unconstrained | `physical_v1` (G1) vs `current_v1` (G3) vs unconstrained diagnostic (G4) | G4 **pilot done** (diagnostic); exhaustive `physical_v1` pilot done; G1/G3 Ready |
| Perturbation budget | Budget A vs B (all grids) | Ready |
| LIME query budget | 25 / 50 / 100 calls | Ready |
| Attacker knowledge | Proposed levels below | Proposed |
| Features | With vs without TTL (WS1c) | Planned, prioritized (TTL option A; G4 points to TTL as the main driver) |

**Proposed attacker-knowledge levels** (decision needed):
- **K1 (strongest):** knows the model, the LIME configuration and the LIME seed. Single-seed search (G1).
- **K2:** knows the model and the LIME configuration, **not** the seed. Multi-seed search (G2).
- **K3 (weakest):** knows neither the model nor the seed; crafts candidates on a **surrogate** model (for example RF) and they are tested on another (XGBoost, MLP). New design; no code yet.

### WS8. Defenses (P12)

- **DEF-1:** average LIME over 5 seeds (301–305). Tested against the G2 adaptive attacker. Reports the change in spoofing success and the extra explanation cost.
- **DEF-2:** explanation-consistency checks, for example flagging an alert when SHAP and LIME disagree beyond the normal band, or when LIME is unstable across seeds. Reports detection of spoofed explanations and the **false-alarm rate on clean validation flows**.
- The models are not changed, so detection quality is unchanged. The final test set reports only DEF-2's false-alarm rate on clean flows (D10).
- **Caveat (updated after the G4 pilot):** the constrained pilots produced no successes. The only candidates beyond the band so far are the G4 diagnostic ones (XGBoost flow 15741, 3 candidates, all breaking checked physical rules). If the final constrained runs also produce none, the defenses can only be evaluated on diagnostic candidates or on the strongest non-successful candidates, and are then reported as **detection of diagnostic explanation manipulation**, not as a defense against a realizable attack. This must be stated.

### WS9. Statistics and reporting (P14)

- **Unit of analysis:** the flow (sample), not the candidate.
- **Intervals (decided, D17):** the final protocol reports **Wilson 95% as the primary interval** and **Clopper–Pearson 95% as a secondary sensitivity check**, at the flow level. Protocol `step6-v2` (pilot) keeps its exact-interval definition.
- **Targeted vs random:** P2 (Wilcoxon signed-rank, Holm over 4 tests), unchanged, with lattice coverage and early stopping.
- **Failure analysis:** for every flow without success, report why: lattice size, LIME bin crossings (quartile edges), whether the edited features ever enter the top-10, early stopping, and the closest candidate to the band.
- **Multiple testing in the exhaustive evaluation:** handled by the confirmation seeds (D14).
- **G4 random vs greedy (exploratory only):** the random control beat shuffled greedy (Wilcoxon signed-rank, p 0.002–0.049). These tests were chosen after seeing the pilot data, were not part of the G4 protocol's primary analysis, and are not corrected for multiplicity. They are reported as **exploratory description** and are **not confirmatory**; no claim rests on them. They motivate, but do not replace, the predeclared P2 targeted-vs-random test in the final protocol.
- Results per attack category and flow type.

### WS10. Pipeline modules still missing (Planned)

1. Multi-seed objective (D4).
2. Held-out script: policy/objective columns, multi-seed overlap check.
3. Packaging of several run folders.
4. Analysis generalization (Feven): several models, budgets A/B, 25/50/100, policies, objectives, flow-level statistics, Wilson intervals.
5. SHAP evaluation module: TreeSHAP for RF/XGBoost **done** (pilot, D11). **New-model layer done (Nov 18):** `src/unsw/v3_final_models.py`, `src/unsw/v3_explain.py` (LIME and TreeSHAP on each model's own features, matched RQ5 metrics, identity-keyed resumable cache). Planned: TreeSHAP background-resampling sensitivity; KernelSHAP for MLP only if amended.
6. G4 unconstrained policy: **done** (pilot run complete). G5 SHAP-targeted search: planned.

### WS11. Paper (P10, P11, P14)

- **Concrete analyst harm (P10):** a worked scenario. An alert on a flow is explained by `sbytes` and packet counts instead of the TTL or state features that actually drove it; the analyst then triages by the wrong cue (for example, treating it as a large legitimate transfer), suppresses similar alerts, or writes a detection rule on the wrong feature. Quantify where possible with the top-1/top-10 changes we measure. Mark it clearly as a scenario.
- **Related work, three clusters (P11):**
  1. explainable AI for intrusion detection;
  2. attacks on and robustness of post-hoc explanations (LIME/SHAP manipulation, explanation stability);
  3. problem-space and constrained adversarial attacks on network intrusion detection (realizable traffic changes, feature-dependency constraints).
  - Include **recent work** from the last two to three years in each cluster. A literature search is still needed; candidate references must be verified before they are cited.
- **Threats to validity:** a small fixed lattice; a seed-noise band from only 6 pairs; dataset artefacts (TTL); the random split vs a time split; LIME configuration choices; unverified timing dependencies; a single dataset.
- **Ethics:** dual use (we describe a manipulation technique); the mitigation is that we publish the defense and limits together; public dataset, no personal data.
- **Reproducibility:** protocols, manifests with hashes, seeds, run folders, tests, the README; model files shared outside Git with hashes.

### WS12. Final evaluation on the official test set (D10)

- **Proposal (decision near the end):** one predeclared run reporting only detection quality (F1, recall, FPR) of the final models, and DEF-2's false-alarm rate on clean test flows.
- **Not allowed:** test flows as attack samples, or test results used to change anything.

### WS13. Repository and deliverables

| Item | Status |
|---|---|
| `feature/step6-v2` on GitHub with code, tables, figures, tests, README | Done (`d04a97c`) |
| Stage 3B commits (`15f4860`, `13154e1`, `691e612`) | Pushed |
| LIME pilot (`8aa992e`), TreeSHAP/G4 code (`9545e66`, `1b255c5`, `ffecbc1`), WS1b (`156a7d9`), v3 preparation (`93d2f36`, `a00ec98`, `b5b5815`), G4 results (`644c041`) | Pushed (`feature/step6-v2` at `644c041`) |
| This roadmap (`docs/research_roadmap.md`) | Reviewed and pushed (Nov 17) |
| RF model file sharing (186 MB, not in Git) | **Not set up.** Do not claim access until shared and verified |
| XGBoost model in Git since Oct 14 | Documented in the README |
| Merge into `main` | Needs approval |

---

## 6. Pilot vs final: what stays valid

| Output | Valid as | Notes |
|---|---|---|
| Step 6 v1 (3 flows, RF) | **Earlier pilot**, reported as is | Never pooled with v2 or final |
| Feven's v1 analysis (18 tables, 8 figures) | v1 results | Unchanged |
| Stage 1 formula check | **Valid for the final design** if the split stays the same | Must be repeated on a time-split dataset (WS1b) |
| `physical_v1` policy and repair module | **Valid** | Re-checked by packet validation (WS4b) |
| Stage 2 feasibility (10 flows) | Pilot | Repeat on the large sample |
| Exhaustive evaluation on 10 flows (run Nov 17: 0 successes) | **Pilot only** (D15) | Current RF/XGBoost, not nested-CV models; no MLP. Never pooled with final results |
| TreeSHAP pilot (0 D11 successes) | **Pilot only** | D11 kept unchanged for these results; not used for the RQ5 comparison (WS6) |
| G4 diagnostic pilot (XGBoost 1/10 flows, RF 0/10, TreeSHAP 0) | **Diagnostic pilot only** | Current models, old split; diagnostic explanation manipulation, not an attack; random-vs-greedy p-values exploratory; never pooled |
| All v1/v2 pilot results | Valid as pilots on the **old split** | 46.6% of old validation rows have an exact copy in sub-train (WS1b); final results use the v3 split |
| Runner, exhaustive evaluator, manifests, tests | **Valid infrastructure** | Need a protocol amendment for new flows, models and pins |
| Current RF and XGBoost files | Pilot models | Final models come from nested CV (WS2) |

**Must be redesigned for the final experiment:**
- the sample set (WS3);
- the models (WS2) and their pinned hashes;
- the protocol (an amendment or a v3 file) with the new flows, models, Wilson intervals and attacker-knowledge levels;
- the grid sizes and runtime estimates (section 8);
- the TTL-ablation and time-split variants, as separate conditions.

---

## 7. Research questions → evidence

| RQ | Evidence needed | Where |
|---|---|---|
| RQ1: Can LIME or SHAP be manipulated? | Exhaustive LIME (pilot, then final); G1/G2 search; G5 TreeSHAP; transfer check. Answer kept in two parts: diagnostic manipulation (G4) vs physically realizable attack (constrained runs) | WS5 |
| RQ2: How much do realistic constraints reduce success? | G1 vs G3 (paired); G4 diagnostic upper bound (**pilot done**: XGBoost 1/10 unconstrained vs 0/10 constrained on the same flows); packet validation of the constraints | WS4, WS5, WS7 |
| RQ3: Do RF, XGBoost and MLP differ (LR optional)? | Same design on all final models | WS2, WS5 |
| RQ4: Can a defense help without hurting detection? | DEF-1, DEF-2, false-alarm rates | WS8 |
| **RQ5: How does explanation robustness differ between LIME and TreeSHAP?** (added Nov 17) | **Primary (descriptive):** matched ranking-change metrics (top-10 Jaccard, Spearman ρ, top-1 change, objective score) for LIME and TreeSHAP on identical flows and candidates, paired by flow. **Sensitivity:** TreeSHAP background resampling, not used as a noise threshold. Success counts under D8/D14 and D11 (unchanged) reported alongside, not compared directly | WS5, WS6, WS9 |

---

## 8. Cost estimates for the final design

**Superseded by measurements (Nov 18):** `docs/protocol_v3_readiness_report.md` (263 flows; Budget A 5,947 and Budget B 26,597 valid candidates; LIME 0.23 s RF, 0.13 s XGBoost, 0.07 s MLP; TreeSHAP 0.6 s RF, 0.02 s XGBoost). The estimates below are kept for the record.

**Assumptions:**
- About 270 flows; the TCP share per category from section 5 gives roughly **110–115 TCP sessions** and **155–160 short flows**.
- Unique legal candidates per flow from the pilot feasibility count: 25 for short flows; 74–580 (mean ≈ 197) for TCP sessions.
- 0.12 s per LIME call, measured in the pilot (RF 0.119 s, XGBoost 0.123 s, current models; 11,180 calls in 23.6 min). MLP and retrained models must be re-measured.

| Quantity | Estimate |
|---|---|
| Unique candidates per model | ≈ 26,000 (very uncertain: one TCP flow alone had 580) |
| Exhaustive LIME calls, 3 models × 5 seeds | ≈ 400,000 |
| Time at 0.12 s per call | ≈ 13–14 hours, in resumable stages (plus confirmation calls) |
| G1-style search grids | Scale by about 27× from the 10-flow numbers; **too large to run in full**. Choose a subset (for example shuffled + random only, 100 calls) when the protocol is amended |

**Recommendation:** with the exhaustive evaluation available, the search grids on the large sample can be reduced to the comparisons that the exhaustive evaluation cannot answer (targeted vs random, multi-seed, unconstrained), and the multi-start method can be dropped there. This is a decision for the protocol amendment.

---

## 9. Order of work

1. ~~Run the 10-flow exhaustive **pilot**~~. **Done Nov 17:** 0 successes on the full `physical_v1` lattice.
2. ~~TreeSHAP pilot and G4 diagnostic~~. **Done Nov 17:** TreeSHAP pilot 0 D11 successes; **G4 diagnostic pilot completed** (400/400 tasks; XGBoost 1/10 flows, diagnostic only; RF 0/10; TreeSHAP 0). The upper bound that RQ2 needed now exists: LIME on XGBoost *can* be moved, but only by edits that break checked rules, mostly TTL edits.
3. ~~WS1b time-split investigation~~ and ~~v3 preparation~~. **Done Nov 17.**
4. ~~Nested-CV timing benchmark~~ and ~~WS2 nested CV (all 8 models)~~. **Done Nov 17–18.**
5. ~~Official-test isolation fix~~. **Done Nov 17** (local commit).
6. ~~WS3 sample draw~~ (**frozen, needs approval**) and ~~explainability compatibility/smoke~~. **Done Nov 18.**
7. **Now: approve protocol v3** (`step6-v3-rc1`) with the scope decisions in `docs/protocol_v3_readiness_report.md`.
8. **WS10:** remaining modules for the approved scope: E0 runner on the new models (built on the v3 layer), held-out/confirmation evaluation, multi-seed objective (if G2), analysis generalization (Feven), parallel-safety benchmark if parallel runs are wanted.
9. **WS4b** packet validation, in parallel (independent of the runs). S2 raw-data time split stays postponed.
10. Final runs: exhaustive (LIME + TreeSHAP), G1/G3 subset, G2, G4, G5, G6, on `no_ttl39` and `full42` models.
11. **WS8** defenses (DEF-1, DEF-2).
12. **WS9/WS11** analysis and paper; **WS12** the single final test-set evaluation.

---

## 10. Decisions needed

| # | Decision | Recommendation |
|---|---|---|
| 1 | ~~SHAP evaluation module + G4 unconstrained diagnostic~~ | **Done:** approved, run and completed Nov 17 (G4 results pushed at `644c041`) |
| 2 | ~~Wilson vs Clopper–Pearson~~ | **Decided (D17):** Wilson primary, Clopper–Pearson secondary |
| 3 | Final sample rule (WS3) | **Approved (Nov 17):** tiered rule on the v3 validation side; the Worms shortfall is reported, never filled from the test set |
| 4 | Nested-CV design: search spaces, trials, selection metric | **Approved (Nov 17)**; practicality of the spaces checked in the timing benchmark, no silent changes |
| 5 | Is LR still in scope (RQ3)? | Optional; keep if time allows |
| 6 | Attacker-knowledge levels K1–K3 | Approve K1/K2 now; K3 if time allows |
| 7 | Time split: main split or robustness check | **Decided:** S2 as an isolated robustness check; raw-data training postponed |
| 7b | Nested-CV design decisions and TTL option A/B | **Approved (Nov 17):** grouped 5×3 nested CV, F1, spaces as drafted, TTL option A; `no_ttl39` trained first |
| 7c | Start the timing benchmark | When you confirm you are home and the laptop is plugged in |
| 7d | Full nested-CV training and the execution order | After the benchmark report |
| 7e | RQ5 analysis (WS6): matched descriptive metrics as primary; TreeSHAP background resampling as a sensitivity analysis (number of draws, background size) | Approve for protocol v3 |
| 8 | Reduced search grids on the large sample | **Now (Nov 18):** measured costs in `docs/protocol_v3_readiness_report.md`; see decisions D-v3-1 … D-v3-9 there |
| 8b | Approve the frozen attack sample (263 flows, Worms 23) | Approve |
| 8c | Approve protocol v3 `step6-v3-rc1` | Approve with the scope choices |
| 8d | Integrate `fix/lock-official-test` and `feature/v3-preparation` into `feature/step6-v2` (fast-forward) and push | After your review |
| 9 | D2c-1 (`sttl` threshold) and D2c-2 (looser policy) | Keep `physical_v1` strict |
| 10 | Final test-set evaluation content (D10) | Decide near the end |
| 11 | How to share the RF model file with the professor | Needs a sharing location you choose |
| 13 | ~~Push the pilot commit~~ | **Done** (pushed) |
| 13b | Push this roadmap commit | **Approved (Nov 17)** |
| 14 | Commit the 17.4 MB explanation cache of the pilot to Git, or keep it on the laptop with its hash recorded | Keep outside Git, as with other raw run files |
