# Step 6 v2: Decisions

**Status:** the overall direction (G1–G6, LR and MLP, SHAP-targeted search, both defense experiments) is **approved by Viacheslav (Nov 17, 2026)**, subject to final feasibility and the professor's requirements. Items still waiting for approval are marked **NEEDS APPROVAL**.
**Branch:** `feature/step6-v2` (pushed to GitHub on Nov 17, 2026; not merged into `main`).
**Evidence used:**
- code on `feature/step6-runner` (`b81f195`) and Feven's `feature/step6-analysis` (`1c5fc37`), merged locally as `01e9ed4`;
- `results/research/unsw_threat_model.md`;
- the stage-1 formula check: `results/step6_v2/physical_formula_check/`, made by `experiments/unsw_physical_formula_check.py`.

---

## 0. Fixed rules

| # | Rule |
|---|---|
| R-1 | **v1 code and results stay unchanged.** This covers protocol `step6-v1`, `results/step6_pilot/`, `results/step6_minimum/`, `results/step6_shared/` and the v1 code paths (`objective_score`, `mutation_proposal_v1`, v1/shuffled/multi-start/random search). It is guarded by `tests/test_step6_v1_baseline.py`: 4 fast checks plus an exact replay of 4 saved v1 tasks |
| R-2 | v2 uses a new protocol file (`experiments/step6_protocol_v2.json`), new run folders (`results/step6_v2_runs/<stage>/`) and resumable tasks |
| R-3 | **The official UNSW-NB15 test set is not used** for search, sample selection, thresholds, tuning or analysis. It may be used only once, in a single **final evaluation** that is fixed in advance (see D10) |
| R-4 | No push and no merge into `main` without approval. Commits use Viacheslav's Git identity, with no AI attribution |

---

## 1. Decisions

### D1. v1 unchanged; v2 separate and resumable (APPROVED)

- v1 results (3 samples, RF, Budget A) are reported as the earlier pilot and are **not pooled** with v2.

### D2. Physical constraint policy `physical_v1`: hybrid, enforced by the search (APPROVED, details updated from the stage-1 data check)

**D2a. Derived features recomputed after every edit.** Only formulas with ≥ 99.9% support on the sub-training split (140,272 rows) are used.

| Formula | Support (sub-train) | Validation (35,069 rows) | Used |
|---|---|---|---|
| `smean = round(sbytes/spkts)` | 100% | 100% | yes |
| `dmean = round(dbytes/dpkts)`, 0 if `dpkts = 0` | 100% | 100% | yes |
| `rate = (spkts+dpkts−1)/dur` (dur > 0, within 1%) | 100% | 100% | yes |
| `rate = 0` when `dur = 0` | 99.86% (3 rows off) | 99.82% | **no**. Edits that would need it are rejected (none of the 10 samples has `dur = 0`) |
| `sload = (sbytes−smean)·8/dur` (within 1%) | 99.975% | 99.986% | yes |
| `dload = (dbytes−dmean)·8/dur` (within 1%) | 99.989% | 99.994% | yes |
| `tcprtt = synack + ackdat` | 100% | 100% | yes (already enforced by the validator) |
| `ct_state_ttl` determined by `(state, sttl, dttl)` | 100% (28 combinations, 0 conflicts) | — | yes, as a lookup table, but see D2c on `sttl` |

**D2b. Zero-coupling rules**

| Rule | Support | Used |
|---|---|---|
| `dpkts = 0` ⇒ `dbytes`, `dmean`, `dload`, `dinpkt`, `djit`, `dloss` = 0 | 100% each | yes |
| Non-TCP ⇒ `swin`, `dwin`, `stcpb`, `dtcpb`, `synack`, `ackdat`, `tcprtt` = 0 | 100% each | yes |
| `spkts ≤ 1` ⇒ `sinpkt = 0` | **18.8%** | **no**. The data contradicts this rule, which was proposed earlier and is now **withdrawn** |
| `spkts ≤ 1` ⇒ `sjit = 0` | **68.8%** | **no**, withdrawn |
| `spkts = 2` ⇒ `sjit = 0` | 99.75% | no (below 99.9%) |

**D2c. Allowed edits** (your changes 1 and 2 applied):

| Feature | Decision | Reason / evidence |
|---|---|---|
| `sbytes` | **allowed, increase only**; repair `smean`, `sload` | Verified formulas |
| `spkts` | **allowed, increase only**; repair `smean`, `rate`, `sload` | Verified formulas. **Note:** for the five 2-packet samples, +1 packet = +50%, which exceeds both budgets, so in practice frozen there |
| `dur` | **excluded** | Your rule 2: `dur` needs `sinpkt` (and `sjit`) to change consistently. The timing formula is **not verified**: `sinpkt = 1000·dur/(spkts−1)` holds for **74.7%** of rows, and 90.7% for 2-packet flows |
| `sttl` | **excluded unless D2c-1 is approved** | `ct_state_ttl` can be repaired by lookup, but see D2c-1 |
| `sloss`, `swin`, `stcpb`, `ackdat` | **excluded** (your rule 1) | Dependencies not verified |
| `sinpkt`, `dinpkt`, `sjit`, `djit` | excluded | No verified formula (`dinpkt` formula: 18.4%) |
| `trans_depth`, `response_body_len` | excluded | Depend on the service |
| `synack`, `dtcpb`, `dwin`, `dloss` | excluded | Teammate table: "No" |
| `dpkts`, `dbytes` | excluded | "Limited" (controlled by the server) |
| `proto`, `service`, `state`, frozen and derived features | not editable | As before |

**D2c-1. `sttl` with lookup repair. NEEDS APPROVAL.**
- For the five 2-packet `INT` samples, the only TTL values seen in the data within ±10%/±25% of 254 are **252** and **255**.
- These combinations are **very rare**: `(INT, 252, 0)` occurs **1 time** and `(INT, 255, 0)` occurs **8 times**, against 63,121 for `(INT, 254, 0)`.
- They would also change `ct_state_ttl` from 2 to 0, and `ct_state_ttl` is one of the strongest features.
- The five TCP samples have **no** allowed TTL alternative.
- **Proposal:** allow an `(state, sttl, dttl)` combination only if it occurs in **≥ 30 sub-train rows**. With this rule `sttl` is **effectively excluded for all 10 samples**.

**Consequence: the attack surface is very small** (important for RQ2).
- Under the verified physical policy, the only possible edits are `sbytes` ↑ (all samples) and `spkts` ↑ (the 5 TCP samples).
- The short flows effectively have **one** editable feature.
- This is a real research finding ("verified constraints leave almost no attack surface"), not a bug. It must be reported per flow type.

**D2c-2. Options to widen the attack surface. NEEDS APPROVAL; recommended: keep it strict and report it.**
- (a) Allow `dur` with a **ratio-preserving** timing repair: `sinpkt`, `sjit`, `dinpkt` and `djit` are multiplied by `dur'/dur`, keeping the sample's own timing ratios. This is an assumption, not a verified formula, so the candidate would be labelled "approximate repair".
- (b) Lower the TTL support threshold in D2c-1.
- (c) Allow `dbytes`/`dpkts` ↑ for TCP flows.

Each option would be a **separately named policy**, so `physical_v1` stays strict.

**D2d. Budget rule.**
- The 5/10-feature limit and the 10%/25% cap count **edited (primary) features only**.
- Recomputed derived features may change by more than the cap. Their changes are reported for every candidate (D9).

**D2e. Feasibility per sample.**
- Count the legal moves per sample × policy × budget before running. A sample with no legal move is reported as "no feasible edit"; it is **not dropped**.

**D2f. Search step grid under `physical_v1`. NEEDS APPROVAL.**
- **Problem:** with only 1–2 editable features and the existing greedy steps (Budget A: +2/5/8/10%, increase only), greedy search has **4 candidates per feature**. It will stop early ("local optimum") and leave much of the LIME budget unused. The random control draws continuous values, so it can try many more distinct candidates. That comparison is not balanced.
- **Proposal:** for `physical_v1` (and the matched random control), use a **finer grid of 1%, 2%, …, 10%** (A) or 1%, …, 25% (B). Unused LIME calls are reported, not hidden.
- The v1 step grid is not changed.

### D3. Ten samples (APPROVED)

| Category | Samples | Flow type |
|---|---|---|
| Generic | 2941, 3040 | UDP/DNS, 2 packets, no reply |
| DoS | 3206 | UDP/DHCP, 2 packets |
| DoS | 27094 | OSPF, 2 packets |
| DoS | 15137 | TCP/SMTP session (CON) |
| Exploits | 4407 | VINES, 2 packets |
| Exploits | 30107, 34186 | TCP/HTTP, TCP/POP3 sessions |
| Fuzzers | 15741, 25740 | TCP/FTP, TCP/HTTP sessions |

- **Selection rule** (written before any v2 run): from the shared 20-sample manifest, take all 5 TCP sessions plus 5 short flows covering every attack type, and keep the 3 v1 samples. The 3 v1 samples were chosen before v2; the other 7 were chosen by flow type and category only.
- **Eligibility check:** every model used (RF, XGBoost, LR, MLP) must predict each sample as attack. A sample failing for one model is reported for that model as "not eligible"; it is not replaced.

### D4. Multi-seed LIME objective (APPROVED)

- **Seeds:** K = 3, seeds **11, 66, 77**.
- **Objective:** mean of the existing objective over the three seeds.
- **Budget counted in actual LIME explanation calls:** the original costs 3, each candidate costs 3.

  | LIME budget | Candidates | Unused calls |
  |---|---|---|
  | 25 | 7 | 1 |
  | 50 | 15 | 2 |
  | 100 | 32 | 1 |

- **Seed families stay disjoint:** held-out 22/33/44/55, order 101–110, control 201–210, defense 301–305.
- **G2 runs shuffled + random only.** Multi-start would get about 2 candidates per restart.

### D5. Multi-start in G1, with matched random controls (APPROVED)

- **Kept in G1 as a secondary method.** Canonical is deterministic (n = 1 per configuration) and descriptive only.
- **Matched random control:** in every comparison it uses the **same** policy, editable features, repair, budget, LIME budget, objective (single- or multi-seed), samples and model. Only the way candidates are proposed differs.
- **Test to implement:** a policy/objective mismatch between a targeted run and its random control must raise an error.
- **Note:** under `physical_v1` with 1–2 editable features, shuffled order and multi-start restarts are nearly the same as canonical. Expect little difference between targeted methods there.

### D6. Current-policy comparison for RQ2 (APPROVED)

- **Policy:** `current_v1` = the unchanged `constraints.py` validator + budget.
- **Grid:** same 10 samples, models, budgets, seeds and LIME budgets as `physical_v1`.
- **Methods:** canonical, shuffled, random (G3).
- **Analysis:** paired by sample.

### D7. G4: diagnostic unconstrained upper bound (APPROVED, label fixed)

- **Label:** G4 is a **diagnostic unconstrained upper bound**, not a controlled comparison of constraints alone.
- **Why:** G4 changes several things at once: allowed features, step sizes, the ±50% cap, no physical checks. So its difference from G1/G3 can't be attributed to "the constraints" alone.
- **Use:** only to answer "can LIME/SHAP on these models be moved beyond seed noise **at all** with large edits?"
- **Settings:**
  - any non-frozen numeric feature, derived features included;
  - up to 10 features, ±50% each;
  - must stay predicted as attack;
  - shuffled + random, 100 LIME calls, 10 samples, RF and XGBoost.

### D8. LIME success criterion (APPROVED; unchanged from v1)

A candidate is a **successful LIME spoof** only if all of these hold:
1. still predicted as attack (P(attack) ≥ 0.5);
2. passes the enforced policy of its condition;
3. passes the **combined held-out test**: top-10 Jaccard **and** Spearman both below the sample's own minimum over the 6 held-out seed pairs (seeds 22/33/44/55);
4. under `physical_v1`: physical status `consistent_with_checked_rules`.

- **Reporting:**
  - top-10-only passes are counted separately, flagging zero-noise samples (e.g. RF 2941);
  - the Part B criterion (seeds 11–55) is reported only for comparison with v1.
- **Under G4:** success is reported as "diagnostic only".

### D9. Always reported, separately from success (APPROVED)

For every candidate:
- **Confidence:** P(attack | x), P(attack | x′), **ΔP**.
- **Perturbation size:**
  - edited primary features (list and count);
  - max and aggregate relative change;
  - **derived features changed by repair**, with their relative change.
- **Constraint status:**
  - `experimental_valid`;
  - teammate-table status;
  - physical status, violations and unverified dependencies.
- **Explanation measures:** search-seed objective, held-out top-10 and Spearman shift, top-1 change, and the SHAP measures (D11).

### D10. Official test set (APPROVED, your rule 9)

- **Locked** for all v2 work.
- **Proposal for the final evaluation (NEEDS APPROVAL, near the end):** one predeclared run on the official test set that reports only the **detection quality** (F1, recall, FPR) of the models used, and the false-alarm rate of DEF-2 on clean test flows.
- **Not allowed:** test-set flows used as attack samples, or test-set results used to change anything.

### D11. SHAP-targeted success criterion (APPROVED in principle; fixed here before any experiment)

**Explainers:**
- RF and XGBoost: Part B's TreeExplainer setup (100 background rows, `default_rng(42)`; RF on the probability scale, XGBoost on the logit scale; one-hot columns summed back to 42 features);
- LR: LinearExplainer;
- MLP: see D12.

**Measures:** computed on |SHAP| over the 42 features, comparing the original x with the candidate x′:
- top-10 Jaccard J(x, x′);
- Spearman ρ(x, x′) with tie-aware ranks;
- top-1 change.

**Reference = the normal disagreement between explainers on the same original flow:**
- J_ref = mean over held-out LIME seeds 22/33/44/55 of the top-10 Jaccard between SHAP(x) and LIME_s(x);
- ρ_ref = the same for Spearman.

These are computed fresh in v2 for every model × sample.

**SHAP success.** A candidate is a **successful SHAP spoof** only if all of these hold:
1. still predicted as attack;
2. passes the enforced policy;
3. physical status OK under `physical_v1`;
4. **J(x, x′) ≤ J_ref and ρ(x, x′) ≤ ρ_ref.**

In plain words: SHAP's explanation must move at least as far as switching to a different explainer would. SHAP is deterministic, so no seed-noise band is needed.

**Also reported, not success criteria:** SHAP top-1 change, and the SHAP shift of the matched random control.

**Example scale from the v1 data (RF):** for sample 2941, the SHAP-vs-LIME top-10 distance is 0.18 and ρ is 0.58. In Part B, random edits kept SHAP ρ ≥ 0.94, so this bar is demanding but meaningful.

**Run settings (G5):**
- the search optimizes the same objective formula, computed on SHAP instead of LIME;
- the budget is counted in SHAP explanation calls (25/50/100, original included);
- the held-out LIME test also applies to G5 candidates, as a transfer check.

### D12. LR and MLP (APPROVED in principle; NEEDS APPROVAL for the training step)

- **Training:**
  - fit `logistic_regression` and the selected `mlp` on the **sub-train split** with the existing preprocessing (`src/unsw/models.py`, selected MLP config);
  - save them as new files `results/models/{key}_validation_pipeline.joblib`, record their hashes in the v2 manifest, and add them to `XAI_MODELS`;
  - RF and XGBoost files are not touched.
- **Training is a new step:** it needs your explicit go-ahead before it runs.
- **Grid (G6):** physical policy, single seed, shuffled + random, Budgets A/B, 100 LIME calls, 10 samples (eligible ones).
- **SHAP:** LR uses LinearExplainer. **MLP: no SHAP in the main grid.** KernelExplainer is too slow; at most a small background on the 10 originals as a check. So RQ1-for-SHAP covers RF, XGBoost and LR.

### D13. Primary analysis (APPROVED; fixed before any v2 run)

**P1, spoofing success (descriptive, RQ1):**
- the number of candidates and samples meeting D8 (LIME) and D11 (SHAP);
- per model × perturbation budget × LIME budget × policy;
- sample-level counts with exact (Clopper–Pearson) 95% intervals (protocol `step6-v2`, pilot). **For the final protocol: Wilson primary, Clopper–Pearson secondary (D17).**

**P2, targeted vs matched random (inferential):**

| Item | Choice |
|---|---|
| Comparison | **Shuffled greedy vs matched random** |
| Condition | `physical_v1`, single-seed objective, **100 LIME calls** |
| Metric | **Held-out Spearman shift**, as a per-sample mean over 10 runs each |
| Test | **Wilcoxon signed-rank**, two-sided, over the 10 samples |
| Effect size | Median difference with a bootstrap 95% CI |
| Family | 4 tests (RF/XGBoost × Budget A/B), Holm-corrected, α = 0.05 |

**If physical search leaves most LIME calls unused:**
- Under D2c the attack surface is tiny, so P2 may show "no difference" mostly because there's nothing to search.
- **Proposed fallback, NOT ADOPTED** (the protocol says `"fallback": "none adopted"`): if more than half of the physical-policy runs stop early, also report the same test under `current_v1` (G3), labelled "secondary, not Holm-family". P2 is reported unchanged, together with lattice coverage and early stopping.

**Secondary** (descriptive or exploratory, labelled as such):
- other methods and budgets;
- multi-seed vs single seed (shuffled, 100 calls, paired by sample);
- physical vs current vs G4 (RQ2);
- model differences (RQ3);
- SHAP-targeted vs matched random;
- LIME bin crossings;
- results per flow type.

**Limitation:** the held-out noise threshold is the minimum of only 6 seed pairs.

---

### D14. Confirmation seeds for the exhaustive evaluation (APPROVED Nov 17, 2026; stage 3B)

- **Seeds:** 401, 402, 403, 404, disjoint from every other seed family (search, held-out, Part B, multi-seed, order, control, defense).
- **Applies to:** every evaluated candidate with `exceeds_heldout_both` (both held-out measures below the original's held-out band), in both physical status groups.
- **Rule:** the candidate and its original are explained under the four seeds; the same combined rule is applied against the original's band on those seeds (6 pairs).
- **Reported as:** `confirmed_lime_success = lime_success AND confirmation_passed`. `lime_success` (D8) is **unchanged** and is reported alongside.
- **Cost:** 4 LIME calls per flagged candidate and per original with a flagged candidate (upper bound: 4 × (candidates + originals) per model).

### D15. The 10-flow exhaustive evaluation is a pilot (Nov 17, 2026; stage 3B)

- Protocol `exhaustive_evaluation.role = "pilot"`: 10 flows, current RF/XGBoost validation models.
- It is **not** the final experimental design (larger fixed per-category sample, nested-CV models, MLP; see the research roadmap) and is **never pooled** with final results.
- Its outcome does not change the success criteria, thresholds or the final sample-selection rule.
- Run folders are named `exhaustive_pilot_*`; a subset run is `exhaustive_smoke_test_*`.

### D16. Pinned model hashes and sample identity (Nov 17, 2026; stage 3B)

- Protocol `expected_model_sha256`: RF `4e49f17d…`, XGBoost `83a34c61…`. Every v2 run refuses to start if a model file is missing or differs; models are never refitted; the exhaustive evaluation re-hashes the files after the run.
- Protocol `sample_identity`: each sample position is pinned to its official-training row index; a run refuses to start if the validation split no longer maps the position to that row, or if the row is not an attack.

### D17. Confidence intervals for the FINAL protocol (APPROVED Nov 17, 2026; recorded before any final result)

- **Final protocol** (the amendment or v3 file for the large-scale experiment, not written yet): every success proportion in P1 is reported with a **Wilson 95% interval as the primary interval** and a **Clopper–Pearson (exact) 95% interval as a secondary sensitivity check**. The unit stays the flow.
- **Protocol `step6-v2` (pilot) is unchanged:** its P1 definition ("exact 95% intervals") stays as declared. The pilot report shows Clopper–Pearson as the v2 definition and Wilson for information only.
- **P2 is unchanged** (Wilcoxon signed-rank, Holm over 4 tests, α = 0.05); the P2 fallback is **not adopted** (D13).

---

## 2. What each research question needs (summary)

| RQ | Needed | Planned in |
|---|---|---|
| RQ1: Can SHAP/LIME be manipulated? | LIME-targeted search + held-out test; SHAP-targeted search + D11; SHAP check of LIME candidates (transfer) | G1, G2, G5 |
| RQ2: How much do constraints reduce success? | Same attack under `current_v1` vs `physical_v1` (paired); G4 as a diagnostic upper bound only | G1 vs G3, G4 |
| RQ3: Do LR, RF, XGBoost and MLP differ? | Same grid on all four models (LR/MLP reduced) | G1, G6 |
| RQ4: Can a defense help without hurting detection? | DEF-1 (5-seed averaged LIME, seeds 301–305, tested against the G2 adaptive attacker) and DEF-2 (consistency checks with the false-alarm rate on clean validation flows). The model is unchanged, so detection quality is unchanged | Post-search |

---

## 3. Stage 1 status (Nov 17, 2026)

**Done:**
- branch `feature/step6-v2` created; Feven's `feature/step6-analysis` merged (`01e9ed4`, no conflicts);
- model hashes confirmed: RF `4e49f17d…`, XGBoost `83a34c61…`, both match the v1 manifest;
- fast tests: **178 passed** (before the new tests), then **17 new fast tests passed**;
- v1 replay (slow): **4/4 saved tasks reproduced exactly**;
- formula check run.

**Not started:** the repair module, protocol v2 and any v2 experiment.

---

## 4. Decisions that still need approval

| # | Item | Recommendation |
|---|---|---|
| 1 | D2c-1: `sttl` only with ≥ 30 rows of support (effectively excluded) | Approve |
| 2 | D2c-2: keep `physical_v1` strict; optionally add a separately named, looser policy | Keep strict; decide later on (a) |
| 3 | D2f: finer step grid (1% steps) for `physical_v1` searches and their random controls | Approve |
| 4 | D10: the content of the single final evaluation on the official test set | Decide near the end |
| 5 | D12: train and save the LR and MLP validation pipelines | Approve before G6 |
| 6 | D13: the fallback rule for P2 | Not adopted (protocol: "none adopted"); P2 unchanged |
| 7 | Professor's deliverables and deadline | Please confirm |
