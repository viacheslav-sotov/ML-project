# Explainable AI for Intrusion Detection Systems: explanation spoofing on UNSW-NB15

Master's research project (Viacheslav Sotov, Feven Ghiday).

**Question:** can small, realistic changes to a malicious network flow keep it detected as an attack but change its **LIME/SHAP explanation** beyond the explainer's own randomness? And how much do realistic (physical) feature constraints limit this?

> **Status (Nov 17, 2026):**
> - **Step 6 v1 complete.** Random Forest, 3 flows, Budget A, 25/50 LIME calls, 123 search tasks + held-out evaluation; **0/186 candidates passed the combined spoofing test**.
> - **Step 6 v2 infrastructure complete:** physical constraint policy, protocol v2, runner, exhaustive evaluator.
> - **No v2 experiment has been run yet.** All results below are labelled as completed or planned.

---

## 1. Repository layout

| Path | Content |
|---|---|
| `src/unsw/` | Library code. `data.py` (data contract), `xai_setup.py` (shared split, models, 20 shared flows), `constraints.py` (**feature taxonomy and validator**), `physical_repair.py` (**physical_v1 policy, repair, physical_report_v2**), `models.py`, `evaluation.py` |
| `experiments/` | Runnable scripts (section 5). The Step 6 core: `unsw_targeted_spoofing.py` (searches), `unsw_step6_runner.py` (**runner**), `unsw_targeted_heldout.py` (held-out LIME), `unsw_constraint_reporting.py`, `unsw_step6_manifest.py`, `unsw_step6_package.py`, `unsw_step6_analysis.py` (**analysis**), `unsw_step6_exhaustive.py`, `unsw_physical_formula_check.py`, `unsw_step6_v2_feasibility.py`, `export_reference_tables.py` |
| `experiments/step6_protocol.json` | **Protocol v1** (predeclared; unchanged since the v1 runs) |
| `experiments/step6_protocol_v2.json` | **Protocol v2** (predeclared before any v2 run) |
| `tests/` | pytest suite (section 6) |
| `results/reference/` | **Feature taxonomy, selected flows, seed families** (generated tables) |
| `results/metrics/shap/shared_validation_sample_identifiers.csv` | The 20 shared validation flows (seed 42) |
| `results/unsw_nb15/network_feature_constraints.csv` | Teammate feature-constraint table (reported, not enforced) |
| `results/research/unsw_threat_model.md` | Threat model and research questions RQ1–RQ4 |
| `results/step6_pilot/`, `results/step6_minimum/` | Step 6 v1 runs: raw outputs incl. per-task histories and **manifests** |
| `results/step6_shared/run_20261106_203918/` | Verified v1 results package (`PACKAGE.md`, `SHA256SUMS`) |
| `results/step6_shared/analysis_run_20261106_203918/` | v1 analysis: report, **18 summary tables**, **8 figures**, metadata |
| `results/step6_v2/` | v2 Stage 1–2 outputs: formula check, feasibility |
| `docs/` | Plans and decisions (`step6_v2_decisions.md`) |
| `results/spoofing_pilot/` | Parts A/B (random perturbations, 600 candidates) |

---

## 2. Setup

**Software used for all results:**
- Python **3.12.0**
- numpy 2.5.3
- pandas 3.0.6
- scikit-learn 1.9.1
- scipy 1.18.1
- xgboost 3.4.1
- lime 0.2.0.1
- shap 0.52.0
- joblib 1.6.0

Every manifest records these versions. The full environment is pinned in `requirements.txt`.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### Data (not in Git)

Download UNSW-NB15 from the UNSW Canberra Cyber site and put the CSV files in `data/raw/`.

**Important:** in this repository the two partition files are **swapped by name**. The code assigns roles by **row count**, not by file name.

| File in `data/raw/` | Role | Rows | SHA-256 |
|---|---|---|---|
| `UNSW_NB15_testing-set.csv` | **official training set** | 175,341 | `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa` |
| `UNSW_NB15_training-set.csv` | **official test set** (locked) | 82,332 | `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559` |

**Split:** `train_test_split(official_train, test_size=0.20, stratify=label, random_state=42)` gives 140,272 sub-train rows and 35,069 validation rows.

**Official test set:** never used by Step 6 (`official_test_used: false` in every manifest). The runner removes it from memory before any search code runs.

### Models (not in Git: `*.joblib` is ignored; the RF file is 186 MB)

| File in `results/models/` | SHA-256 |
|---|---|
| `random_forest_validation_pipeline.joblib` | `4e49f17d9b19747af9e4da472866c4385062d8018582bd7b4872983aefd56531` |
| `xgboost_validation_pipeline.joblib` | `83a34c617a41b886b628ed01257994360f36e8c6863edbc439ee4b201ac4f4b1` |

- **Where they come from:** pipelines fitted on the sub-train split (`src/unsw/xai_setup.py`, SHAP recipe).
- **Get the exact files from the team**, not by refitting. Step 6 checks these hashes and refuses other files. If a file is missing, `load_xai_model()` refits a new model whose hash will differ.

---

## 3. Reproducibility design

**Predeclared protocols:**
- `step6_protocol.json` (v1) and `step6_protocol_v2.json` (v2) fix the models, flows, budgets, seeds, success criteria and primary analysis before any run.
- v1 is unchanged; a test checks its hash against the v1 manifest.

**Manifests:**
- Every run writes `manifest.json` with SHA-256 of data (by role), models, protocol and code functions/files, plus split sizes, flows, LIME configuration, software versions and the Git commit.
- `--resume` refuses if any of these changed.

**Seeds** (all in `results/reference/seed_families.csv`):

| Seed family | Values |
|---|---|
| Split, sample selection, SHAP background | 42 |
| LIME search seed | 11 |
| Held-out LIME seeds | 22, 33, 44, 55 |
| Multi-seed objective (v2) | 11, 66, 77 |
| Order seeds | 101–110 |
| Random-control seeds | 201–210 |
| Defense seeds (v2) | 301–305 |

Every LIME explanation is reseeded.

**Selected flows:**
- v1: 2941, 3206, 4407;
- v2: 2941, 3040, 3206, 27094, 15137, 4407, 30107, 34186, 15741, 25740.

All are from the 20 shared flows (`results/reference/selected_flows.csv`).

**Feature taxonomy and validator:**
- `src/unsw/constraints.py` holds the 42-feature taxonomy (`FEATURE_METADATA`) and `validate_candidate()`.
- `results/reference/feature_taxonomy.csv` combines the taxonomy with the teammate table, the physical_v1 role and the unverified dependencies.

**Physical consistency:**
- Every result reports a physical status: `consistent_with_checked_rules`, `unverified_dependencies` or `violates_checked_rule`.
- "Consistent" means it passes the 11 checked rules. **No candidate is ever called physically valid.**

**Line endings:** text outputs are written with LF; package checksums are LF-normalized, so they verify on any platform.

**Exact v1 replay:** `tests/test_step6_v1_baseline.py` re-runs four saved v1 tasks with the real model and LIME and requires identical results.

---

## 4. Methods in brief

| Item | Description |
|---|---|
| Search objective | `0.7·(1 − top-10 Jaccard) + 0.3·(1 − Spearman)/2` between the original's and the candidate's LIME explanation (same seed) |
| Methods | **Canonical** greedy (fixed feature order); **shuffled** greedy (10 order seeds); **multi-start** greedy (3 climbs sharing one budget); **random control** (same objective, same budget, same legal moves) |
| Budgets | Perturbation budget **A**: ≤ 5 features, ≤ 10% each; **B**: ≤ 10 features, ≤ 25% each, total ≤ 1.5. Explanation budget: **25 / 50 / 100 LIME calls** |
| Success (spoofing) | Still attack; within policy; held-out top-10 Jaccard **and** Spearman below the flow's own seed-noise minimum (seeds 22–55); under physical_v1 also `consistent_with_checked_rules` |
| Constraint policies (v2) | `current_v1` = validator + budget (v1). `physical_v1` = only `sbytes`/`spkts`, increase only, 1% value lattice shared by targeted and random search, derived features (`smean`, `rate`, `sload`, …) recomputed by formulas with ≥ 99.9% support, every candidate re-checked |

---

## 5. Commands

```bash
# --- tests ---
python -m pytest -m "not slow" tests                    # fast suite (250 tests)
python -m pytest tests/test_step6_v1_baseline.py        # v1 checks + exact v1 replay (slow, ~35 s)
python -m pytest tests/test_unsw_step6_analysis.py      # 9 analysis tests

# --- protocols and manifests ---
python experiments/unsw_step6_manifest.py --validate
python experiments/unsw_step6_manifest.py --protocol experiments/step6_protocol_v2.json --validate

# --- Step 6 v1 (as run on Nov 6, 2026) ---
python experiments/unsw_step6_runner.py --models random_forest --samples 2941 3206 4407 --budgets A --explanation-budgets 25 50
python experiments/unsw_targeted_heldout.py --run-dirs results/step6_minimum/run_20261106_203918
python experiments/unsw_step6_package.py --run-dir results/step6_minimum/run_20261106_203918
python experiments/unsw_step6_package.py --verify results/step6_shared/run_20261106_203918
python experiments/unsw_step6_analysis.py

# --- Step 6 v2 (infrastructure ready; runs need approval) ---
python experiments/unsw_physical_formula_check.py       # formula support (done)
python experiments/unsw_step6_v2_feasibility.py         # legal moves per flow (done)
python experiments/export_reference_tables.py           # taxonomy / flows / seeds tables
python experiments/unsw_step6_exhaustive.py --plan-only # exact counts, no LIME
python experiments/unsw_step6_exhaustive.py             # exhaustive evaluation (~11,180 LIME calls)
python experiments/unsw_step6_runner.py --protocol experiments/step6_protocol_v2.json --plan-only \
    --constraint-policies physical_v1 --search-objectives single_seed     # G1 plan
# Any run can be continued with --resume <run folder>.
```

Long runs on Windows: start them detached and check `status.json`. Every task and every exhaustive explanation is saved as soon as it finishes.

---

## 6. Tests

| File | Covers |
|---|---|
| `test_unsw_constraints.py` | taxonomy and validator |
| `test_unsw_spoofing_candidates.py` | Part A generator |
| `test_unsw_targeted_spoofing.py` | searches, accounting, prefix equivalence |
| `test_unsw_targeted_heldout.py` | held-out evaluation |
| `test_unsw_step6_stage0.py`, `test_unsw_step6_runner.py`, `test_unsw_step6_package.py` | protocol, manifest, runner, packaging |
| `test_unsw_step6_analysis.py` | **9 analysis tests** |
| `test_step6_v1_baseline.py` | v1 unchanged (incl. exact replay) |
| `test_unsw_physical_formula_check.py`, `test_unsw_physical_repair.py`, `test_step6_v2_infrastructure.py` | v2 Stages 1–3 |

---

## 7. Results so far

| Experiment | Status | Main result |
|---|---|---|
| Part A/B: 600 random constrained edits (RF, XGB; budgets A/B) | Done (Oct 27–28) | 0/600 changed LIME beyond its seed noise |
| Step 6 v1: RF, 3 flows, Budget A, 25/50 calls (123 tasks) | Done (Nov 6–8) | 0/186 pass the combined test; targeted ≈ random; 0 physically consistent |
| v2 formula check, feasibility (10 flows) | Done (Nov 17) | Only `sbytes`/`spkts` editable under physical_v1; 1,366 valid lattice candidates (333 without unverified dependencies) |
| v2 exhaustive evaluation; G1–G6; DEF-1/2 | **Planned** (needs approval) | — |

---

## 8. Branches

| Branch | Content |
|---|---|
| `main` | Earlier work; nothing from Step 6 merged yet |
| `feature/step6-runner` | Step 6 v1 runner, results, package |
| `feature/step6-analysis` | Feven's v1 analysis |
| `feature/step6-v2` | v1 + analysis merged, plus v2 Stages 1–3 (local until approved) |
