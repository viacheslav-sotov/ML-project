# Explanation-Spoofing Pilot Plan

## Scope and status

This document is a small, reproducible pilot design only. It does not run any adversarial attack, retrain any model, overwrite any existing results, or modify the original experiments. It is grounded in the repository’s current validator and shared sample definitions, and it explicitly separates verified project facts from future implementation choices.

## Verified repository facts

The current project already establishes the following:

- `src/unsw/constraints.py` defines a 42-feature canonical predictor set and enforces a conservative feature taxonomy through `validate_candidate()`.
- `src/unsw/constraints.py` currently labels the features into four categories: `DIRECTLY_CONTROLLABLE`, `DERIVED_DEPENDENT`, `NON_CONTROLLABLE`, and `ATTACK_REQUIRED_PRESERVED`.
- `results/research/unsw_threat_model.md` describes the explanation-spoofing goal as: maintain the model’s malicious prediction while changing the explanation enough to mislead a defender, under realistic feature constraints.
- `results/metrics/shap/shared_validation_sample_identifiers.csv` is the shared validation manifest for the XAI comparison and LIME stability runs.
- `experiments/unsw_lime_stability.py` already defines the ranking-based comparison metrics we can reuse: top-1 agreement, top-5/top-10 Jaccard similarity, rank correlation, and direction agreement.

The repository does not implement the actual explanation-spoofing optimization; this is a future pilot proposal.

## Current validator feature classification

The repo’s validator currently treats the following features as directly controllable, derived/dependent, fixed/non-controllable, or preserved for attack semantics.

### Directly controllable (24)

`dur`, `proto`, `service`, `state`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `sttl`, `dttl`, `sloss`, `dloss`, `sinpkt`, `dinpkt`, `sjit`, `djit`, `swin`, `stcpb`, `dtcpb`, `dwin`, `synack`, `ackdat`, `trans_depth`, `response_body_len`

These are the main traffic-behavior features the current validator allows to vary within valid observed domains, with integer, non-negative, and categorical validity checks.

### Derived or dependent (14)

`rate`, `sload`, `dload`, `tcprtt`, `smean`, `dmean`, `ct_srv_src`, `ct_state_ttl`, `ct_dst_ltm`, `ct_src_dport_ltm`, `ct_dst_sport_ltm`, `ct_dst_src_ltm`, `ct_src_ltm`, `ct_srv_dst`

These are treated as contextual or summary metrics and are not to be modified independently by the validator unless the surrounding flow structure is also recomputed consistently.

### Non-controllable / frozen (1)

`is_sm_ips_ports`

This feature is explicitly treated as non-controllable by the current validator.

### Attack-required or preserved (3)

`is_ftp_login`, `ct_ftp_cmd`, `ct_flw_http_mthd`

These are treated as semantics-sensitive features that should not be changed unless the attack semantics are explicitly being redesigned.

Important note: this is the project’s current synthetic constraint model, not a claim that every feature assignment is physically verifiable in the real network. The project intentionally treats these assignments as conservative design assumptions for future robustness experiments.

## Threat model and assumptions

We adopt the project’s current threat model:

- start from a malicious flow that the classifier already predicts as attack,
- keep the malicious decision intact after perturbation,
- perturb only features currently treated as directly controllable,
- keep derived or contextual features consistent or frozen,
- do not modify the model, training data, labels, or held-out test set,
- require the perturbed flow to pass the repo’s validation checks before considering it a valid candidate.

This is not a physical-world proof that every valid UNSW vector corresponds to a realistic network flow. It is a conservative feature-space attack model designed to stress explanation robustness under the project’s existing assumptions.

## Perturbation budgets

We propose two small, reproducible budgets defined on the same feature scales the project already uses.

### Budget A: low-budget proof-of-concept

- Rule: change at most 5 directly controllable features per sample.
- Measurement: for each changed feature, compute the relative change using the original scale:
  
  $\Delta_i = |x'_i - x_i| / \max(1, |x_i|)$
  
  with a cap of 0.10 per changed feature, and the candidate is accepted only if all edited features remain within their observed validity bounds and preserve integer/non-negative constraints.
- Purpose: a minimal explanation-spoofing stress test designed to move the explanation without large feature edits.

### Budget B: moderate-budget robustness check

- Rule: change at most 10 directly controllable features per sample.
- Measurement: use the same per-feature normalized change formula, but allow a cap of 0.25 per feature and enforce a global aggregate budget:
  
  $\sum_i \Delta_i \le 1.5$
  
  where the sum is taken across the edited direct-controllable features only.
- Purpose: a stronger but still bounded test that keeps the flow realistic and within the validator’s feature rules.

For categorical fields (`proto`, `service`, `state`), the pilot will allow only a valid category change within the known dataset categories. No arbitrary categorical values or hidden ID-like features may be introduced.

## Attack success definition

A candidate perturbation is considered successful only if all conditions below hold.

1. Original flow is malicious and correctly predicted as attack.
2. Modified flow remains classified as malicious.
3. All implemented validator constraints pass via `validate_candidate()` from `src/unsw/constraints.py`.
4. The explanation changes according to the ranking metric defined below.
5. The modified sample remains within the chosen budget and respects the selected feature categories.

The attack is therefore a prediction-preserving explanation change, not a label-flip or model-substitution attack.

## Explanation-change metric

We will use the LIME ranking protocol already implemented in `experiments/unsw_lime_stability.py` for the original and modified explanations.

For each sample, let the original LIME ranking be $R$ and the perturbed ranking be $R'$.

- Top-1 change: `1` if the top feature changes, otherwise `0`.
- Top-5 Jaccard similarity: $J_5 = |T_5 \cap T'_5| / |T_5 \cup T'_5|$
- Top-10 Jaccard similarity: $J_{10} = |T_{10} \cap T'_{10}| / |T_{10} \cup T'_{10}|$
- Rank correlation: Spearman-like correlation on the shared absolute-weight ranking across common features, matching the repo’s current stability comparison logic.
- Model confidence change: $\Delta P = P(attack \mid x') - P(attack \mid x)$

A perturbation is a “meaningful explanation change” if the metric exceeds the same-sample five-seed LIME instability band established by the baseline, rather than merely reflecting normal explainer randomness.

## Matched random-perturbation control

For each candidate sample, we will create a matched control that uses the same number of edited features, the same budget regime, and the same valid feature constraints, but the edits are drawn randomly from the allowed direct-controllable set instead of being optimized. This control answers the question: is the observed explanation change larger than what a non-targeted but valid perturbation would produce?

The control is necessary because the project’s existing five-seed LIME stability experiment already shows that explanations vary naturally under the explainer’s stochasticity. We should compare the attack-induced change against that baseline, not against zero change.

## Use of the five-seed LIME stability baseline

The existing LIME seed-stability run (`experiments/unsw_lime_stability.py`) provides a natural null distribution for explanation change. We will use the same five seeds (`11, 22, 33, 44, 55`) to measure the same sample under unchanged input and compare that distribution to the perturbed-flow change metrics.

The pilot will treat the seed-stability results as the reference band for normal explanation instability.

Operationally:

- compute the original explanation for each selected sample under the five seed settings,
- compute the perturbed explanation under the same five seeds,
- compare top-1, top-5/top-10 Jaccard, and rank correlation against the original-seed baseline,
- require the perturbed sample to exceed the realistic instability band defined by the same-sample five-seed distribution.

This keeps the attack definition anchored to project-generated evidence instead of an arbitrary threshold.

## Selected pilot samples (from the shared 20-sample manifest)

These samples are drawn from the existing shared validation manifest and are not from the held-out test set.

| validation_position | official_id | attack_cat | status |
|---:|---:|---|---|
| 2941 | 134318 | Generic | malicious validation sample |
| 3206 | 64810 | DoS | malicious validation sample |
| 4407 | 92635 | Exploits | malicious validation sample |

These are three concrete pilot candidates for a future implementation. We are not claiming they are the only valid choices; they simply provide a compact, diverse starting set from the project’s existing 20-sample malicious validation subset.

## Success criteria for the pilot

A pilot run is considered meaningful when all of the following happen for at least one selected sample under a budgeted perturbation:

- original flow is malicious and correct,
- modified flow stays malicious,
- all `validate_candidate()` constraints pass,
- the explanation changes more than the same-sample five-seed LIME stability band,
- the modified sample remains inside a defined bounded budget.

If a random control reaches similar explanation change without any targeted optimization, then the pilot should report the effect as weak or non-discriminative rather than as a successful explanation-spoofing result.

## Next implementation steps

1. Build a helper that loads one of the three selected validation samples and its original LIME explanation.
2. Create a constrained candidate generator that edits only the direct-controllable features and enforces the validator.
3. Implement the two budgets as explicit, reproducible search limits.
4. Evaluate a small random search or greedy candidate loop under each budget.
5. Compare the resulting explanation change against the five-seed LIME stability baseline and the matched random-perturbation control.
6. Save all accepted candidates together with their original and perturbed explanations, model confidence, and validator output.

This is intentionally small enough to complete in one focused experimental pass without touching the core model training or original result files.

## Progress Report Summary

We reviewed the existing validator, threat model, and shared sample manifest, and we confirmed that the current project defines a conservative feature taxonomy rather than an implemented attack. We also verified that the shared XAI sample set is a 20-sample malicious validation subset and that the LIME stability code already provides the ranking metrics we want to reuse. We will keep the pilot small and bounded: three selected malicious samples, two perturbation budgets, and a matched random control against the five-seed LIME baseline. Our next step is to implement the candidate generator and evaluation loop without changing the original model-training or result files.
