# UNSW-NB15 Explanation-Spoofing Threat Model

## 1. SYSTEM SETTING

The system is a supervised intrusion-detection pipeline built on UNSW-NB15 flow data.

A network flow is represented by a tabular feature vector containing variables such as protocol, service, state, byte counts, packet counts, timing statistics, and connection-summary counters. The model is trained on the official UNSW training set and is then used to classify a new flow as attack or benign.

In the deployed setting, the workflow is:

- Network traffic is converted into UNSW-style flow features.
- A trained IDS classifier consumes these features.
- The classifier outputs an attack/benign prediction.
- A post-hoc explainer such as SHAP or LIME is applied to the classifier output to explain why the flow was classified as malicious.

This explanation is intended to help a defender or security analyst understand the most influential reasons behind the alert. In this project, the explanation is treated as a diagnostic aid for understanding the model’s decision on a suspicious flow.

The threat model studies a different question: whether an attacker can modify a realistic malicious flow such that the IDS still predicts attack, while the explanation is deliberately changed enough to mislead the analyst about what caused the malicious decision.

---

## 2. ATTACKER GOAL

The attacker aims to produce a feasible modified attack flow x' from an original malicious flow x such that the following hold:

- original prediction = attack
- modified prediction = attack
- explanation for x' differs substantially from explanation for x
- the perturbation remains within the defined realistic feature constraints

Formally, the attacker wants a transformation T such that:

- x is malicious and correctly classified as attack,
- x' = T(x),
- f(x) = attack,
- f(x') = attack,
- E(x') is meaningfully different from E(x),
- x' satisfies the realism constraints for feasible network-flow changes.

Why this matters:

A defender may still receive a correct attack alert, but the explanation may point to misleading features, hide the real attack behavior, or cause an incorrect investigation. The model output remains useful as a detection signal, while the explanatory signal becomes unreliable or intentionally manipulated.

This is an explanation-spoofing problem for a malicious flow under input manipulation, not a dishonest-model-builder attack and not a model-substitution attack.

---

## 3. PREDICTION-PRESERVATION CONDITION

A successful attack must preserve the malicious classification of the flow.

At minimum, the following condition must hold:

- f(x) = attack
- f(x') = attack

The project will also evaluate confidence preservation using the probability or confidence associated with the attack class:

- P(attack | x)
- P(attack | x')
- confidence delta = P(attack | x') - P(attack | x)

The exact tolerance for confidence change will be evaluated and then preregistered before the final attack experiment. The project should not choose an arbitrary confidence threshold prematurely. Instead, the final tolerance rule will be defined in advance of the attack run, based on the project’s evaluation protocol.

The central requirement is:

prediction remains attack, and the confidence change is tracked as part of the attack quality assessment.

---

## 4. EXPLANATION-SPOOFING OBJECTIVE

The explanation-spoofing objective is distinct from the prediction attack objective.

The prediction can remain correct while the explanation becomes misleading.

Let E(x) be the explanation produced for flow x and E(x') be the explanation produced for x'. The attack is considered explanation-spoofing successful when the explanation for x' changes beyond a meaningful threshold relative to the explanation for x.

Possible later metrics include:

- top-k feature overlap
- rank correlation between feature importance rankings
- attribution-vector similarity/distance
- change in top-ranked feature or dominant explanation feature
- targeted explanation manipulation, if later used

The important separation is:

- prediction attack success: x' remains classified as attack
- explanation spoofing success: E(x') differs substantially from E(x)

A flow can satisfy the first condition without satisfying the second. The project is specifically concerned with the second condition while preserving the first.

---

## 5. ATTACKER KNOWLEDGE

The project will define at least two attacker knowledge settings for later ablation.

### A. Explainer-query attacker

This attacker can:

- query the trained IDS for a prediction and confidence,
- query the explainer (e.g., SHAP or LIME),
- observe the explanation values for a chosen flow.

This setting allows optimization against the actual explanation signal used by the analyst.

### B. Model-only attacker

This attacker can:

- query the trained IDS for prediction and confidence,
- but does not directly observe explainer values during optimization.

In this setting, the attacker may still attempt to manipulate explanations indirectly by steering the model decision toward a different feature reliance pattern without direct access to the explanation during optimization.

These two settings are useful for later attacker-knowledge ablation. The project does not implement either setting yet; they are defined here as experimental conditions.

---

## 6. ATTACKER CAPABILITIES

The attacker may modify only features later classified as realistically controllable.

The attacker cannot:

- change the trained model,
- change model parameters,
- change the training data,
- directly change the label,
- directly edit derived features independently if they must be recomputed,
- freely manipulate non-controllable environmental or network features.

The attack is an input-manipulation attack. It does not modify the model or the learning pipeline. It only perturbs feasible flow features while preserving the malicious target class and attempting to substantially alter the explanation.

The exact controllability rules will be specified in the next constraint-specification stage.

---

## 7. FEATURE CONSTRAINT CATEGORIES

The next stage will formally assign UNSW features into the following categories.

### A. Directly controllable

These are features the attacker can change in a realistic way through flow or traffic behavior. Examples may include packet counts, byte counts, timing summaries, or protocol/service metadata that can plausibly vary under an attacker’s control.

### B. Derived or dependent

These features are functions of other features and must remain internally consistent with the modified values. A change to one feature may require corresponding changes to related derived features, otherwise the vector becomes unrealistic or inconsistent.

### C. Non-controllable

These are environmental, context-dependent, or dataset-specific features that cannot reasonably be altered by an attacker in the relevant operating scenario. They are outside the feasible attack surface.

### D. Required or preserved for attack semantics

These features are necessary to keep the flow functionally aligned with an attack pattern. They may not be changed because they define the attack semantics or preserve the malicious intent of the flow.

This category structure is an experimental and design-level formalization. The exact assignment of individual UNSW features will be specified later and must be justified from the dataset semantics and feature definitions rather than assumed.

---

## 8. UNCONSTRAINED VS CONSTRAINED ATTACK

Two evaluation conditions will be defined.

### Unconstrained attack

The attacker is allowed to perturb features without enforcing the complete realistic network-feature rules. This condition acts as an upper-bound or naive explanation attack baseline. It quantifies the maximum attack power when realism is ignored.

### Constrained attack

The attacker is required to satisfy realistic feature constraints, preserve feature consistency, and maintain attack semantics while modifying only feasible features. This is the operationally relevant condition.

Comparing these conditions is central to the research question because it determines whether explanation spoofing depends mainly on the model or on realistic feature-space restrictions. If constrained attacks fail while unconstrained attacks succeed, that suggests that feature realism is a meaningful defense against explanation spoofing.

---

## 9. PERTURBATION BUDGET

Later experiments will include at least two perturbation budgets.

These budgets will eventually be expressed in meaningful original feature units when possible, such as relative changes in packet counts, byte counts, or timing metrics. The exact final budgets will be specified and fixed before the actual attack experiment.

At this stage, the budget is treated as a planning parameter rather than a final experimental choice. The primary point is that explanation spoofing will be studied under bounded perturbation, not unlimited feature editing.

---

## 10. ATTACK SAMPLE ELIGIBILITY

A flow may enter the later attack experiment only if it satisfies the eligibility conditions.

A sample should:

- be a malicious/attack flow,
- be correctly classified as attack by the model before manipulation,
- have a valid SHAP or LIME baseline explanation,
- satisfy any required feature and constraint prerequisites for the chosen attack setting.

Eligibility should later be defined using a fixed, reproducible selection procedure. Where possible, selection should be stratified by attack category to preserve the diversity of malicious flow types. This stage is not implemented yet and is only defined as the sampling protocol for the future attack experiment.

---

## 11. SUCCESS AND FAILURE

### Prediction preservation success

The attack is prediction-preserving if the modified sample remains classified as attack.

### Explanation spoofing success

The attack is explanation-spoofing successful if the explanation changes beyond a threshold established above natural explainer instability.

### Constrained attack success

The attack is fully successful only if both of the following hold:

- prediction remains attack,
- explanation changes beyond the calibrated threshold,
- all feature constraints are satisfied.

### Failure

Failure occurs when one or more required conditions are not met. Examples include:

- the modified sample is reclassified as benign,
- the prediction confidence drops below the acceptable threshold,
- the explanation change is too small,
- feature realism constraints are violated,
- the perturbation leaves the flow semantically invalid or inconsistent.

The explanation-change threshold must eventually be calibrated using the natural SHAP/LIME instability experiment rather than selected arbitrarily. This is essential to avoid misinterpreting regular explainer noise as intentional spoofing.

---

## 12. RESEARCH QUESTIONS CONNECTION

This threat model connects directly to the project’s research questions:

- Can SHAP or LIME explanations for IDS predictions be manipulated under input modification?
- How much does enforcing realistic network constraints reduce attack success?
- Does robustness differ across logistic regression, random forest, XGBoost, and MLP?
- Can a defense reduce explanation spoofing without sacrificing detection quality?

The project studies whether a malicious flow can remain attack-like to the classifier while becoming misleading to the explainer, under realistic feature constraints.

---

## 13. ASSUMPTIONS AND LIMITATIONS

- UNSW-NB15 is an offline benchmark, not live network traffic.
- Some UNSW features are engineered or count-based and may reflect dataset construction rather than raw packet-level behavior.
- Strong predictors may partially reflect dataset-generation structure rather than purely operational network semantics.
- The prepared dataset does not provide a reliable timestamp for implementing the professor’s requested time-based split, so the official UNSW train/test split is preserved and this limitation is documented instead of inventing temporal information.
- Feature controllability must be justified rather than assumed.
- Feasible feature-space perturbations are only an approximation of actual packet-level network manipulation unless they are reconstructed and validated at packet level.

These limitations matter because the project is not claiming that every mathematically valid flow vector corresponds to a physically realizable network event. The threat model is intentionally conservative.

---

## 14. THREAT MODEL SUMMARY TABLE

| Item | Definition |
|---|---|
| attacker objective | Produce a modified malicious flow x' that remains attack-classified but changes the explanation substantially |
| attacker access | Input feature manipulation only; no model replacement or parameter tampering |
| attacker knowledge | Explainer-query attacker or model-only attacker, to be studied later |
| allowed changes | Changes to features later classified as realistically controllable |
| prohibited changes | Model changes, training-data changes, label editing, direct changes to non-controllable features, free modification of derived variables |
| prediction requirement | f(x) = attack and f(x') = attack |
| explanation objective | E(x') differs substantially from E(x) beyond natural explainer instability |
| realism requirement | x' must satisfy realistic feature constraints and preserve attack semantics |
| evaluation conditions | unconstrained and constrained attack settings, compared as experimental baselines |

---

## Research summary

This project studies an input-manipulation threat against IDS explanation systems. The attacker starts from a malicious UNSW flow, perturbs only feasible features, preserves the malicious classification, and attempts to produce a substantially different SHAP or LIME explanation. The goal is not to replace the model, but to manipulate the explanation signal that a defender uses to interpret an alert.

The threat model remains conservative: it separates what is already established in the project from what is only a later experimental definition. It explicitly avoids claiming that any arbitrary UNSW feature vector corresponds to a physically realizable network flow.

---

## Exact attacker goal

The attacker wants to create a feasible modified attack flow x' from an original malicious flow x such that:

- original prediction = attack,
- modified prediction = attack,
- explanation for x' differs substantially from explanation for x,
- perturbation remains within defined realistic feature constraints.

---

## Exact attacker capabilities

The attacker may modify only features later classified as realistically controllable, and cannot change the trained model, model parameters, training data, labels, or non-controllable features. The attack is a feature-space input manipulation, not a model-substitution or scaffolding attack.

---

## Exact attacker knowledge settings

1. Explainer-query attacker: can query the IDS and the explainer for prediction/confidence and explanation values.
2. Model-only attacker: can query the IDS for prediction/confidence but does not directly observe explanation values during optimization.

---

## Prediction-preservation definition

A successful prediction-preserving attack satisfies:

- f(x) = attack
- f(x') = attack

with confidence tracking via:

- P(attack | x)
- P(attack | x')
- confidence delta = P(attack | x') - P(attack | x)

The exact tolerance for confidence change will be preregistered before the final attack experiment.

---

## Explanation-spoofing definition

The explanation is considered spoofed when the explainer output for x' differs substantially from the explainer output for x beyond natural explainer instability, while prediction remains attack. Later metrics may include top-k overlap, rank correlation, attribution similarity, and top-feature changes.

---

## Assumptions and limitations

- UNSW-NB15 is a benchmark dataset, not real-time network traffic.
- Some features are engineered/count-based.
- Strong predictors may reflect dataset-generation structure.
- The dataset does not provide reliable timestamps for a professor-requested time split, so the official split is preserved and this limitation is documented.
- Feature controllability must be justified.
- Feasible perturbations are approximations of realistic network behavior and should not be overclaimed as physically realizable packet-level modifications.

---

## File created

The document was created at:

results/research/unsw_threat_model.md
