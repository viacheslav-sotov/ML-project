# UNSW-NB15 feature-level predictive and leakage analysis

## Scope and rules
This analysis used the official UNSW-NB15 training set only (175,341 rows). The official 82,332-row test set was not used, and the canonical UNSW data loading logic was retained. The `id`, `attack_cat`, and `label` columns were not used as predictors. `label` was the binary target, and `attack_cat` was excluded from prediction and kept for later domain analysis only.

## Method
For every original usable predictor, a reproducible stratified train/validation split was created on the official training data with `random_state=42`. A depth-1 `DecisionTreeClassifier` was fit on each feature individually, with one-hot encoding applied inside the pipeline for categorical features and no encoder fit prior to splitting.

## High-level result
The dataset does not show the kind of near-perfect single-feature separability seen in the previous CIC experiment, but several features are still unusually strong and deserve attention before later explanation work. In particular, count-based flow-statistics and state/protocol-related features rise far above the median. Some of these are plausible indicators of malicious behavior, while others may reflect dataset-generation patterns or target-proxy structure.

## Top 15 strongest individual features
| rank | feature | type | accuracy | precision | recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | sttl | numeric | 0.9220 | 0.8992 | 0.9973 | 0.9457 | 0.8795 |
| 2 | ct_state_ttl | numeric | 0.9045 | 0.8821 | 0.9924 | 0.9340 | 0.8548 |
| 3 | dload | numeric | 0.8589 | 0.8529 | 0.9579 | 0.9023 | 0.8029 |
| 4 | dmean | numeric | 0.8033 | 0.8601 | 0.8492 | 0.8546 | 0.7774 |
| 5 | dloss | numeric | 0.7711 | 0.7898 | 0.9043 | 0.8432 | 0.6958 |
| 6 | sbytes | numeric | 0.7555 | 0.7903 | 0.8722 | 0.8292 | 0.6895 |
| 7 | sloss | numeric | 0.7473 | 0.7900 | 0.8563 | 0.8218 | 0.6856 |
| 8 | spkts | numeric | 0.7409 | 0.7803 | 0.8622 | 0.8192 | 0.6724 |
| 9 | is_sm_ips_ports | numeric | 0.6966 | 0.6917 | 1.0000 | 0.8177 | 0.5250 |
| 10 | response_body_len | numeric | 0.6925 | 0.6917 | 0.9890 | 0.8141 | 0.5249 |
| 11 | ct_dst_sport_ltm | numeric | 0.6806 | 0.6806 | 1.0000 | 0.8100 | 0.7022 |
| 12 | djit | numeric | 0.6806 | 0.6806 | 1.0000 | 0.8100 | 0.6919 |
| 13 | sjit | numeric | 0.6806 | 0.6806 | 1.0000 | 0.8100 | 0.6872 |
| 14 | proto | categorical | 0.6806 | 0.6806 | 1.0000 | 0.8100 | 0.6779 |
| 15 | swin | numeric | 0.6806 | 0.6806 | 1.0000 | 0.8100 | 0.6778 |

## Unusually strong single-feature predictors
The strongest performers are mostly count-derived features and relationship statistics. These are not automatically invalid, but they are highly structured and may be more sensitive to synthetic dataset generation than to genuine user-network behavior. Features that materially exceed the rest of the pack are therefore flagged for additional review before SHAP/LIME.

## Suspected identifier-like or target-proxy features
The strongest features need to be inspected in context. We distinguish between legitimate predictive network behavior and suspicious leakage/proxy behavior as follows.

### A. Legitimate predictive network behavior
- Flow-count variables such as `ct_dst_src_ltm`, `ct_srv_dst`, and similar connection counters can be strongly associated with attack traffic.
- State and protocol summary variables can have genuine discrimination power because they reflect the communication pattern of malicious or benign sessions.

### B. Suspicious leakage / proxy / dataset-artifact behavior
- Count-based features that summarize repeated relationships or traffic patterns may behave like proxies for attack categories or generation artifacts.
- Features with very high purity or near-constant behavior are especially suspicious because they may encode the design of the dataset rather than realistic network semantics.
- `proto`, `service`, and `state` are not leakage by themselves, but they are highly sensitive to synthetic IDS labeling and can act as coarse proxies when combined with engineered counts.

The distinction is important: a feature is not automatically leakage merely because it is predictive. It becomes suspicious when it behaves like an ID or an engineered data-generation artifact rather than a real operational characteristic.

## Near-constant features
- is_ftp_login (numeric): 4 unique values; most common value share = 98.54%
- ct_ftp_cmd (numeric): 4 unique values; most common value share = 98.54%
- is_sm_ips_ports (numeric): 2 unique values; most common value share = 98.42%

## High numeric correlation / redundancy
A Pearson correlation analysis was performed on the original numeric predictors, using a threshold of absolute correlation >= 0.95. The purpose was to identify redundant numeric structure without removing variables.
- is_ftp_login / ct_ftp_cmd: r = 1.0000
- dbytes / dloss: r = 0.9965
- sbytes / sloss: r = 0.9961
- swin / dwin: r = 0.9901
- ct_srv_src / ct_srv_dst: r = 0.9803
- dpkts / dloss: r = 0.9786
- ct_dst_src_ltm / ct_srv_dst: r = 0.9724
- dpkts / dbytes: r = 0.9719
- spkts / sloss: r = 0.9711
- ct_srv_src / ct_dst_src_ltm: r = 0.9671
- spkts / sbytes: r = 0.9638
- ct_dst_ltm / ct_src_dport_ltm: r = 0.9621

## Distribution observations from the strongest features
The strongest individual features generally show a clear shift between normal and attack labels, but the separation is seldom perfect. For most top features, the benign and attack distributions differ strongly while still overlapping to a meaningful degree. This suggests that the dataset is informative and not trivially separable, but that some engineered features still carry a strong synthetic signal.

## Overall conclusion
1. Does UNSW-NB15 contain any single feature that nearly perfectly predicts the binary label?
   - Not according to the official training-set evidence. No feature behaves like a near-perfect label oracle.

2. Is there evidence of obvious target leakage among the model predictors?
   - There is no clear evidence of direct leakage in the sense of a direct label proxy, but some engineered count features and protocol/state features should be treated as suspicious because they may encode generation patterns or category proxies.

3. Which features deserve special attention before SHAP/LIME?
   - The strongest flow-count features and protocol/state-derived summary statistics are the main candidates for careful review.

4. Are highly predictive features legitimate network characteristics or suspicious dataset artifacts?
   - Some are legitimate behavior indicators, but a subset may reflect synthetic dataset design and should be interpreted as structured proxy features rather than purely real-world network behavior.

5. Does UNSW-NB15 appear to have the same extreme separability problem that caused concern with the previous CIC experiment?
   - No. The evidence does not suggest the same near-pathological single-feature separability, although some engineered count features remain unusually strong and should be monitored carefully.

6. Based on this analysis, is it reasonable to continue to the XAI stage with the current feature set?
   - Yes, with caution. The feature set appears usable for explanation-stage work, but the strongest features should be explicitly reviewed for proxy-like structure before interpretation begins.

## Methodological limitations
- This analysis uses a single stratified internal validation split on the official UNSW training set and does not use the official test set.
- The decision-stump baseline is intentionally simple; it is a red-flag screen, not a final model-selection tool.
- Correlation and single-feature analysis can highlight engineered signal but cannot prove leakage without deeper domain investigation.
- Because this is a synthetic intrusion-detection dataset, some highly predictive variables may encode dataset-generation assumptions rather than true real-world network behavior.