# Report tables: exhaustive_pilot_20261117_180620 (pilot)

Pilot results are never pooled with the final large-scale experiment.

```json
{
  "run_id": "exhaustive_pilot_20261117_180620",
  "run_role": "pilot",
  "state": "complete",
  "git": {
    "branch": "feature/step6-v2",
    "commit": "691e612e6da91d71abd039ac8246ace7987b96bb",
    "dirty": false
  },
  "protocol_sha256": "3c5edcec2b0636cb22f9aeb72db9e8751d3a6c658c69dad6b2854e04e14acf9a",
  "model_sha256_at_start": {
    "random_forest": "4e49f17d9b19747af9e4da472866c4385062d8018582bd7b4872983aefd56531",
    "xgboost": "83a34c617a41b886b628ed01257994360f36e8c6863edbc439ee4b201ac4f4b1"
  },
  "model_sha256_after_run": {
    "random_forest": "4e49f17d9b19747af9e4da472866c4385062d8018582bd7b4872983aefd56531",
    "xgboost": "83a34c617a41b886b628ed01257994360f36e8c6863edbc439ee4b201ac4f4b1"
  },
  "official_test_used": false,
  "explanations_computed": 11180,
  "explanations_reused": 0,
  "cached_explanations": 11180,
  "cache_bad_lines": 0,
  "planned_lime_calls": 11180,
  "confirmation_candidates": 0,
  "confirmation_explanations_computed": 0,
  "confirmation_lime_calls_max": 8944,
  "elapsed_minutes": 23.6,
  "per_model": {
    "random_forest": {
      "explanations": 5590,
      "of_which_confirmation": 0,
      "seconds_per_explanation": 0.1194,
      "lime_time_minutes": 11.1
    },
    "xgboost": {
      "explanations": 5590,
      "of_which_confirmation": 0,
      "seconds_per_explanation": 0.123,
      "lime_time_minutes": 11.5
    }
  }
}
```

## candidates_by_flow

| model_key | sample_position | budget | lattice_points | valid | rejected | rejection_reason | valid_consistent_with_checked_rules | still_attack_consistent_with_checked_rules | valid_unverified_dependencies | still_attack_unverified_dependencies |
|---|---|---|---|---|---|---|---|---|---|---|
| random_forest | 2941 | A | 10 | 10 | 0 |  | 10 | 10 | 0 | 0 |
| random_forest | 2941 | B | 25 | 25 | 0 |  | 25 | 25 | 0 | 0 |
| random_forest | 3040 | A | 10 | 10 | 0 |  | 10 | 10 | 0 | 0 |
| random_forest | 3040 | B | 25 | 25 | 0 |  | 25 | 25 | 0 | 0 |
| random_forest | 3206 | A | 10 | 10 | 0 |  | 10 | 10 | 0 | 0 |
| random_forest | 3206 | B | 25 | 25 | 0 |  | 25 | 25 | 0 | 0 |
| random_forest | 4407 | A | 10 | 10 | 0 |  | 10 | 10 | 0 | 0 |
| random_forest | 4407 | B | 25 | 25 | 0 |  | 25 | 25 | 0 | 0 |
| random_forest | 15137 | A | 120 | 120 | 0 |  | 10 | 10 | 110 | 110 |
| random_forest | 15137 | B | 649 | 580 | 69 | violates_checked_rule | 13 | 13 | 567 | 567 |
| random_forest | 15741 | A | 19 | 19 | 0 |  | 9 | 9 | 10 | 10 |
| random_forest | 15741 | B | 103 | 103 | 0 |  | 25 | 25 | 78 | 78 |
| random_forest | 25740 | A | 21 | 21 | 0 |  | 10 | 10 | 11 | 11 |
| random_forest | 25740 | B | 77 | 77 | 0 |  | 25 | 25 | 52 | 52 |
| random_forest | 27094 | A | 10 | 10 | 0 |  | 10 | 10 | 0 | 0 |
| random_forest | 27094 | B | 25 | 25 | 0 |  | 25 | 25 | 0 | 0 |
| random_forest | 30107 | A | 19 | 19 | 0 |  | 9 | 9 | 10 | 10 |
| random_forest | 30107 | B | 74 | 74 | 0 |  | 24 | 24 | 50 | 50 |
| random_forest | 34186 | A | 29 | 29 | 0 |  | 9 | 9 | 20 | 20 |
| random_forest | 34186 | B | 149 | 149 | 0 |  | 24 | 24 | 125 | 125 |
| xgboost | 2941 | A | 10 | 10 | 0 |  | 10 | 10 | 0 | 0 |
| xgboost | 2941 | B | 25 | 25 | 0 |  | 25 | 25 | 0 | 0 |
| xgboost | 3040 | A | 10 | 10 | 0 |  | 10 | 10 | 0 | 0 |
| xgboost | 3040 | B | 25 | 25 | 0 |  | 25 | 25 | 0 | 0 |
| xgboost | 3206 | A | 10 | 10 | 0 |  | 10 | 10 | 0 | 0 |
| xgboost | 3206 | B | 25 | 25 | 0 |  | 25 | 25 | 0 | 0 |
| xgboost | 4407 | A | 10 | 10 | 0 |  | 10 | 10 | 0 | 0 |
| xgboost | 4407 | B | 25 | 25 | 0 |  | 25 | 25 | 0 | 0 |
| xgboost | 15137 | A | 120 | 120 | 0 |  | 10 | 10 | 110 | 110 |
| xgboost | 15137 | B | 649 | 580 | 69 | violates_checked_rule | 13 | 13 | 567 | 567 |
| xgboost | 15741 | A | 19 | 19 | 0 |  | 9 | 9 | 10 | 10 |
| xgboost | 15741 | B | 103 | 103 | 0 |  | 25 | 25 | 78 | 78 |
| xgboost | 25740 | A | 21 | 21 | 0 |  | 10 | 10 | 11 | 11 |
| xgboost | 25740 | B | 77 | 77 | 0 |  | 25 | 25 | 52 | 52 |
| xgboost | 27094 | A | 10 | 10 | 0 |  | 10 | 10 | 0 | 0 |
| xgboost | 27094 | B | 25 | 25 | 0 |  | 25 | 25 | 0 | 0 |
| xgboost | 30107 | A | 19 | 19 | 0 |  | 9 | 9 | 10 | 10 |
| xgboost | 30107 | B | 74 | 74 | 0 |  | 24 | 24 | 50 | 50 |
| xgboost | 34186 | A | 29 | 29 | 0 |  | 9 | 9 | 20 | 20 |
| xgboost | 34186 | B | 149 | 149 | 0 |  | 24 | 24 | 125 | 125 |

## success_by_group

| model_key | physical_status_group | candidates | flows | exceeds_heldout_top10 | exceeds_heldout_spearman | exceeds_heldout_both | lime_success | confirmation_evaluated | confirmation_passed | confirmed_lime_success | flows_with_lime_success | flows_with_confirmed_success | partb_exceeds_seed_noise |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random_forest | consistent_with_checked_rules | 236 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| random_forest | unverified_dependencies | 872 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| xgboost | consistent_with_checked_rules | 236 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| xgboost | unverified_dependencies | 872 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## success_intervals

| model_key | measure | flows_with_success | flows | clopper_pearson_95_low | clopper_pearson_95_high | wilson_95_low | wilson_95_high |
|---|---|---|---|---|---|---|---|
| random_forest | lime_success (D8) | 0 | 10 | 0.0 | 0.3085 | 0.0 | 0.2775 |
| random_forest | confirmed_lime_success | 0 | 10 | 0.0 | 0.3085 | 0.0 | 0.2775 |
| xgboost | lime_success (D8) | 0 | 10 | 0.0 | 0.3085 | 0.0 | 0.2775 |
| xgboost | confirmed_lime_success | 0 | 10 | 0.0 | 0.3085 | 0.0 | 0.2775 |

## explanation_change

| model_key | sample_position | physical_status_group | candidates | band_top10_jaccard_min | band_spearman_min | band_top1_stable | min_heldout_top10_jaccard | median_heldout_top10_jaccard | min_heldout_spearman | median_heldout_spearman | top1_changed_any_seed | top1_changed_3plus_seeds | exceeds_top1_rule | exceeds_top10 | exceeds_spearman | exceeds_both | max_search_seed_objective |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random_forest | 2941 | consistent_with_checked_rules | 25 | 1.0 | 0.8075 | True | 1.0 | 1.0 | 0.9773 | 0.9773 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0074 |
| random_forest | 3040 | consistent_with_checked_rules | 25 | 1.0 | 0.8128 | True | 1.0 | 1.0 | 0.9784 | 0.9784 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0069 |
| random_forest | 3206 | consistent_with_checked_rules | 25 | 0.8182 | 0.7404 | False | 1.0 | 1.0 | 0.9915 | 1.0 | 11 | 0 | 0 | 0 | 0 | 0 | 0.0039 |
| random_forest | 4407 | consistent_with_checked_rules | 25 | 0.8182 | 0.7686 | False | 1.0 | 1.0 | 0.964 | 0.9645 | 25 | 0 | 0 | 0 | 0 | 0 | 0.0042 |
| random_forest | 15137 | consistent_with_checked_rules | 13 | 0.6667 | 0.7708 | True | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| random_forest | 15137 | unverified_dependencies | 567 | 0.6667 | 0.7708 | True | 1.0 | 1.0 | 0.9999 | 1.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0001 |
| random_forest | 15741 | consistent_with_checked_rules | 25 | 0.8182 | 0.8024 | True | 1.0 | 1.0 | 0.9825 | 0.9995 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0002 |
| random_forest | 15741 | unverified_dependencies | 78 | 0.8182 | 0.8024 | True | 1.0 | 1.0 | 0.9835 | 0.9841 | 0 | 0 | 0 | 0 | 0 | 0 | 0.1282 |
| random_forest | 25740 | consistent_with_checked_rules | 25 | 0.6667 | 0.7613 | True | 0.9545 | 1.0 | 0.9856 | 0.9998 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0004 |
| random_forest | 25740 | unverified_dependencies | 52 | 0.6667 | 0.7613 | True | 0.9545 | 1.0 | 0.9998 | 0.9999 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0001 |
| random_forest | 27094 | consistent_with_checked_rules | 25 | 0.8182 | 0.7495 | True | 1.0 | 1.0 | 0.9406 | 0.9413 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0051 |
| random_forest | 30107 | consistent_with_checked_rules | 24 | 0.8182 | 0.7874 | True | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| random_forest | 30107 | unverified_dependencies | 50 | 0.8182 | 0.7874 | True | 1.0 | 1.0 | 0.9862 | 1.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0038 |
| random_forest | 34186 | consistent_with_checked_rules | 24 | 0.8182 | 0.8088 | True | 1.0 | 1.0 | 0.9859 | 1.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0001 |
| random_forest | 34186 | unverified_dependencies | 125 | 0.8182 | 0.8088 | True | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| xgboost | 2941 | consistent_with_checked_rules | 25 | 0.4286 | 0.623 | True | 0.9091 | 0.9091 | 0.9606 | 0.9606 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0035 |
| xgboost | 3040 | consistent_with_checked_rules | 25 | 0.4286 | 0.6433 | True | 0.8636 | 0.8636 | 0.9624 | 0.9624 | 0 | 0 | 0 | 0 | 0 | 0 | 0.004 |
| xgboost | 3206 | consistent_with_checked_rules | 25 | 0.5385 | 0.5164 | True | 0.8182 | 1.0 | 0.9497 | 1.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.1337 |
| xgboost | 4407 | consistent_with_checked_rules | 25 | 0.4286 | 0.4924 | True | 0.8636 | 0.8636 | 0.9844 | 0.9852 | 0 | 0 | 0 | 0 | 0 | 0 | 0.1355 |
| xgboost | 15137 | consistent_with_checked_rules | 13 | 0.5385 | 0.6117 | True | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| xgboost | 15137 | unverified_dependencies | 567 | 0.5385 | 0.6117 | True | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| xgboost | 15741 | consistent_with_checked_rules | 25 | 0.5385 | 0.5827 | True | 0.9545 | 1.0 | 0.9873 | 0.9993 | 0 | 0 | 0 | 0 | 0 | 0 | 0.1281 |
| xgboost | 15741 | unverified_dependencies | 78 | 0.5385 | 0.5827 | True | 1.0 | 1.0 | 0.986 | 0.9876 | 0 | 0 | 0 | 0 | 0 | 0 | 0.001 |
| xgboost | 25740 | consistent_with_checked_rules | 25 | 0.4286 | 0.5255 | True | 1.0 | 1.0 | 0.9909 | 0.9982 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0015 |
| xgboost | 25740 | unverified_dependencies | 52 | 0.4286 | 0.5255 | True | 1.0 | 1.0 | 0.9991 | 0.9996 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| xgboost | 27094 | consistent_with_checked_rules | 25 | 0.5385 | 0.5517 | True | 0.8636 | 0.8636 | 0.9871 | 0.9871 | 0 | 0 | 0 | 0 | 0 | 0 | 0.1323 |
| xgboost | 30107 | consistent_with_checked_rules | 24 | 0.5385 | 0.5935 | True | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| xgboost | 30107 | unverified_dependencies | 50 | 0.5385 | 0.5935 | True | 0.8636 | 1.0 | 0.9907 | 1.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.1324 |
| xgboost | 34186 | consistent_with_checked_rules | 24 | 0.5385 | 0.5566 | True | 0.9091 | 1.0 | 0.9914 | 1.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0004 |
| xgboost | 34186 | unverified_dependencies | 125 | 0.5385 | 0.5566 | True | 1.0 | 1.0 | 1.0 | 1.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |

## attack_preservation

| model_key | sample_position | valid_unique | evaluated | still_attack | p_attack_original | min_p_attack_candidate | max_abs_delta_p | mean_delta_p |
|---|---|---|---|---|---|---|---|---|
| random_forest | 2941 | 25 | 25 | 25 | 1.0 | 0.995 | 0.005 | -0.0004 |
| random_forest | 3040 | 25 | 25 | 25 | 1.0 | 0.985 | 0.015 | -0.0106 |
| random_forest | 3206 | 25 | 25 | 25 | 1.0 | 0.97 | 0.03 | -0.0174 |
| random_forest | 4407 | 25 | 25 | 25 | 1.0 | 0.91 | 0.09 | -0.0628 |
| random_forest | 15137 | 580 | 580 | 580 | 0.99 | 0.98 | 0.01 | -0.0003 |
| random_forest | 15741 | 103 | 103 | 103 | 0.785 | 0.75 | 0.04 | -0.0008 |
| random_forest | 25740 | 77 | 77 | 77 | 0.81 | 0.775 | 0.035 | -0.0005 |
| random_forest | 27094 | 25 | 25 | 25 | 1.0 | 0.86 | 0.14 | -0.06 |
| random_forest | 30107 | 74 | 74 | 74 | 1.0 | 1.0 | 0.0 | 0.0 |
| random_forest | 34186 | 149 | 149 | 149 | 1.0 | 0.995 | 0.005 | -0.0001 |
| xgboost | 2941 | 25 | 25 | 25 | 1.0 | 1.0 | 0.0 | -0.0 |
| xgboost | 3040 | 25 | 25 | 25 | 0.9999 | 0.9999 | 0.0 | -0.0 |
| xgboost | 3206 | 25 | 25 | 25 | 0.9633 | 0.9307 | 0.0327 | -0.0026 |
| xgboost | 4407 | 25 | 25 | 25 | 0.9996 | 0.9939 | 0.0057 | -0.0049 |
| xgboost | 15137 | 580 | 580 | 580 | 0.9968 | 0.9964 | 0.0004 | -0.0 |
| xgboost | 15741 | 103 | 103 | 103 | 0.6963 | 0.5276 | 0.1687 | -0.0523 |
| xgboost | 25740 | 77 | 77 | 77 | 0.8637 | 0.8187 | 0.1062 | 0.0115 |
| xgboost | 27094 | 25 | 25 | 25 | 0.9985 | 0.9794 | 0.0191 | -0.0164 |
| xgboost | 30107 | 74 | 74 | 74 | 0.9981 | 0.9976 | 0.0005 | -0.0001 |
| xgboost | 34186 | 149 | 149 | 149 | 0.9986 | 0.9969 | 0.0017 | -0.0004 |

## budget_effect

| model_key | smallest_budget | physical_status_group | candidates | max_heldout_top10_shift | min_heldout_spearman | max_relative_change | exceeds_both |
|---|---|---|---|---|---|---|---|
| random_forest | A | consistent_with_checked_rules | 97 | 0.0455 | 0.9412 | 0.1 | 0 |
| random_forest | A | unverified_dependencies | 161 | 0.0 | 0.9838 | 0.1 | 0 |
| random_forest | B | consistent_with_checked_rules | 139 | 0.0455 | 0.9406 | 0.25 | 0 |
| random_forest | B | unverified_dependencies | 711 | 0.0455 | 0.9835 | 0.25 | 0 |
| xgboost | A | consistent_with_checked_rules | 97 | 0.1364 | 0.9606 | 0.1 | 0 |
| xgboost | A | unverified_dependencies | 161 | 0.1364 | 0.9865 | 0.1 | 0 |
| xgboost | B | consistent_with_checked_rules | 139 | 0.1818 | 0.9497 | 0.25 | 0 |
| xgboost | B | unverified_dependencies | 711 | 0.1364 | 0.986 | 0.25 | 0 |

## closest_to_band

| model_key | sample_position | smallest_top10_margin | smallest_spearman_margin | closest_candidate | closest_status | closest_top10_jaccard | closest_spearman | band_top10_min | band_spearman_min |
|---|---|---|---|---|---|---|---|---|---|
| random_forest | 2941 | 0.0 | 0.1698 | s2941_99b7e80e2def4c8c | consistent_with_checked_rules | 1.0 | 0.9773 | 1.0 | 0.8075 |
| random_forest | 3040 | 0.0 | 0.1655 | s3040_cca10f505c00af4f | consistent_with_checked_rules | 1.0 | 0.9784 | 1.0 | 0.8128 |
| random_forest | 3206 | 0.1818 | 0.2512 | s3206_ae68afa80f32a399 | consistent_with_checked_rules | 1.0 | 0.9915 | 0.8182 | 0.7404 |
| random_forest | 4407 | 0.1818 | 0.1954 | s4407_313b1ac75e7e26f2 | consistent_with_checked_rules | 1.0 | 0.964 | 0.8182 | 0.7686 |
| random_forest | 15137 | 0.3333 | 0.229 | s15137_3fcc37f83804c8cc | unverified_dependencies | 1.0 | 1.0 | 0.6667 | 0.7708 |
| random_forest | 15741 | 0.1818 | 0.1801 | s15741_e8f04ed33bbbc292 | unverified_dependencies | 1.0 | 0.9841 | 0.8182 | 0.8024 |
| random_forest | 25740 | 0.2879 | 0.2243 | s25740_bf9ecfe873cb80dd | consistent_with_checked_rules | 0.9545 | 0.9998 | 0.6667 | 0.7613 |
| random_forest | 27094 | 0.1818 | 0.1912 | s27094_fb82967e48e28127 | consistent_with_checked_rules | 1.0 | 0.9406 | 0.8182 | 0.7495 |
| random_forest | 30107 | 0.1818 | 0.1988 | s30107_d78a4be241e3cc70 | unverified_dependencies | 1.0 | 0.9862 | 0.8182 | 0.7874 |
| random_forest | 34186 | 0.1818 | 0.1771 | s34186_b93f98b9bde97dae | consistent_with_checked_rules | 1.0 | 0.9859 | 0.8182 | 0.8088 |
| xgboost | 2941 | 0.4805 | 0.3376 | s2941_99b7e80e2def4c8c | consistent_with_checked_rules | 0.9091 | 0.9606 | 0.4286 | 0.623 |
| xgboost | 3040 | 0.4351 | 0.3191 | s3040_f24a7028e5ca337c | consistent_with_checked_rules | 0.8636 | 0.9624 | 0.4286 | 0.6433 |
| xgboost | 3206 | 0.2797 | 0.4333 | s3206_ae68afa80f32a399 | consistent_with_checked_rules | 0.8182 | 0.9497 | 0.5385 | 0.5164 |
| xgboost | 4407 | 0.4351 | 0.492 | s4407_d9b1f6f7a0bb31d1 | consistent_with_checked_rules | 0.8636 | 0.9844 | 0.4286 | 0.4924 |
| xgboost | 15137 | 0.4615 | 0.3883 | s15137_3fcc37f83804c8cc | unverified_dependencies | 1.0 | 1.0 | 0.5385 | 0.6117 |
| xgboost | 15741 | 0.4161 | 0.4033 | s15741_896dbddbaee3b705 | consistent_with_checked_rules | 0.9545 | 0.9877 | 0.5385 | 0.5827 |
| xgboost | 25740 | 0.5714 | 0.4654 | s25740_3f2ce0889be88289 | unverified_dependencies | 1.0 | 0.9994 | 0.4286 | 0.5255 |
| xgboost | 27094 | 0.3252 | 0.4353 | s27094_b9b62ec87c42deb4 | consistent_with_checked_rules | 0.8636 | 0.9871 | 0.5385 | 0.5517 |
| xgboost | 30107 | 0.3252 | 0.3972 | s30107_d78a4be241e3cc70 | unverified_dependencies | 0.8636 | 0.9907 | 0.5385 | 0.5935 |
| xgboost | 34186 | 0.3706 | 0.4348 | s34186_b93f98b9bde97dae | consistent_with_checked_rules | 0.9091 | 0.9914 | 0.5385 | 0.5566 |

## feature_rank

| model_key | sample_position | seeds | rank_sbytes | rank_spkts | rank_smean | rank_sload | rank_rate | any_changed_feature_in_top10 |
|---|---|---|---|---|---|---|---|---|
| random_forest | 2941 | 4 | 34.2 | 18.5 | 17.8 | 21.0 | 29.5 | False |
| random_forest | 3040 | 4 | 33.5 | 18.0 | 17.5 | 34.5 | 33.8 | False |
| random_forest | 3206 | 4 | 21.8 | 17.8 | 35.2 | 21.2 | 34.8 | False |
| random_forest | 4407 | 4 | 39.2 | 16.8 | 17.5 | 19.2 | 30.5 | False |
| random_forest | 15137 | 4 | 13.5 | 18.2 | 31.5 | 33.5 | 25.0 | False |
| random_forest | 15741 | 4 | 23.5 | 29.0 | 22.2 | 33.5 | 28.0 | False |
| random_forest | 25740 | 4 | 22.0 | 27.2 | 21.8 | 33.5 | 28.0 | False |
| random_forest | 27094 | 4 | 39.2 | 15.2 | 15.5 | 34.5 | 33.8 | False |
| random_forest | 30107 | 4 | 22.2 | 22.2 | 30.2 | 28.0 | 21.8 | False |
| random_forest | 34186 | 4 | 26.2 | 17.8 | 23.8 | 33.0 | 27.5 | False |
| xgboost | 2941 | 4 | 11.5 | 33.2 | 7.5 | 30.2 | 23.0 | True |
| xgboost | 3040 | 4 | 11.8 | 32.8 | 7.8 | 32.0 | 30.5 | True |
| xgboost | 3206 | 4 | 7.2 | 33.2 | 14.5 | 26.2 | 31.2 | True |
| xgboost | 4407 | 4 | 24.5 | 32.2 | 6.8 | 27.5 | 21.5 | True |
| xgboost | 15137 | 4 | 31.0 | 30.2 | 13.8 | 30.8 | 36.8 | False |
| xgboost | 15741 | 4 | 7.8 | 32.5 | 7.2 | 30.5 | 31.0 | True |
| xgboost | 25740 | 4 | 6.0 | 32.8 | 5.0 | 31.0 | 29.5 | True |
| xgboost | 27094 | 4 | 25.2 | 31.2 | 7.5 | 31.2 | 27.5 | True |
| xgboost | 30107 | 4 | 8.8 | 31.5 | 14.5 | 37.8 | 34.8 | True |
| xgboost | 34186 | 4 | 10.0 | 30.0 | 9.0 | 27.5 | 33.8 | True |

## bin_crossings

| sample_position | physical_status_group | candidates | candidates_crossing_any_edge | crossing_sbytes | crossing_spkts | crossing_smean | crossing_sload | crossing_rate |
|---|---|---|---|---|---|---|---|---|
| 2941 | consistent_with_checked_rules | 25 | 25 | 25 | 0 | 25 | 0 | 0 |
| 3040 | consistent_with_checked_rules | 25 | 25 | 25 | 0 | 25 | 0 | 0 |
| 3206 | consistent_with_checked_rules | 25 | 3 | 3 | 0 | 0 | 0 | 0 |
| 4407 | consistent_with_checked_rules | 25 | 25 | 0 | 0 | 25 | 0 | 0 |
| 15137 | consistent_with_checked_rules | 13 | 0 | 0 | 0 | 0 | 0 | 0 |
| 15137 | unverified_dependencies | 567 | 0 | 0 | 0 | 0 | 0 | 0 |
| 15741 | consistent_with_checked_rules | 25 | 6 | 0 | 0 | 6 | 0 | 0 |
| 15741 | unverified_dependencies | 78 | 78 | 0 | 78 | 0 | 0 | 0 |
| 25740 | consistent_with_checked_rules | 25 | 12 | 0 | 0 | 12 | 0 | 0 |
| 25740 | unverified_dependencies | 52 | 0 | 0 | 0 | 0 | 0 | 0 |
| 27094 | consistent_with_checked_rules | 25 | 25 | 0 | 0 | 25 | 25 | 0 |
| 30107 | consistent_with_checked_rules | 24 | 0 | 0 | 0 | 0 | 0 | 0 |
| 30107 | unverified_dependencies | 50 | 15 | 0 | 0 | 15 | 0 | 0 |
| 34186 | consistent_with_checked_rules | 24 | 4 | 0 | 0 | 4 | 0 | 0 |
| 34186 | unverified_dependencies | 125 | 0 | 0 | 0 | 0 | 0 | 0 |
