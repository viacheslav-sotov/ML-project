# G4 report tables: g4_diagnostic_pilot_20261117_192235 (DIAGNOSTIC ONLY)

G4 successes are never evidence of a physically realizable network attack.

```json
{
  "run_id": "g4_diagnostic_pilot_20261117_192235",
  "run_role": "diagnostic_pilot",
  "state": "complete",
  "git": {
    "branch": "feature/step6-v2",
    "commit": "ffecbc1a12131cb9e4c33da578802a50f441771e",
    "dirty": false
  },
  "g4_protocol_sha256": "b582ca044dfd9c0c4de9dd707d814b7063cb898c3ab7b9bd377e2a854907b021",
  "model_sha256": {
    "random_forest": "4e49f17d9b19747af9e4da472866c4385062d8018582bd7b4872983aefd56531",
    "xgboost": "83a34c617a41b886b628ed01257994360f36e8c6863edbc439ee4b201ac4f4b1"
  },
  "model_sha256_after_run": {
    "random_forest": "4e49f17d9b19747af9e4da472866c4385062d8018582bd7b4872983aefd56531",
    "xgboost": "83a34c617a41b886b628ed01257994360f36e8c6863edbc439ee4b201ac4f4b1"
  },
  "official_test_used": false,
  "tasks_planned": 400,
  "tasks_completed": 400,
  "tasks_failed": 0,
  "stopped_reasons": {
    "explanation_budget_exhausted": 400
  },
  "search_lime_calls": 40000,
  "search_lime_calls_max": 40000,
  "new_explanations_by_model": {
    "random_forest": 22580,
    "xgboost": 22546
  },
  "seconds_per_new_explanation": {
    "random_forest": 0.1218,
    "xgboost": 0.1368
  },
  "lime_minutes_by_model": {
    "random_forest": 45.8,
    "xgboost": 51.4
  },
  "elapsed_minutes": 129.3,
  "checkpoint_rows": 1200,
  "unique_candidates_evaluated": 1002,
  "physical_status_counts": {
    "consistent_with_checked_rules": 4,
    "unverified_dependencies": 65,
    "violates_checked_rule": 933
  },
  "search_score_mismatches": 0,
  "confirmation_candidates": 3,
  "shap_evaluated": 1002
}
```

## targeted_vs_random

| model_key | sample_position | checkpoint | greedy_runs | greedy_objective_mean | greedy_objective_max | greedy_heldout_spearman_shift_mean | greedy_heldout_top10_shift_mean | greedy_exceeds_both | random_runs | random_objective_mean | random_objective_max | random_heldout_spearman_shift_mean | random_heldout_top10_shift_mean | random_exceeds_both |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random_forest | 2941 | 25 | 10 | 0.0047 | 0.0112 | 0.0222 | 0.0091 | 0 | 10 | 0.0144 | 0.0225 | 0.0629 | 0.0136 | 0 |
| random_forest | 2941 | 50 | 10 | 0.0209 | 0.1391 | 0.0378 | 0.0091 | 0 | 10 | 0.0163 | 0.0225 | 0.0831 | 0.0227 | 0 |
| random_forest | 2941 | 100 | 10 | 0.0244 | 0.1394 | 0.0607 | 0.0227 | 0 | 10 | 0.0296 | 0.1391 | 0.0897 | 0.0318 | 0 |
| random_forest | 3040 | 25 | 10 | 0.0071 | 0.0173 | 0.0328 | 0.0045 | 0 | 10 | 0.013 | 0.0189 | 0.0511 | 0.0045 | 0 |
| random_forest | 3040 | 50 | 10 | 0.0105 | 0.0198 | 0.0487 | 0.0227 | 0 | 10 | 0.0142 | 0.0189 | 0.0635 | 0.0136 | 0 |
| random_forest | 3040 | 100 | 10 | 0.0274 | 0.1478 | 0.074 | 0.0364 | 0 | 10 | 0.0521 | 0.137 | 0.0757 | 0.0409 | 0 |
| random_forest | 3206 | 25 | 10 | 0.0359 | 0.1432 | 0.0604 | 0.0545 | 0 | 10 | 0.1136 | 0.1457 | 0.0679 | 0.0682 | 0 |
| random_forest | 3206 | 50 | 10 | 0.0646 | 0.2602 | 0.0905 | 0.0682 | 0 | 10 | 0.1435 | 0.1495 | 0.0888 | 0.0765 | 0 |
| random_forest | 3206 | 100 | 10 | 0.0829 | 0.2751 | 0.1189 | 0.0864 | 0 | 10 | 0.1444 | 0.1495 | 0.0912 | 0.0765 | 0 |
| random_forest | 4407 | 25 | 10 | 0.0729 | 0.1377 | 0.0529 | 0.0455 | 0 | 10 | 0.1497 | 0.2505 | 0.0725 | 0.1136 | 0 |
| random_forest | 4407 | 50 | 10 | 0.102 | 0.1433 | 0.068 | 0.0455 | 0 | 10 | 0.1631 | 0.2505 | 0.0908 | 0.1364 | 0 |
| random_forest | 4407 | 100 | 10 | 0.134 | 0.1567 | 0.1024 | 0.0682 | 0 | 10 | 0.1766 | 0.2521 | 0.1147 | 0.1227 | 0 |
| random_forest | 15137 | 25 | 10 | 0.0803 | 0.1359 | 0.0186 | 0.0045 | 0 | 10 | 0.2483 | 0.2495 | 0.105 | 0.1364 | 0 |
| random_forest | 15137 | 50 | 10 | 0.1079 | 0.1369 | 0.0318 | 0.0182 | 0 | 10 | 0.2497 | 0.259 | 0.109 | 0.1409 | 0 |
| random_forest | 15137 | 100 | 10 | 0.1591 | 0.2497 | 0.0609 | 0.05 | 0 | 10 | 0.2504 | 0.259 | 0.1061 | 0.1409 | 0 |
| random_forest | 15741 | 25 | 10 | 0.1178 | 0.1409 | 0.0231 | 0.0364 | 0 | 10 | 0.172 | 0.2503 | 0.0876 | 0.1273 | 0 |
| random_forest | 15741 | 50 | 10 | 0.1344 | 0.1423 | 0.037 | 0.05 | 0 | 10 | 0.1844 | 0.2503 | 0.0954 | 0.1447 | 0 |
| random_forest | 15741 | 100 | 10 | 0.1472 | 0.2469 | 0.0453 | 0.0773 | 0 | 10 | 0.207 | 0.2554 | 0.0969 | 0.1705 | 0 |
| random_forest | 25740 | 25 | 10 | 0.0292 | 0.1412 | 0.0257 | 0.0364 | 0 | 10 | 0.1428 | 0.1519 | 0.0946 | 0.0992 | 0 |
| random_forest | 25740 | 50 | 10 | 0.056 | 0.2486 | 0.0423 | 0.0447 | 0 | 10 | 0.1449 | 0.1519 | 0.1084 | 0.1159 | 0 |
| random_forest | 25740 | 100 | 10 | 0.1234 | 0.2493 | 0.0775 | 0.1076 | 0 | 10 | 0.1666 | 0.2494 | 0.1078 | 0.1174 | 0 |
| random_forest | 27094 | 25 | 10 | 0.0387 | 0.1386 | 0.0933 | 0.0318 | 0 | 10 | 0.1273 | 0.1494 | 0.0811 | 0.0091 | 0 |
| random_forest | 27094 | 50 | 10 | 0.0399 | 0.1387 | 0.1033 | 0.0273 | 0 | 10 | 0.1417 | 0.1494 | 0.0906 | 0.0091 | 0 |
| random_forest | 27094 | 100 | 10 | 0.0576 | 0.1432 | 0.1277 | 0.0409 | 0 | 10 | 0.1472 | 0.1494 | 0.1131 | 0.0136 | 0 |
| random_forest | 30107 | 25 | 10 | 0.0329 | 0.1505 | 0.0392 | 0.0182 | 0 | 10 | 0.1084 | 0.1551 | 0.1051 | 0.0727 | 0 |
| random_forest | 30107 | 50 | 10 | 0.0352 | 0.1534 | 0.0509 | 0.0182 | 0 | 10 | 0.1365 | 0.1551 | 0.1183 | 0.0864 | 0 |
| random_forest | 30107 | 100 | 10 | 0.0375 | 0.1537 | 0.0638 | 0.0273 | 0 | 10 | 0.1496 | 0.1551 | 0.1231 | 0.0909 | 0 |
| random_forest | 34186 | 25 | 10 | 0.003 | 0.0074 | 0.0135 | 0.0 | 0 | 10 | 0.0078 | 0.011 | 0.0418 | 0.0227 | 0 |
| random_forest | 34186 | 50 | 10 | 0.0049 | 0.0092 | 0.0246 | 0.0091 | 0 | 10 | 0.0092 | 0.0116 | 0.0537 | 0.0273 | 0 |
| random_forest | 34186 | 100 | 10 | 0.0076 | 0.0137 | 0.0353 | 0.0136 | 0 | 10 | 0.0108 | 0.0144 | 0.0629 | 0.0273 | 0 |
| xgboost | 2941 | 25 | 10 | 0.1676 | 0.2972 | 0.1031 | 0.1829 | 0 | 10 | 0.2252 | 0.3907 | 0.1875 | 0.2773 | 0 |
| xgboost | 2941 | 50 | 10 | 0.1966 | 0.3935 | 0.144 | 0.2466 | 0 | 10 | 0.3017 | 0.3907 | 0.284 | 0.3741 | 0 |
| xgboost | 2941 | 100 | 10 | 0.2188 | 0.3989 | 0.1581 | 0.2786 | 0 | 10 | 0.3251 | 0.3971 | 0.2935 | 0.3914 | 0 |
| xgboost | 3040 | 25 | 10 | 0.1418 | 0.2926 | 0.1097 | 0.1647 | 0 | 10 | 0.2759 | 0.2969 | 0.2667 | 0.3617 | 0 |
| xgboost | 3040 | 50 | 10 | 0.1827 | 0.3933 | 0.1471 | 0.2116 | 0 | 10 | 0.2955 | 0.3047 | 0.3034 | 0.391 | 0 |
| xgboost | 3040 | 100 | 10 | 0.2033 | 0.4035 | 0.1836 | 0.2532 | 0 | 10 | 0.3055 | 0.388 | 0.3031 | 0.3916 | 0 |
| xgboost | 3206 | 25 | 10 | 0.3974 | 0.5511 | 0.2513 | 0.3361 | 0 | 10 | 0.4888 | 0.534 | 0.3433 | 0.4126 | 0 |
| xgboost | 3206 | 50 | 10 | 0.4311 | 0.5518 | 0.2662 | 0.3539 | 0 | 10 | 0.4996 | 0.534 | 0.3487 | 0.4158 | 0 |
| xgboost | 3206 | 100 | 10 | 0.4685 | 0.5621 | 0.3086 | 0.3906 | 0 | 10 | 0.5225 | 0.6231 | 0.3437 | 0.4212 | 0 |
| xgboost | 4407 | 25 | 10 | 0.1654 | 0.4544 | 0.086 | 0.1427 | 0 | 10 | 0.4771 | 0.5373 | 0.3706 | 0.4693 | 0 |
| xgboost | 4407 | 50 | 10 | 0.2033 | 0.4602 | 0.1081 | 0.1775 | 0 | 10 | 0.4939 | 0.5373 | 0.3797 | 0.4725 | 0 |
| xgboost | 4407 | 100 | 10 | 0.2919 | 0.4666 | 0.1919 | 0.2789 | 0 | 10 | 0.5264 | 0.5532 | 0.3751 | 0.4666 | 0 |
| xgboost | 15137 | 25 | 10 | 0.0592 | 0.2839 | 0.0925 | 0.0782 | 0 | 10 | 0.231 | 0.2958 | 0.3041 | 0.2856 | 0 |
| xgboost | 15137 | 50 | 10 | 0.0613 | 0.2899 | 0.0976 | 0.08 | 0 | 10 | 0.2861 | 0.2958 | 0.3708 | 0.3499 | 0 |
| xgboost | 15137 | 100 | 10 | 0.0908 | 0.2924 | 0.1369 | 0.1038 | 0 | 10 | 0.2898 | 0.2958 | 0.3792 | 0.3537 | 0 |
| xgboost | 15741 | 25 | 10 | 0.1321 | 0.3936 | 0.0669 | 0.1216 | 0 | 10 | 0.3475 | 0.473 | 0.3086 | 0.3295 | 0 |
| xgboost | 15741 | 50 | 10 | 0.2569 | 0.486 | 0.1874 | 0.2557 | 1 | 10 | 0.4139 | 0.473 | 0.3726 | 0.378 | 0 |
| xgboost | 15741 | 100 | 10 | 0.3101 | 0.574 | 0.2032 | 0.2744 | 1 | 10 | 0.4427 | 0.4913 | 0.3949 | 0.3996 | 1 |
| xgboost | 25740 | 25 | 10 | 0.1823 | 0.4829 | 0.152 | 0.2004 | 0 | 10 | 0.4264 | 0.54 | 0.3444 | 0.4313 | 0 |
| xgboost | 25740 | 50 | 10 | 0.2118 | 0.4829 | 0.1759 | 0.2424 | 0 | 10 | 0.4927 | 0.54 | 0.4033 | 0.4808 | 0 |
| xgboost | 25740 | 100 | 10 | 0.3198 | 0.4853 | 0.2548 | 0.3408 | 0 | 10 | 0.5025 | 0.54 | 0.4014 | 0.4898 | 0 |
| xgboost | 27094 | 25 | 10 | 0.1758 | 0.4518 | 0.1013 | 0.1531 | 0 | 10 | 0.4888 | 0.5901 | 0.3862 | 0.4528 | 0 |
| xgboost | 27094 | 50 | 10 | 0.2005 | 0.4528 | 0.1234 | 0.178 | 0 | 10 | 0.4895 | 0.5901 | 0.3796 | 0.4524 | 0 |
| xgboost | 27094 | 100 | 10 | 0.2741 | 0.4581 | 0.1918 | 0.2812 | 0 | 10 | 0.5183 | 0.5901 | 0.3809 | 0.4702 | 0 |
| xgboost | 30107 | 25 | 10 | 0.2086 | 0.4569 | 0.0964 | 0.1481 | 0 | 10 | 0.3966 | 0.4569 | 0.3047 | 0.2717 | 0 |
| xgboost | 30107 | 50 | 10 | 0.2311 | 0.4569 | 0.103 | 0.1617 | 0 | 10 | 0.436 | 0.4598 | 0.3313 | 0.2788 | 0 |
| xgboost | 30107 | 100 | 10 | 0.2901 | 0.4721 | 0.1513 | 0.2073 | 0 | 10 | 0.4588 | 0.4662 | 0.371 | 0.2896 | 0 |
| xgboost | 34186 | 25 | 10 | 0.03 | 0.1417 | 0.0234 | 0.0318 | 0 | 10 | 0.1447 | 0.1485 | 0.0348 | 0.1765 | 0 |
| xgboost | 34186 | 50 | 10 | 0.0471 | 0.1447 | 0.0415 | 0.0682 | 0 | 10 | 0.1469 | 0.1485 | 0.0374 | 0.1856 | 0 |
| xgboost | 34186 | 100 | 10 | 0.0897 | 0.153 | 0.0518 | 0.0939 | 0 | 10 | 0.1481 | 0.1529 | 0.0453 | 0.1932 | 0 |

## paired_tests

| model_key | checkpoint | metric | flows | greedy_better | random_better | ties | median_difference | wilcoxon_p_two_sided | label |
|---|---|---|---|---|---|---|---|---|---|
| random_forest | 25 | objective | 10 | 0 | 10 | 0 | -0.0761 | 0.002 | exploratory |
| random_forest | 25 | heldout_spearman_shift | 10 | 1 | 9 | 0 | -0.0345 | 0.0059 | exploratory |
| random_forest | 50 | objective | 10 | 1 | 9 | 0 | -0.07 | 0.0098 | exploratory |
| random_forest | 50 | heldout_spearman_shift | 10 | 2 | 8 | 0 | -0.0372 | 0.0098 | exploratory |
| random_forest | 100 | objective | 10 | 0 | 10 | 0 | -0.0515 | 0.002 | exploratory |
| random_forest | 100 | heldout_spearman_shift | 10 | 2 | 8 | 0 | -0.0283 | 0.0488 | exploratory |
| xgboost | 25 | objective | 10 | 0 | 10 | 0 | -0.1799 | 0.002 | exploratory |
| xgboost | 25 | heldout_spearman_shift | 10 | 0 | 10 | 0 | -0.2004 | 0.002 | exploratory |
| xgboost | 50 | objective | 10 | 0 | 10 | 0 | -0.1809 | 0.002 | exploratory |
| xgboost | 50 | heldout_spearman_shift | 10 | 1 | 9 | 0 | -0.2063 | 0.0039 | exploratory |
| xgboost | 100 | objective | 10 | 0 | 10 | 0 | -0.1506 | 0.002 | exploratory |
| xgboost | 100 | heldout_spearman_shift | 10 | 1 | 9 | 0 | -0.1649 | 0.0039 | exploratory |

## heldout_by_group

| model_key | found_by_method | physical_status_group | candidates | flows | exceeds_top10 | exceeds_spearman | exceeds_both | lime_diagnostic_success | confirmation_evaluated | confirmation_passed | confirmed_diagnostic | flows_with_confirmed | lime_top1_changed_any_seed | lime_top1_rule | min_lime_top5_jaccard | min_lime_top10_jaccard | min_lime_spearman | shap_top1_changed | min_shap_top5_jaccard | min_shap_top10_jaccard | min_shap_spearman | shap_d11_meets_both | min_p_attack |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random_forest | random_control | consistent_with_checked_rules | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.6667 | 0.9545 | 0.9565 | 0 | 1.0 | 1.0 | 0.9979 | 0 | 1.0 |
| random_forest | random_control | unverified_dependencies | 10 | 6 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 3 | 0.6667 | 0.8182 | 0.8887 | 0 | 1.0 | 0.8182 | 0.9298 | 0 | 0.785 |
| random_forest | random_control | violates_checked_rule | 204 | 10 | 23 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 52 | 18 | 0.5476 | 0.7424 | 0.8342 | 11 | 0.4286 | 0.5385 | 0.725 | 0 | 0.61 |
| random_forest | shuffled_greedy | consistent_with_checked_rules | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 | 0.8182 | 0.9872 | 0 | 1.0 | 0.8182 | 0.9626 | 0 | 0.855 |
| random_forest | shuffled_greedy | unverified_dependencies | 15 | 6 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.6667 | 0.7803 | 0.9103 | 2 | 0.4286 | 0.6667 | 0.8911 | 0 | 0.705 |
| random_forest | shuffled_greedy | violates_checked_rule | 275 | 10 | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 74 | 16 | 0.5476 | 0.7803 | 0.7898 | 31 | 0.4286 | 0.5385 | 0.6373 | 0 | 0.525 |
| xgboost | random_control | unverified_dependencies | 15 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 6 | 0.4286 | 0.543 | 0.5922 | 1 | 0.6667 | 0.6667 | 0.9369 | 0 | 0.6382 |
| xgboost | random_control | violates_checked_rule | 188 | 10 | 8 | 3 | 1 | 1 | 1 | 1 | 1 | 1 | 98 | 50 | 0.3839 | 0.4322 | 0.5369 | 4 | 0.4286 | 0.5385 | 0.7429 | 0 | 0.6082 |
| xgboost | shuffled_greedy | consistent_with_checked_rules | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.9167 | 0.8636 | 0.9669 | 0 | 1.0 | 0.8182 | 0.8674 | 0 | 0.7507 |
| xgboost | shuffled_greedy | unverified_dependencies | 25 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 5 | 0.4881 | 0.4918 | 0.5822 | 0 | 0.6667 | 0.5385 | 0.7062 | 0 | 0.6697 |
| xgboost | shuffled_greedy | violates_checked_rule | 266 | 10 | 4 | 4 | 2 | 2 | 2 | 2 | 2 | 1 | 42 | 19 | 0.3839 | 0.4881 | 0.5529 | 2 | 0.4286 | 0.4286 | 0.6795 | 0 | 0.6156 |

## heldout_by_flow

| model_key | sample_position | candidates | band_top10_min | band_spearman_min | min_top10_jaccard | min_spearman | exceeds_both | confirmed | closest_joint_margin | closest_changed_features | closest_method | closest_status | shap_j_ref | shap_rho_ref | min_shap_top10_jaccard | min_shap_spearman | shap_d11_meets_both | min_p_attack |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random_forest | 2941 | 51 | 1.0 | 0.8075 | 0.9091 | 0.8781 | 0 | 0 | 0.0706 | sttl;dload;sinpkt;ct_srv_src;ct_src_dport_ltm | random_control | violates_checked_rule | 0.8182 | 0.605 | 0.6667 | 0.8648 | 0 | 0.82 |
| random_forest | 3040 | 53 | 1.0 | 0.8128 | 0.9091 | 0.8791 | 0 | 0 | 0.0662 | sttl;dload;djit;tcprtt;smean;ct_dst_ltm;ct_src_ltm | random_control | violates_checked_rule | 0.8182 | 0.5553 | 0.6667 | 0.8084 | 0 | 0.79 |
| random_forest | 3206 | 49 | 0.8182 | 0.7404 | 0.8258 | 0.8158 | 0 | 0 | 0.0909 | sttl;dload;sinpkt;djit;ct_dst_ltm;ct_src_dport_ltm;ct_dst_src_ltm | shuffled_greedy | violates_checked_rule | 0.7803 | 0.6014 | 0.5385 | 0.7947 | 0 | 0.79 |
| random_forest | 4407 | 52 | 0.8182 | 0.7686 | 0.8182 | 0.8396 | 0 | 0 | 0.0745 | dur;sttl;sinpkt;dinpkt;synack;smean;ct_src_ltm;ct_srv_dst | random_control | violates_checked_rule | 0.6667 | 0.567 | 0.6667 | 0.7882 | 0 | 0.805 |
| random_forest | 15137 | 46 | 0.6667 | 0.7708 | 0.8182 | 0.8697 | 0 | 0 | 0.1515 | sttl;dttl | random_control | unverified_dependencies | 0.4835 | 0.5287 | 0.5385 | 0.9024 | 0 | 0.84 |
| random_forest | 15741 | 51 | 0.8182 | 0.8024 | 0.7424 | 0.8674 | 0 | 0 | 0.0828 | spkts;sbytes;dbytes;sttl;sload;sinpkt;synack;ct_srv_dst | random_control | violates_checked_rule | 0.456 | 0.4332 | 0.5385 | 0.6373 | 0 | 0.525 |
| random_forest | 25740 | 53 | 0.6667 | 0.7613 | 0.7803 | 0.8342 | 0 | 0 | 0.1451 | sttl;sload;tcprtt;ct_srv_dst | shuffled_greedy | violates_checked_rule | 0.511 | 0.5512 | 0.5385 | 0.7464 | 0 | 0.625 |
| random_forest | 27094 | 54 | 0.8182 | 0.7495 | 0.8636 | 0.7898 | 0 | 0 | 0.0909 | dur;sttl;sinpkt | shuffled_greedy | violates_checked_rule | 0.6667 | 0.4032 | 0.6667 | 0.6791 | 0 | 0.6663 |
| random_forest | 30107 | 48 | 0.8182 | 0.7874 | 0.8636 | 0.8387 | 0 | 0 | 0.0513 | dur;dbytes;sttl;dtcpb;tcprtt | shuffled_greedy | violates_checked_rule | 0.4286 | 0.4359 | 0.5385 | 0.7994 | 0 | 0.72 |
| random_forest | 34186 | 49 | 0.8182 | 0.8088 | 0.8636 | 0.8996 | 0 | 0 | 0.1078 | dur;rate;dttl;sinpkt;sjit;swin;ackdat;smean | random_control | violates_checked_rule | 0.4048 | 0.4906 | 0.6667 | 0.8605 | 0 | 0.655 |
| xgboost | 2941 | 54 | 0.4286 | 0.623 | 0.5705 | 0.6739 | 0 | 0 | 0.1419 | spkts;sttl;dload;djit;tcprtt;ct_state_ttl | shuffled_greedy | violates_checked_rule | 0.3571 | 0.3821 | 0.5385 | 0.8458 | 0 | 0.9902 |
| xgboost | 3040 | 52 | 0.4286 | 0.6433 | 0.543 | 0.6704 | 0 | 0 | 0.1145 | dur;sbytes;sttl;sload;dload;smean;ct_dst_ltm | random_control | violates_checked_rule | 0.456 | 0.3898 | 0.5385 | 0.8598 | 0 | 0.9966 |
| xgboost | 3206 | 47 | 0.5385 | 0.5164 | 0.4881 | 0.5918 | 0 | 0 | 0.0754 | sttl;sload;synack;ackdat;ct_srv_src;ct_srv_dst | random_control | violates_checked_rule | 0.4048 | 0.3856 | 0.5385 | 0.8267 | 0 | 0.6183 |
| xgboost | 4407 | 53 | 0.4286 | 0.4924 | 0.4835 | 0.5819 | 0 | 0 | 0.0895 | spkts;rate;sttl;sinpkt;djit;ct_srv_src | shuffled_greedy | violates_checked_rule | 0.3571 | 0.2843 | 0.6667 | 0.8761 | 0 | 0.9871 |
| xgboost | 15137 | 47 | 0.5385 | 0.6117 | 0.613 | 0.6111 | 0 | 0 | 0.0745 | sttl;ackdat | shuffled_greedy | violates_checked_rule | 0.4048 | 0.3982 | 0.6667 | 0.8881 | 0 | 0.99 |
| xgboost | 15741 | 53 | 0.5385 | 0.5827 | 0.4918 | 0.5529 | 3 | 3 | -0.0298 | sttl;sloss;dloss;sinpkt;synack;ct_dst_src_ltm;ct_srv_dst | shuffled_greedy | violates_checked_rule | 0.3363 | 0.2379 | 0.4286 | 0.6795 | 0 | 0.6082 |
| xgboost | 25740 | 51 | 0.4286 | 0.5255 | 0.4322 | 0.5369 | 0 | 0 | 0.0456 | dur;spkts;sbytes;sttl;sinpkt;stcpb;synack;smean;ct_dst_src_ltm | random_control | violates_checked_rule | 0.2525 | 0.0968 | 0.4286 | 0.8447 | 0 | 0.7043 |
| xgboost | 27094 | 48 | 0.5385 | 0.5517 | 0.4835 | 0.5795 | 0 | 0 | 0.0278 | sttl;sinpkt;djit;synack;smean;ct_srv_dst | random_control | violates_checked_rule | 0.4286 | 0.329 | 0.5385 | 0.8481 | 0 | 0.9416 |
| xgboost | 30107 | 43 | 0.5385 | 0.5935 | 0.6346 | 0.5774 | 0 | 0 | 0.0962 | dbytes;sttl;dinpkt;djit;tcprtt;ackdat;smean | random_control | violates_checked_rule | 0.3333 | 0.2044 | 0.6667 | 0.8484 | 0 | 0.9946 |
| xgboost | 34186 | 48 | 0.5385 | 0.5566 | 0.7803 | 0.9071 | 0 | 0 | 0.3505 | spkts;sbytes;dttl;dinpkt;sjit;tcprtt | shuffled_greedy | violates_checked_rule | 0.4048 | 0.4292 | 0.5385 | 0.8723 | 0 | 0.9961 |

## physical

| model_key | physical_status | candidates | exceeds_both | top_violations |
|---|---|---|---|---|
| random_forest | consistent_with_checked_rules | 2 | 0 |  |
| random_forest | unverified_dependencies | 25 | 0 |  |
| random_forest | violates_checked_rule | 479 | 0 | checked rule violated: tcprtt = synack + ackdat (268); checked rule violated: sload = (sbytes - smean) * 8 / dur, within 1% (246); checked rule violated: smean = round(sbytes / spkts) (226); checked rule violated: rate = (spkts + dpkts - 1) / dur (0 if dur = 0), within 1% (218) |
| xgboost | consistent_with_checked_rules | 2 | 0 |  |
| xgboost | unverified_dependencies | 40 | 0 |  |
| xgboost | violates_checked_rule | 454 | 3 | checked rule violated: sload = (sbytes - smean) * 8 / dur, within 1% (250); checked rule violated: smean = round(sbytes / spkts) (227); checked rule violated: rate = (spkts + dpkts - 1) / dur (0 if dur = 0), within 1% (223); checked rule violated: tcprtt = synack + ackdat (210) |

## perturbation_size

| model_key | found_by_method | candidates | edited_features_median | edited_features_max | max_relative_change_median | max_relative_change_max |
|---|---|---|---|---|---|---|
| random_forest | random_control | 215 | 5.0 | 9 | 0.4591 | 0.5 |
| random_forest | shuffled_greedy | 291 | 4.0 | 9 | 0.5 | 0.5 |
| xgboost | random_control | 203 | 5.0 | 10 | 0.4628 | 0.5 |
| xgboost | shuffled_greedy | 293 | 4.0 | 9 | 0.5 | 0.5 |

## ttl_effect

| model_key | edits_ttl_feature | candidates | median_heldout_spearman | min_heldout_spearman | median_heldout_top10_jaccard | exceeds_both | confirmed | shap_d11_meets_both |
|---|---|---|---|---|---|---|---|---|
| random_forest | False | 233 | 0.9644 | 0.862 | 1.0 | 0 | 0 | 0 |
| random_forest | True | 273 | 0.9037 | 0.7898 | 0.9091 | 0 | 0 | 0 |
| xgboost | False | 167 | 0.9603 | 0.7928 | 0.9091 | 0 | 0 | 0 |
| xgboost | True | 329 | 0.6484 | 0.5369 | 0.6404 | 3 | 3 | 0 |

## features

| model_key | feature | is_ttl | candidates_editing | share_of_candidates | in_most_moving | share_of_most_moving | most_moving_definition |
|---|---|---|---|---|---|---|---|
| random_forest | sttl | True | 195 | 0.3854 | 42 | 0.84 | lowest 10% held-out Spearman |
| random_forest | dload | False | 149 | 0.2945 | 15 | 0.3 | lowest 10% held-out Spearman |
| random_forest | sinpkt | False | 134 | 0.2648 | 21 | 0.42 | lowest 10% held-out Spearman |
| random_forest | ackdat | False | 133 | 0.2628 | 14 | 0.28 | lowest 10% held-out Spearman |
| random_forest | tcprtt | False | 107 | 0.2115 | 14 | 0.28 | lowest 10% held-out Spearman |
| random_forest | smean | False | 104 | 0.2055 | 9 | 0.18 | lowest 10% held-out Spearman |
| random_forest | sbytes | False | 104 | 0.2055 | 10 | 0.2 | lowest 10% held-out Spearman |
| random_forest | dinpkt | False | 95 | 0.1877 | 14 | 0.28 | lowest 10% held-out Spearman |
| random_forest | synack | False | 94 | 0.1858 | 8 | 0.16 | lowest 10% held-out Spearman |
| random_forest | dur | False | 93 | 0.1838 | 16 | 0.32 | lowest 10% held-out Spearman |
| random_forest | rate | False | 86 | 0.17 | 9 | 0.18 | lowest 10% held-out Spearman |
| random_forest | sjit | False | 84 | 0.166 | 5 | 0.1 | lowest 10% held-out Spearman |
| random_forest | sload | False | 80 | 0.1581 | 16 | 0.32 | lowest 10% held-out Spearman |
| random_forest | djit | False | 73 | 0.1443 | 9 | 0.18 | lowest 10% held-out Spearman |
| random_forest | ct_srv_dst | False | 69 | 0.1364 | 8 | 0.16 | lowest 10% held-out Spearman |
| random_forest | spkts | False | 66 | 0.1304 | 9 | 0.18 | lowest 10% held-out Spearman |
| random_forest | dttl | True | 64 | 0.1265 | 5 | 0.1 | lowest 10% held-out Spearman |
| random_forest | ct_state_ttl | True | 61 | 0.1206 | 9 | 0.18 | lowest 10% held-out Spearman |
| random_forest | ct_dst_src_ltm | False | 51 | 0.1008 | 9 | 0.18 | lowest 10% held-out Spearman |
| random_forest | ct_srv_src | False | 50 | 0.0988 | 13 | 0.26 | lowest 10% held-out Spearman |
| random_forest | ct_src_ltm | False | 42 | 0.083 | 7 | 0.14 | lowest 10% held-out Spearman |
| random_forest | dpkts | False | 41 | 0.081 | 3 | 0.06 | lowest 10% held-out Spearman |
| random_forest | dmean | False | 39 | 0.0771 | 2 | 0.04 | lowest 10% held-out Spearman |
| random_forest | stcpb | False | 33 | 0.0652 | 0 | 0.0 | lowest 10% held-out Spearman |
| random_forest | dbytes | False | 31 | 0.0613 | 7 | 0.14 | lowest 10% held-out Spearman |
| random_forest | ct_dst_sport_ltm | False | 30 | 0.0593 | 6 | 0.12 | lowest 10% held-out Spearman |
| random_forest | dtcpb | False | 27 | 0.0534 | 5 | 0.1 | lowest 10% held-out Spearman |
| random_forest | sloss | False | 27 | 0.0534 | 2 | 0.04 | lowest 10% held-out Spearman |
| random_forest | ct_dst_ltm | False | 25 | 0.0494 | 4 | 0.08 | lowest 10% held-out Spearman |
| random_forest | swin | False | 25 | 0.0494 | 2 | 0.04 | lowest 10% held-out Spearman |
| random_forest | ct_src_dport_ltm | False | 24 | 0.0474 | 3 | 0.06 | lowest 10% held-out Spearman |
| random_forest | dloss | False | 23 | 0.0455 | 0 | 0.0 | lowest 10% held-out Spearman |
| random_forest | dwin | False | 9 | 0.0178 | 0 | 0.0 | lowest 10% held-out Spearman |
| xgboost | sttl | True | 278 | 0.5605 | 3 | 1.0 | exceeds held-out band on both |
| xgboost | dinpkt | False | 110 | 0.2218 | 0 | 0.0 | exceeds held-out band on both |
| xgboost | sbytes | False | 108 | 0.2177 | 0 | 0.0 | exceeds held-out band on both |
| xgboost | smean | False | 103 | 0.2077 | 0 | 0.0 | exceeds held-out band on both |
| xgboost | sinpkt | False | 97 | 0.1956 | 2 | 0.6667 | exceeds held-out band on both |
| xgboost | tcprtt | False | 92 | 0.1855 | 1 | 0.3333 | exceeds held-out band on both |
| xgboost | dur | False | 90 | 0.1815 | 0 | 0.0 | exceeds held-out band on both |
| xgboost | synack | False | 90 | 0.1815 | 3 | 1.0 | exceeds held-out band on both |
| xgboost | spkts | False | 85 | 0.1714 | 0 | 0.0 | exceeds held-out band on both |
| xgboost | dload | False | 79 | 0.1593 | 0 | 0.0 | exceeds held-out band on both |
| xgboost | sload | False | 78 | 0.1573 | 0 | 0.0 | exceeds held-out band on both |
| xgboost | djit | False | 76 | 0.1532 | 0 | 0.0 | exceeds held-out band on both |
| xgboost | sjit | False | 74 | 0.1492 | 0 | 0.0 | exceeds held-out band on both |
| xgboost | rate | False | 74 | 0.1492 | 0 | 0.0 | exceeds held-out band on both |
| xgboost | ackdat | False | 73 | 0.1472 | 0 | 0.0 | exceeds held-out band on both |
| xgboost | ct_srv_dst | False | 73 | 0.1472 | 3 | 1.0 | exceeds held-out band on both |
| xgboost | ct_src_ltm | False | 67 | 0.1351 | 0 | 0.0 | exceeds held-out band on both |
| xgboost | ct_state_ttl | True | 54 | 0.1089 | 0 | 0.0 | exceeds held-out band on both |
| xgboost | ct_dst_ltm | False | 50 | 0.1008 | 0 | 0.0 | exceeds held-out band on both |
| xgboost | ct_srv_src | False | 48 | 0.0968 | 0 | 0.0 | exceeds held-out band on both |
| xgboost | ct_dst_src_ltm | False | 47 | 0.0948 | 3 | 1.0 | exceeds held-out band on both |
| xgboost | dttl | True | 47 | 0.0948 | 0 | 0.0 | exceeds held-out band on both |
| xgboost | dtcpb | False | 41 | 0.0827 | 0 | 0.0 | exceeds held-out band on both |
| xgboost | stcpb | False | 37 | 0.0746 | 1 | 0.3333 | exceeds held-out band on both |
| xgboost | ct_dst_sport_ltm | False | 36 | 0.0726 | 0 | 0.0 | exceeds held-out band on both |
| xgboost | dpkts | False | 35 | 0.0706 | 1 | 0.3333 | exceeds held-out band on both |
| xgboost | dmean | False | 34 | 0.0685 | 0 | 0.0 | exceeds held-out band on both |
| xgboost | ct_src_dport_ltm | False | 23 | 0.0464 | 0 | 0.0 | exceeds held-out band on both |
| xgboost | dbytes | False | 23 | 0.0464 | 0 | 0.0 | exceeds held-out band on both |
| xgboost | sloss | False | 22 | 0.0444 | 1 | 0.3333 | exceeds held-out band on both |
| xgboost | dloss | False | 19 | 0.0383 | 1 | 0.3333 | exceeds held-out band on both |
| xgboost | swin | False | 11 | 0.0222 | 0 | 0.0 | exceeds held-out band on both |
| xgboost | dwin | False | 6 | 0.0121 | 0 | 0.0 | exceeds held-out band on both |
