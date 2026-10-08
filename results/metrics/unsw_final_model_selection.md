# UNSW-NB15 final model selection review

## Recommended final four model families
- logistic_regression ? linear benchmark, stable and interpretable, and important as the reference point.
- random_forest ? bagged tree ensemble with strong IDS performance and low variance across folds.
- xgboost ? boosted tree ensemble with strong discrimination and a distinct non-linear tree family.
- mlp ? neural network family with a distinct architecture and non-linear feature interactions.

## Rejected from the final XAI experiment
- decision_tree ? useful as a simple reference but too shallow and less stable for the final study.
- extra_trees ? high-performing but redundant with random forest in the same bagged-tree family.

## Why these four are useful to the research question
These four models span the four major architecture families: linear, bagged tree ensemble, boosted tree ensemble, and neural network. That diversity matters for later XAI robustness testing more than pure rank-by-accuracy.

## Tuning review
- Random Forest baseline: n_estimators=200, random_state=42, n_jobs=-1.
- Best RF randomized search: {'model__n_estimators': 300, 'model__min_samples_split': 2, 'model__min_samples_leaf': 2, 'model__max_features': 'log2', 'model__max_depth': 10, 'model__class_weight': None}
- Best RF search CV F1: 0.9388824697849242
- Baseline RF CV F1: 0.9706396341469276
- Conclusion: RF randomized search did not improve the baseline and was not treated as a meaningful gain.

- Extra Trees baseline: n_estimators=200, random_state=42, n_jobs=-1.
- Best ET randomized search: {'model__n_estimators': 300, 'model__min_samples_split': 2, 'model__min_samples_leaf': 2, 'model__max_features': 'log2', 'model__max_depth': 10, 'model__class_weight': None}
- Best ET search CV F1: 0.9488410344405934
- Baseline ET CV F1: 0.9700823830516361
- Conclusion: ET randomized search did not improve the baseline and was not treated as a meaningful gain.

## Small tuning performed
- XGBoost: 4 validation-set configurations evaluated around the current baseline; the best configuration was selected by validation F1.
- MLP: 4 validation-set configurations evaluated around the current baseline; the best configuration was selected by validation F1.
- Logistic regression: retained at baseline because it serves as the stable linear benchmark.
