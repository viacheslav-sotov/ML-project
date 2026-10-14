# Shared SHAP validation manifest and LIME reuse notes

This folder contains the canonical shared validation sample set for the UNSW-NB15 explanation work.

## Shared sample manifest

The file shared_validation_sample_identifiers.csv stores the validation positions that both Random Forest and XGBoost use for the same 20 correctly classified malicious validation samples. These are the positions in the internal validation split, not the official dataset row IDs.

Use this manifest as the source of truth before any later explanation step, including LIME. Do not generate a second model-specific sample list and assume it matches.

## Reusing the fitted pipelines for LIME

1. Load the fitted pipeline objects from results/models.
2. Read the shared validation positions from shared_validation_sample_identifiers.csv.
3. Select the same rows from the validation split by position.
4. Use the original feature DataFrame and apply the same preprocessing pipeline before building the LIME explainer.
5. Keep the same 20 samples for both models so any later comparison is based on one aligned set.

The fitted model artifacts are:

- random_forest_validation_pipeline.joblib
- xgboost_validation_pipeline.joblib

## Avoid mixing feature spaces

The project uses a ColumnTransformer with one-hot encoding for categorical features and passthrough for numeric features. This means there are two distinct spaces that must not be mixed:

- original UNSW-NB15 columns before preprocessing
- transformed feature columns after one-hot encoding (for example, cat__state_INT and num__sttl)

For LIME, keep one feature space consistently throughout the explanation. If you use the pipeline as a whole, pass the original DataFrame to the pipeline and keep the explanation in the transformed feature space used by the model, or explicitly document the selected feature names. Do not concatenate original columns with one-hot encoded columns, because that changes the meaning of the feature attribution and can break a comparison across models.

## Recommended workflow

- Use the shared validation manifest for the exact sample IDs.
- Reuse the saved fitted pipelines rather than retraining.
- Keep feature-space consistency intact across all later explanation methods.
- Compare only on the same aligned validation subset.
