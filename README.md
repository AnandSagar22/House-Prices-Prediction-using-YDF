# House-Prices-Prediction-using-YDF
This project builds an end-to-end regression pipeline for the Kaggle competition "House Prices — Advanced Regression Techniques" using Yggdrasil Decision Forests (YDF).
It trains a Random Forest on the Ames housing dataset and produces predictions for the test set. The pipeline includes basic feature engineering, 5‑fold cross validation, model inspection, visualizations of prediction quality, and a final submission CSV.
-------------------------------------------------------------------------------------------------------------------

#Why this project matters

The dataset is realistic and mixed-type (numerical + categorical), which makes it a great exercise in data cleaning, feature engineering, and model selection.

Decision forests (Random Forests / Gradient Boosted Trees) are strong baselines for tabular data — they handle categorical data and missing values robustly and give interpretable feature importance.

This project demonstrates a full ML workflow: data preparation → cross-validation → model training → evaluation → visualization → submission. It’s practical and directly reusable for Kaggle-style problems.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# What the repository contains

House_Prices_Prediction_YDF.py — main pipeline (train, CV, visuals, final model, submission).

submission.csv (generated) — final predictions for the test set (Id, SalePrice).

plots/ (generated) — visual outputs saved by the script:

cv_rmse_per_fold.png — RMSE per CV fold

pred_vs_actual.png — predicted vs actual scatter (validation)

residuals_hist.png — residuals histogram

percent_error_hist.png — percent error distribution

permutation_importance_top20.png — top-20 features by permutation importance
