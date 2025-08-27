
# 1) Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance

import ydf


# 2) Configuration
TRAIN_CSV = "train.csv"
TEST_CSV  = "test.csv"
OUTPUT_SUBMISSION = "submission2.csv"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

KFOLD_SPLITS = 5
NUM_TREES = 300
MAX_DEPTH = 16
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# 3) Feature engineering utility

def add_basic_features(df):
    for col in ["TotalBsmtSF", "1stFlrSF"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    if '2ndFlrSF' in df.columns:
        df['2ndFlrSF'] = df['2ndFlrSF'].fillna(0)
    else:
        df['2ndFlrSF'] = 0

    if all(c in df.columns for c in ['TotalBsmtSF','1stFlrSF','2ndFlrSF']):
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    if 'YrSold' in df.columns and 'YearBuilt' in df.columns:
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    if 'YrSold' in df.columns and 'YearRemodAdd' in df.columns:
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

    for bcol in ['FullBath','HalfBath','BsmtFullBath','BsmtHalfBath']:
        if bcol in df.columns:
            df[bcol] = df[bcol].fillna(0)
        else:
            df[bcol] = 0
    df['TotalBath'] = df['FullBath'] + 0.5*df['HalfBath'] + df['BsmtFullBath'] + 0.5*df['BsmtHalfBath']

    if 'LotFrontage' in df.columns:
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

    if 'Alley' in df.columns:
        df['Alley'] = df['Alley'].fillna('NoAlley')

    # Fill object dtypes with 'None' to ensure consistency
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna('None')

    return df


# 4) Load & prepare train data
print("Loading training data from:", TRAIN_CSV)
train_df = pd.read_csv(TRAIN_CSV)
print("Raw train shape:", train_df.shape)
if 'Id' in train_df.columns:
    train_df = train_df.drop('Id', axis=1)

# Target transform
train_df['SalePrice'] = np.log1p(train_df['SalePrice'])

# Feature engineering
train_df = add_basic_features(train_df)

print("Columns after FE sample:", train_df.columns[:30].tolist())


# 5) K-Fold CV: train and collect predictions for visualization
kf = KFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=RANDOM_STATE)
cv_scores = []
all_val_preds_log = []
all_val_true_log = []
all_val_idx = []

fold = 0
for train_idx, val_idx in kf.split(train_df):
    fold += 1
    print(f"\nFold {fold}/{KFOLD_SPLITS}")
    train_pd = train_df.iloc[train_idx].reset_index(drop=True)
    val_pd   = train_df.iloc[val_idx].reset_index(drop=True)

    learner = ydf.RandomForestLearner(label='SalePrice', task=ydf.Task.REGRESSION,
                                      num_trees=NUM_TREES, max_depth=MAX_DEPTH)
    model = learner.train(train_pd)

    preds_log = model.predict(val_pd).squeeze()
    rmse = np.sqrt(mean_squared_error(val_pd['SalePrice'], preds_log))
    print(f" Fold {fold} RMSE (log target): {rmse:.5f}")
    cv_scores.append(rmse)

    # store for aggregate visualization
    all_val_preds_log.append(preds_log)
    all_val_true_log.append(val_pd['SalePrice'].values)
    all_val_idx.append(val_pd.index.values)

# concat arrays
all_val_preds_log = np.concatenate(all_val_preds_log)
all_val_true_log = np.concatenate(all_val_true_log)

print("\nCV summary:")
print("Mean RMSE (log target):", np.mean(cv_scores))
print("Std  RMSE (log target):", np.std(cv_scores))


# 6) Visualizations
# Prepare actual-scale values for plotting
all_val_preds = np.expm1(all_val_preds_log)
all_val_true  = np.expm1(all_val_true_log)
residuals = all_val_preds - all_val_true

# 6.1 CV RMSE per fold
plt.figure(figsize=(8,5))
plt.bar(range(1, KFOLD_SPLITS+1), cv_scores)
plt.xlabel('Fold')
plt.ylabel('RMSE (log target)')
plt.title('CV RMSE per fold')
plt.savefig(os.path.join(PLOTS_DIR, 'cv_rmse_per_fold.png'), bbox_inches='tight')
plt.show()

# 6.2 Predicted vs Actual (original price scale)
plt.figure(figsize=(7,7))
plt.scatter(all_val_true, all_val_preds, alpha=0.4, s=8)
lims = [min(all_val_true.min(), all_val_preds.min()), max(all_val_true.max(), all_val_preds.max())]
plt.plot(lims, lims, 'r--')
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Predicted vs Actual (validation, aggregated folds)')
plt.xscale('log')
plt.yscale('log')
plt.savefig(os.path.join(PLOTS_DIR, 'pred_vs_actual.png'), bbox_inches='tight')
plt.show()

# 6.3 Residuals histogram
plt.figure(figsize=(8,5))
plt.hist(residuals, bins=80)
plt.xlabel('Residual (Pred - Actual)')
plt.ylabel('Count')
plt.title('Residuals distribution (validation)')
plt.savefig(os.path.join(PLOTS_DIR, 'residuals_hist.png'), bbox_inches='tight')
plt.show()

# 6.4 Percent error distribution
pct_errors = (all_val_preds - all_val_true) / all_val_true * 100
plt.figure(figsize=(8,5))
plt.hist(pct_errors, bins=80)
plt.xlabel('Percent error (%)')
plt.ylabel('Count')
plt.title('Percent error distribution (validation)')
plt.savefig(os.path.join(PLOTS_DIR, 'percent_error_hist.png'), bbox_inches='tight')
plt.show()



# 7) Feature importance via custom permutation importance on a subset

import time
from sklearn.metrics import mean_squared_error

# --------------- Safety: ensure we have a trained model object ----------------
# Accept either 'full_model' or 'final_model' if one of them exists; otherwise train now.
if 'full_model' in globals():
    model_to_use = full_model
    print("Using existing variable: full_model")
elif 'final_model' in globals():
    model_to_use = final_model
    print("Using existing variable: final_model")
else:
    print("No pre-trained model found in memory (full_model/final_model). Training a model on full training data now...")
    learner_tmp = ydf.RandomForestLearner(label='SalePrice', task=ydf.Task.REGRESSION,
                                          num_trees=NUM_TREES, max_depth=MAX_DEPTH)
    model_to_use = learner_tmp.train(train_df)
    # Optional: assign to global name for subsequent code
    full_model = model_to_use
    print("Trained and assigned to variable 'full_model'.")

# --------------- Prepare subset for permutation importance ----------------
N_PERMUTE_SAMPLES = 1000  # lower this if it's slow (e.g., 500 or 200)
N_REPEATS = 8             # reduce to 2 or 4 for a quick test
print(f"Permutation importance: using {N_PERMUTE_SAMPLES} samples and {N_REPEATS} repeats (adjust N_PERMUTE_SAMPLES/N_REPEATS if too slow)")

subset = train_df.sample(n=min(N_PERMUTE_SAMPLES, len(train_df)), random_state=RANDOM_STATE).reset_index(drop=True)
X_subset = subset.drop(columns=['SalePrice'])
y_subset = subset['SalePrice']

# --------------- Custom permutation importance function (works with YDF model) ---------------
def compute_permutation_importance(model, X, y, feature_names=None, n_repeats=8, random_state=42):
    rng = np.random.RandomState(random_state)
    if feature_names is None:
        feature_names = X.columns.tolist()
    X_base = X.copy().reset_index(drop=True)
    y_base = np.array(y).reshape(-1)
    # baseline prediction and MSE
    preds_base = model.predict(X_base).squeeze()
    baseline_mse = mean_squared_error(y_base, preds_base)

    n_features = len(feature_names)
    importances = np.zeros((n_features, n_repeats), dtype=float)
    start_time = time.time()

    for i, fname in enumerate(feature_names):
        col_vals = X_base[fname].values.copy()
        for r in range(n_repeats):
            permuted = X_base.copy()
            vals = col_vals.copy()
            rng.shuffle(vals)
            permuted[fname] = vals
            preds_perm = model.predict(permuted).squeeze()
            mse_perm = mean_squared_error(y_base, preds_perm)
            importances[i, r] = mse_perm - baseline_mse
        elapsed = time.time() - start_time
        print(f"Permuted feature {i+1}/{n_features} ({fname}); mean importance: {importances[i].mean():.6f}; elapsed {elapsed:.1f}s")

    importance_means = importances.mean(axis=1)
    importance_stds = importances.std(axis=1)
    return importance_means, importance_stds, feature_names

# --------------- Run importance calculation ---------------
print("\nComputing permutation importance (custom implementation)...")
feature_names = X_subset.columns.tolist()
importance_means, importance_stds, feature_names = compute_permutation_importance(
    model_to_use, X_subset, y_subset, feature_names=feature_names, n_repeats=N_REPEATS, random_state=RANDOM_STATE
)

# --------------- Sort and plot top features ---------------
indices = np.argsort(importance_means)[::-1]
top_k = min(20, len(feature_names))

plt.figure(figsize=(10,6))
plt.barh(range(top_k)[::-1], importance_means[indices][:top_k])
plt.yticks(range(top_k)[::-1], [feature_names[i] for i in indices[:top_k]])
plt.xlabel('Mean increase in MSE (permutation)')
plt.title('Permutation feature importance (top features)')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'permutation_importance_top20.png'), bbox_inches='tight')
plt.show()


# 8) Final model already trained (full_model). Predict on test set and save submission
print('\nLoading test set and preparing submission...')
test_df = pd.read_csv(TEST_CSV)
ids = test_df['Id'] if 'Id' in test_df.columns else None
if 'Id' in test_df.columns:
    test_df = test_df.drop('Id', axis=1)

# Apply same feature engineering
test_df = add_basic_features(test_df)

preds_log_test = full_model.predict(test_df).squeeze()
preds_test = np.expm1(preds_log_test)

submission = pd.DataFrame({'Id': ids, 'SalePrice': preds_test})
submission.to_csv(OUTPUT_SUBMISSION, index=False)
print('Submission saved to', OUTPUT_SUBMISSION)
print('Plots saved to', PLOTS_DIR)


# 9) End notes
# - The plots are saved in the 'plots' directory. Adjust NUM_TREES / MAX_DEPTH for more tuning.
# - Permutation importance is model- and data-dependent; for more reliable ranking, increase n_repeats and sample size.
# - You can display the model summary using full_model.describe() if your ydf build supports it.
