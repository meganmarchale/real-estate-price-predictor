# %% [markdown]
# ## Baseline CatBoost Regression (GPU Fallback Compatible)
# 
# This notebook cell trains **two baseline CatBoost models** to predict real estate prices:
# 
# - One with **all filtered features** (low-variance features removed)
# - One with **top 30 features** selected via Random Forest importance
# 
# The goal is to quickly evaluate CatBoost performance **without Optuna tuning**, while ensuring **GPU compatibility with CPU fallback**.
# 
# 
# ###  Library Imports
# 
# Includes:
# - `catboost` for gradient boosting on decision trees
# - `sklearn` for feature selection and evaluation
# - Custom modules: 
#   - `DataLoader` to load the cleaned ML dataset
#   - `ModelEvaluator`, `ModelVisualizer`, `ExperimentTracker` for tracking, evaluation and plotting
# 
# 
# 
# ### Data Preparation
# 
# - Loads the cleaned dataset via `DataLoader`
# - Removes low-variance features with `VarianceThreshold`
# - Uses `RandomForestRegressor` to extract the **top 30 most important features**
# 
# 
# ### Model Training with GPU Fallback
# 
# Defines a helper `train_with_fallback(...)` function that:
# - Tries to train CatBoost on GPU (`task_type="GPU"`)
# - If it fails (e.g., no GPU), it **automatically falls back to CPU**
# 
# Models are trained on:
# - `X_reduced`: all filtered features
# - `X_top`: only the top 30 features
# 
# 
# ### Model Evaluation
# 
# Each model is evaluated using:
# - **MAE** (Mean Absolute Error)
# - **RMSE** (Root Mean Squared Error)
# - **R<sup>2</sup>** (Coefficient of Determination)
# 
# Results are logged using `ExperimentTracker`.
# 
# 
# ### Diagnostics & Visualization
# 
# - Summary of model metrics is displayed in tables
# - Diagnostic plots generated with `ModelVisualizer`:
#   - Residuals
#   - Prediction distribution
#   - Error across price ranges
# - A **side-by-side residual plot** compares both models
# 
# 
# ### Test Mode Support
# 
# If `TEST_MODE = True`:
# - Dataset size and number of iterations are reduced
# - Useful for debugging and fast iterations
# 
# ### Summary
# 
# This cell provides a **fast, robust, GPU-aware CatBoost baseline**, ideal for:
# - Comparing with tuned models (e.g., XGBoost + Optuna)
# - Benchmarking preprocessing strategies
# - Validating GPU/CPU execution paths
# 

# %%
import sys, os
# Add the project root to the Python path
project_root = os.path.abspath("../..")
sys.path.append(project_root)

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from utils.constants import ML_READY_DATA_FILE, TEST_MODE
from utils.data_loader import DataLoader
from utils.model_evaluator import ModelEvaluator
from utils.experiment_tracker import ExperimentTracker
from utils.model_visualizer import ModelVisualizer

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

if TEST_MODE:
    print("TEST_MODE is ON – using reduced dataset and fewer estimators.")
else:
    print("TEST_MODE is OFF – full training in progress.")

# Step 1: Load data
loader = DataLoader(ML_READY_DATA_FILE)
df = loader.load_data()
X = df.drop(columns=["price"])
y = df["price"]

# Step 2: Remove low-variance features
selector = VarianceThreshold(threshold=0.01)
X_reduced = X.loc[:, selector.fit(X).get_support()]

# Step 3: Extract top 30 features via RandomForest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_reduced, y)
top_features = pd.Series(rf_model.feature_importances_, index=X_reduced.columns).nlargest(30).index.tolist()
X_top = X_reduced[top_features]

# Step 4: Define baseline CatBoost parameters
default_params = {
    "iterations": 500 if not TEST_MODE else 50,
    "depth": 6,
    "learning_rate": 0.1,
    "loss_function": "RMSE",
    "verbose": 0,
    "random_seed": 42
}

# Step 5: Train models with fallback logic
def train_with_fallback(X_train, y_train, model_name):
    try:
        print(f"⏳ Training {model_name} on GPU...")
        model = CatBoostRegressor(**default_params, task_type="GPU", devices="0")
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"⚠️ GPU training failed for {model_name}. Falling back to CPU. Reason:\n{e}")
        model = CatBoostRegressor(**default_params, task_type="CPU")
        model.fit(X_train, y_train)
    return model

# Train on all features
model_all = train_with_fallback(X_reduced, y, "CatBoost (All Features)")
y_pred_all = model_all.predict(X_reduced)

# Train on top features
model_top = train_with_fallback(X_top, y, "CatBoost (Top RF Features)")
y_pred_top = model_top.predict(X_top)

# Step 6: Evaluate
evaluator_all = ModelEvaluator("CatBoost (All Features)")
mae_all, rmse_all, r2_all = evaluator_all.evaluate(y, y_pred_all)

evaluator_top = ModelEvaluator("CatBoost (Top RF Features)")
mae_top, rmse_top, r2_top = evaluator_top.evaluate(y, y_pred_top)

# Step 7: Log
tracker = ExperimentTracker()
df_metrics_all = tracker.log_and_get_evaluations(
    model="CatBoost (All Features)",
    experiment="CatBoost Baseline (All Features)",
    mae=mae_all,
    rmse=rmse_all,
    r2=r2_all,
)

df_metrics_top = tracker.log_and_get_evaluations(
    model="CatBoost (Top RF Features)",
    experiment="CatBoost Baseline (Top RF Features)",
    mae=mae_top,
    rmse=rmse_top,
    r2=r2_top,
)

# Step 8: Summary
print("Summary (All Features):")
evaluator_all.display_model_summary(df_metrics_all)

print("Summary (Top RF Features):")
evaluator_top.display_model_summary(df_metrics_top)

# Step 9: Diagnostics
print("Diagnostics (All Features):")
ModelVisualizer(model_all, X_reduced, y, "CatBoost (All Features)").plot_all_diagnostics()

print("Diagnostics (Top RF Features):")
ModelVisualizer(model_top, X_top, y, "CatBoost (Top RF Features)").plot_all_diagnostics()

# Step 10: Side-by-side residuals
ModelEvaluator.plot_price_range_residuals_side_by_side(
    y, y_pred_all, y_pred_top,
    model_names=("CatBoost (All Features)", "CatBoost (Top RF Features)")
)

"""
Optional:
ModelEvaluator.plot_shap_comparison_beeswarm(
    model_all=model_all,
    x_all=X_reduced,
    model_top=model_top,
    x_top=X_top
)
"""


# %% [markdown]
# # CatBoost + Optuna Hyperparameter Tuning Pipeline
# 
# 
# ## Data Preparation
# 
# - Load the cleaned ML-ready dataset from a CSV file using `DataLoader`.
# - Drop the target variable `price` to separate `X` and `y`.
# - Apply `VarianceThreshold` to remove low-variance features (threshold = 0.01).
# - Use a `RandomForestRegressor` to rank feature importance.
# - Select the **top 30 most important features** for one of the models.
# 
# 
# ##  Hyperparameter Tuning (Optuna)
# 
# Define the function `tune_catboost_with_optuna(...)` that:
# 
# - Runs an Optuna optimization loop.
# - Evaluates model performance with **5-Fold Cross-Validation**.
# - Minimizes the **Root Mean Squared Error (RMSE)**.
# 
# ### Tuned Hyperparameters:
# 
# - `iterations`  
# - `depth`  
# - `learning_rate`  
# - `l2_leaf_reg`  
# - `bagging_temperature`  
# - `random_strength`
# 
# 
# 
# ## Train Final Models
# 
# Two models are trained:
# 
# - One using **all filtered features**
# - One using the **top 30 features**
# 
# Each is trained using the **best parameters** found by Optuna.
# 
# 
# 
# ##  Evaluation
# 
# Models are evaluated using:
# 
# - `MAE`: Mean Absolute Error  
# - `RMSE`: Root Mean Squared Error  
# - `R<sup>2</sup>`: Coefficient of determination  
# 
# Results are logged with `ExperimentTracker`.
# 
# 
# 
# ##  Diagnostics
# 
# - Summary tables displayed with `ModelEvaluator`
# - Residuals & diagnostic plots from `ModelVisualizer`
# - Optionally, **SHAP values** can be plotted to understand feature importance
# 
# 
# 
# ## Test Mode (Optional)
# 
# When `TEST_MODE = True`, the pipeline uses:
# 
# - A smaller dataset  
# - Fewer Optuna trials (`n_trials = 3`)  
# 
# To speed up execution and debugging.
# 

# %%
import sys, os

# Add the project root to the Python path
project_root = os.path.abspath("../..")
sys.path.append(project_root) 

import optuna
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error
from utils.constants import ML_READY_DATA_FILE, TEST_MODE
from utils.data_loader import DataLoader
from utils.model_evaluator import ModelEvaluator
from utils.experiment_tracker import ExperimentTracker
from utils.model_visualizer import ModelVisualizer


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
 

if TEST_MODE:
    print("TEST_MODE is ON – running in test mode (reduced data, fewer trials).")
else:
    print("TEST_MODE is OFF – full training is active.")


# Step 1: Load and clean dataset
loader = DataLoader(ML_READY_DATA_FILE)
df = loader.load_data()
X = df.drop(columns=["price"])
y = df["price"]

# Step 2: Remove low-variance features
selector = VarianceThreshold(threshold=0.01)
X_reduced = X.loc[:, selector.fit(X).get_support()]

# Step 3: Extract top 30 features using Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_reduced, y)
top_features = pd.Series(rf_model.feature_importances_, index=X_reduced.columns).nlargest(30).index.tolist()
X_top = X_reduced[top_features]

# Step 4: Define Optuna objective for CatBoost
def tune_catboost_with_optuna(X_data, y_data, n_trials=50):
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 300, 1000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
        }

        model = CatBoostRegressor(
            **params,
            verbose=0,
            loss_function="RMSE",
            random_state=42
        )

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv.split(X_data):
            X_train, X_val = X_data.iloc[train_idx], X_data.iloc[val_idx]
            y_train, y_val = y_data.iloc[train_idx], y_data.iloc[val_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            scores.append(root_mean_squared_error(y_val, preds))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study

# Step 5: Tune and train both models
n_trials = 3 if TEST_MODE else 50

# All Features
study_all = tune_catboost_with_optuna(X_reduced, y, n_trials)
model_all = CatBoostRegressor(**study_all.best_params, verbose=0, loss_function="RMSE", random_state=42)
model_all.fit(X_reduced, y)
y_pred_all = model_all.predict(X_reduced)

# Top RF Features
study_top = tune_catboost_with_optuna(X_top, y, n_trials)
model_top = CatBoostRegressor(**study_top.best_params, verbose=0, loss_function="RMSE", random_state=42)
model_top.fit(X_top, y)
y_pred_top = model_top.predict(X_top)

# Step 6: Evaluation
evaluator_all = ModelEvaluator("CatBoost + Optuna CV (All Features)")
mae_all, rmse_all, r2_all = evaluator_all.evaluate(y, y_pred_all)

evaluator_top = ModelEvaluator("CatBoost + Optuna CV (Top RF Features)")
mae_top, rmse_top, r2_top = evaluator_top.evaluate(y, y_pred_top)

# Step 7: Log results
tracker = ExperimentTracker()
df_metrics_all = tracker.log_and_get_evaluations(
    model="CatBoost + Optuna CV (All Features)",
    experiment="CatBoost with Optuna (All Features)",
    mae=mae_all,
    rmse=rmse_all,
    r2=r2_all,
)

df_metrics_top = tracker.log_and_get_evaluations(
    model="CatBoost + Optuna CV (Top RF Features)",
    experiment="CatBoost with Optuna (Top RF Features)",
    mae=mae_top,
    rmse=rmse_top,
    r2=r2_top,
)

# Step 8: Summary & Diagnostics
print("Evaluation Summary (All Features):")
evaluator_all.display_model_summary(df_metrics_all)

print("Evaluation Summary (Top RF Features):")
evaluator_top.display_model_summary(df_metrics_top)

print("Diagnostics (All Features):")
ModelVisualizer(model_all, X_reduced, y, "CatBoost + Optuna CV (All Features)").plot_all_diagnostics()

print("Diagnostics (Top RF Features):")
ModelVisualizer(model_top, X_top, y, "CatBoost + Optuna CV (Top RF Features)").plot_all_diagnostics()

# Step 9: Residuals & SHAP Comparison
ModelEvaluator.plot_price_range_residuals_side_by_side(
    y, y_pred_all, y_pred_top,
    model_names=("CatBoost (All Features)", "CatBoost (Top RF Features)")
)

"""
ModelEvaluator.plot_shap_comparison_beeswarm(
    model_all=model_all,
    x_all=X_reduced,
    model_top=model_top,
    x_top=X_top
)
"""

# %% [markdown]
# # Saving CatBoost + Optuna Hyperparameter Tuning Models (`.pkl`) After Training
# 
# After training CatBoost models with Optuna tuning, it is essential to persist the trained models using `.pkl` files. The following script handles this process by creating timestamped filenames and saving them to the appropriate directory.
# 
# 
# ## What the Script Does
# 
# 1. **Appends the project root** to the Python path (for relative imports).
# 2. **Generates a timestamped filename**, with an optional `_TEST` suffix if `TEST_MODE` is enabled.
# 3. **Ensures the target directory exists** and removes any conflicting file with the same name.
# 4. **Saves both models**:
#    - One trained with **all features**.
#    - One trained with the **top 30 features** (selected via Random Forest).
# 
# 
# 

# %%
import sys, os

# Add the project root to the Python path
project_root = os.path.abspath("../..")
sys.path.append(project_root)

import joblib
from datetime import datetime
from utils.constants import TEST_MODE, MODELS_DIR

# Create timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# Add suffix if in TEST mode
suffix = "_TEST" if TEST_MODE else ""

# Define subdirectory for .pkl files
PKL_DIR = os.path.join(MODELS_DIR, "pkl")
if os.path.isfile(PKL_DIR):
    os.remove(PKL_DIR)  # Remove conflicting file if it exists
os.makedirs(PKL_DIR, exist_ok=True)

# Build filenames
filename_all = f"catboost_optuna_all_{timestamp}{suffix}.pkl"
filename_top = f"catboost_optuna_top30_{timestamp}{suffix}.pkl"

# Save models
joblib.dump(model_all, os.path.join(PKL_DIR, filename_all))
joblib.dump(model_top, os.path.join(PKL_DIR, filename_top))

print(f"[✔] CatBoost models saved to '{PKL_DIR}' as:\n - {filename_all}\n - {filename_top}")


# %% [markdown]
# # Saving CatBoost Models and Their Feature Lists
# 
# After training CatBoost models using **all features** and **top 30 features**, it is important to save both the models (`.pkl`) and their associated feature lists (`.json`). This ensures proper inference and avoids feature mismatch errors.
# 
# 
# ## Directory Structure
# 
# All model files are stored under the `models/` directory:
# 
# ## Why Save JSON Feature Files
# 
# Each `.pkl` model is trained with a specific list of input features:
# 
# - **All features model** uses all selected input columns (after variance threshold)
# - **Top30 model** uses only the top 30 features selected by Random Forest
# 
# Saving these lists in `.json` format allows:
# 
# - Correct feature alignment during inference
# - Avoiding runtime errors (e.g., feature mismatch)
# - Easier model reproducibility
# 
# 
# 
# ## Naming Convention
# 
# Each `.json` file has the same base name as the `.pkl` model it corresponds to.
# 
# | Model file (`.pkl`)                              | Feature file (`.json`)                             |
# |--------------------------------------------------|----------------------------------------------------|
# | `catboost_optuna_all_20250629_1030_TEST.pkl`     | `catboost_optuna_all_20250629_1030_TEST.json`     |
# | `catboost_optuna_top30_20250629_1030_TEST.pkl`   | `catboost_optuna_top30_20250629_1030_TEST.json`   |
# 
# ---

# %%
import json

# Define subdirectory for JSON feature files
FEATURES_DIR = os.path.join(MODELS_DIR, "features")
os.makedirs(FEATURES_DIR, exist_ok=True)

# Save features used for each model (All Features and Top30)
feature_file_all = filename_all.replace(".pkl", ".json")
feature_file_top = filename_top.replace(".pkl", ".json")

with open(os.path.join(FEATURES_DIR, feature_file_all), "w") as f:
    json.dump(list(X_reduced.columns), f, indent=2)

with open(os.path.join(FEATURES_DIR, feature_file_top), "w") as f:
    json.dump(top_features, f, indent=2)

print(f"[✔] Associated feature JSON files saved to '{FEATURES_DIR}' as:\n - {feature_file_all}\n - {feature_file_top}")



