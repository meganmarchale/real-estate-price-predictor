# %% [markdown]
# ## Feature Engineering & Preprocessing (Preparing ML-Ready Data)
# 
# Feature Engineering is the process of creating, selecting, and transforming variables (features) to improve the performance of machine learning models.
# 
# In this notebook, we will apply several techniques to transform the cleaned dataset into a more informative and ML-ready format.
# 
# ### Objectives of this step:
# 
# 1. **Remove or encode non-numeric columns**  
#    Convert categorical variables to numeric formats using techniques such as One-Hot Encoding or Label Encoding.
# 
# 2. **Handle date or time-related features**  
#    Derive new features from construction year (e.g., building age), or other temporal indicators.
# 
# 3. **Create new derived features**  
#    Add variables like:
#    - `building_age` = `current_year - buildingConstructionYear`
# 
# 4. **(Optional) Normalize or scale selected features**  
#    This step is not required for tree-based models (e.g., XGBoost, Random Forest), but might be included later if experimenting with distance-based algorithms.
# 
# 5. **Reduce dimensionality or drop irrelevant features**  
#    Focus only on the most relevant variables based on domain knowledge or correlation analysis.
# 
# At the end of this notebook, we will generate a dataset ready for training baseline models.
# 

# %%
import sys, os

# Add the project root to the Python path
project_root = os.path.abspath("../..")
sys.path.append(project_root)

# Imports from local modules
import pandas as pd
from utils.data_cleaner import DataCleaner
from utils.data_loader import DataLoader
from utils.experiment_tracker import ExperimentTracker
from utils.constants import CLEANED_DIR
 
tracker = ExperimentTracker()
last_cleanded_dataset = tracker.get_latest_cleaned_file(CLEANED_DIR)

# Replace boolean values with 0 and 1, drop unnecessary columns, and clean the DataFrame
loader = DataLoader(last_cleanded_dataset)
df = loader.load_data()

df = loader.clean_booleans(df, bool_cols=["hasLivingRoom", "hasTerrace"])

df = loader.drop_columns(df, columns_to_drop=["Unnamed: 0", "id", "url"])
df = loader.drop_na_targets(df, target_col="price")

df.head()


# %%
#  Create new derived features BEFORE pipeline
df["building_age"] = 2025 - df["buildingConstructionYear"]

# %%


# %%
from utils.preprocessing_pipeline import PreprocessingPipeline


# 2. Initialize preprocessing pipeline


pipeline = PreprocessingPipeline(
    df=df,
    target_col="price",
    drop_cols=["price_per_m2", "log_price"],  
)

df_encoded = pipeline.fit_transform()

# Check if unwanted columns are still present
for col in ["price_per_m2", "log_price"]:
    if col in df_encoded.columns:
        print(f"❌ Unwanted column still present: {col}")
    else:
        print(f"[INFO] Column removed: {col}")


# 3. Save full and sample dataset

import os
import shutil
from utils.constants import ML_READY_DIR, ML_READY_DATA_FILE, ML_READY_SAMPLE_XLSX


# Clean and recreate ml_ready directory
if os.path.exists(ML_READY_DIR):
    shutil.rmtree(ML_READY_DIR)
os.makedirs(ML_READY_DIR, exist_ok=True)

# Save to CSV and Excel
df_encoded.to_csv(ML_READY_DATA_FILE, index=False)
df_encoded.head(10).to_excel(ML_READY_SAMPLE_XLSX, index=False)

print(f"Dataset ready. Shape: {df_encoded.shape}")
print(f"Saved to: {ML_READY_DATA_FILE}")
print(f"Excel sample: {ML_READY_SAMPLE_XLSX}")


