# %% [markdown]
# # Load dataset and clean
# - Step 1 – Drop columns with too many missing value
# - Step 2 – Keep only rows with at least 70% non-missing values
# - Step 3 – Remove outliers based on key numerical columns

# %%
import sys, os

# Add the project root to the Python path
project_root = os.path.abspath("../..")
sys.path.append(project_root)

# Imports from local modules
import pandas as pd
from utils.data_cleaner import DataCleaner
from utils.constants import RAW_DATA_FILE, CLEANED_DATA_FILE, ML_READY_DATA_FILE



# === Load raw dataset ===
print(f"Loading dataset from: {RAW_DATA_FILE}")
df_raw = pd.read_csv(RAW_DATA_FILE)
print(f"Initial shape: {df_raw.shape}")

# === Apply data cleaning steps ===
cleaner = DataCleaner(missing_threshold=0.5, row_threshold=0.7)

# Step 1: Drop columns with too many missing values
df_step1 = cleaner.drop_columns_with_missing_values(df_raw)

# Step 2: Drop rows with too many missing values
df_step2 = cleaner.drop_rows_with_missing_values(df_step1)

# Step 3: Remove outliers based on numerical rules
df_cleaned = cleaner.remove_outliers(df_step2)

# === Save cleaned dataset ===
os.makedirs(os.path.dirname(CLEANED_DATA_FILE), exist_ok=True)
df_cleaned.to_csv(CLEANED_DATA_FILE, index=False)
print(f"\nCleaned dataset saved to: {CLEANED_DATA_FILE}")

# === Save a 10-row sample as Excel file for review ===
os.makedirs(os.path.dirname(ML_READY_DATA_FILE), exist_ok=True)
excel_sample_path = ML_READY_DATA_FILE.replace(".csv", "_sample10.xlsx")
df_cleaned.head(10).to_excel(excel_sample_path, index=False)
print(f"Sample Excel file saved to: {excel_sample_path}")


# %% [markdown]
# ## Load and Explore the Dataset
# 
# - Load dataset
# - df.head(), df.info(), df.describe()
# - Visual summary: distributions, correlations, missing values

# %%
# Load and Explore the Dataset

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Load the cleaned dataset
print(f"Loading dataset from: {CLEANED_DATA_FILE}")
df_cleaned = pd.read_csv(CLEANED_DATA_FILE)

# Display basic structure
print("Dataset loaded successfully!")
print("Shape:", df_cleaned.shape)

# Display the first rows
df_cleaned.head()

# Check column types and non-null counts
df_cleaned.info()

# Summary statistics for numerical columns
df_cleaned.describe()





# %%
# Plot distributions of key numerical variables
numerical_cols = ['price', 'bedroomCount', 'bathroomCount', 'habitableSurface']
df_cleaned[numerical_cols].hist(figsize=(12, 8), bins=30, color="steelblue", edgecolor="black")
plt.suptitle("Distribution of Numerical Features", fontsize=14)
plt.tight_layout()
plt.show()

# %%
# Correlation heatmap for numerical variables
plt.figure(figsize=(10, 6))
sns.heatmap(df_cleaned[numerical_cols].corr(), annot=True, cmap="Blues", fmt=".2f")
plt.title("Correlation Matrix of Key Numerical Features")
plt.tight_layout()
plt.show()

# %% [markdown]
# # Visualize missing values using missingno

# %%
import missingno as msno

# Load the cleaned dataset
print(f"Loading dataset from: {CLEANED_DATA_FILE}")
df_cleaned = pd.read_csv(CLEANED_DATA_FILE)

# Visualize missing values using missingno
msno.matrix(df_cleaned, figsize=(14, 5), sparkline=False)
plt.title("Missing Data Matrix")
plt.show()


