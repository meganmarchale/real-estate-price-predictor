# %%
import sys, os
import json
from glob import glob
import pandas as pd
from joblib import load
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.abspath("../..")
sys.path.append(project_root)

# Import project constants
from utils.constants import ML_READY_DATA_FILE, MODELS_DIR

# Define the correct predictions directory (at the root)
PREDICTIONS_DIR = os.path.abspath(os.path.join(project_root, "predictions"))

# Step 1: Load the machine learning-ready dataset
if not os.path.exists(ML_READY_DATA_FILE):
    raise FileNotFoundError(f"ML-ready dataset not found at: {ML_READY_DATA_FILE}")

print(f"Using ML-ready dataset: {os.path.basename(ML_READY_DATA_FILE)}")
df = pd.read_csv(ML_READY_DATA_FILE)

# Step 2: Randomly select 10 properties for inference
df_sample = df.sample(n=10, random_state=42).reset_index(drop=True)
print("10 random properties selected for prediction.")

# Remove non-feature columns
base_features = df_sample.drop(columns=["id", "url"], errors="ignore")

# Step 3: Load all .pkl models and perform predictions
models_pkl_dir = os.path.join(MODELS_DIR, "pkl")
models_json_dir = os.path.join(MODELS_DIR, "features")

pkl_files = glob(os.path.join(models_pkl_dir, "*.pkl"))

if not pkl_files:
    raise ValueError(f"No .pkl models found in: {models_pkl_dir}")

predictions = df_sample.copy()

for pkl_path in pkl_files:
    model_name = os.path.basename(pkl_path).replace(".pkl", "")
    json_filename = model_name + ".json"
    json_path = os.path.join(models_json_dir, json_filename)

    if not os.path.exists(json_path):
        print(f"Skipping model '{model_name}': missing features file '{json_filename}'")
        continue

    try:
        # Load model and its features
        model = load(pkl_path)
        with open(json_path, "r") as f:
            features = json.load(f)

        # Subset input and predict
        X_input = base_features[features]
        preds = model.predict(X_input)
        predictions[model_name] = preds

        print(f"Prediction completed for model: {model_name}")

    except Exception as e:
        print(f"❌ Failed prediction for model '{model_name}': {e}")

# Step 4: Save predictions to CSV (in correct predictions/ folder)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
output_file = f"inference_predictions_{timestamp}.csv"
output_path = os.path.join(PREDICTIONS_DIR, output_file)

predictions.to_csv(output_path, index=False)
print(f"Inference predictions saved to: {output_path}")



