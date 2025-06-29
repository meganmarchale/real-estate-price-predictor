# %%
# === [0. Imports] ===
import os
import sys
from pathlib import Path
import pandas as pd

# === [1. Detect project root robustly (supports nbclient + script + notebook)] ===
def get_project_root(marker_file=".git", fallback_name="real-estate-price-predictor"):
    """
    Traverse upward from the current working directory to find the project root.
    Looks for a marker (like .git) or fallback folder name.
    """
    current = Path.cwd().resolve()
    for parent in [current] + list(current.parents):
        if (parent / marker_file).exists() or fallback_name.lower() in parent.name.lower():
            return parent
    raise RuntimeError(f"❌ Could not find project root using marker '{marker_file}' or fallback '{fallback_name}'")

# === [2. Add project root to sys.path for local imports] ===
project_root = get_project_root()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"Using detected project root: {project_root}")

# === [3. Local imports from utils] ===
from utils.experiment_tracker import ExperimentTracker
from utils.constants import METRICS_DB_PATH, CLEANED_DIR
from utils.model_evaluator import ModelEvaluator
from utils.model_table import ModelComparativeTable

# === [4. Load pre-study model metrics (CSV)] ===
model_pre_study_path = project_root / "data" / "ml_pre_study_metrics" / "model_metrics.csv"
if not model_pre_study_path.exists():
    raise FileNotFoundError(f"❌ File not found: {model_pre_study_path}")
print(f"✅ Found file: {model_pre_study_path}")

# === [5. Create model comparison object] ===
mcp = ModelComparativeTable()

# === [6. Display pre-study summary (from CSV)] ===
try:
    df_csv = pd.read_csv(model_pre_study_path)
    if df_csv.empty or "r2" not in df_csv.columns or df_csv["r2"].dropna().empty:
        raise ValueError("⚠️ CSV file is empty or missing valid 'r2' values.")
    mcp.display_model_summary_pre_study(model_pre_study_path)
except Exception as e:
    print("⚠️ Failed to display pre-study summary:", e)

# === [7. Display live summary (from SQLite)] ===
try:
    if not mcp.df_all_evals.empty:
        mcp.display_model_summary()
    else:
        print("⚠️ No experiment logs found in SQLite tracker.")
except Exception as e:
    print("⚠️ Failed to display model summary:", e)



