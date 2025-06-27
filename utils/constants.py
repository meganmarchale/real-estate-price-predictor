
# Output folders
MODEL_OUTPUT_DIR = "../data/ML"
RESULTS_DIR = "../results"
MODELS_DIR = "../models"
LOGS_DIR = "../logs"

# Number of top features to keep
N_TOP_FEATURES = 30

# Default random state for reproducibility
RANDOM_STATE = 42

# Price ranges for analysis (used in plotting)
PRICE_BINS = [0, 250_000, 500_000, 750_000, 1_000_000, float("inf")]
PRICE_BIN_LABELS = ["<250k", "250k–500k", "500k–750k", "750k–1M", ">1M"]

RAW_DATA_FILE = "../data/immoweb_real_estate_raw.csv"
CLEANED_DATA_FILE = "../data/immoweb_real_estate_cleaned_dataset.csv"
ML_READY_DATA_FILE = "../data/immoweb_real_estate_ml_ready.csv"

# Target variable for prediction
TARGET_COLUMN = "price"

# Columns to drop (e.g. ID, URL)
COLUMNS_TO_DROP = ["id", "url"]

# constants.py
TEST_MODE = True  