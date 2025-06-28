
import glob
import sys, os
import sqlite3
import json
from datetime import datetime
import pandas as pd

from utils.constants import METRICS_DB_PATH, CLEANED_DIR



class ExperimentTracker:
    """
    Class to track cleaning versions and model evaluation metrics in a SQLite database.
    """

    def __init__(self, db_path=METRICS_DB_PATH):
        """
        Initialize the experiment tracker and create necessary tables if they do not exist.
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """
        Create tables for cleaning versions and model evaluations if they do not exist.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Table for cleaning steps
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cleaning_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    version TEXT,
                    description TEXT,
                    rows_after_cleaning INTEGER,
                    details TEXT
                )
            """)

            # Table for model evaluations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    model TEXT,
                    dataset TEXT,
                    experiment TEXT,
                    cleaning_version_id INTEGER,
                    mae REAL,
                    rmse REAL,
                    r2 REAL,
                    FOREIGN KEY(cleaning_version_id) REFERENCES cleaning_versions(id)
                )
            """)

    def log_cleaning(self, version: str, description: str, rows_after_cleaning: int, steps_dict: dict) -> int:
        """
        Log a data cleaning version into the database.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        details_json = json.dumps(steps_dict, indent=2)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO cleaning_versions (timestamp, version, description, rows_after_cleaning, details)
                VALUES (?, ?, ?, ?, ?)
            """, (timestamp, version, description, rows_after_cleaning, details_json))
            conn.commit()
            cleaning_id = cursor.lastrowid

        print(f"[✓] Cleaning version '{version}' logged with ID {cleaning_id}")
        return cleaning_id

    def log_model_evaluation(self, model: str, dataset: str, experiment: str,
                            cleaning_version_id: int, mae: float, rmse: float, r2: float) -> None:
        """
        Log a model evaluation into the database, unless it already exists.

        Prevents duplicate logging based on model, dataset, experiment, and cleaning version.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check if this exact model evaluation already exists
            cursor.execute("""
                SELECT COUNT(*) FROM model_evaluations
                WHERE model = ? AND dataset = ? AND experiment = ? AND cleaning_version_id = ?
            """, (model, dataset, experiment, cleaning_version_id))
            exists = cursor.fetchone()[0]

            if exists:
                print(f"Model evaluation for '{model}' already exists. Skipping log.")
            else:
                cursor.execute("""
                    INSERT INTO model_evaluations (timestamp, model, dataset, experiment,
                                                cleaning_version_id, mae, rmse, r2)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (timestamp, model, dataset, experiment, cleaning_version_id, mae, rmse, r2))
                conn.commit()
                print(f"[✓] Model evaluation for '{model}' logged.")

    def get_cleaning_versions(self) -> pd.DataFrame:
        """
        Retrieve all cleaning versions from the database.
        """
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql("SELECT * FROM cleaning_versions ORDER BY timestamp DESC", conn)

    

    def get_latest_cleaned_file(self, directory: str, pattern: str = "immoweb_real_estate_cleaned_for_ml_*.csv") -> str:
        """
        Return the most recently modified cleaned_for_ml file from the given directory.

        Parameters:
            directory (str): Path to the directory containing the files.
            pattern (str): Glob pattern to match cleaned files.

        Returns:
            str: Full path to the most recently modified file.
        """
        search_path = os.path.join(directory, pattern)
        files = glob.glob(search_path)
        if not files:
            raise FileNotFoundError(f"No cleaned files matching pattern '{pattern}' found in {directory}")
        latest_file = max(files, key=os.path.getmtime)
        return latest_file

    def get_latest_cleaning_version_id(self, directory: str, pattern: str = "immoweb_real_estate_cleaned_for_ml_*.csv") -> str:
        """
        Extract the cleaning version ID from the most recently cleaned dataset file.

        Parameters:
        - directory (str): Path to the directory containing cleaned files.
        - pattern (str): Glob pattern to match cleaned files.

        Returns:
        - str: Cleaning version ID (extracted from filename).
        """
        latest_cleaned_file = self.get_latest_cleaned_file(directory, pattern)
        filename = os.path.basename(latest_cleaned_file)
        
        if "_for_ml_" in filename:
            return filename.split("_for_ml_")[1].replace(".csv", "")
        else:
            raise ValueError(f"Filename does not contain a valid cleaning version ID: {filename}")

    def get_model_evaluations(self, dataset_name: str, cleaning_version_id: int) -> pd.DataFrame:
        """
        Fetch logged model evaluation metrics from SQLite for a specific dataset and cleaning version.

        Args:
            dataset_name (str): Name of the dataset (e.g. 'immoweb_real_estate_ml_ready.csv')
            cleaning_version_id (int): Cleaning version identifier

        Returns:
            pd.DataFrame: DataFrame of model evaluations
        """
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT * FROM model_evaluations
            WHERE dataset = ? AND cleaning_version_id = ?
        """
        df = pd.read_sql_query(query, conn, params=(dataset_name, cleaning_version_id))
        conn.close()
        return df.drop_duplicates()
    


    def log_and_get_evaluations(self, model: str, experiment: str, mae: float, rmse: float, r2: float,
                                pattern: str = "immoweb_real_estate_cleaned_for_ml_*.csv") -> pd.DataFrame:
        """
        Log a model evaluation for the latest cleaned dataset and return the metrics DataFrame.

        Args:
            model (str): Model name (e.g., "LinearRegression")
            experiment (str): Name of the experiment (e.g., "Baseline Linear Regression")
            mae (float): Mean Absolute Error
            rmse (float): Root Mean Squared Error
            r2 (float): R-squared
            cleaned_dir (str): Directory where cleaned datasets are stored
            pattern (str): Filename pattern to identify the latest cleaned dataset

        Returns:
            pd.DataFrame: DataFrame containing all model evaluations for that dataset and version
        """
        # Get the latest cleaned dataset file and extract its name
        latest_cleaned_file = self.get_latest_cleaned_file(CLEANED_DIR, pattern)
        dataset_name = os.path.basename(latest_cleaned_file)

        # Get the associated cleaning version ID
        cleaning_version_id_str = self.get_latest_cleaning_version_id(CLEANED_DIR, pattern)
        cleaning_version_id = int(cleaning_version_id_str)

        # Log the evaluation (with duplication check handled internally)
        self.log_model_evaluation(
            model=model,
            dataset=dataset_name,
            experiment=experiment,
            cleaning_version_id=cleaning_version_id,
            mae=mae,
            rmse=rmse,
            r2=r2
        )

        # Return the corresponding metrics DataFrame
        return self.get_model_evaluations(dataset_name, cleaning_version_id)
