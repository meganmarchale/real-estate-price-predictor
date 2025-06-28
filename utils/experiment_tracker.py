
import glob
import sys, os
import sqlite3
import json
from datetime import datetime
import pandas as pd

from utils.constants import METRICS_DB_PATH



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
        Log a model evaluation into the database.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
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

    def get_model_evaluations(self, join_cleaning: bool = True) -> pd.DataFrame:
        """
        Retrieve all model evaluations, optionally joined with cleaning versions.
        """
        with sqlite3.connect(self.db_path) as conn:
            if join_cleaning:
                query = """
                    SELECT m.*, c.version AS cleaning_version_name
                    FROM model_evaluations m
                    LEFT JOIN cleaning_versions c ON m.cleaning_version_id = c.id
                    ORDER BY m.timestamp DESC
                """
            else:
                query = "SELECT * FROM model_evaluations ORDER BY timestamp DESC"
            return pd.read_sql(query, conn)
        
    

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
