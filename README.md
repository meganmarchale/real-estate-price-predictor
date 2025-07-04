# Prerequisites

Make sure **Python 3.12.10** is installed.

- [Download Python 3.12.10](https://www.python.org/downloads/release/python-31210/)
- Add Python to your system `PATH`

> ⚠️ This project is validated under Python 3.12.10. Other versions may result in compatibility issues.

# Real Estate Data Analysis & Price Predictor

A modular and extensible project for predicting real estate prices across multiple cities and model types.  
Compatible with both **local environments** and **Databricks** (MLflow, DBFS, Delta).



## Environment Setup

Use the virtual environment to manage dependencies cleanly:

```bash
chmod +x setup-env.sh
./setup-env.sh
```


##  Project Structure

```text

real-estate-price-predictor/
├── configs/                         # YAML configuration files (e.g. column mapping)
│   └── feature_mapping.yaml         # Mapping of column variants to standard names
│
├── data/                            # Input CSV datasets
│   ├── cleaned/                     # Cleaned datasets ready for modeling
│   ├── ml_ready/                    # ML-ready final datasets
│   ├── raw/                         # Original raw exports
│   └── outputs/                     # Any intermediate or test outputs
│
├── database/                        # SQLite database storage for evaluation and cleaning logs
│   └── metrics.db                   # Centralized database for model metrics and data cleaning versions
│
├── dbfs_models/                     # Output directory for models if using Databricks
│
├── local_models/                    # Trained models stored locally
│   ├── rf/                          # Random Forest models by dataset
│   ├── lgbm/                        # LightGBM models by dataset
│   └── lr/                          # Linear Regression models by dataset
│
├── ml_models/                       # Core machine learning model definitions
│   ├── __init__.py                  # Makes this a Python package
│   ├── base_model.py                # Abstract base class for model interfaces
│   ├── rf_model.py                  # Random Forest implementation
│   ├── lgbm_model.py                # LightGBM implementation (Distributed Gradient Boosting Machine)
│   ├── lr_model.py                  # Linear Regression implementation
│   └── model_factory.py             # Factory to retrieve the correct model class
│
├── notebooks/                       # Jupyter notebooks for exploration and training
│   ├── exploration/                 # Notebooks for EDA per source
│   └── pipeline/                    # Modular notebooks (cleaning, training, tuning, export,etc.)
│       ├── 00_setup_env.ipynb           # Setup virtual environment and dependencies
│       ├── 01_exploration.ipynb         # Data exploration and inspection
│       ├── 02_preprocessing.ipynb       # Data cleaning and feature engineering
│       ├── 03_train_model.ipynb         # Training individual models
│       ├── 04_evaluate_model.ipynb      # Evaluation metrics and visualizations
│       ├── 05_register_model.ipynb      # Optional model registry logic
│       └── 06_batch_train_all.ipynb     # Loop training over all datasets
│
├── scripts/                         # Executable Python scripts
│   ├── train_all_datasets.py        # Main script to train all models for all datasets
│   ├── train_all_datasets.sh        # Bash script to launch training from terminal
│   ├── train_and_register.py        # Alternate script to train and register models
│   └── train_and_register.sh        # Bash wrapper for above
│
├── tests/                           # Unit tests
│   ├── __init__.py                  # Init file for test package
│   └── test_model_training.py       # Basic test for training pipeline
│
├── utils/                           # Utility scripts and helpers
│   ├── column_mapper.py             # Logic to standardize columns across datasets
│   ├── constants.py                 # Global constants (e.g., target column)
│   ├── logger.py                    # Logging utilities
│   ├── paths.py                     # Helper functions for path management
│   ├── model_evaluator.py           # Centralized logic to log model evaluations (MAE, RMSE, R<sup>2</sup>) to SQLite
│   ├── data_cleaner.py              # Cleans data and logs decisions (outliers, filters, price range, etc.)
│   └── preprocessing.py             # Custom preprocessing functions
│
├── .gitignore                       # Git ignored files list
├── README.md                        # Project overview and documentation
├── requirements.txt                 # Python package dependencies
└── setup-env.sh                     # Script to initialize virtual environment


real-estate-price-predictor/
├── configs/                         # YAML configuration files (e.g. column mapping)
│   └── feature_mapping.yaml         # Mapping of column variants to standard names
│
├── data/                            # Input CSV datasets
│   └── immovlan_real_estate.csv     # Sample real estate dataset
│
├── dbfs_models/                     # Output directory for models if using Databricks
│
├── local_models/                    # Trained models stored locally
│   ├── rf/                          # Random Forest models by dataset
│   ├── lgbm/                        # LightGBM models by dataset
│   └── lr/                          # Linear Regression models by dataset
│
├── ml_models/                       # Core machine learning model definitions
│   ├── __init__.py                  # Makes this a Python package
│   ├── base_model.py                # Abstract base class for model interfaces
│   ├── rf_model.py                  # Random Forest implementation
│   ├── lgbm_model.py                # LightGBM implementation (Distributed Gradient Boosting Machine)
│   ├── lr_model.py                  # Linear Regression implementation
│   └── model_factory.py             # Factory to retrieve the correct model class
│
├── notebooks/                       # Jupyter notebooks for exploration and training
│   ├── 00_setup_env.ipynb           # Setup virtual environment and dependencies
│   ├── 01_exploration.ipynb         # Data exploration and inspection
│   ├── 02_preprocessing.ipynb       # Data cleaning and feature engineering
│   ├── 03_train_model.ipynb         # Training individual models
│   ├── 04_evaluate_model.ipynb      # Evaluation metrics and visualizations
│   ├── 05_register_model.ipynb      # Optional model registry logic
│   └── 06_batch_train_all.ipynb     # Loop training over all datasets
│
├── scripts/                         # Executable Python scripts
│   ├── train_all_datasets.py        # Main script to train all models for all datasets
│   ├── train_all_datasets.sh        # Bash script to launch training from terminal
│   ├── train_and_register.py        # Alternate script to train and register models
│   └── train_and_register.sh        # Bash wrapper for above
│
├── tests/                           # Unit tests
│   ├── __init__.py                  # Init file for test package
│   └── test_model_training.py       # Basic test for training pipeline
│
├── utils/                           # Utility scripts and helpers
│   ├── column_mapper.py             # Logic to standardize columns across datasets
│   ├── constants.py                 # Global constants (e.g., target column)
│   ├── logger.py                    # Logging utilities
│   ├── paths.py                     # Helper functions for path management
│   └── preprocessing.py             # Custom preprocessing functions
│
├── .gitignore                       # Git ignored files list
├── README.md                        # Project overview and documentation
├── requirements.txt                 # Python package dependencies
└── setup-env.sh                     # Script to initialize virtual environment
```

# SQLite Integration & Model/Cleaning Logging with Cecil

To enable reproducible tracking of both **model performance metrics** and **data cleaning decisions**, this project uses a centralized **SQLite database** managed by the internal class `Cecil`.

## Location

- **Database file**: `database/metrics.db`
- Automatically created if it does not exist.

## What Cecil Logs

The class `Cecil` is responsible for logging:

| Type                    | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `model_metrics`         | Model name, dataset, algorithm, MAE, RMSE, R<sup>2</sup>, features used, timestamp     |
| `data_cleaning_log`     | Version name, rules applied (e.g., price cap, filters), row counts, date     |
| `pipeline_run` (TBD)    | Full pipeline metadata (duration, success/fail, user/system info)           |

## Usage

The database is automatically updated by notebooks/scripts like:

- `010_data_load_clean.ipynb` → Logs cleaning decisions
- `070_evaluation.ipynb` → Logs MAE, RMSE, R<sup>2</sup> per model
- `080_export_model.ipynb` → Can log exported models and versions
- All interactions go through the `Cecil` class under `utils/cecil.py` (recommended)

You can easily query logged entries via Python:

```python
import sqlite3
conn = sqlite3.connect("database/metrics.db")
pd.read_sql("SELECT * FROM model_metrics", conn)









## Features
- Modular architecture with reusable model classes (OOP)
- Supports multiple cities, datasets, and model types
- Compatible with local `.csv` files and Databricks tables
- Easy model comparison and export (`.pkl` or MLflow)
- Optional API via FastAPI for inference


# 2. Data Analysis



# 3. Model training


![picture 0](images/a2b25867ed7b3e42fc0ceacc68656e03fd009a21e5ee5e8add10e78a494023b3.png)  
![picture 1](images/a2b25867ed7b3e42fc0ceacc68656e03fd009a21e5ee5e8add10e78a494023b3.png)  
