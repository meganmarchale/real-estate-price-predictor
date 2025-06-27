import optuna
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import root_mean_squared_error
from datetime import datetime
import pandas as pd
import os

from constants import TEST_MODE, MODEL_OUTPUT_DIR, LOGS_DIR
from utils.model_evaluator import ModelEvaluator

class Tuner:
    def __init__(self, model_name="XGBoostRegressor", n_trials=None, output_dir=LOGS_DIR):
        self.model_name = model_name
        self.n_trials = n_trials if n_trials is not None else (3 if TEST_MODE else 50)
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(self.output_dir, f"tuning_metrics_{self.timestamp}.csv")
        self.evaluator = ModelEvaluator(model_name, output_dir)
        self.records = []

        os.makedirs(self.output_dir, exist_ok=True)

        if TEST_MODE:
            print("⚠️ Running in TEST MODE – reduced trials, subsampled data.")

    def _objective(self, trial, X, y):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
        }

        # Reduce iterations if in test mode
        if TEST_MODE:
            params["n_estimators"] = 50

        model = XGBRegressor(**params)
        kf = KFold(n_splits=3, shuffle=True, random_state=42)

        mae_scores = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            mae = mean_absolute_error(y_val, y_pred)
            mae_scores.append(mae)

        return sum(mae_scores) / len(mae_scores)

    def run(self, X, y):
        """
        Runs hyperparameter tuning using Optuna for the model specified in the tuner.
        This method optionally subsamples the data if in TEST_MODE, performs hyperparameter optimization
        using Optuna, retrains the best model on the full dataset, evaluates its performance, and saves
        the results and evaluation metrics to CSV files.
        Args:
          X (pd.DataFrame): Feature matrix.
          y (pd.Series or np.ndarray): Target variable.
        Returns:
          Tuple[XGBRegressor, dict]: The best trained model and the best hyperparameters found.
        """
        # Optionally subsample if in TEST_MODE
        if TEST_MODE:
            X = X.sample(n=1000, random_state=42)
            y = y.loc[X.index]

        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self._objective(trial, X, y), n_trials=self.n_trials)

        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best value (MAE): {study.best_value:.4f}")
        print(f"Best hyperparameters: {study.best_params}")

        # Retrain on full data with best params
        best_model = XGBRegressor(**study.best_params)
        best_model.fit(X, y)
        y_pred = best_model.predict(X)
        mae, rmse, r2 = self.evaluator.evaluate(y, y_pred)

        # Save evaluation record
        self.evaluator.save_to_csv()

        # Save Optuna results
        df = pd.DataFrame([{
            "timestamp": self.timestamp,
            "model": self.model_name,
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4),
            **study.best_params
        }])
        df.to_csv(self.log_path, index=False)
        print(f"Tuning metrics saved to {self.log_path}")

        return best_model, study.best_params
