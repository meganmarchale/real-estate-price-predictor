import os
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import root_mean_squared_error

class ModelEvaluator:
    """
    model_evaluator.py

    This module provides the `ModelEvaluator` class for evaluating and comparing regression models.
    It computes key performance metrics (MAE, RMSE, R²), stores them in memory, and exports them to a timestamped CSV file.
    It also provides functionality to compare multiple evaluations and highlight the best model.

    Classes:
        - ModelEvaluator: Evaluate model performance and save/compare results.

    Typical usage:
        >>> evaluator = ModelEvaluator("XGBoost Regressor")
        >>> mae, rmse, r2 = evaluator.evaluate(y_test, y_pred)
        >>> evaluator.save_to_csv()
        >>> df_results = evaluator.compare_models()
        >>> print(df_results)

    Dependencies:
        - pandas
        - datetime
        - os
        - scikit-learn (sklearn.metrics)
    """

    def __init__(self, model_name, output_dir="../data/ML"):
        self.model_name = model_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_filename = f"model_metrics_{self.timestamp}.csv"
        self.output_path = os.path.join(output_dir, self.output_filename)
        self.records = []

        os.makedirs(output_dir, exist_ok=True)

    def evaluate(self, y_true, y_pred):
        """
        Evaluates regression model predictions using MAE, RMSE, and R² metrics.

        Args:
          y_true (array-like): True target values.
          y_pred (array-like): Predicted target values by the model.

        Returns:
          tuple: A tuple containing:
            - mae (float): Mean Absolute Error of the predictions.
            - rmse (float): Root Mean Squared Error of the predictions.
            - r2 (float): R² (coefficient of determination) score of the predictions.

        Side Effects:
          - Prints the evaluation results.
          - Appends the evaluation metrics and metadata to the `records` attribute.
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        self._print_results(mae, rmse, r2)
        self.records.append({
            "timestamp": self.timestamp,
            "model": self.model_name,
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4)
        })
        return mae, rmse, r2

    def save_to_csv(self):
        """
        Saves the evaluation records to a CSV file.

        Converts the list of records stored in `self.records` into a pandas DataFrame
        and writes it to a CSV file at the location specified by `self.output_path`.
        The CSV file will not include the DataFrame index.

        Prints the path to the saved results upon completion.
        """
        df = pd.DataFrame(self.records)
        df.to_csv(self.output_path, index=False)
        print(f"\nResults saved to: {self.output_path}")

    def compare_models(self):
        """
        Compares recorded model evaluation metrics and identifies the best model based on the highest R² score.

        Returns:
          pandas.DataFrame: A DataFrame containing all model records with an additional 'best' column.
                    The row corresponding to the model with the highest R² score is marked with '✓' in the 'best' column.
                    Returns None if there are no records to compare.
        """
        df = pd.DataFrame(self.records)
        if df.empty:
            print("No model records to compare.")
            return None
        best_idx = df["r2"].idxmax()
        df["best"] = ""
        df.loc[best_idx, "best"] = "✓"
        return df

    def _print_results(self, mae, rmse, r2):
        """
        Prints the evaluation metrics for the model in a formatted manner.

        Args:
          mae (float): Mean Absolute Error of the model predictions.
          rmse (float): Root Mean Squared Error of the model predictions.
          r2 (float): R-squared (coefficient of determination) of the model predictions.

        Outputs:
          Prints the model name and the provided evaluation metrics (MAE, RMSE, R²) to the console.
        """
        print(f"Evaluation – {self.model_name}")
        print(f"  MAE:  {mae:,.2f} €")
        print(f"  RMSE: {rmse:,.2f} €")
        print(f"  R²:   {r2:.4f}")
        print("-" * 40)
