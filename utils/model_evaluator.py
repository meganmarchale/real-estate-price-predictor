import os
import pandas as pd
from IPython.display import display
from datetime import datetime
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import root_mean_squared_error
from utils.experiment_tracker import ExperimentTracker
from utils.model_visualizer import ModelVisualizer
import matplotlib.pyplot as plt
import seaborn as sns



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

    def __init__(self, model_name):
        self.model_name = model_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.records = []

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




    def display_model_summary(self, df: pd.DataFrame):
        """
        Display and enhance the model evaluation summary from a provided DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing model evaluation metrics
        """
        if df.empty:
            print("⚠️ No model evaluation records found.")
            return

        best_idx = df["r2"].idxmax()
        df["best"] = ""
        df.loc[best_idx, "best"] = "✓"

        def get_model_type(name):
            if "Linear" in name:
                return "Linear"
            elif "Random Forest" in name:
                return "Tree"
            elif any(boost in name for boost in ["XGBoost", "LightGBM", "CatBoost"]):
                return "Boosting"
            elif "Stacked" in name:
                return "Ensemble"
            else:
                return "Other"

        df["type"] = df["model"].apply(get_model_type)
        df["rank_r2"] = df["r2"].rank(method="min", ascending=False).astype(int)
        df["mae"] = df["mae"].apply(lambda x: f"{x:,.2f} €".replace(",", " "))
        df["rmse"] = df["rmse"].apply(lambda x: f"{x:,.2f} €".replace(",", " "))
        df["r2"] = df["r2"].round(4)

        def parse_euro(value):
            return float(value.replace(" ", "").replace(" €", ""))

        df["rmse/mae"] = df.apply(
            lambda row: round(parse_euro(row["rmse"]) / parse_euro(row["mae"]), 2),
            axis=1
        )

        def highlight_top_3(row):
            if row["rank_r2"] == 1:
                return ['background-color: lightgreen'] * len(row)
            elif row["rank_r2"] == 2:
                return ['background-color: #d0f0c0'] * len(row)
            elif row["rank_r2"] == 3:
                return ['background-color: #e6f5d0'] * len(row)
            return [''] * len(row)

        print("=== Model Evaluation Summary ===")
        display(df.style.apply(highlight_top_3, axis=1))
        print(f"\n👉 Best model based on R²: {df.loc[best_idx, 'model']} ✓")




    def evaluate_and_track_model(self, model, X_test, y_test, y_pred, model_name, experiment_name=None):
        """
        Evaluation pipeline (simplified):
        - evaluates the model
        - logs metrics
        - displays summary
        - runs visual diagnostics (residuals only)

        Args:
            model: Trained model
            X_test (DataFrame): Test features
            y_test (Series): True target values
            y_pred (array-like): Model predictions
            model_name (str): Name of the model
            experiment_name (str, optional): Experiment label
        """

        # Evaluate
        mae, rmse, r2 = self.evaluate(y_test, y_pred)

        # Log
        tracker = ExperimentTracker()
        df_metrics = tracker.log_and_get_evaluations(
            model=model_name,
            experiment=experiment_name or f"Run {self.timestamp}",
            mae=mae,
            rmse=rmse,
            r2=r2,
        )

        # Display summary
        self.display_model_summary(df_metrics)

        # Basic visual diagnostics
        visualizer = ModelVisualizer(model, X_test, y_test, model_name=model_name)
        visualizer.plot_all_diagnostics()
        visualizer.plot_price_range_residuals()


    @staticmethod
    def plot_price_range_residuals_static(y_true, y_pred, model_name="Model"):
        """
        Plots the distribution of residuals across price ranges using a boxplot.
        """
        residuals = y_true - y_pred
        df = pd.DataFrame({
            "price": y_true,
            "residuals": residuals
        })

        bins = [0, 250_000, 500_000, 750_000, 1_000_000, float("inf")]
        labels = ["<250k", "250k–500k", "500k–750k", "750k–1M", ">1M"]
        df["price_range"] = pd.cut(df["price"], bins=bins, labels=labels)

        plt.figure(figsize=(10, 6))
        sns.boxplot(x="price_range", y="residuals", data=df, palette="muted")
        plt.axhline(0, linestyle="--", color="red")
        plt.title(f"Residuals by Price Range – {model_name}")
        plt.xlabel("Price Range (€)")
        plt.ylabel("Residual (€)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_price_range_residuals_side_by_side(y_true, y_pred_1, y_pred_2, model_names=("Model 1", "Model 2")):
        """
        Display side-by-side boxplots of residuals by price range for two models.
        """


        # Compute residuals
        residuals_1 = y_true - y_pred_1
        residuals_2 = y_true - y_pred_2

        # Create DataFrames for both models
        df_1 = pd.DataFrame({
            "Price": y_true,
            "Residuals": residuals_1,
            "Model": model_names[0]
        })
        df_2 = pd.DataFrame({
            "Price": y_true,
            "Residuals": residuals_2,
            "Model": model_names[1]
        })

        # Combine both DataFrames
        df = pd.concat([df_1, df_2], axis=0)

        # Bin prices into ranges
        bins = [0, 250_000, 500_000, 750_000, 1_000_000, float("inf")]
        labels = ["<250k", "250k–500k", "500k–750k", "750k–1M", ">1M"]
        df["Price Range"] = pd.cut(df["Price"], bins=bins, labels=labels)

        # Plot boxplot by price range and model
        plt.figure(figsize=(14, 6))
        sns.boxplot(data=df, x="Price Range", y="Residuals", hue="Model", palette="Set2")
        plt.axhline(0, linestyle="--", color="red")
        plt.title("Residuals by Price Range – Model Comparison")
        plt.xlabel("Price Range (€)")
        plt.ylabel("Residual (€)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

