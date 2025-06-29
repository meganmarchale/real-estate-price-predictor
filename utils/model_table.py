import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import VarianceThreshold
from utils.constants import TEST_MODE  
from utils.experiment_tracker import ExperimentTracker
from IPython.display import display, HTML

class ModelComparativeTable:


    def __init__(self):
        # Load all experiment evaluations from tracker
        self.df_all_evals = ExperimentTracker().get_all_experiment_df()

        # Check if DataFrame is valid before enrichment
        if self.df_all_evals.empty or "r2" not in self.df_all_evals.columns:
            print("⚠️ No model evaluations found in experiment tracker.")
            self.df_all_evals = pd.DataFrame()  # avoid cascading errors
        else:
            # Safe enrichment if DataFrame is non-empty and contains 'r2'
            self.df_all_evals = self.enrich_model_summary(self.df_all_evals)


    def enrich_model_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Marqueur de meilleur modèle
        best_idx = df["r2"].idxmax()
        df["best"] = ""
        df.loc[best_idx, "best"] = "✓"

        # Type de modèle
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
        df["r2"] = df["r2"].round(4)

        # Format euros
        df["mae"] = df["mae"].apply(lambda x: f"{x:,.2f} €".replace(",", " "))
        df["rmse"] = df["rmse"].apply(lambda x: f"{x:,.2f} €".replace(",", " "))

        # Ratio rmse / mae
        def parse_euro(value):
            return float(value.replace(" ", "").replace(" €", ""))

        df["rmse/mae"] = df.apply(
            lambda row: round(parse_euro(row["rmse"]) / parse_euro(row["mae"]), 2),
            axis=1
        )

        # Move 'best' column to the end
        best_col = df.pop("best")
        df["best"] = best_col

        return df









    def display_model_summary_pre_study(self, csv_path):
        """
        Display and enhance the model evaluation summary from a CSV log file.

        Args:
            csv_path (str): Path to the CSV file containing model metrics.
        """
        # Load and clean
        summary_df = pd.read_csv(csv_path).drop_duplicates()

        # Identify best model (highest R²)
        best_idx = summary_df["r2"].idxmax()
        summary_df["best"] = ""
        summary_df.loc[best_idx, "best"] = "✓"

        # Add model type
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

        summary_df["type"] = summary_df["model"].apply(get_model_type)

        # Add ranking by R²
        summary_df["rank_r2"] = summary_df["r2"].rank(method="min", ascending=False).astype(int)

        # Format monetary values
        summary_df["mae"] = summary_df["mae"].apply(lambda x: f"{x:,.2f} €".replace(",", " "))
        summary_df["rmse"] = summary_df["rmse"].apply(lambda x: f"{x:,.2f} €".replace(",", " "))
        summary_df["r2"] = summary_df["r2"].round(4)

        # Add rmse/mae ratio
        def parse_euro(value):
            return float(value.replace(" ", "").replace(" €", ""))

        summary_df["rmse/mae"] = summary_df.apply(lambda row: round(parse_euro(row["rmse"]) / parse_euro(row["mae"]), 2), axis=1)

        # Optional: sort by R² if needed (but keep original index)
        # summary_df = summary_df.sort_values(by="r2", ascending=False).reset_index(drop=True)

        # Highlight top 3
        def highlight_top_3(row):
            if row["rank_r2"] == 1:
                return ['background-color: lightgreen'] * len(row)
            elif row["rank_r2"] == 2:
                return ['background-color: #d0f0c0'] * len(row)
            elif row["rank_r2"] == 3:
                return ['background-color: #e6f5d0'] * len(row)
            return [''] * len(row)

        # Display
        print("=== Model Evaluation Summary ===")
        display(summary_df.style.apply(highlight_top_3, axis=1))
        print(f"\n👉 Best model based on 'r2': {summary_df.loc[best_idx, 'model']} ✓")


      
    def display_model_summary(self):
        df = self.df_all_evals.copy()

        if df.empty:
            print("⚠️ No model evaluation records found.")
            return

        df = df.drop_duplicates().copy()

        # Ensure float types before any calculation
        for col in ["mae", "rmse", "r2"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

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
        df["r2"] = df["r2"].round(4)

        # Compute ratio
        df["rmse/mae"] = df.apply(
            lambda row: round(row["rmse"] / row["mae"], 2)
            if pd.notnull(row["rmse"]) and pd.notnull(row["mae"]) and row["mae"] != 0 else None,
            axis=1
        )

        # Create formatted columns for display only
        df["mae_display"] = df["mae"].apply(lambda x: f"{x:,.2f} €".replace(",", " ") if pd.notnull(x) else "N/A")
        df["rmse_display"] = df["rmse"].apply(lambda x: f"{x:,.2f} €".replace(",", " ") if pd.notnull(x) else "N/A")

        # Replace values only for display
        df["mae"] = df["mae_display"]
        df["rmse"] = df["rmse_display"]
        df.drop(columns=["mae_display", "rmse_display"], inplace=True)

        def highlight_top_3(row):
            if row["rank_r2"] == 1:
                return ['background-color: lightgreen'] * len(row)
            elif row["rank_r2"] == 2:
                return ['background-color: #d0f0c0'] * len(row)
            elif row["rank_r2"] == 3:
                return ['background-color: #e6f5d0'] * len(row)
            return [''] * len(row)

        print("=== Model Evaluation Summary ===")
        styled = df.style.apply(highlight_top_3, axis=1)
        display(HTML(styled.to_html()))

        best_model_name = df.loc[best_idx, "model"]
        print(f"\n👉 Best model based on R²: {best_model_name} ✓")
    