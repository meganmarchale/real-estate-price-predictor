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

        # === AJOUT DU PRINT DEBUG ICI ===
        print("\n=== AVANT ENRICHISSEMENT ===")
        print(self.df_all_evals[["model", "mae", "rmse", "r2"]].head(10))
        print(self.df_all_evals.dtypes)
        # ================================


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

        # Ajoute les colonnes d'affichage, ne modifie PAS les colonnes numériques originales
        df["mae_display"] = df["mae"].apply(lambda x: f"{x:,.2f} €".replace(",", " ") if pd.notnull(x) else "N/A")
        df["rmse_display"] = df["rmse"].apply(lambda x: f"{x:,.2f} €".replace(",", " ") if pd.notnull(x) else "N/A")

        # Ratio rmse / mae, version calcul sur les floats uniquement
        df["rmse/mae"] = df.apply(
            lambda row: round(row["rmse"] / row["mae"], 2)
            if pd.notnull(row["rmse"]) and pd.notnull(row["mae"]) and row["mae"] != 0 else None,
            axis=1
        )

        # Move 'best' column to the end
        best_col = df.pop("best")
        df["best"] = best_col

        return df





    def display_model_summary_from_db(db_path):
        # Charger depuis la base SQLite
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM model_evaluations", conn)
        conn.close()

        # DEBUG types
        print("==== Colonnes et types ====")
        print(df.dtypes)
        print(df.head(5))

        # Conversion sécurisée en float
        for col in ["mae", "rmse", "r2"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Calcul du ratio, robustement
        df["rmse/mae"] = df.apply(
            lambda row: round(row["rmse"] / row["mae"], 2)
            if pd.notnull(row["rmse"]) and pd.notnull(row["mae"]) and row["mae"] != 0 else None,
            axis=1
        )

        # Trouver le meilleur modèle (R² max)
        best_idx = df["r2"].idxmax()
        df["best"] = ""
        if pd.notnull(best_idx):
            df.loc[best_idx, "best"] = "✓"

        # Types de modèles (optionnel)
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

        # Formatage pour display (jamais dans le calcul !)
        df["mae_display"] = df["mae"].apply(lambda x: f"{x:,.2f} €".replace(",", " ") if pd.notnull(x) else "N/A")
        df["rmse_display"] = df["rmse"].apply(lambda x: f"{x:,.2f} €".replace(",", " ") if pd.notnull(x) else "N/A")

        # Affichage simple dans le terminal (tu peux remplacer par display() si notebook)
        print("=== Résumé modèles ===")
        print(df[["model", "mae_display", "rmse_display", "r2", "rmse/mae", "best", "type"]])

        # Retourne le DataFrame final si besoin
        return df



      
    def display_model_summary(self):
        df = self.df_all_evals.copy()

        if df.empty:
            print("⚠️ No model evaluation records found.")
            return

        df = df.drop_duplicates().copy()

        # S’assurer que c’est bien en float
        for col in ["mae", "rmse", "r2"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        best_idx = df["r2"].idxmax()
        df["best"] = ""
        df.loc[best_idx, "best"] = "✓"

        # Ajout du type de modèle
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

        # Calcul ratio
        df["rmse/mae"] = df.apply(
            lambda row: round(row["rmse"] / row["mae"], 2)
            if pd.notnull(row["rmse"]) and pd.notnull(row["mae"]) and row["mae"] != 0 else None,
            axis=1
        )

        # Colonnes formatées pour l’affichage (pas pour les calculs)
        df["mae_display"] = df["mae"].apply(lambda x: f"{x:,.2f} €".replace(",", " ") if pd.notnull(x) else "N/A")
        df["rmse_display"] = df["rmse"].apply(lambda x: f"{x:,.2f} €".replace(",", " ") if pd.notnull(x) else "N/A")

        # Colonnes à afficher
        show_cols = [
            "model", "type", "mae_display", "rmse_display", "r2", "rmse/mae", "rank_r2", "best"
        ]

        # Coloration conditionnelle
        def highlight_top_3(row):
            if row["rank_r2"] == 1:
                return ['background-color: lightgreen'] * len(row)
            elif row["rank_r2"] == 2:
                return ['background-color: #d0f0c0'] * len(row)
            elif row["rank_r2"] == 3:
                return ['background-color: #e6f5d0'] * len(row)
            return [''] * len(row)

        print("=== Model Evaluation Summary ===")
        display(df[show_cols].style.apply(highlight_top_3, axis=1))

        best_model_name = df.loc[best_idx, "model"]
        print(f"\n👉 Best model based on R²: {best_model_name} ✓")

    