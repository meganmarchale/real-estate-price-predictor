import os,sys
# Add the project root to the Python path
project_root = os.path.abspath("../..")
sys.path.append(project_root)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import VarianceThreshold
from utils.constants import TEST_MODE  

class ModelVisualizer:
    def __init__(self, model, X, y, model_name="Model"):
        self.model = model
        self.X = X
        self.y = y
        self.model_name = model_name
        self.y_pred = model.predict(X)
        self.residuals = self.y - self.y_pred


    def plot_all_diagnostics(self):
        """
        Display residuals vs predicted, residual distribution, and actual vs predicted in a single row.

        This method uses the internal model, features (X), target (y), and predictions (y_pred)
        to plot three diagnostic charts side by side:
            1. Residuals vs Predicted
            2. Residual Distribution
            3. Predicted vs Actual

        Returns:
            None. Displays the plots in a single matplotlib figure.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Residuals vs Predicted
        sns.scatterplot(x=self.y_pred / 1e6, y=self.residuals, ax=axes[0])
        axes[0].axhline(0, color='red', linestyle='--')
        axes[0].set_xlabel("Predicted Price (€M)")
        axes[0].set_ylabel("Residual (€)")
        axes[0].set_title(f"Residuals vs Predicted – {self.model_name}")

        # Plot 2: Residual Distribution
        sns.histplot(self.residuals, bins=60, kde=False, color="orange", ax=axes[1])
        axes[1].set_xlabel("Prediction Error (€)")
        axes[1].set_title(f"Residual Distribution – {self.model_name}")

        # Plot 3: Actual vs Predicted
        sns.scatterplot(x=self.y / 1e6, y=self.y_pred / 1e6, ax=axes[2])
        max_price_m = max(self.y.max(), self.y_pred.max()) / 1e6
        axes[2].plot([0, max_price_m], [0, max_price_m], color='red', linestyle='--')
        axes[2].set_xlabel("Actual Price (€M)")
        axes[2].set_ylabel("Predicted Price (€M)")
        axes[2].set_title(f"Predicted vs Actual – {self.model_name}")

        plt.tight_layout()
        plt.show()


    def plot_residuals_vs_predicted(self):
        """
        Plots the residuals versus the predicted values for the model.

        This method creates a scatter plot where the x-axis represents the predicted prices (in millions of euros)
        and the y-axis represents the residuals (difference between actual and predicted values). A horizontal red dashed
        line at y=0 is added to help visualize the distribution of residuals around zero. The plot is titled with the model's name.

        Requires:
            - self.y_pred: Array-like, predicted values from the model.
            - self.residuals: Array-like, residuals (actual - predicted).
            - self.model_name: String, name of the model for plot title.

        Displays:
            - A matplotlib figure showing residuals vs. predicted values.
        """
        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=self.y_pred / 1e6, y=self.residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("Predicted Price (€M)")
        plt.ylabel("Residual (€)")
        plt.title(f"Residuals vs Predicted – {self.model_name}")
        plt.show()

    def plot_residual_distribution(self):
        """
        Plots the distribution of residuals (prediction errors) for the model.

        This method creates a histogram of the residuals using Seaborn and Matplotlib,
        allowing visual assessment of the prediction errors. The x-axis represents the
        prediction error in euros, and the plot is titled with the model's name.

        Returns:
            None
        """
        plt.figure(figsize=(6, 5))
        sns.histplot(self.residuals, bins=60, kde=False, color="orange")
        plt.xlabel("Prediction Error (€)")
        plt.title(f"Residual Distribution – {self.model_name}")
        plt.show()

    def plot_predicted_vs_actual(self):
        """
        Plots a scatter plot comparing actual versus predicted property prices.

        This method visualizes the relationship between the actual prices (`self.y`) and the predicted prices (`self.y_pred`)
        for a real estate price prediction model. Both actual and predicted prices are scaled to millions of euros for clarity.
        A reference line (y = x) is included to indicate perfect prediction. The plot is labeled and titled with the model's name.

        Returns:
            None
        """
        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=self.y / 1e6, y=self.y_pred / 1e6)
        plt.plot([0, 2], [0, 2], color='red', linestyle='--')
        plt.xlabel("Actual Price (€M)")
        plt.ylabel("Predicted Price (€M)")
        plt.title(f"Predicted vs Actual – {self.model_name}")
        plt.show()

    def plot_price_range_residuals(self):
        """
        Plots the distribution of residuals across different price ranges using a boxplot.
        This method creates a new DataFrame by copying the feature set and adding columns for residuals and actual prices.
        It then categorizes the prices into predefined bins (e.g., <250k, 250k–500k, etc.) and visualizes the residuals
        for each price range using a seaborn boxplot. A horizontal line at zero is added for reference.
        The plot helps to assess whether the model's residuals are distributed differently across various price segments.
        Assumes the following instance attributes:
            - self.X: pandas DataFrame of features.
            - self.y: array-like of actual prices.
            - self.residuals: array-like of residuals (predicted - actual).
            - self.model_name: string, name of the model.
        Returns:
            None. Displays the plot.
        """
        df = self.X.copy()
        df["residuals"] = self.residuals
        df["price"] = self.y
        bins = [0, 250_000, 500_000, 750_000, 1_000_000, float("inf")]
        labels = ["<250k", "250k–500k", "500k–750k", "750k–1M", ">1M"]
        df["price_range"] = pd.cut(df["price"], bins=bins, labels=labels)

        plt.figure(figsize=(8, 5))
        sns.boxplot(x="price_range", y="residuals", data=df, palette="muted")
        plt.axhline(0, color="red", linestyle="--")
        plt.title(f"Residuals by Price Range – {self.model_name}")
        plt.xlabel("Price Range")
        plt.ylabel("Residual (€)")
        plt.show()

    def plot_feature_importance(self):
        """
        Plots the top 15 most important features based on model coefficients or importances.

        Works with models that have:
        - `feature_importances_` (e.g., tree-based models like RandomForest, XGBoost)
        - `coef_` (e.g., LinearRegression, Ridge, Lasso)

        Raises:
            AttributeError: If neither attribute is found.
        """
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = self.model.coef_
        else:
            raise AttributeError(
                f"The model of type {type(self.model)} does not support feature importance or coefficients."
            )

        features = self.X.columns
        df_feat = pd.DataFrame({
            "feature": features,
            "importance": importances
        })

        df_feat = df_feat.reindex(df_feat.importance.abs().sort_values(ascending=False).index)[:15]

        plt.figure(figsize=(8, 5))
        sns.barplot(data=df_feat, y="feature", x="importance", palette="viridis")
        plt.title(f"Top 15 Feature Importances – {self.model_name}")
        plt.xlabel("Importance (Coefficient or Score)")
        plt.tight_layout()
        plt.show()

    def plot_shap_summary(self):
        """
        Generates and displays a SHAP summary beeswarm plot for the model's predictions.

        Supports linear models like LinearRegression using shap.LinearExplainer.
        """


        try:
            # LinearExplainer is appropriate for models like sklearn's LinearRegression
            explainer = shap.LinearExplainer(self.model, self.X, feature_perturbation="interventional")
            shap_values = explainer.shap_values(self.X)

            # Summary plot
            shap.summary_plot(shap_values, self.X, plot_type="beeswarm", max_display=20)
        except (ValueError, AttributeError, ImportError) as e:
            print(f"SHAP summary plot failed: {e}")



    




    def plot_permutation_importance(self, scoring="neg_mean_absolute_error", n_repeats=30, top_n_features=20):
        """
        Plot permutation feature importance for the trained model.
        Handles both standalone models and pipeline objects (with transformers).
        """
        if self.model is None:
            print("Model not provided.")
            return

        try:
            # Default: use X as-is
            X_to_use = self.X
            model_to_use = self.model

            # If it's a pipeline, extract the preprocessor and regressor
            if hasattr(self.model, "named_steps"):
                if "preprocessor" in self.model.named_steps and "regressor" in self.model.named_steps:
                    preprocessor = self.model.named_steps["preprocessor"]
                    model_to_use = self.model.named_steps["regressor"]

                    # Apply preprocessing
                    X_transformed = preprocessor.transform(self.X)

                    # Get feature names after preprocessing
                    feature_names = preprocessor.get_feature_names_out()
                    X_to_use = pd.DataFrame(X_transformed, columns=feature_names, index=self.X.index)

            # Run permutation importance
            result = permutation_importance(
                model_to_use,
                X_to_use,
                self.y,
                scoring=scoring,
                n_repeats=n_repeats,
                random_state=42,
                n_jobs=-1
            )

            # Sort and plot
            sorted_idx = result.importances_mean.argsort()[::-1][:top_n_features]
            plt.figure(figsize=(10, 6))
            plt.barh(
                range(top_n_features),
                result.importances_mean[sorted_idx],
                xerr=result.importances_std[sorted_idx],
                align='center'
            )
            plt.yticks(range(top_n_features), [X_to_use.columns[i] for i in sorted_idx])
            plt.xlabel("Permutation Importance (mean decrease in score)")
            plt.title(f"Permutation Importance – {self.model_name}")
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"⚠️ Permutation importance failed: {e}")