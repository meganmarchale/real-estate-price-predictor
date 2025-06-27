# plotting.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

class ModelVisualizer:
    def __init__(self, model, X, y, model_name="Model"):
        self.model = model
        self.X = X
        self.y = y
        self.model_name = model_name
        self.y_pred = model.predict(X)
        self.residuals = self.y - self.y_pred

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
        Plots the top 15 most important features used by the model.
        This method retrieves feature importances from the trained model, pairs them with their corresponding feature names,
        and displays a horizontal bar plot of the top 15 features ranked by importance. The plot provides a visual summary
        of which features contribute most to the model's predictions.
        Requirements:
            - The model must implement a `get_feature_importance()` method.
            - `self.X` must be a DataFrame containing the feature columns.
            - Requires matplotlib and seaborn for plotting.
        Raises:
            AttributeError: If the model does not have a `get_feature_importance()` method.
        """
        importances = self.model.get_feature_importance()
        features = self.X.columns
        df_feat = pd.DataFrame({"feature": features, "importance": importances})
        df_feat = df_feat.sort_values("importance", ascending=False).head(15)

        plt.figure(figsize=(8, 5))
        sns.barplot(data=df_feat, y="feature", x="importance")
        plt.title(f"Top 15 Feature Importances – {self.model_name}")
        plt.show()

    def plot_shap_summary(self):
        """
        Generates and displays a SHAP summary beeswarm plot for the model's predictions.

        This method uses the SHAP library to explain the output of the trained model on the feature set `self.X`.
        It computes SHAP values using the model stored in `self.model` and visualizes the top 20 most important features
        using a beeswarm plot, which shows the distribution of SHAP values for each feature.

        Returns:
            None
        """
        explainer = shap.Explainer(self.model)
        shap_values = explainer(self.X)
        shap.plots.beeswarm(shap_values, max_display=20)
