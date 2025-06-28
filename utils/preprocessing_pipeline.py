import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

class PreprocessingPipeline:
    """
    A class to preprocess a DataFrame for machine learning tasks.
    """

    def __init__(self, df, target_col="price", drop_cols=None):
        self.df = df.copy()
        self.target_col = target_col
        self.drop_cols = drop_cols if drop_cols is not None else []

    def fit_transform(self):
        # Remove any unwanted columns (e.g., "url", "log_price")
        df = self.df.drop(columns=self.drop_cols, errors="ignore")

        # Extract target column and remove it from the feature set
        y = df[self.target_col].values
        df = df.drop(columns=[self.target_col])

        # Identify column types
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

        # Define preprocessing pipeline
        preprocessor = ColumnTransformer(transformers=[
            ("num", SimpleImputer(strategy="median"), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ], remainder="passthrough")

        # Fit and transform the features
        X_prepared = preprocessor.fit_transform(df)

        # Get final column names
        cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)
        final_columns = numerical_cols + list(cat_features) + [
            col for col in df.columns if col not in (numerical_cols + categorical_cols)
        ]

        # Create processed DataFrame
        df_processed = pd.DataFrame(X_prepared, columns=final_columns)
        df_processed[self.target_col] = y

        return df_processed
