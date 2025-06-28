import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

class PreprocessingPipeline:
    def __init__(self, df, target_col="price", drop_cols=[]):
        self.df = df.copy()
        self.target_col = target_col
        self.drop_cols = drop_cols

    def fit_transform(self):
        # Remove specified columns
        df = self.df.drop(columns=self.drop_cols, errors="ignore")

        # Identify categorical and numerical columns
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

        # Remove target if present
        if self.target_col in numerical_cols:
            numerical_cols.remove(self.target_col)

        # Preprocessing pipeline
        preprocessor = ColumnTransformer(transformers=[
            ("num", SimpleImputer(strategy="median"), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ], remainder="passthrough")

        X = df.drop(columns=[self.target_col])
        y = df[self.target_col].values

        X_prepared = preprocessor.fit_transform(X)
        cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)
        final_columns = numerical_cols + list(cat_features) + [
            col for col in X.columns if col not in (numerical_cols + categorical_cols)
        ]

        df_processed = pd.DataFrame(X_prepared, columns=final_columns)
        df_processed[self.target_col] = y

        return df_processed
