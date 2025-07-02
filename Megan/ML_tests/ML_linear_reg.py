import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

from utils.data_cleaner import FullDataCleaner  # Fusion de DataCleaner & FeaturesCleaner
from utils._5_evaluation import evaluate_model  # Ton script d'évaluation


# Load data
df = pd.read_csv("/Users/meganmarchale/Documents/BeCode/real-estate-price-predictor/scripts/Megan/ML_tests/X_train.csv") 
y = pd.read_csv("/Users/meganmarchale/Documents/BeCode/real-estate-price-predictor/scripts/Megan/ML_tests/y_train.csv").squeeze()  

df = df.dropna(subset=["habitableSurface"])

# Pipeline
preprocessing = Pipeline([
    ("data_cleaning", FullDataCleaner())
])

# Adapt columns
numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = df.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

# Columns transformer
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
])

pipeline = Pipeline([
    ("cleaning", preprocessing),
    ("preprocessing", preprocessor),
    ("model", LinearRegression())
])

# Train
pipeline.fit(df, y)

# Evaluate
X_test = pd.read_csv("/Users/meganmarchale/Documents/BeCode/real-estate-price-predictor/scripts/Megan/ML_tests/X_test.csv")
y_test = pd.read_csv("/Users/meganmarchale/Documents/BeCode/real-estate-price-predictor/scripts/Megan/ML_tests/y_test.csv").squeeze()


X_test = X_test.dropna(subset=["habitableSurface"])

evaluate_model(pipeline, X_test, y_test)

# Save results
joblib.dump(pipeline, "linear_regression_model.joblib")
print("\nModel saved: 'linear_regression_model.joblib'")
