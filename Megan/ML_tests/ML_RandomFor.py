from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.pipeline import Pipeline
from utils.data_cleaner import FullDataCleaner
from utils._5_evaluation import evaluate_model

# Load data
X_train = pd.read_csv("/Users/meganmarchale/Documents/BeCode/real-estate-price-predictor/scripts/Megan/ML_tests/X_train.csv")
X_test = pd.read_csv("/Users/meganmarchale/Documents/BeCode/real-estate-price-predictor/scripts/Megan/ML_tests/X_test.csv")
y_train = pd.read_csv("/Users/meganmarchale/Documents/BeCode/real-estate-price-predictor/scripts/Megan/ML_tests/y_train.csv").values.ravel()
y_test = pd.read_csv("/Users/meganmarchale/Documents/BeCode/real-estate-price-predictor/scripts/Megan/ML_tests/y_test.csv").values.ravel()



# Pipeline
pipeline = Pipeline([
    ('cleaning', FullDataCleaner()),
    ('model', RandomForestRegressor(random_state=42))
])

# Fit
pipeline.fit(X_train, y_train)

# Evaluate
evaluate_model(pipeline, X_test, y_test)

