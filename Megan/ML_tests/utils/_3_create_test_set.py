import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("/Users/meganmarchale/Documents/BeCode/real-estate-price-predictor/data/cleaned_data_megan.csv")

# Split X / y
y = df["price"]
X = df.drop(["price", "price_square_meter"], axis=1)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\nTest set is ready")