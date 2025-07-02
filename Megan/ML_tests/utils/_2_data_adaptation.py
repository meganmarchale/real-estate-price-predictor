
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df = df.copy()
        df = df.dropna(subset=["price", "habitableSurface"])
        df = df[
            (df["price"].between(30_000, 1_400_000)) &
            (df["habitableSurface"].between(10, 600)) &
            (df["bedroomCount"].between(0, 10)) &
            (df["bathroomCount"].between(0, 5)) &
            (df["toiletCount"].between(0, 5)) &
            (df["facedeCount"].between(1, 4))
        ]
        return df


class FeaturesCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df = df.copy()

        return df

