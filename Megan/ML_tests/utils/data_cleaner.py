"""Lighter version"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FullDataCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # ✅ Ne dropper que habitableSurface (car 'price' n'est plus dans X)
        df = df.dropna(subset=["habitableSurface"])

        # Nettoyage basique (tu peux adapter selon tes besoins)
        if "province" in df.columns:
            df["province"] = df["province"].str.strip().str.lower()

        if "type" in df.columns:
            df["type"] = df["type"].str.strip().str.lower()

        if "buildingCondition" in df.columns:
            df["buildingCondition"] = df["buildingCondition"].str.strip().str.lower()

        return df

    # Cleaning 
    def _drop_columns_with_missing_values(self, df):
        cols_to_drop = df.columns[df.isnull().mean() > self.missing_threshold]
        return df.drop(columns=cols_to_drop)

    def _drop_useless_columns(self, df):
        cols_to_drop = [
            "floorCount", "livingRoomSurface", "floodZoneType", "terraceOrientation",
            "roomCount", "streetFacadeWidth", "hasVisiophone", "hasLivingRoom"
        ]
        return df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    def _remove_outliers(self, df):
        return df[
            (df["price"].between(50_000, 1_400_000)) &
            (df["habitableSurface"].between(10, 600)) &
            (df["bedroomCount"].between(0, 10)) &
            (df["bathroomCount"].between(0, 5)) &
            (df["toiletCount"].between(0, 5)) &
            (df["facedeCount"].between(1, 4))
        ]

    # Feature Engineering
    def _fill_epc(self, df):
        mode_epc = df.groupby('buildingCondition')['epcScore'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown")
        df['epcScore'] = df['epcScore'].fillna(df['buildingCondition'].map(mode_epc))
        df['epcScore'].fillna("unknown", inplace=True)
        df['buildingCondition'].fillna("unknown", inplace=True)
        return df

    def _handle_toilet_bathroom(self, df):
        df = df[~((df["toiletCount"].isna()) & (df["bathroomCount"].isna()))]
        df = df[~((df["toiletCount"] > 0) & (df["bathroomCount"].isna()))]
        df['toiletCount'] = df['toiletCount'].fillna(0)
        return df

    def _drop_high_missing_columns(self, df):
        exceptions = ['hasSwimmingPool', 'hasGarden', 'gardenSurface']
        missing_percent = df.isnull().mean() * 100
        cols_to_drop = [col for col in missing_percent.index if missing_percent[col] > 80 and col not in exceptions]
        return df.drop(columns=cols_to_drop)

    def _engineer_kitchen(self, df):
        df['kitchenSurface'].fillna(0, inplace=True)
        df['kitchenType'].fillna('Not installed', inplace=True)
        df['kitchen_installed'] = ~((df['kitchenSurface'] == 0) & (df['kitchenType'] == 'Not installed'))
        return df.drop(['kitchenSurface', 'kitchenType'], axis=1)

    def _fill_missing_features(self, df):
        fill_values = {
            'terraceSurface': 0,
            'hasTerrace': False,
            'parkingCountOutdoor': 0,
            'parkingCountIndoor': 0,
            'hasLift': 'False',
            'hasSwimmingPool': 'False',
            'bedroomCount': 'missing',
            'hasBasement': 'False'
        }
        return df.fillna(value=fill_values)

    def _categorize_terrace(self, df):
        non_zero = df[df['terraceSurface'] > 0]['terraceSurface']
        q33, q66 = non_zero.quantile([0.33, 0.66])

        def categorize(row):
            if not row['hasTerrace'] or row['terraceSurface'] == 0:
                return 'No terrace'
            elif row['terraceSurface'] <= q33:
                return 'Small'
            elif row['terraceSurface'] <= q66:
                return 'Medium'
            else:
                return 'Big'

        df['terraceSurface'] = df.apply(categorize, axis=1)
        return df.drop('hasTerrace', axis=1)

    def _categorize_garden(self, df):
        if 'hasGarden' in df.columns and 'gardenSurface' in df.columns:
            def categorize(row):
                return 'No garden' if not row['hasGarden'] or row['gardenSurface'] == 0 else row['gardenSurface']

            df['gardenSurface'] = df.apply(categorize, axis=1)
            df.drop('hasGarden', axis=1, inplace=True)
            df['gardenSurface'].fillna('missing', inplace=True)
        return df

    def _build_features(self, df):
        df['buildingConstructionYear'] = df['buildingConstructionYear'].fillna('missing')
        df['facedeCount'] = df['facedeCount'].fillna(0)
        df.loc[df['facedeCount'] > 4, 'facedeCount'] = 0
        df['heatingType'] = df['heatingType'].fillna('unknown')
        df.loc[df['type'].isin(["APARTMENT", "APARTMENT_GROUP"]), 'landSurface'] = 0
        df.dropna(subset=["landSurface"], inplace=True)
        return df

    def _add_price_square_meter(self, df):
        df["price_square_meter"] = df["price"] / df["habitableSurface"]
        return df
