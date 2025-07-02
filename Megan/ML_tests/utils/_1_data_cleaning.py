import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DataCleaner(BaseEstimator, TransformerMixin):
    """Cleaning raw data."""
    def __init__(self, missing_threshold=1):
        self.missing_threshold = missing_threshold
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.clean_dataset(X.copy())

    def drop_columns_with_missing_values(self, df):
        initial_col_count = df.shape[1]
        cols_to_drop = df.columns[df.isnull().mean() > self.missing_threshold]
        df_cleaned = df.drop(columns=cols_to_drop)

        print(f"\n=== Drop columns with missing values ===")
        print(f"Initial: {initial_col_count}, Dropped: {len(cols_to_drop)}, Remaining: {df_cleaned.shape[1]}")
        return df_cleaned

    def drop_useless_columns(self, df):
        initial_col_count = df.shape[1]
        cols_to_drop = [
            "floorCount", "livingRoomSurface", "floodZoneType", "terraceOrientation",
            "roomCount", "streetFacadeWidth", "hasVisiophone", "hasLivingRoom"
        ]
        df_cleaned = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

        print(f"\n=== Drop columns with high level of missing values ===")
        print(f"Initial: {initial_col_count}, Dropped: {len(cols_to_drop)}, Remaining: {df_cleaned.shape[1]}")
        return df_cleaned
    
    def remove_outliers(self, df):
        before_rows = df.shape[0]
        df_filtered = df[
            (df["price"].between(50_000, 1_400_000)) &
            (df["habitableSurface"].between(10, 600)) &
            (df["bedroomCount"].between(0, 10)) &
            (df["bathroomCount"].between(0, 5)) &
            (df["toiletCount"].between(0, 5)) &
            (df["facedeCount"].between(1, 4))
        ]
        print(f"\n=== Drop outliers ===")
        print(f"Outliers removed: {before_rows - df_filtered.shape[0]}")
        return df_filtered

    def clean_dataset(self, df):
        df = self.drop_columns_with_missing_values(df)
        df = df.dropna(subset=["price", "habitableSurface"])
        print(f"After dropping NA in price/surface: {df.shape}")
        df = self.drop_useless_columns(df)
        df = self.remove_outliers(df)
        return df


class FeaturesCleaner:
    """Handle and tranform variables."""
    def __init__(self):
        pass

    def fill_epc(self, df):
        mode_epc = df.groupby('buildingCondition')['epcScore'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown")
        df['epcScore'] = df['epcScore'].fillna(df['buildingCondition'].map(mode_epc))
        df['epcScore'].fillna("unknown", inplace=True)
        df['buildingCondition'].fillna("unknown", inplace=True)
        print(f"EPC score and builgin condition OK")
        return df

    def handle_toilet_bathroom(self, df):
        df = df[~((df["toiletCount"].isna()) & (df["bathroomCount"].isna()))]
        df = df[~((df["toiletCount"] > 0) & (df["bathroomCount"].isna()))]
        df['toiletCount'] = df['toiletCount'].fillna(0)
        return df

    def drop_high_missing_columns(self, df):
        exceptions = ['hasSwimmingPool', 'hasGarden', 'gardenSurface']
        missing_percent = df.isnull().mean() * 100
        cols_to_drop = [col for col in missing_percent.index if missing_percent[col] > 80 and col not in exceptions]
        print(f"Dropping {len(cols_to_drop)} features with >70% missing values.")
        return df.drop(columns=cols_to_drop)

    def engineer_kitchen(self, df):
        df['kitchenSurface'].fillna(0, inplace=True)
        df['kitchenType'].fillna('Not installed', inplace=True)
        df['kitchen_installed'] = ~((df['kitchenSurface'] == 0) & (df['kitchenType'] == 'Not installed'))
        print(f"Kitchen types handled.")
        return df.drop(['kitchenSurface', 'kitchenType'], axis=1)

    def fill_missing_features(self, df):
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

    def categorize_terrace(self, df):
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

    def categorize_garden(self, df):
        if 'hasGarden' in df.columns and 'gardenSurface' in df.columns:
            def categorize(row):
                return 'No garden' if not row['hasGarden'] or row['gardenSurface'] == 0 else row['gardenSurface']

            df['gardenSurface'] = df.apply(categorize, axis=1)
            df.drop('hasGarden', axis=1, inplace=True)
            df['gardenSurface'].fillna('missing', inplace=True)
        return df

    def build_features(self, df):
        df['buildingConstructionYear'] = df['buildingConstructionYear'].fillna('missing')
        df['facedeCount'] = df['facedeCount'].fillna(0)
        df.loc[df['facedeCount'] > 4, 'facedeCount'] = 0
        df['heatingType'] = df['heatingType'].fillna('unknown')
        df.loc[df['type'].isin(["APARTMENT", "APARTMENT_GROUP"]), 'landSurface'] = 0
        df.dropna(subset=["landSurface"], inplace=True)
        return df

    def add_price_square_meter(self, df):
        df["price_square_meter"] = df["price"]/df["habitableSurface"]
        return df

    def engineer_all(self, df):
        df = self.fill_epc(df)
        df = self.handle_toilet_bathroom(df)
        df = self.drop_high_missing_columns(df)
        df = self.engineer_kitchen(df)
        df = self.fill_missing_features(df)
        df = self.categorize_terrace(df)
        df = self.categorize_garden(df)
        df = self.build_features(df)
        df = self.add_price_square_meter(df)
        return df


def main():
    input_path = "/Users/meganmarchale/Documents/BeCode/real-estate-price-predictor/data/immoweb_real_estate.csv"
    output_path = "/Users/meganmarchale/Documents/BeCode/real-estate-price-predictor/data/cleaned_data_megan.csv"

    df = pd.read_csv(input_path)

    cleaner = DataCleaner()
    engineer = FeaturesCleaner()

    print("Step 1: Cleaning raw data.")
    df = cleaner.clean_dataset(df)

    print("Step 2: Features handling.")
    df = engineer.engineer_all(df)

    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path}")


if __name__ == "__main__":
    main()
