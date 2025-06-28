import pandas as pd

class DataCleaner:
    """A class for cleaning real estate datasets by handling missing values and removing outliers."""
    def __init__(self, missing_threshold=0.5, row_threshold=0.7):
        self.missing_threshold = missing_threshold
        self.row_threshold = row_threshold

    def drop_columns_with_missing_values(self, df):
        """
        Drops columns from the DataFrame that have a proportion of missing values greater than the specified threshold.
        Parameters:
          df (pandas.DataFrame): The input DataFrame to process.
        Returns:
          pandas.DataFrame: A new DataFrame with columns containing excessive missing values removed.
        Prints:
          - Initial number of columns.
          - Number of columns dropped.
          - Number of remaining columns.
          - List of columns that were dropped.
        Note:
          The threshold for missing values is defined by the instance attribute `self.missing_threshold`.
        """
        initial_col_count = df.shape[1]
        cols_to_drop = df.columns[df.isnull().mean() > self.missing_threshold]
        df_cleaned = df.drop(columns=cols_to_drop)
        dropped_col_count = len(cols_to_drop)

        print("\n=== Drop columns with missing values ===")
        print(f"Initial number of columns: {initial_col_count}")
        print(f"Number of columns dropped: {dropped_col_count}")
        print(f"Remaining columns: {df_cleaned.shape[1]}")
        print(f"Columns dropped: {list(cols_to_drop)}")

        return df_cleaned

    def drop_rows_with_missing_values(self, df):
        """
        Remove rows from the DataFrame that have more missing values than allowed by the row_threshold.
        Parameters:
          df (pd.DataFrame): The input DataFrame to clean.
        Returns:
          pd.DataFrame: A DataFrame with rows containing too many missing values removed.
        Notes:
          - The minimum required number of non-null columns per row is calculated as:
            int(df.shape[1] * self.row_threshold)
          - Prints summary statistics about the number of rows before and after cleaning.
        """
        min_required = int(df.shape[1] * self.row_threshold)
        before_rows = df.shape[0]
        df_cleaned = df.dropna(thresh=min_required)
        after_rows = df_cleaned.shape[0]

        print("\n=== Drop rows with too many missing values ===")
        print(f"Threshold: ≥ {min_required} non-null columns")
        print(f"Rows before: {before_rows}, after: {after_rows}, removed: {before_rows - after_rows}")

        return df_cleaned

    def remove_outliers(self, df):
        """
        Removes outlier rows from the given DataFrame based on predefined value ranges for key columns.
        Parameters:
          df (pandas.DataFrame): The input DataFrame containing real estate data.
        Returns:
          pandas.DataFrame: A filtered DataFrame with outlier rows removed.
        The following filters are applied:
          - 'price' must be between 50,000 and 1,200,000 (inclusive)
          - 'habitableSurface' must be between 15 and 500 (inclusive)
          - 'bedroomCount' must be between 0 and 10 (inclusive)
          - 'bathroomCount' must be between 0 and 5 (inclusive)
          - 'toiletCount' must be between 0 and 5 (inclusive)
          - 'buildingConstructionYear' must be between 1850 and 2025 (inclusive)
        Prints a summary of the number of rows before and after outlier removal.
        """
        before_rows = df.shape[0]
        df_filtered = df[
            (df["price"].between(50_000, 1_200_000)) &
            (df["habitableSurface"].between(15, 500)) &
            (df["bedroomCount"].between(0, 10)) &
            (df["bathroomCount"].between(0, 5)) &
            (df["toiletCount"].between(0, 5)) &
            (df["buildingConstructionYear"].between(1850, 2025))
        ]
        after_rows = df_filtered.shape[0]

        print("\n=== Outlier Removal Summary ===")
        print(f"Rows before: {before_rows}, after: {after_rows}, removed: {before_rows - after_rows}")

        return df_filtered

    def clean_dataset(self, df):
        """
        Cleans the input DataFrame by performing a series of data cleaning steps.

        Steps performed:
          1. Drops columns that contain missing values.
          2. Drops rows that contain missing values.
          3. Removes outliers from the DataFrame.

        Args:
          df (pandas.DataFrame): The input DataFrame to be cleaned.

        Returns:
          pandas.DataFrame: The cleaned DataFrame after applying all cleaning steps.
        """
        df = self.drop_columns_with_missing_values(df)
        df = self.drop_rows_with_missing_values(df)
        df = self.remove_outliers(df)
        return df
