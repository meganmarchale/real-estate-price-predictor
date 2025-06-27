import pandas as pd

class DataLoader:
    def __init__(self, path):
        self.path = path

    def load_data(self):
        """
        Loads data from a CSV file specified by self.path, removes rows with missing values, and returns the resulting DataFrame.

        Returns:
          pandas.DataFrame: The cleaned DataFrame with all rows containing NaN values removed.
        """
        df = pd.read_csv(self.path).dropna()
        return df

    def split_X_y(self, df, target_column="price"):
        """
        Splits a DataFrame into features (X) and target (y) based on the specified target column.

        Parameters:
          df (pd.DataFrame): The input DataFrame containing features and target.
          target_column (str, optional): The name of the target column to separate. Defaults to "price".

        Returns:
          Tuple[pd.DataFrame, pd.Series]: A tuple containing the features DataFrame (X) and the target Series (y).

        Notes:
          - Feature column names are sanitized by replacing non-alphanumeric characters with underscores.
          - Duplicate columns in X are removed.
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X.columns = X.columns.str.replace('[^a-zA-Z0-9_]', '_', regex=True)
        X = X.loc[:, ~X.columns.duplicated()]
        return X, y
