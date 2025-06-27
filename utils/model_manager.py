from catboost import CatBoostRegressor

class ModelManager:
    """A class to manage the training, saving, and loading of a CatBoost regression model."""
    def __init__(self, model=None):
        self.model = model or CatBoostRegressor(verbose=0)

    def train(self, X, y):
        """
        Trains the underlying model using the provided feature matrix X and target vector y.

        Parameters:
          X (array-like or pandas.DataFrame): Feature matrix used for training the model.
          y (array-like or pandas.Series): Target values corresponding to X.

        Returns:
          model: The trained model instance.
        """
        self.model.fit(X, y)
        return self.model

    def save(self, path):
        """
        Saves the trained model to the specified file path.

        Args:
          path (str): The file path where the model will be saved.
        """
        self.model.save_model(path)

    def load(self, path):
        """
        Loads a pre-trained model from the specified file path.

        Args:
          path (str): The file path from which to load the model.

        Returns:
          object: The loaded model instance.
        """
        self.model.load_model(path)
        return self.model
