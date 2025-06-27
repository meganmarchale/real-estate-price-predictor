import os
import joblib
from datetime import datetime

class ModelManager:
    """A class to manage the saving and loading of machine learning models."""
    def __init__(self, model_dir="../models", use_timestamp=True):
        self.model_dir = model_dir
        self.use_timestamp = use_timestamp
        os.makedirs(self.model_dir, exist_ok=True)

    def _generate_model_path(self, model_name, extension="pkl"):
        """
        Generate a file path for the model, with timestamp if enabled.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if self.use_timestamp else ""
        filename = f"{model_name}_{timestamp}.{extension}" if timestamp else f"{model_name}.{extension}"
        return os.path.join(self.model_dir, filename)

    def save_model(self, model, model_name, extension="pkl"):
        """
        Save the model to disk (joblib by default).
        """
        path = self._generate_model_path(model_name, extension)
        joblib.dump(model, path)
        print(f"Model saved to: {path}")
        return path

    def load_model(self, path):
        """
        Load a model from disk.
        """
        model = joblib.load(path)
        print(f"Model loaded from: {path}")
        return model
