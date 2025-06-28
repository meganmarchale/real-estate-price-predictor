import os
import sys
import time
import papermill as pm
from typing import List

# Add the project root to the Python path
project_root = os.path.abspath("../..")
sys.path.append(project_root)

class NotebookPipelineRunner:
    def __init__(self, notebook_paths: List[str]) -> None:
        self.notebook_paths = notebook_paths

    def run_pipeline(self) -> None:
        print(">>> Running pipeline notebooks with Papermill...\n")
        for notebook in self.notebook_paths:
            print(f"[RUNNING] {notebook}")
            output_path = notebook.replace(".ipynb", "_executed.ipynb")
            start = time.time()
            try:
                pm.execute_notebook(
                    input_path=notebook,
                    output_path=output_path,
                    log_output=True
                )
                duration = time.time() - start
                print(f"[SUCCESS] Finished: {notebook} in {duration:.2f} seconds\n")
            except Exception as e:
                print(f"[ERROR] Failed to execute {notebook} — {str(e)}")
                raise

        print("All notebooks executed successfully.")

if __name__ == "__main__":
    pipeline = [
        "notebooks/pipeline/010_data_load_clean.ipynb",
        "notebooks/pipeline/020_visualization_clean_for_ml.ipynb",
        "notebooks/pipeline/030_preprocessing.ipynb",
        "notebooks/pipeline/040_train_baseline_model.ipynb",
        "notebooks/pipeline/050_tune_xgboost.ipynb",
        "notebooks/pipeline/060_tune_catboost.ipynb"
    ]

    runner = NotebookPipelineRunner(notebook_paths=pipeline)
    runner.run_pipeline()
