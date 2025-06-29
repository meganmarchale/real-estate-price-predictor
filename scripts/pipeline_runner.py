import os
import sys
import time
import papermill as pm
import papermill
from typing import List
from pathlib import Path

# Version check
REQUIRED_PAPERMILL_VERSION = "2.4.0"
if papermill.__version__ != REQUIRED_PAPERMILL_VERSION:
    raise RuntimeError(
        f"Papermill version {REQUIRED_PAPERMILL_VERSION} required, found {papermill.__version__}"
    )

print(f"Using papermill version: {papermill.__version__}")

# Compatibility fix for Windows Python 3.12+
if sys.platform.startswith("win"):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def get_project_root(marker=".git", fallback_name="real-estate-price-predictor"):
    current = os.path.abspath(__file__)
    current_dir = os.path.dirname(current)
    for parent in Path(current_dir).resolve().parents:
        if (parent / marker).exists():
            return str(parent.resolve())
        if fallback_name.lower() in parent.name.lower():
            return str(parent.resolve())
    raise RuntimeError("Could not detect project root")

project_root = get_project_root()
sys.path.insert(0, project_root)

class PapermillPipelineRunner:
    def __init__(self, notebook_paths: List[str]) -> None:
        self.notebook_paths = [p for p in notebook_paths if "_executed" not in p]

    def run_pipeline(self) -> None:
        print("\n=== Running pipeline with Papermill ===\n")

        log_dir = os.path.join(project_root, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log")

        with open(log_path, "a", encoding="utf-8", errors="replace") as logf:
            logf.write(f"Pipeline started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

            for notebook_path in self.notebook_paths:
                try:
                    self.run_notebook(notebook_path, logf)
                except Exception as e:
                    logf.write(f"\n[ERROR] Failed: {notebook_path}\n{str(e)}\n")
                    print(f"[ERROR] Failed: {notebook_path}")
                    break

        print("\n=== Pipeline completed ===")

    def run_notebook(self, notebook_path: str, logf) -> None:
        print(f"\nRunning notebook: {notebook_path}")
        start_time = time.time()

        input_path = os.path.join(project_root, notebook_path)
        output_path = input_path.replace(".ipynb", "_executed.ipynb")

        pm.execute_notebook(
            input_path=input_path,
            output_path=output_path,
            kernel_name="python3",
            log_output=True
        )

        elapsed = time.time() - start_time
        logf.write(f"Finished {notebook_path} in {elapsed:.2f} seconds\n")
        print(f"Finished: {notebook_path} ({elapsed:.2f} seconds)")

if __name__ == "__main__":
    pipeline = [
        "notebooks/pipeline/010_data_load_clean.ipynb",
        "notebooks/pipeline/020_visualization_clean_for_ml.ipynb",
        "notebooks/pipeline/030_preprocessing.ipynb",
        "notebooks/pipeline/040_train_baseline_model.ipynb",
        "notebooks/pipeline/050_tune_xgboost.ipynb",
        "notebooks/pipeline/060_tune_catboost.ipynb",
        "notebooks/pipeline/070_evaluation.ipynb",
        "notebooks/pipeline/080_inference.ipynb"
    ]

    runner = PapermillPipelineRunner(notebook_paths=pipeline)
    runner.run_pipeline()
