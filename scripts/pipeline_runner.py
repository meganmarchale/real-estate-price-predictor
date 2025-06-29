import os
import sys
import time
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from typing import List
from pathlib import Path

# Fix asyncio issue under Windows
if sys.platform.startswith("win"):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Detect project root
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

class NotebookPipelineRunner:
    def __init__(self, notebook_paths: List[str]) -> None:
        self.notebook_paths = [p for p in notebook_paths if "_executed" not in p]

    def run_pipeline(self) -> None:
        print("\n=== Running pipeline notebooks ===\n")

        log_path = os.path.join(project_root, "logs", f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        try:
            with open(log_path, "a", encoding="utf-8", errors="replace") as logf:
                logf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Pipeline started\n")
        except Exception as e:
            print("[WARNING] Failed to write to log file:", e)

        for notebook_path in self.notebook_paths:
            self.run_notebook(notebook_path)

        print("\n=== Pipeline completed ===")

    def run_notebook(self, notebook_path: str) -> None:
        print("\nRunning notebook:", notebook_path)

        start_time = time.time()

        try:
            abs_path = os.path.join(project_root, notebook_path)
            with open(abs_path, encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)

            client = NotebookClient(nb, timeout=1200, kernel_name="python3")
            client.execute()

            output_path = abs_path.replace(".ipynb", "_executed.ipynb")
            with open(output_path, "w", encoding="utf-8", errors="replace") as f:
                nbformat.write(nb, f)

            for i, cell in enumerate(nb.cells):
                if cell.cell_type == "code":
                    print("\n--- Cell", i, "---")
                    for output in cell.get("outputs", []):
                        if output.output_type == "stream":
                            print(output.text)
                        elif output.output_type == "execute_result":
                            print(output["data"].get("text/plain", ""))
                        elif output.output_type == "error":
                            print("[ERROR]", output.ename + ":", output.evalue)

            elapsed = time.time() - start_time
            print("Notebook finished:", notebook_path, f"({elapsed:.2f} seconds)")

        except CellExecutionError as e:
            print("[ERROR] Execution failed in notebook:", notebook_path)
            print(str(e))
            raise

if __name__ == "__main__":
    if "_executed" in os.getcwd():
        print("[ERROR] Execution aborted: running from an '_executed' directory.")
        sys.exit(1)

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

    runner = NotebookPipelineRunner(notebook_paths=pipeline)
    runner.run_pipeline()
