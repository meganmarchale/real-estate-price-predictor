import os
import sys
import time
import nbformat
from nbclient import NotebookClient
from rich.console import Console
from typing import List

# Patch for Windows + ZMQ compatibility
if sys.platform.startswith("win"):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add the project root to the Python path
project_root = os.path.abspath("../..")
sys.path.append(project_root)

console = Console()

class NotebookPipelineRunner:
    def __init__(self, notebook_paths: List[str]) -> None:
        self.notebook_paths = notebook_paths

    def run_pipeline(self) -> None:
        console.print("\n[bold blue]>>> Running pipeline notebooks with nbclient...\n[/bold blue]")
        for notebook_path in self.notebook_paths:
            self.run_notebook(notebook_path)
        console.print("[bold green]All notebooks executed successfully.[/bold green]")

    def run_notebook(self, notebook_path: str) -> None:
        console.print(f"[bold yellow][RUNNING][/bold yellow] {notebook_path}")
        start = time.time()

        with open(notebook_path, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        client = NotebookClient(nb, timeout=1200, kernel_name='python3')
        try:
            client.execute()

            # Display outputs of each code cell
            for i, cell in enumerate(nb.cells):
                if cell.cell_type == 'code':
                    for output in cell.get("outputs", []):
                        if output.output_type == "stream":
                            console.print(output.text.strip())
                        elif output.output_type == "execute_result":
                            console.print(output["data"].get("text/plain", "").strip())
                        elif output.output_type == "error":
                            console.print(f"[red]{''.join(output.get('traceback', []))}[/red]")

            # Save the executed notebook
            output_path = notebook_path.replace(".ipynb", "_executed.ipynb")
            with open(output_path, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)

            duration = time.time() - start
            console.print(f"[bold green]\n[SUCCESS][/bold green] Finished: {notebook_path} in {duration:.2f} seconds\n")

        except Exception as e:
            console.print(f"[bold red][ERROR][/bold red] Failed: {notebook_path} with error:\n{e}")
            raise

if __name__ == "__main__":
    pipeline = [
        "notebooks/pipeline/010_data_load_clean.ipynb",
        "notebooks/pipeline/020_visualization_clean_for_ml.ipynb",
        "notebooks/pipeline/030_preprocessing.ipynb",
        "notebooks/pipeline/040_train_baseline_model.ipynb",
        "notebooks/pipeline/050_tune_xgboost.ipynb",
        "notebooks/pipeline/060_tune_catboost.ipynb",
        "notebooks/pipeline/070_evaluation.ipynb",
        "notebooks/pipeline/080_export_model.ipynb",
        "notebooks/pipeline/090_inference.ipynb"
    ]

    runner = NotebookPipelineRunner(notebook_paths=pipeline)
    runner.run_pipeline()
