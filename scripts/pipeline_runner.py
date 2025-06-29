import os
import sys
import time
import nbformat
from nbclient import NotebookClient
from typing import List
from rich.console import Console

# Patch for Windows + asyncio compatibility
if sys.platform.startswith("win"):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add project root to sys.path
project_root = os.path.abspath("../..")
sys.path.append(project_root)

console = Console(force_terminal=True, color_system="truecolor")

class NotebookPipelineRunner:
    def __init__(self, notebook_paths: List[str]) -> None:
        self.notebook_paths = [
            p for p in notebook_paths if "_executed" not in p
        ]

    def run_pipeline(self) -> None:
        console.print("\n[bold blue]Running pipeline notebooks with nbclient...\n[/bold blue]")

        with open("pipeline_run.log", "a", encoding="utf-8") as logf:
            logf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Pipeline started\n")

        for notebook_path in self.notebook_paths:
            self.run_notebook(notebook_path)

        console.print("\n[bold green]👏 All notebooks executed successfully.[/bold green]")

    def run_notebook(self, notebook_path: str) -> None:
        console.print(f"\n➡️  [bold yellow]Running:[/bold yellow] {notebook_path}")
        start = time.time()

        try:
            with open(notebook_path, encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)

            client = NotebookClient(nb, timeout=1200, kernel_name='python3')
            client.execute()

            output_path = notebook_path.replace(".ipynb", "_executed.ipynb")
            with open(output_path, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)

            # === Display cell outputs ===
            for i, cell in enumerate(nb.cells):
                if cell.cell_type == 'code':
                    outputs = cell.get("outputs", [])
                    for out in outputs:
                        if out.output_type == "stream" and out.name == "stdout":
                            console.print(f"[dim]--- Cell {i} ---[/dim]")
                            console.print(out.text.rstrip())

            duration = time.time() - start
            console.print(f"🟢 [bold green]Finished:[/bold green] {notebook_path} in {duration:.2f} seconds")

        except Exception as e:
            console.print(f"[bold red]🔴 Failed:[/bold red] {notebook_path} with error:\n{e}")
            raise

if __name__ == "__main__":
    if "_executed" in os.getcwd():
        print("🔴 Execution aborted: running from an '_executed' directory.")
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
