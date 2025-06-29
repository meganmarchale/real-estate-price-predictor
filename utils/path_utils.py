import sys
from pathlib import Path

# === [1. Robust project root detection for scripts + nbclient + notebooks] ===
import os
import sys
from pathlib import Path

def get_project_root(marker=".git", fallback_name="real-estate-price-predictor"):
    try:
        # Works in script or notebook (Jupyter, nbclient)
        base_path = Path(__file__).resolve()
    except NameError:
        # __file__ not defined (inside Jupyter or nbclient): fallback to notebook path
        try:
            # Works with nbclient: __vsc_ipynb_file__ may be set by VSCode
            from IPython import get_ipython
            base_path = Path(get_ipython().run_line_magic('pwd', '')).resolve()
        except:
            base_path = Path.cwd().resolve()

    for parent in [base_path] + list(base_path.parents):
        if (parent / marker).exists():
            return parent.resolve()

    for parent in base_path.parents:
        if fallback_name.lower() in parent.name.lower():
            return parent.resolve()

    raise RuntimeError(f"❌ Could not find project root using marker '{marker}' or fallback '{fallback_name}'")

project_root = get_project_root()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"📁 Using detected project root: {project_root}")
