import sys
from pathlib import Path

def setup_project_root(marker_file=".git"):
    """
    Trouve automatiquement le dossier racine du projet
    et l’ajoute au sys.path pour les imports.
    Compatible avec les notebooks et les scripts Python.
    """
    try:
        current = Path(__file__).resolve()  # scripts normaux
    except NameError:
        current = Path.cwd()  # notebooks

    for parent in current.parents:
        if (parent / marker_file).exists():
            project_root = parent
            break
    else:
        project_root = current.parents[1]  # fallback

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    return project_root
