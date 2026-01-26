import os
from pathlib import Path

# Project Root (calculated relative to this file: src/ttstl/config.py -> ../../)
# This assumes the file is located at <repo_root>/src/ttstl/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data Directory
# Users can override the default location via an environment variable
DATA_DIR = Path(os.getenv("TTSTL_DATA_DIR", PROJECT_ROOT / "data"))

# Manifest Path
MANIFEST_PATH = DATA_DIR / "data_manifest.json"

# Dataset Mapping
# Maps logical dataset names to their physical directory paths
DATASETS = {"synthetic": DATA_DIR / "synthetic_data", "intc": DATA_DIR / "intc_data"}
