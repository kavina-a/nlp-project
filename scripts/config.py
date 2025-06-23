# config.py
import logging
import os
import sys
from pathlib import Path

import mlflow

# Directories
ROOT_DIR = Path(__file__).parent.parent.absolute()
LOGS_DIR = Path(ROOT_DIR, "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
EFS_DIR = Path(f"/efs/shared_storage/madewithml/{os.environ.get('GITHUB_USERNAME', '')}")
try:
    Path(EFS_DIR).mkdir(parents=True, exist_ok=True)
except OSError:
    EFS_DIR = Path(ROOT_DIR, "efs")
    Path(EFS_DIR).mkdir(parents=True, exist_ok=True)

# Config MLflow
MODEL_REGISTRY = Path(f"{EFS_DIR}/mlflow")
Path(MODEL_REGISTRY).mkdir(parents=True, exist_ok=True)
MLFLOW_TRACKING_URI = "file://" + str(MODEL_REGISTRY.absolute())
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

