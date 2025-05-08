# Re-import dependencies after code execution state reset
import os
from pathlib import Path

# Redefine paths
project_root = "/mnt/data/insurance-ci-cd-ml-project"
config_py_path = os.path.join(project_root, "prediction_model", "config", "config.py")

# New content for config.py adapted to insurance dataset
updated_config_py = """
from pathlib import Path

# Directories
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_FILE = PACKAGE_ROOT / "insurance.csv"

# Model config
TARGET = "charges"
FEATURES = [
    "age",
    "sex",
    "bmi",
    "children",
    "smoker",
    "region"
]

NUMERICAL_VARS = [
    "age",
    "bmi",
    "children"
]

CATEGORICAL_VARS = [
    "sex",
    "smoker",
    "region"
]

PIPELINE_NAME = "insurance_model"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output_v1.pkl"
"""

# Write the updated configuration to config.py
os.makedirs(os.path.dirname(config_py_path), exist_ok=True)
with open(config_py_path, "w") as f:
    f.write(updated_config_py)

config_py_path
