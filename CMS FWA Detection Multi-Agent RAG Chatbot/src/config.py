from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
POLICY_DIR = DATA_DIR / "cms_policy"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_DIR = ROOT_DIR / "vectorstore"
MODEL_DIR = ROOT_DIR / "models"
