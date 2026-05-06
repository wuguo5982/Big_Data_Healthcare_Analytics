
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

CLAIMS_PATH = DATA_DIR / "claims.csv"
CLINICAL_NOTES_PATH = DATA_DIR / "clinical_notes.csv"
POLICY_PATH = DATA_DIR / "cms_fwa_policy.txt"
AUDIT_RULES_PATH = DATA_DIR / "audit_rules.json"

DEFAULT_LLM_MODEL = "gpt-4o-mini"
PROJECT_TITLE = "Improved Healthcare FWA with Grounded RAG"
