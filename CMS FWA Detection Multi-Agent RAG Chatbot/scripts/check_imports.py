"""
Import smoke test for CI/CD.

This test confirms the core modules can be imported without launching Streamlit/Flask.
"""


from pathlib import Path
import sys
import importlib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))



MODULES = [
    "src.config",
    "src.generate_sample_data",
    "src.data_loader",
    "src.feature_engineering",
    "src.fraud_models",
    "src.graph_detection",
    "src.guardrails",
    "src.notes_nlp",
    "src.prompts",
    "src.rag_pipeline",
    "agents.fwa_agents",
]

def main() -> None:
    errors = []
    for module in MODULES:
        try:
            importlib.import_module(module)
            print(f"Imported: {module}")
        except Exception as exc:
            errors.append((module, str(exc)))

    if errors:
        for module, err in errors:
            print(f"ERROR importing {module}: {err}")
        raise SystemExit(1)

    print("All core imports passed.")

if __name__ == "__main__":
    main()
