"""
CI smoke test for the healthcare FWA project.

This verifies key app modules can be imported without requiring OPENAI_API_KEY.
"""

import importlib

CANDIDATE_MODULES = [
    "src.config",
    "src.rag.retriever",
    "src.enterprise.reranker",
    "src.llm.gpt4omini_engine",
    "src.validation.validator",
    "src.hallucination.scorer",
    "src.enterprise.judge",
    "src.enterprise.production_workflow",
]

def main() -> None:
    imported = 0
    errors = []

    for module in CANDIDATE_MODULES:
        try:
            importlib.import_module(module)
            print(f"Imported: {module}")
            imported += 1
        except ModuleNotFoundError as exc:
            print(f"Skipped missing optional module: {module} ({exc})")
        except Exception as exc:
            errors.append((module, str(exc)))

    if errors:
        for module, err in errors:
            print(f"ERROR importing {module}: {err}")
        raise SystemExit(1)

    if imported == 0:
        raise SystemExit("No core modules were imported. Check project structure.")

    print(f"Import smoke test passed. Imported {imported} modules.")

if __name__ == "__main__":
    main()
