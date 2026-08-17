import argparse
import json
from src.factory import build_workflow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default="P0095")
    parser.add_argument(
        "--query",
        default="Summarize readmission risk and care-coordination priorities."
    )
    args = parser.parse_args()

    workflow = build_workflow()
    result = workflow.invoke(args.patient, args.query)

    print("\n=== MULTI-AGENT PLAN ===")
    print(" -> ".join(result["plan"]))

    print("\n=== RISK ===")
    print(json.dumps(result["risk_summary"], indent=2))

    print("\n=== EVIDENCE ===")
    print(json.dumps(result["evidence_summary"]["evidence"], indent=2))

    print("\n=== SAFETY ===")
    print(json.dumps(result["safety"], indent=2))

    print("\n=== FINAL REPORT ===")
    print(result["final_report"])

if __name__ == "__main__":
    main()
