"""Run with: uv run --no-sync blood_cancer --topic 'acute myeloid leukemia'"""
import os
os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("SSL_CERT_DIR", None)
os.environ.pop("SSLKEYLOGFILE", None)

import argparse
from datetime import date
from pathlib import Path
import truststore
truststore.inject_into_ssl()
from blood_cancer.crew import BloodCancerCrew

def run():
    parser = argparse.ArgumentParser(description="Educational blood cancer research (not clinical advice)")
    parser.add_argument("--topic", default="acute myeloid leukemia", help="Disease or research topic, not personal patient data")
    parser.add_argument("--question", default="Summarize diagnostic workup, relevant laboratory tests, and treatment evidence")
    args = parser.parse_args()
    Path("output").mkdir(exist_ok=True)
    inputs = {"topic": args.topic, "question": args.question, "current_date": date.today().isoformat()}
    print("Research-only prototype. Do not enter patient identifiers or use for diagnosis or treatment decisions.")
    BloodCancerCrew().crew().kickoff(inputs=inputs)

if __name__ == "__main__":
    run()
