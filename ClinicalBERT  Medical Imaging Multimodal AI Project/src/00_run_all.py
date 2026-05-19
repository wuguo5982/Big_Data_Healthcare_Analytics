"""Run the full NIH ClinicalBERT multimodal pipeline step by step."""

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


PIPELINE_SCRIPTS = [
    "01_prepare_nih_subset.py",
    "02_generate_ehr_and_notes.py",
    "03_extract_clinicalbert_embeddings.py",
    "04_train_clinicalbert_ml.py",
    "05_train_multimodal_fusion.py",
    "06_generate_explainable_outputs.py",
]


def run_script(script_name: str) -> None:
    """
    Run one pipeline script.

    If the script fails, the pipeline will stop automatically.
    """

    script_path = PROJECT_ROOT / "src" / script_name

    print("\n" + "=" * 80)
    print(f"Running: {script_name}")
    print("=" * 80)

    subprocess.run(
        [sys.executable, str(script_path)],
        check=True,
        cwd=str(PROJECT_ROOT),
    )

    print(f"Finished: {script_name}")


def main() -> None:
    """
    Run the complete multimodal AI pipeline.

    Pipeline steps:
    1. Prepare NIH image subset
    2. Generate synthetic EHR + clinical notes
    3. Extract ClinicalBERT embeddings
    4. Train traditional ML models
    5. Train multimodal fusion model
    6. Generate explainable JSON outputs
    """

    for script_name in PIPELINE_SCRIPTS:
        run_script(script_name)

    print("\nAll pipeline steps completed successfully.")


if __name__ == "__main__":
    main()