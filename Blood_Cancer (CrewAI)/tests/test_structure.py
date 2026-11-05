from pathlib import Path

def test_config_files_present():
    base = Path(__file__).resolve().parents[1]
    assert (base / "src/blood_cancer/config/agents.yaml").exists()
    assert (base / "src/blood_cancer/config/tasks.yaml").exists()
