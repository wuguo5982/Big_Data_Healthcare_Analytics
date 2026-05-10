def test_core_imports():
    import src.config
    import src.generate_sample_data
    import src.data_loader
    import agents.fwa_agents


def test_streamlit_and_flask_files_exist():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    assert (root / "streamlit_app.py").exists()
    assert (root / "app.py").exists()
    assert (root / "Dockerfile").exists()
