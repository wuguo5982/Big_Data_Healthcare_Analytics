
def test_production_imports():
    from src.enterprise.production_workflow import run_production_analysis
    assert callable(run_production_analysis)
