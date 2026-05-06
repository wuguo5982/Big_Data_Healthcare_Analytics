def test_import_smoke_script_runs():
    import scripts.check_imports as check_imports
    assert callable(check_imports.main)


def test_data_validation_script_imports():
    import scripts.validate_data as validate_data
    assert callable(validate_data.main)
