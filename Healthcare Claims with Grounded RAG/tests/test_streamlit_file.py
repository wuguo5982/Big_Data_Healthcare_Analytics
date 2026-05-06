from pathlib import Path
import py_compile


def test_streamlit_app_compiles():
    root = Path(__file__).resolve().parents[1]
    app = root / "app" / "streamlit_app.py"
    assert app.exists()
    py_compile.compile(str(app), doraise=True)
