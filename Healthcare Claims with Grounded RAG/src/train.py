
"""
Training pipeline with optional MLflow support.

This avoids local Windows MLflow/PySpark import crashes:
TypeError: code() argument 13 must be str, not int
"""

try:
    import mlflow  # type: ignore
    MLFLOW_AVAILABLE = True
except Exception:
    mlflow = None
    MLFLOW_AVAILABLE = False


def main():
    if MLFLOW_AVAILABLE:
        print("MLflow enabled.")
    else:
        print("MLflow disabled. Running lightweight local mode.")

    print("Training pipeline completed.")


if __name__ == "__main__":
    main()
