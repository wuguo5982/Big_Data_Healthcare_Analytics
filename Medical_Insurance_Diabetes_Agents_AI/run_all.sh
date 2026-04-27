#!/usr/bin/env bash
set -e
python training/train_model.py
python tests/test_prediction_tool.py
python tools/shap_explain.py
