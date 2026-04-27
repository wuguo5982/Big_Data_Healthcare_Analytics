#!/usr/bin/env bash
set -e
python -m src.generate_sample_data
python store_index.py
python -m src.train_model
streamlit run streamlit_app.py
