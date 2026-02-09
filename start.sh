#!/usr/bin/env bash
set -e

export API_BASE="http://localhost:8000"

# API
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Streamlit
streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501