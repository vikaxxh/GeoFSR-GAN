#!/bin/bash
set -e

if [ "$1" = 'api' ]; then
    echo "Starting GeoFSR-GAN FastAPI Backend Server on port 8000..."
    exec uvicorn api.main:app --host 0.0.0.0 --port 8000
elif [ "$1" = 'ui' ]; then
    echo "Starting GeoSR Streamlit Web Application on port 8501..."
    exec streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
else
    exec "$@"
fi
