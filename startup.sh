#!/bin/bash
# Startup script for FastAPI application

# Navigate to the application directory
cd /home/site/wwwroot

# Install dependencies
pip install -r requirements.txt

# Start Uvicorn with live auto-reload
uvicorn backend_app:app --host 0.0.0.0 --port 8000 --reload
