#!/usr/bin/env bash
# backend/alpaca/start_portfolio_service.sh

source /opt/venv/bin/activate

cd "$(dirname "$0")"

# Create necessary directory structure
mkdir -p /app/backend
if [ ! -L "/app/backend/alpaca" ]; then
  ln -sf /app /app/backend/alpaca
fi

# Add the app directory to PYTHONPATH
export PYTHONPATH=/app

# Start the Uvicorn server with the correct module path
exec python -m uvicorn backend.alpaca.api.portfolio_service:app --host 0.0.0.0 --port 8001 --reload

