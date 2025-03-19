#!/bin/bash
cd "$(dirname "$0")"

# pip install -r req.txt

python -m uvicorn portfolio_service:app --host 0.0.0.0 --port 8001 --reload

