#!/bin/bash

echo "🚀 Starting Dockerfile watchers..."

# Watcher for the 'dashboard' service
watchexec --watch ./dashboard/Dockerfile \
  -- 'echo " H Dockerfile for dashboard changed, rebuilding..." && docker compose build dashboard' &

# Watcher for the 'portfolio-service'
watchexec --watch ./backend/alpaca/Dockerfile \
  -- 'echo " H Dockerfile for portfolio-service changed, rebuilding..." && docker compose build portfolio-service' &

# Keep the script running and wait for background jobs
# Press Ctrl+C to stop all watchers
wait