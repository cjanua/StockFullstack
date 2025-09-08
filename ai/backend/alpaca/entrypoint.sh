#!/usr/bin/env bash
set -e

VENV_MARKER="/opt/venv/bin/activate"

if [ ! -f req.txt ]; then
  echo "Error: req.txt not found."
  exit 1
fi

# CORRECTED the command from 'sha2sum' to 'sha256sum'
hash=$(sha256sum req.txt | cut -d ' ' -f 1)
hash_file="/opt/venv/.deps-hash"

if [ ! -f "$VENV_MARKER" ] || [ ! -f "$hash_file" ] || [ "$(cat "$hash_file")" != "$hash" ]; then
  echo "🐍 Virtual environment is missing or dependencies have changed. Installing..."
  python -m venv /opt/venv
  source /opt/venv/bin/activate
  pip install --no-cache-dir -r req.txt
  echo "$hash" > "$hash_file"
  echo "✅ Installation complete."
else
  echo "✅ Dependencies are unchanged. Skipping install."
fi

exec "$@"