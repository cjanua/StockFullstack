#!/usr/bin/env bash
# dashboard/entrypoint.sh
set -e

# Ensure package files are present (mounted from host)
if [ ! -f /app/package.json ] || [ ! -f /app/package-lock.json ]; then
    echo "Error: package.json or package-lock.json not found."
    exit 1
fi

# Compute hash of package.json and package-lock.json
hash=$(cat /app/package.json /app/package-lock.json | sha256sum | cut -d ' ' -f 1)

# Path to hash file in node_modules (persisted in volume)
hash_file="/app/node_modules/.deps-hash"

if [ ! -f "$hash_file" ] || [ "$(cat "$hash_file")" != "$hash" ]; then
    echo "Dependencies changed or first run. Installing..."
    npm ci
    npm install --save-exact --save-dev typescript
    echo "$hash" > "$hash_file"
else
    echo "Dependencies unchanged. Skipping install."
fi

# Execute the original CMD
exec "$@"