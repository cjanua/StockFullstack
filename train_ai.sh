#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="ai-training-stockfullstack"
PROJECT_PATH="$SCRIPT_DIR"  # Adjust if your project root differs

print_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --clean           Clean model result cache inside the container"
    echo "  -h, --help        Show this help message"
}

clean=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            print_usage
            exit 0
            ;;
        --clean)
            clean=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

if ! docker ps | grep -q "$CONTAINER_NAME"; then
    echo "Starting Docker Compose environment..."
    docker-compose -f docker-compose.yaml -f docker-compose.dev.yaml up -d ai-training
fi

if [ "$clean" = true ]; then
    echo "Cleaning model cache inside the container..."
    docker exec "$CONTAINER_NAME" rm -rf /workspace/model_res/cache/*
fi

docker exec -it "$CONTAINER_NAME" python /workspace/ai/main.py
