#!/usr/bin/env bash
# scripts/train_ai.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="ai-training-stockfullstack"
PROJECT_PATH="$SCRIPT_DIR"  # Adjust if your project root differs

print_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --clean           Clean model result cache inside the container"
    echo "  --gpu-profile     Run with CUDA profiling to optimize performance"
    echo "  --batch-size N    Set custom batch size for training (default: auto)"
    echo "  --no-cache        Force rebuild of Docker image with no cache"
    echo "  -h, --help        Show this help message"
}

clean=false
gpu_profile=false
batch_size=""
no_cache=false

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
        --gpu-profile)
            gpu_profile=true
            shift
            ;;
        --batch-size)
            batch_size="$2"
            shift 2
            ;;
        --no-cache)
            no_cache=true
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
    compose_cmd="docker-compose -f docker-compose.yaml -f docker-compose.dev.yaml"
    
    if [ "$no_cache" = true ]; then
        echo "Building with --no-cache flag..."
        $compose_cmd build --no-cache ai-training
    fi
    
    $compose_cmd up -d ai-training
fi

if [ "$clean" = true ]; then
    echo "Cleaning model cache inside the container..."
    docker exec "$CONTAINER_NAME" rm -rf /workspace/model_res/cache/*
fi

# Set environment variables for optimization
ENV_VARS=""

if [ -n "$batch_size" ]; then
    ENV_VARS+="BATCH_SIZE=$batch_size "
    echo "Setting custom batch size: $batch_size"
fi

if [ "$gpu_profile" = true ]; then
    ENV_VARS+="CUDA_LAUNCH_BLOCKING=1 "
    echo "Enabling CUDA profiling for performance analysis"
fi

# Add PYTORCH_CUDA_ALLOC_CONF for better memory management
ENV_VARS+="PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 "

echo "Starting training with optimized settings..."
docker exec -it "$CONTAINER_NAME" /bin/bash -c "${ENV_VARS} python ai/main.py" > ./ai.log 2>&1