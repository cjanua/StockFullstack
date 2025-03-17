#!/bin/bash
# direnv allow
pwd
source ./scripts/arch/archreq.sh
direnv allow

cd frontend


# Function to print usage instructions
print_usage() {
    echo "Usage: $0 [command arguments...] [--feature-flag]"
    echo "  run               Run Next.js Server"
    echo "  -fr               Enable Build"
    echo "  --clean           Clean Next.js Cache"
    echo "  --format          Prettier"
}

# Function to check if an array contains a value
contains_element() {
    local element="$1"
    shift
    local arr=("$@")
    for i in "${arr[@]}"; do
        if [[ "$i" == "$element" ]]; then
            return 0
        fi
    done
    return 1
}

# Store original arguments
original_args=("$@")
dev=true
filtered_args=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -req|--update)
            direnv allow
            exit 0
            ;;
        -fr|--build)
            dev=false
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        --clean)
            rm -rf .next
            shift
            ;;
        --format)
            bun run format
            shift
            ;;
        *)
            filtered_args+=("$1")
            shift
            ;;
    esac
done

# Check if command was provided
if [ ${#filtered_args[@]} -eq 0 ]; then
    bun run --watch --hot --exposed -b dev 
elif $feature_enabled && contains_element "-fr" "${original_args[@]}"; then
    bun run --silent build && bun run --silent start
else
    bun run --watch --hot --exposed -b dev
fi