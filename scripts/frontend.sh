#!/bin/bash

pwd
cd frontend


# Function to print usage instructions
print_usage() {
    echo "Usage: $0 [--feature-flag] [command arguments...]"
    echo "  --feature-flag    Enable special feature"
    echo "  command          Command to execute"
    echo "  arguments        Arguments to pass to command"
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
feature_enabled=false
filtered_args=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --feature-flag)
            feature_enabled=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            filtered_args+=("$1")
            shift
            ;;
    esac
done

# Check if command was provided
if [ ${#filtered_args[@]} -eq 0 ]; then
    echo "Error: No command provided"
    print_usage
    exit 1
fi

# Execute command with feature flag if it was provided in original args
if $feature_enabled && contains_element "--feature-flag" "${original_args[@]}"; then
    echo "Executing command with feature flag"
    "${filtered_args[@]}" --feature-flag
else
    echo "Executing command without feature flag"
    "${filtered_args[@]}"
fi