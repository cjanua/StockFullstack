#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
pushd "$SCRIPT_DIR" >/dev/null
eval "$(direnv export bash)"
popd >/dev/null

# Function to print usage instructions
print_usage() {
    echo "Usage: $0 [command arguments...]"
    echo "  --clean           Clean model result cache"
    echo "  --monitor         Monitor GPU usage with rocm-smi using tmux"
    echo "  -h, --help        Show this help message"
}

# Store original arguments
original_args=("$@")
monitor=false
filtered_args=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            print_usage
            exit 0
            ;;
        --clean)
            echo "Cleaning model cache at $PROJECT_PATH/model_res/cache/..."
            rm -rf "$PROJECT_PATH"/model_res/cache/*
            shift
            ;;
        # --monitor)
        #     monitor=true
        #     shift
        #     ;;
        *)
            filtered_args+=("$1")
            shift
            ;;
    esac
done

# if [ "$monitor" = true ]; then
#     if ! command -v tmux &> /dev/null; then
#         echo "Error: tmux is not installed. Please install it to use --monitor." >&2
#         exit 1
#     fi
#     if ! command -v rocm-smi &> /dev/null; then
#         echo "Error: rocm-smi is not installed. Please ensure ROCm is installed to use --monitor." >&2
#         exit 1
#     fi

#     SESSION_NAME="stock_training_$$"
#     echo "Starting training session in tmux (session: $SESSION_NAME)..."
#     # Run the main command and kill the tmux session when it's done.
#     tmux new-session -d -s "$SESSION_NAME" "uv run stock-run ; tmux kill-session -t $SESSION_NAME"
#     tmux split-window -h -t "$SESSION_NAME" "watch -n 1 rocm-smi"
#     tmux attach-session -t "$SESSION_NAME"
# else
uv run stock-run
# fi