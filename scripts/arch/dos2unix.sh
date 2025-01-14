#!/bin/bash

# Print colored status messages
print_status() {
    local color=$1
    local message=$2
    case $color in
        "green")  echo -e "\033[32m$message\033[0m" ;;
        "yellow") echo -e "\033[33m$message\033[0m" ;;
        "red")    echo -e "\033[31m$message\033[0m" ;;
        *)        echo "$message" ;;
    esac
}

# File extensions to clean
FILE_EXTENSIONS=(
    "sh"
    "py"
    "js"
    "jsx"
    "ts"
    "tsx"
    "json"
    "yml"
    "yaml"
    "md"
    "css"
    "scss"
    "html"
    "txt"
)

# Build extension pattern
EXTENSION_PATTERN=""
for ext in "${FILE_EXTENSIONS[@]}"; do
    EXTENSION_PATTERN="$EXTENSION_PATTERN -o -name '*.$ext'"
done
EXTENSION_PATTERN=${EXTENSION_PATTERN:3} # Remove initial "-o "

print_status "yellow" "Starting Windows artifact cleanup..."

# Get list of tracked files respecting .gitignore
TRACKED_FILES=$(git ls-files)

# Clean only tracked files with specified extensions
while IFS= read -r file; do
    if [[ -f "$file" ]] && [[ "$file" =~ \.(sh|py|js|jsx|ts|tsx|json|yml|yaml|md|css|scss|html|txt)$ ]]; then
        sed -i 's/\r$//' "$file"
        if [[ "$file" =~ \.sh$ ]]; then
            chmod +x "$file"
        fi
        if [ "$1" == "--verbose" ] || [ "$1" == "-v" ]; then
            print_status "yellow" "Cleaned: $file"
        fi
    fi
done <<< "$TRACKED_FILES"

print_status "green" "Cleanup completed successfully!"