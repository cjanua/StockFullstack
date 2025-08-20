run_file_cleanup() {
    # Array of paths to clean. '.' means the whole project directory.
    local paths_to_clean=(".")

    # File extensions to clean
    local text_file_extensions=(
        "sh" "py" "js" "jsx" "ts" "tsx" "json" "yml" "yaml" "md"
        "css" "scss" "html" "txt" "envrc" ".env"
    )

    # Build grep pattern
    local extension_pattern=$(IFS="|"; echo "${text_file_extensions[*]}")

    echo "Ensuring correct file permissions and line endings..."

    for path in "${paths_to_clean[@]}"; do
        # Find text files and scripts in bin/ tracked by git
        git ls-files "$path" | grep -E "\.($extension_pattern)$|/bin/" | while IFS= read -r file; do
            if [[ -f "$file" ]]; then
                # Convert DOS/Windows line endings to Unix
                sed -i 's/\r$//' "$file"

                # Make shell scripts and bin executables executable
                if [[ "$file" == *.sh ]] || [[ "$file" == bin/* ]]; then
                    chmod +x "$file"
                fi
            fi
        done
    done
}