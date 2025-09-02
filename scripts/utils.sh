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

generate_ssh_key() {
    local key_path="$HOME/.ssh/id_rsa"
    local pub_key_path="${key_path}.pub"

    # Ensure .ssh directory exists with correct permissions
    mkdir -p "$HOME/.ssh"
    chmod 700 "$HOME/.ssh"

    # Check if the default key already exists
    if [[ -f "$key_path" ]]; then
        echo "SSH key already exists at $key_path."
        read -p "Do you want to overwrite it? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborting key generation."
            if [[ -f "$pub_key_path" ]]; then
                echo "Existing public key:"
                cat "$pub_key_path"
            fi
            return 1
        fi
    fi

    # Generate the SSH key
    # -t rsa: specifies RSA key type
    # -b 4096: specifies a 4096-bit key
    # -f "$key_path": specifies the file to save the key
    # -N "": specifies an empty passphrase
    ssh-keygen -t rsa -b 4096 -f "$key_path" -N ""

    echo "SSH key generated successfully."
    echo "Public key ($pub_key_path):"
    cat "$pub_key_path"
}