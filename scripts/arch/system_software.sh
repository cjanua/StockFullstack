#!/bin/bash
chmod +x ./scripts/arch/dos2unix.sh
./scripts/arch/dos2unix.sh

# Directory for temporary files
TIMESTAMP_DIR="/tmp/pacman_update_check"
TIMESTAMP_FILE="${TIMESTAMP_DIR}/last_update"

# Create directory if it doesn't exist
mkdir -p "$TIMESTAMP_DIR"

# Function to check if update is needed
need_update() {
    # If timestamp file doesn't exist, update is needed
    if [ ! -f "$TIMESTAMP_FILE" ]; then
        return 0
    fi

    # Get last update timestamp
    last_update=$(cat "$TIMESTAMP_FILE")
    current_time=$(date +%s)
    
    # Calculate time difference in seconds (86400 seconds = 1 day)
    time_diff=$((current_time - last_update))
    
    # Return true (0) if more than a day has passed, false (1) otherwise
    [ $time_diff -ge 86400 ]
}

# Check if update is needed
if need_update; then
    echo "Running daily pacman sync..."
    sudo pacman -Sy
    # Store current timestamp
    date +%s > "$TIMESTAMP_FILE"
else
    echo "Pacman sync already performed today, skipping..."
fi

sudo pacman -S --needed base-devel git

if ! which yay >/dev/null 2>&1; then
    git clone https://aur.archlinux.org/yay.git
    cd yay
    makepkg -si
    cd ..
    rm -rf ./yay
fi

yay -Sy

if ! which direnv >/dev/null 2>&1; then
    yay -Sy direnv
    echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
    source ~/.bashrc
fi

if ! which python >/dev/null 2>&1; then
    sudo pacman -S python
fi