if ! which bun >/dev/null 2>&1; then
    curl -fsSL https://bun.sh/install | bash
    source ~/.bashrc 
fi

if ! which git >/dev/null 2>&1; then
    sudo pacman -S git
fi

if ! which gh >/dev/null 2>&1; then
    yay -Sy github-cli
fi

if ! which podman >/dev/null 2>&1; then
    # Install Docker and related tools
    sudo pacman -S podman podman-compose openssh fuse-overlayfs
    # if [ ! -f /etc/containers/registries.conf ] || ! grep -q "unqualified-search-registries" /etc/containers/registries.conf; then
    #     echo "Adding registry configuration..."
    #     echo "$REGISTRY_CONFIG" | sudo tee /etc/containers/registries.conf > /dev/null
    # else
    #     echo "Registry configuration already exists, skipping..."
    # fi
    # First, we'll remove the old storage system completely
    # Create a user-owned directory for container storage
    mkdir -p ~/podman-storage

    # Update the storage configuration
    cat > ~/.config/containers/storage.conf << EOF
[storage]
driver = "vfs"
runroot = "/home/$USER/podman-storage/run"
graphroot = "/home/$USER/podman-storage/root"
EOF

    podman pull redis
fi

# if ! which snyk >/dev/null 2>&1; then
#     curl https://static.snyk.io/cli/latest/snyk-linux -o snyk
#     chmod +x ./snyk
#     mv ./snyk /usr/local/bin/ 
# fi

source ./venv/bin/activate 
if ! which uv >/dev/null 2>&1; then
    pip install --upgrade uv 
fi

uv pip install -r req.txt

cd frontend
bun install
cd ..

chmod +x ./backend/alpaca/apca.py
chmod +x ./run.sh

has_systemd=$(ps -p 1 | grep -c systemd)
if [ "$has_systemd" -eq 1 ]; then
    echo "Systemd is running"
else
    sudo tee -a /etc/wsl.conf > /dev/null << EOF
[boot]
systemd=true
EOF
    echo "Restart you WSL Image to run systemd"
fi

if ! grep -q "cd /home/wsluser/Stocks" ~/.bashrc; then
    echo 'cd /home/wsluser/Stocks' >> ~/.bashrc
    echo 'docker-compose down' >> ~/.bashrc
    echo 'cd -' >> ~/.bashrc
    source ~/.bashrc
fi