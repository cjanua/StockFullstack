if ! which git >/dev/null 2>&1; then
    sudo pacman -S git
fi

if ! which bun >/dev/null 2>&1; then
    curl -fsSL https://bun.sh/install | bash
    source ~/.bashrc 
fi

# if ! which gh >/dev/null 2>&1; then
#     yay -Sy github-cli
# fi

if ! which podman >/dev/null 2>&1; then
    # Install Podman
    sudo pacman -S podman podman-compose openssh fuse-overlayfs

    mkdir -p ~/podman-storage

    # Update the storage configuration
    cat > ~/.config/containers/storage.conf << EOF
[storage]
driver = "vfs"
runroot = "/home/$USER/podman-storage/run"
graphroot = "/home/$USER/podman-storage/root"
EOF
fi
podman pull docker.io/library/redis:latest
# podman pull docker.io/quantconnect/lean:latest
if ! which docker >/dev/null 2>&1; then
    sudo pacman -S wget
    wget https://download.docker.com/linux/static/stable/x86_64/docker-27.5.1.tgz -qO- | tar xvfz - docker/docker --strip-components=1
    sudo mv ./docker /usr/local/bin
    yay -S docker-desktop
    systemctl --user start docker-desktop
    systemctl --user enable docker-desktop
fi

podman pull redis
# podman pull quantconnect/lean



source ./venv/bin/activate 
if ! which uv >/dev/null 2>&1; then
    pip install --upgrade uv 
fi

uv pip install -r req.txt

cd dashboard
bun install
cd ..

# chmod +x ./backend/alpaca/apca.py
chmod +x ./run.sh

has_systemd=$(ps -p 1 | grep -c systemd)
if [ "$has_systemd" -eq 1 ]; then
    echo ""
else
    sudo tee -a /etc/wsl.conf > /dev/null << EOF
[boot]
systemd=true
EOF
    echo "Restart your system to run systemd"
fi

if ! grep -q "cd /home/wsluser/Stocks" ~/.bashrc; then
    echo 'cd /home/wsluser/Stocks' >> ~/.bashrc
    echo 'docker-compose down' >> ~/.bashrc
    echo 'cd -' >> ~/.bashrc
    source ~/.bashrc
fi