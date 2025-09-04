#!/usr/bin/env bash
set -e

# Counter for sudo commands
SUDO_COUNT=1

# Redefine sudo to include counter
sudo() {
  echo "SUDO COMMAND #$SUDO_COUNT: $@"
  SUDO_COUNT=$((SUDO_COUNT+1))
  command sudo "$@"
}

packages_to_install=""
for pkg in ca-certificates curl gnupg lsb-release git; do
  if ! dpkg -s "$pkg" &> /dev/null; then
    packages_to_install="$packages_to_install $pkg"
  fi
done

if [ -n "$packages_to_install" ]; then
  sudo apt-get install -y $packages_to_install
fi


if [ ! -d /etc/apt/keyrings ]; then
  sudo mkdir -p /etc/apt/keyrings
fi

if [ ! -f /etc/apt/keyrings/docker.gpg ]; then
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
fi

if ! (cat /etc/apt/sources.list.d/docker.list | grep -q "signed-by=/etc/apt/keyrings/docker.gpg"); then
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
fi


packages_to_install=""
for pkg in docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin; do
  if ! dpkg -s "$pkg" &> /dev/null; then
    packages_to_install="$packages_to_install $pkg"
  fi
done

if [ -n "$packages_to_install" ]; then
  sudo apt-get install -y $packages_to_install
  sudo usermod -aG docker $USER
  if ! groups | grep -q docker ; then
    newgrp docker
    echo "You may need to restart your session for the changes to take effect."
    echo "Re-run installation script to continue"
    exit 1
  fi
fi


if ! command -v amdgpu-install &> /dev/null; then
  wget https://repo.radeon.com/amdgpu-install/6.4.2.1/ubuntu/noble/amdgpu-install_6.4.60402-1_all.deb
  sudo apt install -y ./amdgpu-install_6.4.60402-1_all.deb

  amdgpu-install --list-usecase

  sudo apt install -y acl
  sudo usermod -aG render,video $USER
fi


# if [ -e /dev/dxg ] && ! getfacl -p /dev/dxg | grep -q "user:$USER:rw-"; then
#   sudo setfacl -m u:$USER:rw /dev/dxg
# fi

if [ ! -d ~/.nvm ]; then
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
fi

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

if ! nvm ls | grep -q "v22."; then
  nvm install 22
fi

if ! node -v | grep -q "v22."; then
  nvm use 22
  nvm alias default 22
fi

if ! command -v watchexec &> /dev/null; then
  echo "--> watchexec not found. Installing via Cargo (Rust's package manager)..."
  
  # Install rustup and cargo if they aren't already installed
  if ! command -v cargo &> /dev/null; then
    echo "--> Installing the Rust toolchain (rustup, cargo)..."
    # The '-y' flag makes the installation non-interactive
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # Add cargo to the current shell's PATH
    source "$HOME/.cargo/env"
  fi


  echo "--> Using Cargo to install watchexec-cli..."
  cargo install watchexec-cli
  
  echo "--> Verifying installation..."
  watchexec --version
fi

if ! command -v uv &> /dev/null
then
    echo "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    export PATH="$HOME/.cargo/bin:$PATH"
fi