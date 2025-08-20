#!/usr/bin/env bash
# scripts/ubuntu/install_deps.sh
set -e  # Exit on any error



if ! command -v nix &> /dev/null
then
  sh <(curl --proto '=https' --tlsv1.2 -L https://nixos.org/nix/install) --daemon

  echo "Enabling Nix flakes..."

  sudo mkdir -p /etc/nix

  echo "experimental-features = nix-command flakes" | sudo tee -a /etc/nix/nix.conf > /dev/null

  sudo systemctl restart nix-daemon
  echo "Nix flakes enabled."

  if [ -e /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh ]; then
    . /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh
  elif [ -e /etc/profile.d/nix.sh ]; then
    . /etc/profile.d/nix.sh
  fi

  echo "Git: Adding project directory to be owned by user"
  git config --global --add safe.directory /mnt/d/Repos/Stocks
fi

if ! command -v direnv &> /dev/null
then
  echo "Installing direnv..."
  sudo apt update
  sudo apt install -y direnv

  # Add direnv hook to .bashrc if not already present
  if ! grep -q 'eval "$(direnv hook bash)"' ~/.bashrc; then
    echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
    echo "Added direnv hook to ~/.bashrc"
  else
    echo "direnv hook already present in ~/.bashrc"
  fi
fi

if ! command -v amdgpu-install &> /dev/null; then
  wget https://repo.radeon.com/amdgpu-install/6.4.2.1/ubuntu/noble/amdgpu-install_6.4.60402-1_all.deb
  sudo apt install -y ./amdgpu-install_6.4.60402-1_all.deb

  amdgpu-install --list-usecase
fi

sudo usermod -aG render,video $USER


sudo apt update
sudo apt install -y acl

if [ -e /dev/dxg ]; then
  sudo setfacl -m u:$USER:rw /dev/dxg
fi

echo "Setup complete. You can now use Nix flakes."

rocminfo
rocm-smi