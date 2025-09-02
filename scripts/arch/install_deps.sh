#!/usr/bin/env bash
# scripts/arch/install_deps.sh

# Define the path to the package list and read it
PACKAGE_LIST_FILE="$(dirname "$0")/packages.txt"
PACKAGES_TO_INSTALL=$(cat "$PACKAGE_LIST_FILE")

# Check if all packages are installed. If not, install the missing ones.
if ! pacman -Q $PACKAGES_TO_INSTALL &> /dev/null; then
    sudo pacman -S --noconfirm --needed $PACKAGES_TO_INSTALL
fi

if ! command -v pyenv &> /dev/null
then
    sudo pacman -S --noconfirm pyenv
     
fi
if ! grep -q 'eval "$(pyenv init -)"' ~/.bashrc; then
    sudo pacman -S base-devel openssl zlib xz tk libffi
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
fi


if ! command -v uv &> /dev/null
then
    echo "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    export PATH="$HOME/.cargo/bin:$PATH"
fi


if ! command -v direnv &> /dev/null
then
    sudo pacman -S direnv
fi

if ! command -v docker &> /dev/null
then
    sudo pacman -S docker
fi


if ! grep -q 'eval "$(direnv hook bash)"' ~/.bashrc; then
    echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
    source ~/.bashrc
fi

if ! command -v rocminfo &> /dev/null
then
    sudo pacman -S rocm-hip-sdk rocm-opencl-sdk rocm-smi-lib hipblas rocblas hipblaslt rocminfo
    sudo usermod -aG video,render cjanua

fi

if ! grep -q 'export HSA_OVERRIDE_GFX_VERSION=11.0.0' ~/.bashrc; then
    echo 'export HSA_OVERRIDE_GFX_VERSION=11.0.0' >> ~/.bashrc
    source ~/.bashrc
fi
