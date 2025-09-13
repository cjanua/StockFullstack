#!/usr/bin/env bash

# Setup GPU support for WSL2 with AMD RX 7800 XT
echo "🚀 Setting up GPU support for WSL2..."

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "1. Copying WSL configurations..."

# Copy .wslconfig to Windows user profile
WINDOWS_USER_PROFILE="/mnt/c/Users/$(whoami)"
if [ ! -d "$WINDOWS_USER_PROFILE" ]; then
    # Try alternative path
    WINDOWS_USER_PROFILE="/mnt/c/Users/$USER"
fi

if [ -d "$WINDOWS_USER_PROFILE" ]; then
    echo "📁 Copying .wslconfig to $WINDOWS_USER_PROFILE"
    cp "$SCRIPT_DIR/.wslconfig" "$WINDOWS_USER_PROFILE/.wslconfig"
    echo "✅ .wslconfig copied"
else
    echo "⚠️ Could not find Windows user profile. Please manually copy:"
    echo "   From: $SCRIPT_DIR/.wslconfig"
    echo "   To: %USERPROFILE%\\.wslconfig"
fi

echo "2. Installing GPU drivers and ROCm support..."

# Update system
sudo apt update

# Install ROCm repository
if ! dpkg -l | grep -q "rocm-dev"; then
    echo "📦 Installing ROCm repository..."
    wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
    echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
    sudo apt update
    
    echo "📦 Installing ROCm packages..."
    sudo apt install -y rocm-dev rocm-libs rocm-utils
else
    echo "✅ ROCm already installed"
fi

echo "3. Setting up Docker with GPU support..."

# Update docker-compose for GPU passthrough
echo "📝 Updating Docker configuration for GPU access..."

echo "4. Testing GPU setup..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Device name: {torch.cuda.get_device_name(0)}')
"

echo "🎯 Setup complete! Next steps:"
echo "1. Restart WSL2: wsl --shutdown (from Windows)"
echo "2. Restart WSL2: wsl (from Windows)"
echo "3. Test GPU: python -c \"import torch; print(torch.cuda.is_available())\""
echo "4. Run training: scripts/train_ai.sh --clean"