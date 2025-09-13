#!/bin/bash
# Test GPU passthrough in WSL2

echo "Testing GPU passthrough in WSL2..."

echo "1. Checking WSL2 GPU support..."
if [ -d "/usr/lib/wsl" ]; then
    echo "✅ WSL2 detected"
    ls -la /usr/lib/wsl/lib/ | head -10
else
    echo "❌ WSL GPU libraries not found"
fi

echo -e "\n2. Testing in Docker container..."
if docker exec ai-training-stockfullstack python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
" 2>/dev/null; then
    echo "✅ GPU working in container"
else
    echo "❌ GPU not working in container"
fi

echo -e "\n3. Recommendations:"
echo "For best performance with your AMD RX 7800 XT:"
echo "  Option A: Use Windows native Python with ROCm"
echo "  Option B: Enable WSL2 GPU passthrough (requires driver updates)"
echo "  Option C: Continue with CPU (slower but works)"