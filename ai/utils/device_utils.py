# Device management utilities for CUDA/CPU handling
import torch


def get_device():
    """Get the best available device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        return device
    else:
        device = torch.device("cpu")
        print("ℹ️  Using CPU (CUDA not available)")
        return device


def move_to_device(tensor_or_model, device=None):
    """Move tensor or model to the specified device."""
    if device is None:
        device = get_device()
    
    return tensor_or_model.to(device, non_blocking=True)


def ensure_same_device(tensor, reference_model):
    """Ensure tensor is on the same device as the model."""
    model_device = next(reference_model.parameters()).device
    return tensor.to(model_device, non_blocking=True)


def print_device_info():
    """Print comprehensive device information."""
    print("🔧 Device Information:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA devices: {torch.cuda.device_count()}")
        print(f"   Current device: {torch.cuda.current_device()}")
        print(f"   Device name: {torch.cuda.get_device_name(0)}")
        print(f"   Device capability: {torch.cuda.get_device_capability(0)}")
        
        # Memory info
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**2   # MB
        print(f"   Memory allocated: {memory_allocated:.1f} MB")
        print(f"   Memory reserved: {memory_reserved:.1f} MB")
    else:
        print("   Using CPU computation")


def clear_cuda_cache():
    """Clear CUDA cache if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 CUDA cache cleared")


def setup_cuda_optimizations():
    """Setup CUDA optimizations for better performance."""
    if torch.cuda.is_available():
        # Enable tensor core usage for compatible hardware
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        print("⚡ CUDA optimizations enabled")
