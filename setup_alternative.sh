#!/bin/bash

echo "=== Alternative Setup Script for NCCL Issues ==="

# Activate virtual environment
source ../myproject311/bin/activate

# Function to test NCCL
test_nccl() {
    python -c "
import torch
import torch.distributed as dist
try:
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'CUDA version: {torch.version.cuda}')
    if torch.cuda.is_available():
        print(f'GPU count: {torch.cuda.device_count()}')
        print(f'NCCL available: {dist.is_nccl_available()}')
        if dist.is_nccl_available():
            print(f'NCCL version: {torch.cuda.nccl.version()}')
        print('✅ NCCL test passed')
    exit(0)
except Exception as e:
    print(f'❌ NCCL test failed: {e}')
    exit(1)
"
}

# Option 1: Try CUDA 12.1 (most recent)
echo "=== Option 1: Installing PyTorch with CUDA 12.1 ==="
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if test_nccl; then
    echo "✅ CUDA 12.1 installation successful!"
else
    echo "❌ CUDA 12.1 failed, trying CUDA 11.8..."
    
    # Option 2: Try CUDA 11.8
    echo "=== Option 2: Installing PyTorch with CUDA 11.8 ==="
    pip uninstall torch torchvision torchaudio -y
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    if test_nccl; then
        echo "✅ CUDA 11.8 installation successful!"
    else
        echo "❌ CUDA 11.8 failed, trying CPU-only version..."
        
        # Option 3: CPU-only fallback
        echo "=== Option 3: Installing CPU-only PyTorch ==="
        pip uninstall torch torchvision torchaudio -y
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        
        echo "⚠️  Using CPU-only PyTorch. Training will be much slower."
        echo "Consider using a single GPU with CUDA_VISIBLE_DEVICES=0"
    fi
fi

# Install remaining packages
echo "=== Installing remaining packages ==="
pip install --upgrade pip wheel

# Try Unsloth installation
echo "Installing Unsloth..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" || {
    echo "⚠️  Unsloth installation failed, continuing without it..."
    echo "You can run the training without Unsloth optimizations"
}

# Install other packages
pip install transformers peft datasets trl tensorboard accelerate bitsandbytes

echo "=== Final verification ==="
python -c "
import torch
print(f'Final PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'NCCL available: {torch.distributed.is_nccl_available()}')

try:
    import unsloth
    print('✅ Unsloth available')
except ImportError:
    print('❌ Unsloth not available (will use standard training)')

try:
    import transformers, peft, datasets, trl
    print('✅ All training packages available')
except ImportError as e:
    print(f'❌ Missing packages: {e}')
"

echo "=== Setup complete! ==="
echo "If NCCL issues persist, use single GPU training:"
echo "CUDA_VISIBLE_DEVICES=0 python GPRO_matmul_ec2.py" 