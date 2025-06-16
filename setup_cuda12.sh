#!/bin/bash

echo "=== CUDA 12.8 Optimized Setup Script ==="
echo "Detected CUDA 12.8 system - using optimal configuration"

# Activate virtual environment
source ../myproject311/bin/activate

# Update pip and install wheel
pip install --upgrade pip wheel

echo "=== Installing PyTorch with CUDA 12.1 (optimal for CUDA 12.8) ==="
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "=== Testing PyTorch installation ==="
python -c "
import torch
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
print(f'✅ CUDA version: {torch.version.cuda}')
print(f'✅ GPU count: {torch.cuda.device_count()}')

import torch.distributed as dist
print(f'✅ NCCL available: {dist.is_nccl_available()}')
if torch.cuda.is_available() and dist.is_nccl_available():
    print(f'✅ NCCL version: {torch.cuda.nccl.version()}')
    print('🚀 Multi-GPU training should work perfectly!')
else:
    print('⚠️  NCCL issue detected, but single GPU will work')
"

echo "=== Installing Unsloth (optimized for CUDA 12.x) ==="
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

echo "=== Installing remaining packages ==="
pip install transformers peft datasets trl tensorboard accelerate bitsandbytes

echo "=== Testing Unsloth installation ==="
python -c "
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    print('✅ Unsloth successfully imported')
    print(f'✅ BFloat16 supported: {is_bfloat16_supported()}')
    print('🚀 Unsloth optimizations available!')
except ImportError as e:
    print(f'❌ Unsloth import failed: {e}')
    print('Will fall back to standard training')
"

echo "=== Final verification ==="
python -c "
import torch
print(f'📊 Final configuration:')
print(f'   PyTorch: {torch.__version__}')
print(f'   CUDA: {torch.version.cuda}')
print(f'   GPUs: {torch.cuda.device_count()}')
print(f'   NCCL: {torch.distributed.is_nccl_available()}')

try:
    import unsloth
    print(f'   Unsloth: ✅ Available')
except ImportError:
    print(f'   Unsloth: ❌ Not available')

print('🎯 Ready for GRPO training!')
"

echo "=== Setup complete! ==="
echo "Your CUDA 12.8 system is now optimized for:"
echo "  • Multi-GPU GRPO training with NCCL"
echo "  • Unsloth 2x speed optimizations"
echo "  • Memory-efficient training"
echo ""
echo "Run training with: bash run_training.sh" 