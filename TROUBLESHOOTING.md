# NCCL and Unsloth Troubleshooting Guide

## Problem: `undefined symbol: ncclCommInitRankScalable`

This error indicates NCCL (NVIDIA Collective Communications Library) compatibility issues.

### Quick Solutions (in order of preference):

### 1. Use the Updated Installation Script
```bash
bash run_training.sh
```
This automatically tries different PyTorch versions and falls back gracefully.

### 2. Use the Alternative Setup Script
```bash
bash setup_alternative.sh
```
This script tests multiple CUDA versions and provides detailed diagnostics.

### 3. Manual Installation Options

#### Option A: CUDA 12.1 (Recommended)
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Option B: CUDA 11.8 (Fallback)
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Option C: CPU-only (Last Resort)
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. Single GPU Training (Bypass NCCL)
If distributed training fails, use single GPU:
```bash
CUDA_VISIBLE_DEVICES=0 python GPRO_matmul_ec2.py
```

## Unsloth Integration Benefits

When Unsloth works properly, you get:
- **2x faster training**
- **50% less memory usage**
- **Automatic optimization** for your GPU type
- **Better convergence** with optimized chat templates

## Fallback Behavior

The updated script now:
1. **Tries to import Unsloth** - if successful, uses optimizations
2. **Falls back to standard PyTorch** - if Unsloth fails
3. **Adjusts batch sizes accordingly** - larger with Unsloth, smaller without
4. **Handles both single and multi-GPU** - automatically

## Verification Commands

Test your installation:
```bash
# Test PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Test CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Test NCCL
python -c "import torch; print(f'NCCL: {torch.distributed.is_nccl_available()}')"

# Test Unsloth
python -c "import unsloth; print('Unsloth: Available')" || echo "Unsloth: Not available"

# Test GPU count
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

## Expected Output Configurations

### With Unsloth (Optimal):
```
[UNSLOTH] Successfully imported Unsloth optimizations
[UNSLOTH] Using bfloat16 (optimal for modern GPUs)
[CONFIG] Multi-GPU mode detected: 4 GPUs
[UNSLOTH] Using Unsloth optimizations for improved memory efficiency
```

### Without Unsloth (Fallback):
```
[WARNING] Unsloth not available: No module named 'unsloth'
[FALLBACK] Will use standard PyTorch/transformers training
[STANDARD] Using float16 (fallback for older GPUs)
[CONFIG] Multi-GPU mode detected: 4 GPUs
[STANDARD] Using standard training (more conservative memory usage)
```

## Performance Expectations

| Mode | Batch Size/GPU | Memory Usage | Training Speed |
|------|---------------|--------------|----------------|
| Unsloth Multi-GPU | 2 | ~50% less | ~2x faster |
| Standard Multi-GPU | 1 | Normal | Normal |
| Unsloth Single-GPU | 4 | ~50% less | ~2x faster |
| Standard Single-GPU | 2 | Normal | Normal |

## If All Else Fails

1. **Use single GPU training**: `CUDA_VISIBLE_DEVICES=0 python GPRO_matmul_ec2.py`
2. **Check system compatibility**: Ensure CUDA drivers match PyTorch version
3. **Try CPU training**: Slower but will work on any system
4. **Check EC2 instance type**: Some instances have NCCL restrictions

## Common Error Messages and Solutions

### `ImportError: No module named 'unsloth'`
**Solution**: Script will automatically fall back to standard training.

### `RuntimeError: CUDA out of memory`
**Solution**: Script automatically uses smaller batch sizes without Unsloth.

### `Process group initialization failed`
**Solution**: Script will fall back to single GPU training.

### `undefined symbol: ncclCommInitRankScalable`
**Solution**: Use the updated installation scripts above. 