#!/bin/bash

# Activate virtual environment
source ../myproject311/bin/activate

# Update pip and install wheel
pip install --upgrade pip wheel

# Install PyTorch with CUDA 12.1 (compatible with your CUDA 12.8 system)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Unsloth first (it will handle some dependencies optimally)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install other required packages
pip install transformers peft datasets trl tensorboard accelerate bitsandbytes

# Verify installations
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'NCCL version: {torch.cuda.nccl.version()}')" || echo "NCCL not available"

# Create necessary directories
mkdir -p /home/ec2-user/matmul_outputs/models
mkdir -p /home/ec2-user/matmul_outputs/tensorboard_logs

# Generate dataset first
echo "Generating training dataset..."
python dataset.py

# Set Hugging Face token (replace with your token)
# export HUGGING_FACE_HUB_TOKEN="your_token_here"

# Check if NCCL is working properly
echo "Testing NCCL compatibility..."
python -c "
import torch
import torch.distributed as dist
try:
    if torch.cuda.is_available():
        print(f'CUDA devices: {torch.cuda.device_count()}')
        print('NCCL backend available:', dist.is_nccl_available())
        print('Testing basic NCCL functionality...')
        # Test if we can initialize NCCL
        if dist.is_nccl_available():
            print('NCCL should work for distributed training')
        else:
            print('WARNING: NCCL not properly available')
    else:
        print('CUDA not available')
except Exception as e:
    print(f'NCCL test failed: {e}')
    print('Falling back to single GPU training')
"

# Check number of available GPUs
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Detected $GPU_COUNT GPUs"

if [ "$GPU_COUNT" -gt 1 ]; then
    echo "Attempting distributed training with $GPU_COUNT GPUs..."
    
    # Try torchrun (newer, more reliable than torch.distributed.launch)
    torchrun \
        --nproc_per_node=$GPU_COUNT \
        --master_port=29500 \
        GPRO_matmul_ec2.py
    
    # If torchrun fails, try the older method
    if [ $? -ne 0 ]; then
        echo "torchrun failed, trying torch.distributed.launch..."
        python -m torch.distributed.launch \
            --nproc_per_node=$GPU_COUNT \
            --master_port=29500 \
            --use_env \
            GPRO_matmul_ec2.py
    fi
    
    # If both fail, fall back to single GPU
    if [ $? -ne 0 ]; then
        echo "Distributed training failed, falling back to single GPU..."
        CUDA_VISIBLE_DEVICES=0 python GPRO_matmul_ec2.py
    fi
else
    echo "Single GPU detected, running single GPU training..."
    CUDA_VISIBLE_DEVICES=0 python GPRO_matmul_ec2.py
fi 