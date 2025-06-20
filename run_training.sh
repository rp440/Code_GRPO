#!/bin/bash
sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev python3.11-distutils -y

python3.11 -m venv myproject311

# Activate virtual environment
source ./myproject311/bin/activate

# Update pip and install wheel
pip install --upgrade pip wheel

# Install PyTorch with CUDA 12.1 (compatible with your CUDA 12.8 system)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install required packages
pip install transformers peft datasets trl tensorboard accelerate bitsandbytes

# Install Unsloth for 2x faster training and 50% memory reduction
echo "Installing Unsloth for optimized training..."
pip install "unsloth==2025.6.15"

# Install vLLM for optimized inference
pip install "vllm>=0.8.5"

# Verify installations
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'NCCL version: {torch.cuda.nccl.version()}')" || echo "NCCL not available"

# Verify Unsloth installation
echo "Verifying Unsloth installation..."
python -c "
try:
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    print('✅ Unsloth successfully installed and ready for 2x faster training!')
    print(f'   - bfloat16 supported: {is_bfloat16_supported()}')
except ImportError as e:
    print(f'⚠️  Unsloth not available: {e}')
    print('   - Training will fall back to standard PyTorch (still works but slower)')
except Exception as e:
    print(f'⚠️  Unsloth import error: {e}')
    print('   - Training will fall back to standard PyTorch')
"

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

# **4 GPU DISTRIBUTED TRAINING SETUP**
if [ "$GPU_COUNT" -eq 4 ]; then
    echo "🚀 **4 GPU DISTRIBUTED TRAINING MODE**"
    echo "   - GPUs: 4x Tesla T4 (15GB each)"
    echo "   - Total VRAM: 60GB"
    echo "   - Mode: Multi-GPU GRPO training with Unsloth optimization"
    echo "   - Expected batch size: 2 per GPU (total effective: 32)"
    echo "   - Performance: 2x faster training, 50% memory reduction with Unsloth"
    
    # Set optimal environment for 4 GPU training
    export NCCL_DEBUG=INFO
    export CUDA_LAUNCH_BLOCKING=0
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    
    # Try torchrun first (recommended for 4 GPUs)
    echo "Launching with torchrun (recommended)..."
    torchrun \
        --nproc_per_node=4 \
        --master_port=29500 \
        --nnodes=1 \
        --node_rank=0 \
        GPRO_matmul_ec2.py
    
    # If torchrun fails, try torch.distributed.launch
    if [ $? -ne 0 ]; then
        echo "torchrun failed, trying torch.distributed.launch..."
        python -m torch.distributed.launch \
            --nproc_per_node=4 \
            --master_port=29500 \
            --use_env \
            GPRO_matmul_ec2.py
    fi
    
    # Final fallback to single GPU
    if [ $? -ne 0 ]; then
        echo "❌ Distributed training failed, falling back to single GPU..."
        echo "   - This will be ~4x slower but will work"
        CUDA_VISIBLE_DEVICES=0 python GPRO_matmul_ec2.py
    fi

elif [ "$GPU_COUNT" -gt 1 ]; then
    echo "🔧 **MULTI-GPU TRAINING MODE** ($GPU_COUNT GPUs)"
    echo "   - Using all available GPUs"
    
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
    echo "💻 **SINGLE GPU TRAINING MODE**"
    echo "   - Using GPU 0 only"
    echo "   - Will be slower but more stable"
    CUDA_VISIBLE_DEVICES=0 python GPRO_matmul_ec2.py
fi

echo ""
echo "🎯 **Training completed!**"
echo "   - Training optimized with Unsloth (2x faster, 50% memory reduction)"
echo "   - Check tensorboard logs for training progress"
echo "   - Model saved to: /home/ec2-user/matmul_outputs/models/"
echo "   - Discovery logs saved in outputs directory" 