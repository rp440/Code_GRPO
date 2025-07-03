#!/bin/bash
#  apt update
#  apt install software-properties-common -y
#  add-apt-repository ppa:deadsnakes/ppa -y
#  apt update
#  apt install python3.11 python3.11-venv python3.11-dev python3.11-distutils -y

# python3.11 -m venv myproject311
 python3 -m venv myproject311


# Activate virtual environment
source ./myproject311/bin/activate

# Update pip and install wheel
pip install --upgrade pip wheel

# Option 1: Install from frozen requirements (recommended for exact reproducibility)
# if [ -f "requirements.txt" ]; then
#     echo "Installing from frozen requirements.txt for exact reproducibility..."
#     pip install -r requirements.txt
# else
#     echo "requirements.txt not found, installing packages individually..."
    
    # Install PyTorch with CUDA 12.1 (compatible with your CUDA 12.8 system)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    # Install required packages
    pip install transformers peft datasets trl tensorboard accelerate bitsandbytes

    # Install vLLM for optimized inference
    pip install "vllm>=0.8.5"


# Verify installations
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'NCCL version: {torch.cuda.nccl.version()}')" || echo "NCCL not available"

# Verify transformers installation
echo "Verifying transformers installation..."
python3 -c "
try:
    import transformers
    import peft
    print('✅ Transformers and PEFT successfully installed!')
    print(f'   - Transformers version: {transformers.__version__}')
    print(f'   - Using 4-bit quantization with 512 token context length')
except ImportError as e:
    print(f'⚠️  Import error: {e}')
    print('   - Please check your installations')
except Exception as e:
    print(f'⚠️  Error: {e}')
"

# Install rclone and sync files from Google Drive
echo "Installing rclone and syncing files from Google Drive..."
sudo apt update && sudo apt install -y rclone

# Configure rclone (interactive; skip if already configured)
rclone config

# Copy files from Google Drive to local workspace (update YOUR/DRIVE/PATH as needed)
rclone copy gdrive:YOUR/DRIVE/PATH ./Code_GRPO/ --progress

# Create necessary directories
mkdir -p ~/matmul_outputs/models
mkdir -p ~/matmul_outputs/tensorboard_logs

# Generate dataset first
echo "Generating training dataset..."
python3 dataset.py

# Set Hugging Face token (replace with your token)
# export HUGGING_FACE_HUB_TOKEN="your_token_here"

# Check if NCCL is working properly
echo "Testing NCCL compatibility..."
python3 -c "
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
echo "Detecting GPUs..."
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>&1)

# Debug: Show raw output
echo "Raw GPU detection output: '$GPU_COUNT'"

# Handle case where GPU detection fails or returns non-numeric
if [ -z "$GPU_COUNT" ] || ! [[ "$GPU_COUNT" =~ ^[0-9]+$ ]]; then
    echo "⚠️  GPU detection failed or returned non-numeric value, defaulting to 0 GPUs (CPU only)"
    echo "   Raw output was: '$GPU_COUNT'"
    GPU_COUNT=0
fi

echo "✅ Detected $GPU_COUNT GPUs"

# **2 GPU DISTRIBUTED TRAINING SETUP**
if [ "$GPU_COUNT" -eq 2 ]; then
    echo "🚀 **2 GPU DISTRIBUTED TRAINING MODE**"
    echo "   - GPUs: 2x GPU (detected)"
    echo "   - Mode: Multi-GPU GRPO training with 4-bit quantization"
    echo "   - Expected batch size: 8 per GPU × 16 grad_acc = 256 total effective"
    echo "   - Configuration: 4-bit quantization with 512 token context length"
    
    # Set optimal environment for 2 GPU training
    export NCCL_DEBUG=INFO
    export CUDA_LAUNCH_BLOCKING=0
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    
    # Try torchrun first (recommended for 2 GPUs)
    echo "Launching with torchrun (recommended)..."
    torchrun \
        --nproc_per_node=2 \
        --master_port=29500 \
        --nnodes=1 \
        --node_rank=0 \
        GPRO_matmul_ec2.py
    
    # If torchrun fails, try torch.distributed.launch
    if [ $? -ne 0 ]; then
        echo "torchrun failed, trying torch.distributed.launch..."
        python3 -m torch.distributed.launch \
            --nproc_per_node=2 \
            --master_port=29500 \
            --use_env \
            GPRO_matmul_ec2.py
    fi
    
    # Final fallback to single GPU
    if [ $? -ne 0 ]; then
        echo "❌ Distributed training failed, falling back to single GPU..."
        echo "   - This will be ~2x slower but will work"
        CUDA_VISIBLE_DEVICES=0 python3 GPRO_matmul_ec2.py
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
        python3 -m torch.distributed.launch \
            --nproc_per_node=$GPU_COUNT \
            --master_port=29500 \
            --use_env \
            GPRO_matmul_ec2.py
    fi
    
    # If both fail, fall back to single GPU
    if [ $? -ne 0 ]; then
        echo "Distributed training failed, falling back to single GPU..."
        CUDA_VISIBLE_DEVICES=0 python3 GPRO_matmul_ec2.py
    fi
else
    echo "💻 **SINGLE GPU TRAINING MODE**"
    echo "   - Using GPU 0 only"
    echo "   - Will be slower but more stable"
    CUDA_VISIBLE_DEVICES=0 python3 GPRO_matmul_ec2.py
fi

echo ""
echo "🎯 **Training completed!**"
echo "   - Training using 4-bit quantization with 512 token context length"
echo "   - Check tensorboard logs for training progress"
echo "   - Model saved to: ~/matmul_outputs/models/"
echo "   - Discovery logs saved in outputs directory" 