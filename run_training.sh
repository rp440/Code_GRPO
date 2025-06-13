#!/bin/bash

# Activate virtual environment
 source  ../myproject311/bin/activate

# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft datasets trl tensorboard accelerate

# Create necessary directories
mkdir -p /home/ec2-user/matmul_outputs/models
mkdir -p /home/ec2-user/matmul_outputs/tensorboard_logs

# Generate dataset first
echo "Generating training dataset..."
python dataset.py

# Set Hugging Face token (replace with your token)
# export HUGGING_FACE_HUB_TOKEN="your_token_here"

# Run the training script (single GPU first to avoid distributed issues)
# For distributed training, uncomment the lines below:
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    GPRO_matmul_ec2.py

# Single GPU training (more stable)
CUDA_VISIBLE_DEVICES=0 python GPRO_matmul_ec2.py 