#!/bin/bash

# EC2 GRPO Matrix Multiplication Inference Launcher
# Optimized for EC2 deployment with monitoring and resource management

set -e

echo "🚀 Starting EC2 GRPO Matrix Multiplication Inference"
echo "=================================================="

# Check if running on EC2
if [ -f /sys/hypervisor/uuid ] && [ `head -c 3 /sys/hypervisor/uuid` == ec2 ]; then
    echo "✅ Running on EC2 instance"
    INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type)
    echo "   Instance Type: $INSTANCE_TYPE"
else
    echo "⚠️  Not detected as EC2 instance"
fi

# System information
echo "🖥️  System Information:"
echo "   CPU Cores: $(nproc)"
echo "   Memory: $(free -h | awk '/^Mem:/ {print $2}')"
echo "   Disk Space: $(df -h / | awk 'NR==2 {print $4}')"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    GPU_AVAILABLE=true
else
    echo "⚠️  No GPU detected - using CPU mode"
    GPU_AVAILABLE=false
fi

# Create necessary directories
echo "📁 Setting up directories..."
mkdir -p models
mkdir -p logs

# Check for final adapter and download if needed
echo "🔍 Checking for final adapter..."
python3 download_final_adapter.py

# Check Python environment
echo "🐍 Python Environment:"
python3 --version

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv --version

# Check if virtual environment exists
# if [ ! -d "venv" ]; then
#     echo "📦 Creating virtual environment..."
#     python3 -m venv venv
# fi

# # Activate virtual environment
# echo "🔄 Activating virtual environment..."
# source venv/bin/activate

# Install/upgrade requirements
echo "📦 Installing requirements..."
sudo $HOME/.local/bin/uv pip install --system -r requirements_ec2.txt

# Set environment variables for optimal performance
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

if [ "$GPU_AVAILABLE" = true ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "🎯 GPU mode enabled"
else
    export CUDA_VISIBLE_DEVICES=""
    echo "🎯 CPU mode enabled"
fi

# Memory optimization for EC2
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)

# Function to handle cleanup on exit
cleanup() {
    echo "🧹 Cleaning up..."
    if [ "$GPU_AVAILABLE" = true ]; then
        nvidia-smi --gpu-reset || true
    fi
    echo "✅ Cleanup completed"
}

# Set trap for cleanup
trap cleanup EXIT

# Pre-flight checks
echo "🔍 Pre-flight checks..."

# Check disk space (need at least 10GB)
AVAILABLE_SPACE=$(df / | awk 'NR==2 {print $4}')
if [ $AVAILABLE_SPACE -lt 10485760 ]; then  # 10GB in KB
    echo "❌ Insufficient disk space. Need at least 10GB free."
    exit 1
fi

# Check memory (need at least 8GB)
AVAILABLE_MEM=$(free | awk 'NR==2{printf "%.0f", $7/1024/1024}')
if [ $AVAILABLE_MEM -lt 8 ]; then
    echo "❌ Insufficient memory. Need at least 8GB available."
    exit 1
fi

echo "✅ All pre-flight checks passed"

# Performance monitoring in background
if command -v htop &> /dev/null; then
    echo "📊 Starting performance monitoring (htop available)"
else
    echo "📊 Performance monitoring (install htop for better monitoring)"
fi

# Start the inference application
echo "🎬 Launching inference application..."
echo "=================================================="

# Log system stats before starting
echo "📊 System stats at launch:" >> logs/system_stats.log
date >> logs/system_stats.log
free -h >> logs/system_stats.log
if [ "$GPU_AVAILABLE" = true ]; then
    nvidia-smi >> logs/system_stats.log
fi
echo "---" >> logs/system_stats.log

# Run the main application
python3 inference_ec2.py 2>&1 | tee logs/inference_$(date +%Y%m%d_%H%M%S).log

echo "🏁 Inference session completed" 