#!/bin/bash
echo "=== Fixing Unsloth Version Compatibility Issue ==="
echo "Error: PeftModelForCausalLM_fast_forward not defined"
echo "This indicates a version mismatch between Unsloth and PEFT libraries"
echo ""

# Check current versions
echo "Current versions:"
python -c "import unsloth; print(f'Unsloth: {unsloth.__version__}')" 2>/dev/null || echo "Unsloth: Not installed"
python -c "import peft; print(f'PEFT: {peft.__version__}')" 2>/dev/null || echo "PEFT: Not installed"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null || echo "Transformers: Not installed"

echo ""
echo "=== Installing Compatible Versions ==="

# Uninstall existing versions
pip uninstall -y unsloth peft

# Install compatible versions
echo "Installing Unsloth with compatible PEFT version..."
pip install "unsloth[conda] @ git+https://github.com/unslothai/unsloth.git"

# Alternative: Install specific compatible versions
echo "If above fails, trying alternative installation..."
pip install transformers==4.45.0 peft==0.12.0
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git@main"

echo ""
echo "=== Verification ==="
python -c "
try:
    from unsloth import FastLanguageModel
    from unsloth.models.qwen2 import PeftModelForCausalLM_fast_forward
    print('[SUCCESS] Unsloth imports work correctly!')
except ImportError as e:
    print(f'[ERROR] Still having issues: {e}')
    print('[FALLBACK] Script will use standard training instead')
"

echo ""
echo "=== Alternative: Quick Fix for Current Session ==="
echo "If the version issue persists, the script will automatically fallback to standard training."
echo "This will still work but without Unsloth's speed optimizations."
echo ""
echo "To manually run with standard training, set UNSLOTH_AVAILABLE=False in the script." 