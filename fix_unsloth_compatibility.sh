#!/bin/bash
echo "=== FIXING UNSLOTH COMPATIBILITY ISSUE ==="
echo "Problem: PeftModelForCausalLM_fast_forward not defined"
echo "Solution: Install compatible versions"
echo ""

# Check current environment
echo "Current Python environment:"
which python
python --version

echo ""
echo "Step 1: Uninstalling incompatible versions..."
pip uninstall -y unsloth peft transformers trl

echo ""
echo "Step 2: Installing compatible base libraries..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "Step 3: Installing compatible transformers and peft..."
pip install transformers==4.44.0
pip install peft==0.12.0

echo ""
echo "Step 4: Installing compatible TRL..."
pip install trl==0.10.1

echo ""
echo "Step 5: Installing compatible Unsloth..."
pip install "unsloth[conda] @ git+https://github.com/unslothai/unsloth.git@24.8"

echo ""
echo "Step 6: Verification test..."
python -c "
import sys
print('Python version:', sys.version)
try:
    import unsloth
    print('✅ Unsloth version:', unsloth.__version__)
    
    from unsloth import FastLanguageModel
    print('✅ FastLanguageModel import: SUCCESS')
    
    from unsloth.models.qwen2 import PeftModelForCausalLM_fast_forward
    print('✅ PeftModelForCausalLM_fast_forward: SUCCESS')
    
    import transformers, peft, trl
    print('✅ Transformers:', transformers.__version__)
    print('✅ PEFT:', peft.__version__)
    print('✅ TRL:', trl.__version__)
    
    print('')
    print('🎉 ALL COMPATIBILITY ISSUES FIXED!')
    print('You can now run the training script successfully.')
    
except Exception as e:
    print('❌ Error:', e)
    print('Manual fix needed - check versions manually')
"

echo ""
echo "=== ALTERNATIVE QUICK FIX ==="
echo "If the above doesn't work, try this simpler approach:"
echo ""
echo "pip uninstall -y unsloth"
echo "pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'"
echo ""
echo "This installs the latest stable version with all dependencies." 