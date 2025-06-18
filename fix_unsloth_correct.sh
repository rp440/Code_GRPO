#!/bin/bash
echo "=== CORRECT UNSLOTH COMPATIBILITY FIX ==="
echo "Fixing the version conflicts and unsloth-zoo issues..."
echo ""

# Step 1: Complete cleanup including unsloth-zoo
echo "Step 1: Complete cleanup..."
pip uninstall -y unsloth unsloth-zoo peft transformers trl

echo ""
echo "Step 2: Installing the latest compatible versions..."
# Use latest versions that work together
pip install transformers>=4.51.3,<4.52.0
pip install peft>=0.12.0
pip install trl>=0.10.1

echo ""
echo "Step 3: Installing Unsloth without version tag..."
# Install latest main branch which should have the fixes
pip install "unsloth[conda] @ git+https://github.com/unslothai/unsloth.git"

echo ""
echo "Step 4: Verification test..."
python -c "
try:
    print('Testing Unsloth imports...')
    from unsloth import FastLanguageModel
    print('✅ FastLanguageModel: SUCCESS')
    
    # Test the specific function that was failing
    from unsloth.models.qwen2 import PeftModelForCausalLM_fast_forward
    print('✅ PeftModelForCausalLM_fast_forward: SUCCESS')
    
    import transformers, peft, trl
    print('✅ Transformers:', transformers.__version__)
    print('✅ PEFT:', peft.__version__)
    print('✅ TRL:', trl.__version__)
    
    print('')
    print('🎉 COMPATIBILITY FIXED! Training script should work now.')
    
except ImportError as e:
    print('❌ Import Error:', e)
    print('')
    print('Trying alternative approach...')
    
except Exception as e:
    print('❌ Other Error:', e)
"

echo ""
echo "=== ALTERNATIVE IF ABOVE FAILS ==="
echo "If imports still fail, try this nuclear option:"
echo ""
echo "# Complete environment reset"
echo "pip uninstall -y unsloth unsloth-zoo peft transformers trl"
echo "pip install --no-deps transformers==4.51.3"
echo "pip install --no-deps peft==0.13.0"
echo "pip install --no-deps trl==0.10.1"
echo "pip install --no-deps 'unsloth @ git+https://github.com/unslothai/unsloth.git'"
echo ""
echo "The --no-deps flag prevents dependency conflicts." 