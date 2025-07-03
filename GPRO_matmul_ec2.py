"""
GRPO Matrix Multiplication DSL Training Script for Distributed/Multi-GPU Training

Key Features:
1. Load previous LoRA checkpoint but discard optimizer state
2. Create fresh AdamW optimizer with configurable learning rate  
3. Reset value head parameters for fresh training
4. Simple reward function: +0.1 for DSL tags, -1 per multiplication, -100 for wrong answers
5. Standard PyTorch/transformers training with distributed support
6. Using Qwen2-1.5B model

Configuration:
- Set LOAD_FROM_CHECKPOINT = False to train from scratch
- Adjust CHECKPOINT_PATH to your previous model path
- Modify NEW_LEARNING_RATE as needed
- Supports both single GPU and multi-GPU distributed training
- Automatic detection of distributed environment

Installation Requirements:
- Standard dependencies: pip install -U trl peft transformers datasets huggingface_hub accelerate torch bitsandbytes
"""

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel, LoraConfig, get_peft_model
import os
import shutil
import ast
from datetime import datetime
from datasets import load_dataset
from huggingface_hub import login
from trl import GRPOConfig, GRPOTrainer
import random
import json
import math
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import warnings
import logging

# Suppress expected gradient checkpointing warnings - these are normal with Qwen2
warnings.filterwarnings("ignore", message=".*Caching is incompatible with gradient checkpointing.*")
warnings.filterwarnings("ignore", message=".*is incompatible with gradient checkpointing.*") 
warnings.filterwarnings("ignore", message=".*adient checkpointing.*")  # Catches partial messages
logging.getLogger("transformers.models.qwen2.modeling_qwen2").setLevel(logging.ERROR)
print("[INFO] Suppressed expected Qwen2 gradient checkpointing warnings for cleaner output")

# Standard PyTorch/transformers training (no Unsloth)
UNSLOTH_AVAILABLE = False

def is_bfloat16_supported():
    """Check if bfloat16 is supported on the current GPU"""
    return torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False

# --- Initialize Distributed Training Environment ---
# CRITICAL: Set memory optimization environment variables BEFORE any CUDA operations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
print("[MEMORY] Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

# CRITICAL: Initialize distributed environment before model loading
if 'RANK' in os.environ:
    print(f"[DISTRIBUTED] Initializing distributed training...")
    print(f"[DISTRIBUTED] RANK: {os.environ.get('RANK')}, LOCAL_RANK: {os.environ.get('LOCAL_RANK')}, WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    print(f"[DISTRIBUTED] Set CUDA device to: {torch.cuda.current_device()}")
    
    # GRPO memory optimization for distributed training
    torch.cuda.empty_cache()
    print(f"[MEMORY] Cleared CUDA cache for rank {os.environ.get('RANK')}")
else:
    print("[SINGLE GPU] Running in single GPU mode")
    torch.cuda.empty_cache()
    print("[MEMORY] Cleared CUDA cache")

# --- 1. Paths Configuration ---
# Base directory for all outputs - use current user's home directory
BASE_OUTPUT_DIR = os.path.expanduser("~/matmul_outputs")
MODEL_SAVE_DIR = os.path.join(BASE_OUTPUT_DIR, "models")
TENSORBOARD_LOGS_DIR = os.path.join(BASE_OUTPUT_DIR, "tensorboard_logs")
DATASET_PATH = "./matrix_io_data_for_grpo.jsonl"

# Create necessary directories
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOGS_DIR, exist_ok=True)

# --- 2. Core DSL Logic and Matrix Operations ---
class DSLExecutor:
    def __init__(self, matrix_a, matrix_b):
        self.variables = {}
        if not (isinstance(matrix_a, list) and len(matrix_a) == 2 and
                isinstance(matrix_a[0], list) and len(matrix_a[0]) == 2 and
                all(isinstance(el, (int, float)) for row in matrix_a for el in row) and
                isinstance(matrix_b, list) and len(matrix_b) == 2 and
                isinstance(matrix_b[0], list) and len(matrix_b[0]) == 2 and
                all(isinstance(el, (int, float)) for row in matrix_b for el in row)):
            raise ValueError("Test matrices A and B for DSLExecutor must be 2x2 lists of lists of numbers.")
        for r_idx in range(2):
            for c_idx in range(2):
                self.variables[f'A[{r_idx},{c_idx}]'] = matrix_a[r_idx][c_idx]
                self.variables[f'B[{r_idx},{c_idx}]'] = matrix_b[r_idx][c_idx]

    def _get_value(self, var_name):
        try:
            return ast.literal_eval(var_name)
        except (ValueError, SyntaxError, TypeError):
            if var_name not in self.variables:
                raise ValueError(f"Variable '{var_name}' not found. Available: {list(self.variables.keys())}")
            return self.variables[var_name]

    def execute_step(self, step_line):
        original_step_line = step_line
        step_line = step_line.strip()
        if not step_line: return
        if '=' not in step_line:
            raise ValueError(f"Malformed DSL step (missing '='): '{original_step_line}'")

        target_var, expression = [s.strip() for s in step_line.split('=', 1)]
        
        # Handle chained operations (e.g., M1 + M4 - M5 + M7)
        result = self._evaluate_expression(expression, original_step_line)
        self.variables[target_var] = result
    
    def _evaluate_expression(self, expression, original_step_line):
        """Evaluate an expression that may contain chained +/- operations or complex parenthetical expressions"""
        expression = expression.strip()
        
        # Simple assignment (no operators) - check for variable names or numbers only
        assign_match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*(?:\[[0-9,\s]*\])?|\-?\d+(?:\.\d+)?)\s*$", expression)
        if assign_match:
            return self._get_value(assign_match.group(1).strip())
        
        # Handle parenthetical expressions first - simple cases like (A[0,0] - A[1,1]) * (B[1,0] + B[1,1])
        paren_mult_match = re.match(r"^\s*\(([^)]+)\)\s*\*\s*\(([^)]+)\)\s*$", expression)
        if paren_mult_match:
            left_expr = paren_mult_match.group(1).strip()
            right_expr = paren_mult_match.group(2).strip()
            left_val = self._evaluate_simple_expression(left_expr)
            right_val = self._evaluate_simple_expression(right_expr)
            return left_val * right_val
        
        # Handle expression * variable or variable * expression cases
        mult_var_match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*(?:\[[0-9,\s]*\])?)\s*\*\s*\(([^)]+)\)\s*$", expression)
        if mult_var_match:
            var_name = mult_var_match.group(1).strip()
            expr = mult_var_match.group(2).strip()
            var_val = self._get_value(var_name)
            expr_val = self._evaluate_simple_expression(expr)
            return var_val * expr_val
            
        var_mult_match = re.match(r"^\s*\(([^)]+)\)\s*\*\s*([A-Za-z_][A-Za-z0-9_]*(?:\[[0-9,\s]*\])?)\s*$", expression)
        if var_mult_match:
            expr = var_mult_match.group(1).strip()
            var_name = var_mult_match.group(2).strip()
            expr_val = self._evaluate_simple_expression(expr)
            var_val = self._get_value(var_name)
            return expr_val * var_val
        
        # Single binary operation (backward compatibility)
        binary_match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*(?:\[[0-9,\s]*\])?|\-?\d+(?:\.\d+)?)\s*([*+-])\s*([A-Za-z_][A-Za-z0-9_]*(?:\[[0-9,\s]*\])?|\-?\d+(?:\.\d+)?)\s*$", expression)
        if binary_match:
            op1_name = binary_match.group(1).strip()
            operator = binary_match.group(2).strip()
            op2_name = binary_match.group(3).strip()
            val1 = self._get_value(op1_name)
            val2 = self._get_value(op2_name)
            if operator == '+': return val1 + val2
            elif operator == '*': return val1 * val2
            elif operator == '-': return val1 - val2
            else: raise ValueError(f"Unsupported operator '{operator}' in expression: '{expression}'")
        
        # Chained operations (e.g., M2 + M3 - M5 + M6)
        return self._evaluate_chained_expression(expression, original_step_line)
    
    def _evaluate_simple_expression(self, expression):
        """Evaluate simple expressions within parentheses (addition/subtraction only)"""
        expression = expression.strip()
        
        # Single variable or number
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*(?:\[[0-9,\s]*\])?$|^\-?\d+(?:\.\d+)?$", expression):
            return self._get_value(expression)
        
        # Split by + and - operators
        tokens = re.split(r'(\s*[+-]\s*)', expression)
        if len(tokens) == 1:
            return self._get_value(expression.strip())
        
        # First term
        result = self._get_value(tokens[0].strip())
        
        # Process remaining terms
        i = 1
        while i < len(tokens):
            if i + 1 >= len(tokens):
                break
            operator = tokens[i].strip()
            operand = tokens[i + 1].strip()
            
            if not operator or not operand:
                i += 2
                continue
                
            value = self._get_value(operand)
            
            if operator == '+': 
                result += value
            elif operator == '-': 
                result -= value
            else: 
                raise ValueError(f"Unsupported operator '{operator}' in simple expression: '{expression}'")
            
            i += 2
        
        return result
    
    def _evaluate_chained_expression(self, expression, original_step_line):
        """Evaluate chained +/- operations like M2 + M3 - M5 + M6"""
        # Split by + and - while preserving operators
        tokens = re.split(r'(\s*[+-]\s*)', expression)
        if len(tokens) == 1:
            raise ValueError(f"Malformed expression: '{expression}' in DSL line: '{original_step_line}'")
        
        # First term (no leading operator)
        first_term = tokens[0].strip()
        result = self._get_value(first_term)
        
        # Process remaining terms with their operators
        i = 1
        while i < len(tokens):
            if i + 1 >= len(tokens):
                break
            operator = tokens[i].strip()
            operand = tokens[i + 1].strip()
            
            # Skip empty tokens from splitting
            if not operator or not operand:
                i += 2
                continue
                
            value = self._get_value(operand)
            
            if operator == '+': 
                result += value
            elif operator == '-': 
                result -= value
            else: 
                raise ValueError(f"Unsupported operator '{operator}' in chained expression: '{expression}'")
            
            i += 2
        
        return result

    def run_dsl_and_get_c(self, dsl_script_string):
        steps = dsl_script_string.strip().split('\n')
        for step in steps:
            clean_step = step.strip()
            if clean_step: self.execute_step(step)
        c_matrix = [[None, None], [None, None]]
        required_c_elements = ['C[0,0]', 'C[0,1]', 'C[1,0]', 'C[1,1]']
        for elem in required_c_elements:
            if elem not in self.variables:
                raise ValueError(f"DSL script did not compute {elem}. Vars: {self.variables}")
            r, c = map(int, elem[2:-1].split(','))
            c_matrix[r][c] = self.variables[elem]
        if any(el is None for row in c_matrix for el in row):
             raise ValueError(f"DSL did not compute all C elements. C: {c_matrix}, Vars: {self.variables}")
        return c_matrix

def manual_matrix_multiply_2x2(A, B):
    C = [[0,0], [0,0]]
    C[0][0] = A[0][0] * B[0][0] + A[0][1] * B[1][0]
    C[0][1] = A[0][0] * B[0][1] + A[0][1] * B[1][1]
    C[1][0] = A[1][0] * B[0][0] + A[1][1] * B[1][0]
    C[1][1] = A[1][0] * B[0][1] + A[1][1] * B[1][1]
    return C

def count_multiplications_in_dsl(dsl_script):
    """
    Count the number of multiplication operations in a DSL script.
    Handles various formats including parenthetical expressions and chained operations.
    """
    count = 0
    lines = dsl_script.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or not '=' in line:
            continue
            
        # Split at = to get the expression part
        parts = line.split('=', 1)
        if len(parts) != 2:
            continue
            
        expression = parts[1].strip()
        
        # Count multiplication operators in the expression
        # But be careful not to count multiplications inside variable names or matrix indices
        
        # Patterns that indicate multiplication:
        # 1. (expr) * (expr) - parenthetical multiplication
        # 2. var * (expr) - variable times parenthetical expression
        # 3. (expr) * var - parenthetical expression times variable
        # 4. var * var - simple variable multiplication
        # 5. A[i,j] * B[k,l] - matrix element multiplication
        
        mult_patterns = [
            r'\([^)]+\)\s*\*\s*\([^)]+\)',  # (expr) * (expr)
            r'[A-Za-z_][A-Za-z0-9_]*(?:\[[0-9,\s]*\])?\s*\*\s*\([^)]+\)',  # var * (expr)
            r'\([^)]+\)\s*\*\s*[A-Za-z_][A-Za-z0-9_]*(?:\[[0-9,\s]*\])?',  # (expr) * var
            r'[A-Za-z_][A-Za-z0-9_]*(?:\[[0-9,\s]*\])?\s*\*\s*[A-Za-z_][A-Za-z0-9_]*(?:\[[0-9,\s]*\])?',  # var * var
        ]
        
        # Find all multiplication operations in this line
        temp_expression = expression
        line_mult_count = 0
        
        for pattern in mult_patterns:
            matches = re.findall(pattern, temp_expression)
            line_mult_count += len(matches)
            
            # Remove found matches to avoid double counting
            temp_expression = re.sub(pattern, 'PLACEHOLDER', temp_expression)
        
        # Additional check: count remaining * operators that might not match patterns above
        # But exclude those inside matrix indices [i,j] or function calls
        remaining_expression = temp_expression
        # Remove matrix indices and function-like constructs
        remaining_expression = re.sub(r'\[[^\]]+\]', '', remaining_expression)
        remaining_expression = re.sub(r'PLACEHOLDER', '', remaining_expression)
        
        # Count remaining * that aren't already accounted for
        additional_mults = remaining_expression.count('*')
        
        # Sanity check: a well-formed multiplication line should have exactly 1 multiplication
        # If we find more than 1, log it but proceed
        total_line_mults = max(line_mult_count, additional_mults)
        if total_line_mults > 1:
            print(f"[MULT COUNT WARNING] Line '{line}' appears to have {total_line_mults} multiplications")
        
        # Count ALL multiplications in this line, not just limit to 1
        count += total_line_mults
            
    return count

def _generate_random_2x2_matrix_for_inference(low=-99, high=99):
    return [[random.randint(low, high) for _ in range(2)] for _ in range(2)]

A_INFERENCE_MATRIX = _generate_random_2x2_matrix_for_inference()
B_INFERENCE_MATRIX = _generate_random_2x2_matrix_for_inference()
C_EXPECTED_INFERENCE_RESULT = manual_matrix_multiply_2x2(A_INFERENCE_MATRIX, B_INFERENCE_MATRIX)

# --- 3. GRPO Configuration and System Prompt ---
BASE_MODEL_NAME_FOR_FINETUNING = "Qwen/Qwen2-1.5B"


TRAINED_MODEL_DIR_NAME = f"{BASE_MODEL_NAME_FOR_FINETUNING.split('/')[-1]}-GRPO-MatMulDSL-JSONL"
LOCAL_TRAINED_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, TRAINED_MODEL_DIR_NAME)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Configuration for checkpoint loading and fresh training
CHECKPOINT_PATH = os.path.expanduser("~/final_adapter_dir")  # Path to existing DSL-STF LoRA adapter
NEW_LEARNING_RATE = 2e-5 # Learning rate for fresh optimizer
LOAD_FROM_CHECKPOINT = True  # Set to False to train from scratch

# Training configuration - Auto-detect distributed vs single GPU
EPOCHS = 1

# Auto-configure based on distributed environment
if 'WORLD_SIZE' in os.environ:
    # Multi-GPU distributed training
    NUM_GPUS = int(os.environ['WORLD_SIZE'])
    BATCH_SIZE_PER_GPU = 8   # Higher than Colab for better GPU utilization
    GRAD_ACC_STEPS = 16      # Keep same effective batch size
    print(f"[CONFIG] Multi-GPU mode detected: {NUM_GPUS} GPUs")
    print(f"[CONFIG] Distributed training configuration")
else:
    # Single GPU training
    NUM_GPUS = 1
    BATCH_SIZE_PER_GPU = 4   # Higher than Colab for better GPU utilization
    GRAD_ACC_STEPS = 16      # Keep same effective batch size
    print(f"[CONFIG] Single GPU mode")
    print(f"[CONFIG] Standard training configuration")

# Calculate total batch size
TOTAL_BATCH_SIZE = BATCH_SIZE_PER_GPU * NUM_GPUS * GRAD_ACC_STEPS
print(f"[CONFIG] Batch size: {BATCH_SIZE_PER_GPU}, Gradient accumulation: {GRAD_ACC_STEPS}")

# Memory optimization settings
USE_GRADIENT_CHECKPOINTING = False  # Disabled to avoid gradient issues with PEFT
MAX_GRAD_NORM = 0.5
WARMUP_RATIO = 0.05

model_config_desc = f"lr{NEW_LEARNING_RATE}_epochs{EPOCHS}_batch{TOTAL_BATCH_SIZE}_gradacc{GRAD_ACC_STEPS}_gpu{NUM_GPUS}_t4"
model_name = f"{TRAINED_MODEL_DIR_NAME}_{model_config_desc}_{timestamp}"
tensorboard_name = f"runs_{model_config_desc}_{timestamp}"

FINAL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, model_name)
FINAL_TENSORBOARD_PATH = os.path.join(TENSORBOARD_LOGS_DIR, tensorboard_name)

print(f"[SAVE CONFIG] Model will be saved to: {FINAL_MODEL_PATH}")
print(f"[SAVE CONFIG] TensorBoard logs will be saved to: {FINAL_TENSORBOARD_PATH}")
print(f"[GPU CONFIG] Using {NUM_GPUS} GPU(s) with {BATCH_SIZE_PER_GPU} batch size per GPU")
print(f"[TRAINING CONFIG] Max completion length: 512 tokens, Max prompt length: 256 tokens")
print(f"[TRAINING CONFIG] Effective batch size: {TOTAL_BATCH_SIZE} (batch_size={BATCH_SIZE_PER_GPU} × grad_acc={GRAD_ACC_STEPS})")

SYSTEM_MESSAGE = """You are an AI assistant specialized in generating Domain Specific Language (DSL) scripts for 2x2 matrix multiplication. You can provide explanations, but must wrap your DSL code in <DSL></DSL> tags.
  EXAMPLE DSL OUTPUT FORMAT: For matrices A=[[1,2],[3,4]] and B=[[5,6],[7,8]], a valid response would be:  
<DSL>
M1 = (A[0,0] - A[1,1]) * (B[1,0] + B[1,1])
M2 = (A[1,0] + A[0,1]) * (B[0,1])
M3 = A[1,1] * (B[0,0] - B[1,1])
M4 = A[0,0] * (B[1,1] + B[0,0])
M5 = (A[0,1] + A[1,1]) * (B[0,0])
M6 = (A[1,0] - A[0,0]) * (B[1,0] - B[0,1])
M7 = (A[0,0] + A[1,0]) * (B[0,1] - B[1,1])

C[0,0] = M2 + M3 - M5 + M6
C[0,1] = M1 + M6 - M4
C[1,0] = M7 - M2 + M1
C[1,1] = M3 - M4 + M5 - M7
</DSL>

This uses 7 multiplications, but can be optimized using techniques like Strassen's algorithm.  YOUR TASK: Generate a DSL script that performs 2x2 matrix multiplication using 7 or fewer multiplications. You may provide explanations outside the DSL tags, but the actual code must be within <DSL></DSL> tags. 
DSL SYNTAX RULES: - M variables: Store multiplication results (e.g., M1 = A[0,0] * B[0,0]) - S variables:
 Store addition/subtraction results (e.g., S1 = M1 + M2) - Matrix elements: A[row,col] and B[row,col] where row,col ∈ {0,1} - Final output: C[row,col] = result - Operations: + (addition), * (multiplication), - (subtraction) - Variable assignment: VAR = expression 
  REQUIREMENTS: - Use ≤7 multiplications total within the <DSL></DSL> tags - Compute all four elements: C[0,0], C[0,1], C[1,0], C[1,1] - Wrap DSL code in <DSL></DSL> tags - You may add explanations outside the tags  If you cannot determine a valid sequence, output: Error: Cannot determine full sequence."""

DEFAULT_USER_PROMPT_FOR_DSL_GENERATION = "Generate the DSL script to calculate C = A * B for the given 2x2 matrices, using 7 or fewer multiplications.First think about answer then respond in <DSL></DSL> tags."

# --- 4. Dataset Preparation from JSONL ---
def preprocess_jsonl_data(item):
    """
    Prepares a single item from the JSONL dataset.
    """
    matrix_a = None
    matrix_b = None

    # Try to get matrices as lists first
    if "matrix_A_list" in item and "matrix_B_list" in item:
        try:
            matrix_a = item["matrix_A_list"]
            matrix_b = item["matrix_B_list"]
            if not (isinstance(matrix_a, list) and isinstance(matrix_b, list)):
                matrix_a, matrix_b = None, None
        except:
            matrix_a, matrix_b = None, None
            
    # Fallback: Try to get matrices as strings and parse them
    if matrix_a is None and "A_matrix_str" in item:
        try:
            matrix_a = ast.literal_eval(item["A_matrix_str"])
        except:
            print(f"Warning: Could not parse A_matrix_str: {item.get('A_matrix_str')}")
            matrix_a = [[0,0],[0,0]]
    if matrix_b is None and "B_matrix_str" in item:
        try:
            matrix_b = ast.literal_eval(item["B_matrix_str"])
        except:
            print(f"Warning: Could not parse B_matrix_str: {item.get('B_matrix_str')}")
            matrix_b = [[0,0],[0,0]]

    # If matrices couldn't be loaded, use defaults
    if matrix_a is None or matrix_b is None:
        print(f"Error: Could not determine matrix A or B for dataset item: {item}. Using placeholder matrices.")
        matrix_a = [[1,1],[1,1]]
        matrix_b = [[1,1],[1,1]]

    # Calculate expected C
    try:
        expected_c = manual_matrix_multiply_2x2(matrix_a, matrix_b)
    except Exception as e:
        print(f"Error calculating expected_C for A={matrix_a}, B={matrix_b}: {e}. Using placeholder C.")
        expected_c = [[0,0],[0,0]]

    user_content = item.get("user_query", DEFAULT_USER_PROMPT_FOR_DSL_GENERATION)

    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_content},
        ],
        "A_matrix_str": str(matrix_a),
        "B_matrix_str": str(matrix_b),
        "expected_C_str": str(expected_c),
    }

print(f"Loading dataset from: {DATASET_PATH}")
print("*** IMPORTANT: Make sure you have run dataset.py to generate the dataset file first! ***")

try:
    # Check if dataset file exists (must be generated by dataset.py)
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset file not found at {DATASET_PATH}")
        print("[REQUIRED] Please run dataset.py first to generate the training dataset!")
        print("[INFO] The dataset.py script should create the JSONL file with matrix multiplication examples.")
        exit(1)
    
    print(f"[SUCCESS] Found dataset file at: {DATASET_PATH}")
    
    # Check file size to ensure it's not empty
    file_size = os.path.getsize(DATASET_PATH)
    print(f"[INFO] Dataset file size: {file_size} bytes")
    
    if file_size == 0:
        print("[ERROR] Dataset file is empty!")
        print("[REQUIRED] Please run dataset.py to generate proper training data.")
        exit(1)
    
    # Load the dataset generated by dataset.py
    raw_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    train_dataset_for_grpo = raw_dataset.map(preprocess_jsonl_data)

    print(f"[SUCCESS] Processed dataset. Number of samples: {len(train_dataset_for_grpo)}")
    
    if len(train_dataset_for_grpo) == 0:
        print("[ERROR] No valid samples found after processing!")
        print("[CHECK] Check the format of your dataset.py generated file.")
        exit(1)
    
    print(f"[SAMPLE] First processed sample keys: {list(train_dataset_for_grpo[0].keys())}")
    print(f"[SAMPLE] Sample A matrix: {train_dataset_for_grpo[0]['A_matrix_str']}")
    print(f"[SAMPLE] Sample B matrix: {train_dataset_for_grpo[0]['B_matrix_str']}")
    
except Exception as e:
    print(f"[ERROR] Failed to load or process dataset from {DATASET_PATH}: {e}")
    print("[TIP] Make sure dataset.py has been run and generated a valid JSONL file.")
    print("[FORMAT] Expected format: Each line should be a JSON object with matrix data.")
    exit(1)

# --- 5. Model Loading and PEFT Setup ---
print(f"Loading base model with standard approach: {BASE_MODEL_NAME_FOR_FINETUNING}")

# Determine optimal dtype
if is_bfloat16_supported():
    dtype = torch.bfloat16
    print("[STANDARD] Using bfloat16 (optimal for modern GPUs)")
else:
    dtype = torch.float16
    print("[STANDARD] Using float16 (fallback for older GPUs)")

# Configure device map for distributed vs single GPU
if 'RANK' in os.environ:
    device_map = None
    print("[DISTRIBUTED] Loading model without device_map for distributed training")
else:
    device_map = "auto"
    print("[SINGLE GPU] Loading model with device_map='auto'")

# Load model using standard transformers approach with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME_FOR_FINETUNING,
    torch_dtype=dtype,
    device_map=device_map,
    trust_remote_code=True,
    quantization_config=bnb_config,
)

tokenizer_for_training = AutoTokenizer.from_pretrained(
    BASE_MODEL_NAME_FOR_FINETUNING,
    trust_remote_code=True,
)

# Configure model for gradient checkpointing and GRPO compatibility
base_model.config.use_cache = False
print("[FIX] Set use_cache=False to avoid gradient checkpointing conflicts")

# Additional explicit cache disabling for Qwen2 layers
if hasattr(base_model.config, 'use_cache'):
    base_model.config.use_cache = False
if hasattr(base_model, 'generation_config') and hasattr(base_model.generation_config, 'use_cache'):
    base_model.generation_config.use_cache = False

# Explicitly disable gradient checkpointing on base model if it exists
if hasattr(base_model, 'gradient_checkpointing') and base_model.gradient_checkpointing:
    base_model.gradient_checkpointing = False
    print("[FIX] Disabled gradient checkpointing on base model")

print("[FIX] Explicitly disabled caching on all model components")
print("[INFO] Note: Any gradient checkpointing warnings you see are expected and harmless")

# Configure tokenizer
if tokenizer_for_training.pad_token is None:
    tokenizer_for_training.pad_token = tokenizer_for_training.eos_token
if tokenizer_for_training.padding_side == 'right':
    tokenizer_for_training.padding_side = 'left'

# Load previous LoRA checkpoint or create fresh LoRA
if LOAD_FROM_CHECKPOINT:
    print(f"Loading previous LoRA checkpoint from: {CHECKPOINT_PATH}")
    try:
        model_peft = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
        print("[SUCCESS] Successfully loaded previous LoRA checkpoint")
        
        # CRITICAL: Enable gradients for LoRA parameters after loading checkpoint
        for name, param in model_peft.named_parameters():
            if 'lora_' in name.lower():
                param.requires_grad = True
        print("[FIX] Enabled gradients for LoRA parameters")
        
    except Exception as e:
        print(f"[WARNING] Could not load checkpoint ({e}), creating fresh LoRA...")
        LOAD_FROM_CHECKPOINT = False  # Force fresh creation
        
if not LOAD_FROM_CHECKPOINT:
    print("Creating fresh LoRA configuration...")
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    model_peft = get_peft_model(base_model, lora_config)
    
    # Immediately enable training mode and check gradients
    model_peft.train()
    print("[PEFT SETUP] Enabling gradients for all LoRA parameters...")
    for name, param in model_peft.named_parameters():
        if 'lora_' in name.lower():
            param.requires_grad = True
            print(f"[PEFT SETUP] Enabled: {name}")
    
    # Verify PEFT setup
    trainable_count = sum(1 for p in model_peft.parameters() if p.requires_grad)
    print(f"[PEFT SETUP] {trainable_count} trainable parameters after PEFT setup")

# Configure gradient checkpointing and training mode
print("[STANDARD] Using standard training mode")
model_peft.train()

# =========================
#  New PEFT Stacking Logic
# =========================
# The goal is to stack a fresh GRPO LoRA adapter on top of a previously trained DSL-STF
# adapter. The DSL-STF adapter should remain FROZEN (non-trainable) while the new GRPO
# adapter receives gradients.

# Path to the already-trained DSL-STF LoRA adapter.  Update this path to point at your
# actual adapter directory.
STF_ADAPTER_PATH = os.path.expanduser("~/final_adapter_dir")  # <-- CHANGE ME

try:
    print(f"[STACKED PEFT] Loading frozen DSL-STF adapter from: {STF_ADAPTER_PATH}")
    # If we already have a PeftModel, simply load another adapter into it; otherwise, create one.
    if isinstance(model_peft, PeftModel):
        model_peft.load_adapter(STF_ADAPTER_PATH, adapter_name="stf", is_trainable=False)
    else:
        model_peft = PeftModel.from_pretrained(model_peft, STF_ADAPTER_PATH, is_trainable=False)
    print("[STACKED PEFT] Successfully loaded DSL-STF adapter (frozen).")
except Exception as e:
    print(f"[STACKED PEFT] WARNING: Could not load DSL-STF adapter: {e}.  Proceeding without it.")

# Add the fresh GRPO adapter that we will fine-tune.
print("[STACKED PEFT] Adding new GRPO adapter on top of (base + STF)")
grpo_lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
)

# If `model_peft` is already a PeftModel (expected) we can simply add another adapter.
if isinstance(model_peft, PeftModel):
    model_peft.add_adapter("grpo", grpo_lora_config)
    model_peft.train_adapter("grpo")  # makes ONLY the GRPO adapter trainable
    model_peft.set_adapter("grpo")
else:
    # Fallback – should not happen, but handle just in case.
    model_peft = get_peft_model(model_peft, grpo_lora_config)

# Explicitly freeze any parameter that does NOT belong to the GRPO adapter.
for name, param in model_peft.named_parameters():
    if "lora_" in name.lower():
        param.requires_grad = ".grpo" in name.lower()
    else:
        param.requires_grad = False

print("[STACKED PEFT] Trainable parameters after stacking:")
model_peft.print_trainable_parameters()
# =========================
#  End PEFT Stacking Logic
# =========================

# Explicitly disable gradient checkpointing on PEFT model
if hasattr(model_peft, 'gradient_checkpointing_enable'):
    # Don't call enable, and if it's already enabled, try to disable
    if hasattr(model_peft, 'gradient_checkpointing') and model_peft.gradient_checkpointing:
        model_peft.gradient_checkpointing = False
        print("[FIX] Disabled gradient checkpointing on PEFT model")
    
# Force use_cache=False on all model components (critical for GRPO)
if hasattr(model_peft, 'config'):
    model_peft.config.use_cache = False
if hasattr(model_peft, 'base_model') and hasattr(model_peft.base_model, 'config'):
    model_peft.base_model.config.use_cache = False
print("[CRITICAL FIX] Ensured use_cache=False for GRPO compatibility")

# CRITICAL: Ensure all trainable parameters have gradients enabled
trainable_params_count = 0
for name, param in model_peft.named_parameters():
    if param.requires_grad:
        trainable_params_count += 1
    else:
        # Force enable gradients for PEFT parameters
        if 'lora_' in name.lower() and '.grpo' in name.lower():
            param.requires_grad = True
            trainable_params_count += 1
            print(f"[FIX] Enabled gradients for: {name}")

print(f"[GRADIENT CHECK] {trainable_params_count} parameters have gradients enabled")

# Verify gradients are properly enabled
if trainable_params_count == 0:
    print("[ERROR] No parameters have gradients enabled!")
    # Force enable gradients only for the GRPO adapter parameters
    for name, param in model_peft.named_parameters():
        if 'lora_' in name.lower() and '.grpo' in name.lower():
            param.requires_grad = True
            print(f"[FORCE FIX] Enabled gradients for (GRPO): {name}")

model_peft.print_trainable_parameters()

# Fresh optimizer with new learning rate and gradient clipping
from torch.optim import AdamW

# Get only trainable parameters for the optimizer
trainable_params_for_optimizer = [param for param in model_peft.parameters() if param.requires_grad]
print(f"[OPTIMIZER] Creating optimizer with {len(trainable_params_for_optimizer)} trainable parameter groups")

if len(trainable_params_for_optimizer) == 0:
    print("[ERROR] No trainable parameters found for optimizer!")
    print("[DEBUG] All model parameters:")
    for name, param in model_peft.named_parameters():
        print(f"  {name}: requires_grad={param.requires_grad}")
    raise ValueError("No trainable parameters found")

optimizer = AdamW(
    trainable_params_for_optimizer,
    lr=NEW_LEARNING_RATE,
    weight_decay=0.01,  # Added weight decay for better regularization
    eps=1e-8  # Increased epsilon for numerical stability with FP16
)
print(f"[SUCCESS] Created fresh AdamW optimizer with lr={NEW_LEARNING_RATE}")

# Verify optimizer has parameters with gradients
optimizer_param_count = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
print(f"[OPTIMIZER CHECK] Optimizer managing {optimizer_param_count:,} parameters")

# Additional safety check for gradient computation
print("[GRADIENT TEST] Testing gradient computation...")
test_params_with_grad = sum(1 for p in model_peft.parameters() if p.requires_grad)
print(f"[GRADIENT TEST] Found {test_params_with_grad} parameters with requires_grad=True")

# Reset value head parameters if it exists
try:
    if hasattr(model_peft, 'v_head') and model_peft.v_head is not None:
        model_peft.v_head.reset_parameters()
        print("[SUCCESS] Reset value head parameters")
    else:
        print("[INFO] No value head found to reset")
except Exception as e:
    print(f"[WARNING] Could not reset value head: {e}")

# Configure model pad token
model_peft.config.pad_token_id = tokenizer_for_training.eos_token_id

# --- 6. Reward Function for GRPO ---
# CRITICAL: GRPO requires multiple generations per prompt for preference learning
_num_generations_per_prompt_for_reward = 4  # MUST be >= 2 for GRPO to work effectively
_reward_call_count = 0
_best_n_mults = float('inf')  # Track the best number of multiplications found so far

# Simple reward hyperparameters
TAG_BONUS = 0.1          # +0.1 for DSL tags
MULT_PENALTY = -1.0      # -1 for each multiplication
WRONG_ANSWER_PENALTY = -100.0  # -100 for wrong answers

# Alias for backward compatibility (avoid NameError in any leftover references)
WEIRD_ANSWER_PENALTY = WRONG_ANSWER_PENALTY

# Discovery logging setup
timestamp_for_discoveries = datetime.now().strftime("%Y%m%d_%H%M%S")
DISCOVERIES_LOG_FILE = os.path.join(BASE_OUTPUT_DIR, f"dsl_discoveries_{timestamp_for_discoveries}.txt")

def log_discovery(message, dsl_script=None):
    """Log discoveries to both file and console"""
    full_message = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
    print(f"**DISCOVERY** {full_message}")
    
    # Write to local file
    with open(DISCOVERIES_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{full_message}\n")
        if dsl_script:
            f.write(f"DSL Script:\n{dsl_script}\n")
            f.write("-" * 50 + "\n")

# Initialize discovery log
log_discovery(f"DISCOVERY LOG STARTED - Training Session {timestamp_for_discoveries}")
log_discovery(f"Simple Scoring: +0.1 for DSL tags, -1 per multiplication, -100 for wrong answers")

def matrix_dsl_reward(completions, prompts=None, completion_ids=None, **kwargs):
    global _num_generations_per_prompt_for_reward, _reward_call_count, _best_n_mults
    _reward_call_count += 1
    
    A_matrix_str_list = kwargs["A_matrix_str"]
    B_matrix_str_list = kwargs["B_matrix_str"]
    expected_C_str_list = kwargs["expected_C_str"]
    rewards = []

    print(f"\n=== NEW REWARD CALCULATION CALL #{_reward_call_count} ===")
    print(f"Processing {len(completions)} completions...")
    print(f"Current best multiplications: {_best_n_mults if _best_n_mults != float('inf') else 'None found yet'}")

    for i, dsl_script_raw_content in enumerate(completions):
        prompt_idx = i // _num_generations_per_prompt_for_reward
        current_A_str = A_matrix_str_list[prompt_idx]
        current_B_str = B_matrix_str_list[prompt_idx]
        current_expected_C_str = expected_C_str_list[prompt_idx]

        try:
            A = ast.literal_eval(current_A_str)
            B = ast.literal_eval(current_B_str)
            expected_C = ast.literal_eval(current_expected_C_str)
        except Exception as e:
            print(f"  Completion {i}: Error parsing matrices: {e}")
            rewards.append(WEIRD_ANSWER_PENALTY + TAG_BONUS)
            continue

        # Handle different completion formats and extract content
        if isinstance(dsl_script_raw_content, list):
            if len(dsl_script_raw_content) == 1:
                item = dsl_script_raw_content[0]
                if isinstance(item, dict) and 'role' in item and 'content' in item:
                    processed_content = item['content']
                elif isinstance(item, str):
                    processed_content = item
                else:
                    processed_content = str(item)
            else:
                processed_content = str(dsl_script_raw_content)
        elif isinstance(dsl_script_raw_content, dict) and 'role' in dsl_script_raw_content and 'content' in dsl_script_raw_content:
            processed_content = dsl_script_raw_content['content']
        elif isinstance(dsl_script_raw_content, str):
            processed_content = dsl_script_raw_content
        else:
            processed_content = str(dsl_script_raw_content)

        # Extract DSL content (with or without tags)
        dsl_match = re.search(r'<DSL>(.*?)</DSL>', processed_content, re.DOTALL | re.IGNORECASE)
        if dsl_match:
            dsl_content = dsl_match.group(1).strip()
            print(f"  Completion {i}: [PASS] DSL tags found!")
            tag_bonus = TAG_BONUS
        else:
            dsl_content = processed_content
            print(f"  Completion {i}: [WARNING] No <DSL></DSL> tags found")
            tag_bonus = 0.0
        
        # Clean up special tokens from DSL content
        temp_tokens_to_remove = ["<|im_end|>", "<|endoftext|>", "<|file_separator|>"]
        if hasattr(tokenizer_for_training, 'all_special_tokens') and tokenizer_for_training.all_special_tokens is not None:
            valid_special_tokens = [
                str(t) for t in tokenizer_for_training.all_special_tokens
                if t is not None and isinstance(t, str) and t
            ]
            temp_tokens_to_remove.extend(valid_special_tokens)
        
        if tokenizer_for_training.eos_token and isinstance(tokenizer_for_training.eos_token, str):
            temp_tokens_to_remove.append(tokenizer_for_training.eos_token)

        unique_tokens_to_remove = list(set(t for t in temp_tokens_to_remove if t))

        for token_str in unique_tokens_to_remove:
            dsl_content = dsl_content.replace(token_str, "")

        lines = dsl_content.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        final_dsl_script = "\n".join(cleaned_lines)

        if not final_dsl_script or final_dsl_script.strip().lower() == "error: cannot determine full sequence.":
            print(f"  Completion {i}: Empty or error DSL script. Reward: {WEIRD_ANSWER_PENALTY + tag_bonus:.1f}")
            rewards.append(WEIRD_ANSWER_PENALTY + tag_bonus)
            continue
            
        # Parse and Execute DSL
        num_multiplications = 0
        reward = 0.0
        
        try:
            # Count multiplications more accurately
            num_multiplications = count_multiplications_in_dsl(final_dsl_script)
            
            # Execute DSL
            executor = DSLExecutor(A, B)
            C_dsl = executor.run_dsl_and_get_c(final_dsl_script)

            # SIMPLE SCORING SYSTEM
            if C_dsl == expected_C:
                # Correct answer: penalty only for number of multiplications
                reward = num_multiplications * MULT_PENALTY  # -1 per multiplication
                print(f"  Completion {i}: **CORRECT** {num_multiplications}-mult, reward={reward:.1f}")
                
                if num_multiplications < _best_n_mults:
                    _best_n_mults = num_multiplications
                    log_discovery(f"NEW BEST SOLUTION! {num_multiplications} multiplications", final_dsl_script)
                    log_discovery(f"Test matrices: A={A}, B={B}, Expected C={expected_C}")
                    
            else:
                # Wrong answer: large penalty
                reward = WRONG_ANSWER_PENALTY  # -100
                print(f"  Completion {i}: **INCORRECT** {num_multiplications}-mult, reward={reward:.1f}")
                print(f"  Completion {i}: Expected: {expected_C}, Got: {C_dsl}")

        except Exception as e:
            # FAILED EXECUTION - wrong answer penalty
            reward = WRONG_ANSWER_PENALTY  # -100
            print(f"  Completion {i}: **EXECUTION FAILED**: {str(e)[:100]}...")
            print(f"  Completion {i}: Wrong answer penalty: {reward:.1f}")
        
        # Final reward with tag bonus
        final_reward = reward + tag_bonus
        rewards.append(final_reward)
        
        if tag_bonus > 0:
            print(f"  Completion {i}: Final reward: {final_reward:.1f} (base: {reward:.1f}, tag bonus: +{tag_bonus:.1f})")
        else:
            print(f"  Completion {i}: Final reward: {final_reward:.1f}")
        
    print(f"Batch average reward: {sum(rewards)/len(rewards):.2f}")
    print(f"Current global best: {_best_n_mults if _best_n_mults != float('inf') else 'None'} multiplications")
    print("=" * 50)
    
    return rewards

# --- 7. Training Arguments and GRPOTrainer ---
print("Configuring training arguments for GRPO...")
# Configure optimal precision settings
use_bf16 = is_bfloat16_supported()  # Automatic detection
use_fp16 = not use_bf16

# Configure distributed training arguments
distributed_args = {}
if 'RANK' in os.environ:
    distributed_args.update({
        # CRITICAL: Disable problematic distributed features
        "ddp_broadcast_buffers": False,
        # Note: world_size, process_index, local_rank are handled automatically by torch.distributed.launch
    })
    print(f"[DISTRIBUTED] Configured for distributed training")
    print(f"[DISTRIBUTED] Environment: RANK={os.environ.get('RANK')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}")
else:
    distributed_args.update({
        # Keep minimal for single GPU
    })
    print(f"[SINGLE GPU] Configured for single GPU training")

# Setup TensorBoard logging directory
local_tensorboard_dir = os.path.join(FINAL_MODEL_PATH, "runs")
os.makedirs(local_tensorboard_dir, exist_ok=True)

# Always use local directory for TensorBoard logging to avoid connection issues
actual_tensorboard_dir = local_tensorboard_dir

training_args_grpo = GRPOConfig(
    output_dir=FINAL_MODEL_PATH,
    learning_rate=NEW_LEARNING_RATE,  # Use the configurable learning rate
    remove_unused_columns=False,
    gradient_accumulation_steps=GRAD_ACC_STEPS,
    num_train_epochs=EPOCHS,
    bf16=use_bf16, 
    fp16=use_fp16,
    per_device_train_batch_size=BATCH_SIZE_PER_GPU,
    # Training configuration settings
    max_completion_length=512,
    num_generations=_num_generations_per_prompt_for_reward,
    max_prompt_length=256,
    logging_steps=5,
    save_strategy="steps",  # Change to steps for checkpoint saving every 100 steps
    save_steps=100,  # Save checkpoint every 100 steps
    logging_dir=actual_tensorboard_dir,  # TensorBoard logs to the correct directory
    report_to="tensorboard",
    push_to_hub=False,
    dataloader_drop_last=True,  # Changed to True to ensure consistent batches
    warmup_steps=2,
    # Data loading configuration
    dataloader_num_workers=0,
    dataloader_prefetch_factor=None,
    # Memory optimization settings
    gradient_checkpointing=False,  # Explicitly disabled to prevent warnings
    max_grad_norm=MAX_GRAD_NORM,
    warmup_ratio=WARMUP_RATIO,
    # Additional GRPO settings
    group_by_length=False,
    dataloader_pin_memory=False,
    ddp_find_unused_parameters=False,
    torch_compile=False,
    # Apply distributed configuration
    **distributed_args,
)

model_peft.config.pad_token_id = tokenizer_for_training.pad_token_id

# Validate dataset before training
print(f"Dataset validation:")
print(f"  - Dataset size: {len(train_dataset_for_grpo)}")
print(f"  - Dataset features: {train_dataset_for_grpo.features}")

if len(train_dataset_for_grpo) == 0:
    print("[ERROR] Dataset is empty!")
    exit()

# Check if dataset is sufficient for distributed training
min_samples_needed = BATCH_SIZE_PER_GPU * NUM_GPUS * GRAD_ACC_STEPS * 2
if len(train_dataset_for_grpo) < min_samples_needed:
    print(f"[WARNING] Dataset size ({len(train_dataset_for_grpo)}) is smaller than recommended minimum ({min_samples_needed})")
    print(f"[WARNING] This may cause issues with distributed training across {NUM_GPUS} GPUs")
    print(f"[RECOMMENDATION] Run dataset.py to generate more samples")
else:
    print(f"[SUCCESS] Dataset size is sufficient for distributed training")

trainer_grpo = GRPOTrainer(
    model=model_peft,
    # NOTE: tokenizer parameter removed - old tokenizer API causes UnboundLocalError: 'current_batch' 
    # GRPOTrainer will use the tokenizer from the model automatically
    reward_funcs=[matrix_dsl_reward],
    args=training_args_grpo,
    train_dataset=train_dataset_for_grpo,
    optimizers=(optimizer, None),  # Use our custom optimizer, no scheduler
)

print("Starting GRPO training...")
print(f"  - Per-device batch size: {training_args_grpo.per_device_train_batch_size}")
print(f"  - Gradient accumulation steps: {training_args_grpo.gradient_accumulation_steps}")
print(f"  - Total effective batch size: {TOTAL_BATCH_SIZE}")
print(f"  - Learning rate: {training_args_grpo.learning_rate}")
print(f"  - Epochs: {training_args_grpo.num_train_epochs}")
print(f"  - Max completion length: {training_args_grpo.max_completion_length}")
print(f"  - Max prompt length: {training_args_grpo.max_prompt_length}")
print(f"  - Generations per prompt: {_num_generations_per_prompt_for_reward}")
print(f"  - Mode: Standard PyTorch training")
print(f"TensorBoard logs will be saved locally in: {actual_tensorboard_dir}")
print(f"To view TensorBoard, run: %load_ext tensorboard")
print(f"%tensorboard --logdir {actual_tensorboard_dir}")

# Add pre-training diagnostics
print("\n=== PRE-TRAINING DIAGNOSTICS ===")
print(f"GPU Memory before training:")
if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
    memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3
    print(f"  Allocated: {memory_allocated:.2f} GiB")
    print(f"  Reserved: {memory_reserved:.2f} GiB")

print(f"Dataset sample check:")
print(f"  First sample prompt length: {len(str(train_dataset_for_grpo[0]['prompt']))}")
print(f"  First sample A matrix: {train_dataset_for_grpo[0]['A_matrix_str']}")

# Clear cache before training
torch.cuda.empty_cache()
print("Cleared CUDA cache before training")

print("\n=== PRE-TRAINING FINAL CHECKS ===")
# Minimal gradient sanity-check (no dummy forward pass)
trainable_exists = any(p.requires_grad for p in model_peft.parameters())
if not trainable_exists:
    raise ValueError("No trainable parameters have gradients enabled.")
print("[FINAL CHECK] Gradient flags verified (✓)")

print("\n=== STARTING GRPO TRAINING ===")
print("If training hangs on 'Loading data', it's likely a memory issue...")

try:
    trainer_grpo.train()
    print("GRPO Training finished successfully!")
except Exception as e:
    print(f"[ERROR] Training failed: {e}")
    print("Try reducing batch size further or using a smaller model")
    # Don't exit, continue with inference to test what we have
    pass

# Log final discovery summary
final_best = _best_n_mults if _best_n_mults != float('inf') else 'None found'
log_discovery(f"TRAINING COMPLETED - Final best: {final_best} multiplications")
log_discovery(f"Simple reward system: +0.1 for DSL tags, -1 per multiplication, -100 for wrong answers")
log_discovery(f"Discoveries log saved to: {DISCOVERIES_LOG_FILE}")

# --- 8. Save Model ---
print(f"Saving fine-tuned model to {FINAL_MODEL_PATH}...")
trainer_grpo.save_model(FINAL_MODEL_PATH)

# --- 9. Inference and Verification ---
print("\n--- Inference with GRPO Fine-tuned Model ---")
try:
    print(f"Loading fine-tuned model from: {FINAL_MODEL_PATH}")
    inference_base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME_FOR_FINETUNING, torch_dtype="auto", trust_remote_code=True)
    inference_model = PeftModel.from_pretrained(inference_base_model, FINAL_MODEL_PATH)
    inference_model = inference_model.merge_and_unload()
    
    inference_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME_FOR_FINETUNING, trust_remote_code=True)
    if inference_tokenizer.pad_token is None: 
        inference_tokenizer.pad_token = inference_tokenizer.eos_token
    if inference_tokenizer.padding_side == 'right': 
        inference_tokenizer.padding_side = 'left'
    print("[SUCCESS] Fine-tuned model loaded for inference.")
except Exception as e:
    print(f"[ERROR] Error loading fine-tuned model: {e}")
    print("[FALLBACK] Falling back to base model for inference...")
    try:
        inference_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME_FOR_FINETUNING, torch_dtype="auto", trust_remote_code=True)
        inference_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME_FOR_FINETUNING, trust_remote_code=True)
        if inference_tokenizer.pad_token is None: 
            inference_tokenizer.pad_token = inference_tokenizer.eos_token
        if inference_tokenizer.padding_side == 'right': 
            inference_tokenizer.padding_side = 'left'
        print("[FALLBACK] Using base model for inference.")
    except Exception as fallback_e:
        print(f"[ERROR] Error loading base model for inference: {fallback_e}. Exiting.")
        exit()

text_gen_pipeline = pipeline(
    "text-generation", 
    model=inference_model, 
    tokenizer=inference_tokenizer
)

user_query_for_inference = DEFAULT_USER_PROMPT_FOR_DSL_GENERATION
user_query_for_inference += f" (Using A={str(A_INFERENCE_MATRIX)}, B={str(B_INFERENCE_MATRIX)})"

inference_chat_messages = [
    {"role": "system", "content": SYSTEM_MESSAGE},
    {"role": "user", "content": user_query_for_inference}
]
formatted_inference_prompt = inference_tokenizer.apply_chat_template(
    inference_chat_messages, tokenize=False, add_generation_prompt=True)

print(f"\nGenerating DSL for A={A_INFERENCE_MATRIX}, B={B_INFERENCE_MATRIX}")
print(f"Expected result: {C_EXPECTED_INFERENCE_RESULT}")

outputs = text_gen_pipeline(
    formatted_inference_prompt, 
    max_new_tokens=350, 
    do_sample=False, 
    temperature=0.1, 
    top_p=0.9,
    pad_token_id=inference_tokenizer.pad_token_id, 
    eos_token_id=inference_tokenizer.eos_token_id
)

generated_full_text = outputs[0]['generated_text']
assistant_reply_raw = generated_full_text
if generated_full_text.startswith(formatted_inference_prompt):
    assistant_reply_raw = generated_full_text[len(formatted_inference_prompt):].strip()
else:
    assistant_marker = "<|im_start|>assistant"
    last_occurrence_idx = generated_full_text.rfind(assistant_marker)
    if last_occurrence_idx != -1:
        start_of_reply_idx = generated_full_text.find("\n", last_occurrence_idx)
        if start_of_reply_idx != -1:
            assistant_reply_raw = generated_full_text[start_of_reply_idx+1:].strip()

tokens_to_clean = ["<|im_end|>", "<|endoftext|>"] + ([inference_tokenizer.eos_token] if inference_tokenizer.eos_token else [])
unique_tokens_to_clean = list(set(t for t in tokens_to_clean if t))
for token in unique_tokens_to_clean:
    if assistant_reply_raw.endswith(token):
        assistant_reply_raw = assistant_reply_raw[:-len(token)].strip()

print(f"\n--- Raw Assistant's Reply (DSL Script) ---\n{assistant_reply_raw}\n------------------------------------")
print("\n--- Verifying Generated DSL ---")
lines = assistant_reply_raw.split('\n')
cleaned_lines = [line.strip() for line in lines if line.strip()]
final_generated_dsl = "\n".join(cleaned_lines)

if not final_generated_dsl or final_generated_dsl.strip().lower() == "error: cannot determine full sequence.":
    print("[FAILED] Model did not generate a valid DSL script or explicitly errored.")
else:
    try:
        executor = DSLExecutor(A_INFERENCE_MATRIX, B_INFERENCE_MATRIX)
        C_generated = executor.run_dsl_and_get_c(final_generated_dsl)
        num_mults_generated = sum(1 for line in final_generated_dsl.split('\n') if re.search(r"=\s*[\w\[\],\s\.\d\-]+\s*\*\s*[\w\[\],\s\.\d\-]+", line.strip()))

        print(f"  Generated DSL executed. Multiplications: {num_mults_generated}")
        print(f"  Resulting C: {C_generated}")
        print(f"  Expected C:  {C_EXPECTED_INFERENCE_RESULT}")

        if C_generated == C_EXPECTED_INFERENCE_RESULT:
            print(f"[PASSED] Algorithmically correct.")
            if num_mults_generated <= 7: 
                print(f"    [EFFICIENT] {num_mults_generated} multiplications (<= 7).")
            else: 
                print(f"    [SUBOPTIMAL] {num_mults_generated} multiplications (> 7).")
        else:
            print("[FAILED] Algorithmically INCORRECT.")
    except ValueError as e: 
        print(f"[FAILED] Invalid DSL or execution error: {e}")
    except Exception as e: 
        print(f"[FAILED] Unexpected verification error: {e}")

print("\n--- Testing Enhanced DSL with Chained Operations ---")
# Test the enhanced DSL executor with the user's example
test_dsl_example = """M1 = (A[0,0] - A[1,1]) * (B[1,0] + B[1,1])
M2 = (A[1,0] + A[0,1]) * (B[0,1])
M3 = A[1,1] * (B[0,0] - B[1,1])
M4 = A[0,0] * (B[1,1] + B[0,0])
M5 = (A[0,1] + A[1,1]) * (B[0,0])
M6 = (A[1,0] - A[0,0]) * (B[1,0] - B[0,1])
M7 = (A[0,0] + A[1,0]) * (B[0,1] - B[1,1])

C[0,0] = M2 + M3 - M5 + M6
C[0,1] = M1 + M6 - M4
C[1,0] = M7 - M2 + M1
C[1,1] = M3 - M4 + M5 - M7"""

print("Testing DSL example with chained operations:")
print(f"Input DSL:\n{test_dsl_example}")

try:
    # Test multiplication counting
    mult_count = count_multiplications_in_dsl(test_dsl_example)
    print(f"\n[MULT COUNT] Found {mult_count} multiplications (expected: 7)")
    
    # Test DSL execution
    test_executor = DSLExecutor(A_INFERENCE_MATRIX, B_INFERENCE_MATRIX)
    test_result = test_executor.run_dsl_and_get_c(test_dsl_example)
    expected_result = manual_matrix_multiply_2x2(A_INFERENCE_MATRIX, B_INFERENCE_MATRIX)
    
    print(f"\n[DSL TEST] Test matrices: A={A_INFERENCE_MATRIX}, B={B_INFERENCE_MATRIX}")
    print(f"[DSL TEST] Expected result: {expected_result}")
    print(f"[DSL TEST] DSL result:      {test_result}")
    
    if test_result == expected_result:
        print(f"[DSL TEST] ✓ Enhanced DSL executor works correctly!")
        print(f"[DSL TEST] ✓ Chained operations (M2 + M3 - M5 + M6) handled properly")
        print(f"[DSL TEST] ✓ Parenthetical expressions parsed correctly")
        if mult_count == 7:
            print(f"[DSL TEST] ✓ Multiplication counting is accurate (7/7)")
        else:
            print(f"[DSL TEST] ⚠ Multiplication count discrepancy: got {mult_count}, expected 7")
    else:
        print(f"[DSL TEST] ✗ DSL execution failed - results don't match")
        
except Exception as e:
    print(f"[DSL TEST] ✗ Error testing enhanced DSL: {e}")

print(f"\n*** SCRIPT COMPLETED SUCCESSFULLY ***")
print(f"[SAVED] Model saved to: {FINAL_MODEL_PATH}")
print(f"[LOGS] TensorBoard logs (local): {actual_tensorboard_dir}")
print(f"[DISCOVERIES] Local: {DISCOVERIES_LOG_FILE}")
final_best_summary = _best_n_mults if _best_n_mults != float('inf') else 'None found'
print(f"[BEST SOLUTION] {final_best_summary} multiplications")
print(f"[TIP] To restart TensorBoard: %tensorboard --logdir {actual_tensorboard_dir}") 