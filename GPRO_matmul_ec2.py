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
from dataclasses import dataclass
from transformers import TrainerCallback

# ==============================================
#  CENTRAL TRAINING CONFIGURATION (ONE-STOP VIEW)
# ==============================================
@dataclass
class TrainConfig:
    """All tunable knobs for GRPO training & sampling."""
    # --- core hyper-parameters ---
    learning_rate: float = 2e-6
    epochs: int = 1
    grad_acc_steps: int = 8

    # --- batch / sequence lengths ---
    batch_size_per_gpu: int = 6   # reduced batch size due to 1024-token completions
    max_completion_length: int = 750
    max_prompt_length: int = 256

    # --- generation / exploration ---
    num_generations: int = 8                 # increased exploration depth
    temperature: float = 1.4                 # higher entropy for exploration

    # ---------- new exploration levers ----------
    beta: float = 0.01           # KL penalty coefficient
    scale_rewards: bool = False  # disable reward scaling
    top_k: int | None = None     # allow disabling top-k sampling
    top_p: float = 0.95          # nucleus sampling threshold

    # reward shaping
    shaping_coeff: float = -0.1  # additional per-multiplication penalty

    # ---------- advanced GRPO hyper-params ----------
    epsilon: float = 0.2                        # PPO-style ratio clip
    epsilon_high: float = 0.28                 # tighter two-sided clip
    delta: float = 0.3                         # alt name used in some versions
    loss_type: str = "dr_grpo"                # use DR-GRPO to reduce length bias
    mask_truncated_completions: bool = True    # ignore sequences that hit max tokens in loss
    num_iterations: int = 2                    # more inner optimisation steps
    generation_batch_size: int | None = None   # oversampling knob
    steps_per_generation: int | None = None    # decouples sampling from optimisation
    disable_dropout: bool = True               # eliminates stochastic KL noise

    # --- logging / checkpointing ---
    logging_steps: int = 5
    save_steps: int = 10
    warmup_steps: int = 2
    warmup_ratio: float = 0.05

    # --- verbose logging helpers ---
    log_completions: bool = True
    num_completions_to_print: int = 2
    token_entropy_percentile_threshold: float = 0.05

# Instantiate a global so the rest of the script can reference it
CFG = TrainConfig()

# Convenience aliases so existing variables continue to work
NEW_LEARNING_RATE = CFG.learning_rate
EPOCHS = CFG.epochs

_num_generations_per_prompt_for_reward = CFG.num_generations  # synced with config

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

    # ------------------------------------------------------
    # Helper: canonicalize a DSL variable name so that minor
    # spacing / case differences (e.g. "a[0, 1]", "C[1, 0]")
    # no longer break execution.
    # ------------------------------------------------------
    def _canonicalize_var_name(self, name: str) -> str:
        name = name.strip()
        # Upper-case first letter (so a→A, b→B, c→C, m→M, etc.)
        if name and name[0].isalpha():
            name = name[0].upper() + name[1:]
        # Remove optional spaces inside the [row,col] indices
        name = re.sub(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]', r'[\1,\2]', name)
        return name

    def _get_value(self, var_name):
        var_name = self._canonicalize_var_name(var_name)
        try:
            # Attempt literal eval first (for numeric constants or lists)
            return ast.literal_eval(var_name)
        except (ValueError, SyntaxError, TypeError):
            if var_name not in self.variables:
                raise ValueError(f"Variable '{var_name}' not found. Available: {list(self.variables.keys())}")
            return self.variables[var_name]

    def execute_step(self, step_line):
        original_step_line = step_line
        step_line = step_line.strip()
        # Allow users to end statements with a semicolon, e.g. "C[0,0] = ...;"
        if step_line.endswith(';'):
            step_line = step_line[:-1].rstrip()
        if not step_line: return
        if '=' not in step_line:
            raise ValueError(f"Malformed DSL step (missing '='): '{original_step_line}'")

        target_var, expression = [s.strip() for s in step_line.split('=', 1)]
        
        # Canonicalize target variable name to maintain consistency (e.g. C[0, 0] -> C[0,0])
        target_var_canon = self._canonicalize_var_name(target_var)
        
        # Handle chained operations (e.g., M1 + M4 - M5 + M7)
        result = self._evaluate_expression(expression, original_step_line)
        self.variables[target_var_canon] = result
    
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
        # Allow the first term to be a complex expression (e.g., contains '*')
        result = self._evaluate_expression(first_term, original_step_line) if ('*' in first_term or '+' in first_term or '-' in first_term[1:]) else self._get_value(first_term)
        
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
                
            # Recursively evaluate the operand so it can itself be a multiplication or parenthetical expr
            value = self._evaluate_expression(operand, original_step_line) if ('*' in operand or '+' in operand or '-' in operand[1:]) else self._get_value(operand)
            
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
CHECKPOINT_PATH = "./final_adapter"  # Path to existing DSL-STF LoRA adapter (updated)
NEW_LEARNING_RATE = 2e-6 # Learning rate for fresh optimizer
LOAD_FROM_CHECKPOINT = True  # Set to False to train from scratch

# Training configuration - Auto-detect distributed vs single GPU
EPOCHS = 1

# Auto-configure based on distributed environment
if 'WORLD_SIZE' in os.environ:
    # Multi-GPU distributed training
    NUM_GPUS = int(os.environ['WORLD_SIZE'])
    # BATCH_SIZE_PER_GPU = 6   # reduced batch size due to 1024-token completions
    # GRAD_ACC_STEPS = 8      # Keep same effective batch size
    print(f"[CONFIG] Multi-GPU mode detected: {NUM_GPUS} GPUs")
    print(f"[CONFIG] Distributed training configuration")
else:
    # Single GPU training
    NUM_GPUS = 1
    # BATCH_SIZE_PER_GPU = 4   # reduced batch size due to 1024-token completions (single-GPU)
    # GRAD_ACC_STEPS = 16      # Keep same effective batch size
    print(f"[CONFIG] Single GPU mode")
    print(f"[CONFIG] Standard training configuration")

# Calculate total batch size
TOTAL_BATCH_SIZE = CFG.batch_size_per_gpu * NUM_GPUS * CFG.grad_acc_steps
print(f"[CONFIG] Batch size: {CFG.batch_size_per_gpu}, Gradient accumulation: {CFG.grad_acc_steps}")

# Memory optimization settings
USE_GRADIENT_CHECKPOINTING = False  # Disabled to avoid gradient issues with PEFT
MAX_GRAD_NORM = 0.5
WARMUP_RATIO = 0.05

model_config_desc = f"lr{NEW_LEARNING_RATE}_epochs{EPOCHS}_batch{TOTAL_BATCH_SIZE}_gradacc{CFG.grad_acc_steps}_gpu{NUM_GPUS}_t4"
model_name = f"{TRAINED_MODEL_DIR_NAME}_{model_config_desc}_{timestamp}"
tensorboard_name = f"runs_{model_config_desc}_{timestamp}"

FINAL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, model_name)
FINAL_TENSORBOARD_PATH = os.path.join(TENSORBOARD_LOGS_DIR, tensorboard_name)

print(f"[SAVE CONFIG] Model will be saved to: {FINAL_MODEL_PATH}")
print(f"[SAVE CONFIG] TensorBoard logs will be saved to: {FINAL_TENSORBOARD_PATH}")
print(f"[GPU CONFIG] Using {NUM_GPUS} GPU(s) with {CFG.batch_size_per_gpu} batch size per GPU")
print(f"[TRAINING CONFIG] Max completion length: 512 tokens, Max prompt length: 256 tokens")
print(f"[TRAINING CONFIG] Effective batch size: {TOTAL_BATCH_SIZE} (batch_size={CFG.batch_size_per_gpu} × grad_acc={CFG.grad_acc_steps})")

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
            # print(f"[PEFT SETUP] Enabled: {name}")  # Suppressed verbose per-parameter log
    
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
STF_ADAPTER_PATH = "./final_adapter"  # path to frozen DSL-STF LoRA adapter

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
    # Attempt to make ONLY the GRPO adapter trainable using the PEFT helper.
    try:
        model_peft.train_adapter("grpo")  # Preferred API (PEFT ≥0.6)
    except AttributeError:
        print("[STACKED PEFT] 'train_adapter' not found in current PEFT version – falling back to manual gradient control.")
        # Manually enable gradients for GRPO LoRA parameters only; others stay frozen.
        for name, param in model_peft.named_parameters():
            if "lora_" in name.lower():
                param.requires_grad = ".grpo" in name.lower()
            else:
                param.requires_grad = False

    # Activate the GRPO adapter for forward passes.
    try:
        model_peft.set_adapter("grpo")
    except AttributeError:
        print("[STACKED PEFT] 'set_adapter' not found in current PEFT version – continuing without explicit activation.")
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
_num_generations_per_prompt_for_reward = CFG.num_generations  # synced with TrainConfig (must be >= 2)
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
        # If the DSL tags were present but contain no valid lines, do NOT award the tag bonus
        if not cleaned_lines:
            tag_bonus = 0.0  # No bonus for empty DSL content
            print(f"  Completion {i}: [WARNING] <DSL> tags are empty – tag bonus removed")
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
            shaping_term = CFG.shaping_coeff * num_multiplications
            
            # Execute DSL
            executor = DSLExecutor(A, B)
            C_dsl = executor.run_dsl_and_get_c(final_dsl_script)

            # SIMPLE SCORING SYSTEM
            if C_dsl == expected_C:
                # Correct answer: penalty only for number of multiplications
                reward = num_multiplications * MULT_PENALTY  # -1 per multiplication
                reward += shaping_term
                print(f"  Completion {i}: **CORRECT** {num_multiplications}-mult, reward={reward:.1f}")
                
                if num_multiplications < _best_n_mults:
                    _best_n_mults = num_multiplications
                    log_discovery(f"NEW BEST SOLUTION! {num_multiplications} multiplications", final_dsl_script)
                    log_discovery(f"Test matrices: A={A}, B={B}, Expected C={expected_C}")
                    
            else:
                # Wrong answer: large penalty
                reward = WRONG_ANSWER_PENALTY  # -100
                reward += (num_multiplications * MULT_PENALTY) + shaping_term
                print(f"  Completion {i}: **INCORRECT** {num_multiplications}-mult, reward={reward:.1f}")
                print(f"  Completion {i}: Expected: {expected_C}, Got: {C_dsl}")

        except Exception as e:
            # FAILED EXECUTION - wrong answer penalty
            reward = WRONG_ANSWER_PENALTY  # -100
            print(f"  Completion {i}: **EXECUTION FAILED**: {str(e)[:100]}...")
            print(f"  Completion {i}: Wrong answer penalty: {reward:.1f}")
        
        # Final reward with tag bonus
        final_reward = reward + tag_bonus  # tag bonus added after shaping already in reward
        rewards.append(final_reward)
        
        if tag_bonus > 0:
            print(f"  Completion {i}: Final reward: {final_reward:.1f} (base: {reward:.1f}, tag bonus: +{tag_bonus:.1f})")
        else:
            print(f"  Completion {i}: Final reward: {final_reward:.1f}")
        
    print(f"Batch average reward: {sum(rewards)/len(rewards):.2f}")
    print(f"Current global best: {_best_n_mults if _best_n_mults != float('inf') else 'None'} multiplications")
    print("=" * 50)
    
    return rewards

# ==================================================
#  Metric Logger Callback – prints exploration stats
# ==================================================
class ExplorationMetricCallback(TrainerCallback):
    """Logs entropy, KL, and advantage mean each time trainer logs."""
    def __init__(self):
        self.initial_entropy = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        entropy = None
        kl = None
        adv = None
        # try common keys
        for k, v in logs.items():
            lk = k.lower()
            if 'entropy' in lk and entropy is None:
                entropy = v
            if ('kl' in lk or 'kl_div' in lk) and kl is None:
                kl = v
            if 'advantage' in lk and adv is None:
                adv = v
        # store initial entropy
        if entropy is not None and self.initial_entropy is None:
            self.initial_entropy = entropy
        # Build message
        msg_parts = []
        if entropy is not None and self.initial_entropy is not None:
            ratio = 100.0 * entropy / (self.initial_entropy + 1e-8)
            msg_parts.append(f"Entropy {entropy:.3f} ({ratio:.1f}% of start)")
        if kl is not None:
            msg_parts.append(f"KL {kl:.3f}")
        if adv is not None:
            msg_parts.append(f"Adv ⌀ {adv:.3f}")
        if msg_parts:
            print("[METRICS] " + " | ".join(msg_parts))

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
    gradient_accumulation_steps=CFG.grad_acc_steps,
    num_train_epochs=EPOCHS,
    bf16=use_bf16, 
    fp16=use_fp16,
    per_device_train_batch_size=CFG.batch_size_per_gpu,
    # Training configuration settings
    max_completion_length=CFG.max_completion_length,
    num_generations=CFG.num_generations,
    temperature=CFG.temperature,
    top_k=CFG.top_k,
    top_p=CFG.top_p,
    beta=CFG.beta,
    scale_rewards=CFG.scale_rewards,
    max_prompt_length=CFG.max_prompt_length,
    logging_steps=CFG.logging_steps,
    save_strategy="steps",  # Change to steps for checkpoint saving every 100 steps
    save_steps=CFG.save_steps,   # Save checkpoint every N steps
    logging_dir=actual_tensorboard_dir,  # TensorBoard logs to the correct directory
    report_to="tensorboard",
    push_to_hub=False,
    dataloader_drop_last=True,  # Changed to True to ensure consistent batches
    warmup_steps=CFG.warmup_steps,
    # Data loading configuration
    dataloader_num_workers=0,
    dataloader_prefetch_factor=None,
    # Memory optimization settings
    gradient_checkpointing=False,  # Explicitly disabled to prevent warnings
    max_grad_norm=MAX_GRAD_NORM,
    warmup_ratio=CFG.warmup_ratio,
    # --- advanced GRPO knobs ---
    epsilon=CFG.epsilon,
    epsilon_high=CFG.epsilon_high,
    delta=CFG.delta,
    loss_type=CFG.loss_type,
    mask_truncated_completions=CFG.mask_truncated_completions,
    num_iterations=CFG.num_iterations,
    generation_batch_size=CFG.generation_batch_size,
    steps_per_generation=CFG.steps_per_generation,
    disable_dropout=CFG.disable_dropout,
    # Additional GRPO settings
    group_by_length=False,
    dataloader_pin_memory=False,
    ddp_find_unused_parameters=False,
    torch_compile=False,
    # --- logging helpers ---
    # log_completions=CFG.log_completions,
    # num_completions_to_print=CFG.num_completions_to_print,
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
min_samples_needed = CFG.batch_size_per_gpu * NUM_GPUS * CFG.grad_acc_steps * 2
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
    callbacks=[ExplorationMetricCallback()],
)

# ---------------------------------------------
# Checkpoint-resumption helper: scan EVERY run directory inside
# matmul_outputs/models/ and pick the highest-step `checkpoint-*`.
# This makes resumption independent of batch size, GPU count, etc.
# ---------------------------------------------

def _find_latest_checkpoint(base_models_dir: str):
    """Return path to checkpoint with the largest global step under base_models_dir."""
    latest_ckpt = None
    latest_step = -1
    for run_name in os.listdir(base_models_dir):
        run_dir = os.path.join(base_models_dir, run_name)
        if not os.path.isdir(run_dir):
            continue
        for entry in os.listdir(run_dir):
            if not entry.startswith("checkpoint-"):
                continue
            try:
                step_num = int(re.findall(r"\d+", entry)[-1])
            except (IndexError, ValueError):
                continue
            if step_num > latest_step:
                latest_step = step_num
                latest_ckpt = os.path.join(run_dir, entry)
    return latest_ckpt

# Scan for newest checkpoint across ALL runs
latest_ckpt_path = _find_latest_checkpoint(MODEL_SAVE_DIR)

if latest_ckpt_path:
    print(f"[CHECKPOINT] Latest checkpoint found: {latest_ckpt_path}")
else:
    print("[CHECKPOINT] No checkpoint found – starting fresh training")

print("Starting GRPO training (with auto-resume)…")
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
    if latest_ckpt_path:
        trainer_grpo.train(resume_from_checkpoint=latest_ckpt_path)
    else:
        trainer_grpo.train()
    print("GRPO Training finished successfully!")

    # ---- Post-training checkpoint sanity print ----
    try:
        final_ckpts = [d for d in os.listdir(FINAL_MODEL_PATH) if d.startswith("checkpoint-")]
        print(f"[CHECKPOINT] After training, checkpoints present: {final_ckpts}")
    except Exception as e:
        print(f"[CHECKPOINT] Warning: could not list checkpoint directories ({e})")

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
    temperature=CFG.temperature,
    top_p=CFG.top_p,
    top_k=CFG.top_k,
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