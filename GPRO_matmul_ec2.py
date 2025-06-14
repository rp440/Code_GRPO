"""
GPRO Matrix Multiplication DSL Training Script for EC2

Key Features:
1. Load previous LoRA checkpoint but discard optimizer state
2. Create fresh AdamW optimizer with configurable learning rate  
3. Reset value head parameters for fresh training
4. New L1 error-based reward function:
   - Correct result: score = (10 - mul_ops)
   - Incorrect result: score = (-dist/5 - 0.2*mul_ops)

Configuration:
- Set LOAD_FROM_CHECKPOINT = False to train from scratch
- Adjust CHECKPOINT_PATH to your previous model path
- Modify NEW_LEARNING_RATE as needed
- Configured for 4 GPUs with distributed training
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

# --- Initialize Distributed Training Environment ---
# CRITICAL: Initialize distributed environment before model loading
if 'RANK' in os.environ:
    print(f"[DISTRIBUTED] Initializing distributed training...")
    print(f"[DISTRIBUTED] RANK: {os.environ.get('RANK')}, LOCAL_RANK: {os.environ.get('LOCAL_RANK')}, WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    print(f"[DISTRIBUTED] Set CUDA device to: {torch.cuda.current_device()}")
else:
    print("[SINGLE GPU] Running in single GPU mode")

# --- 1. Paths Configuration ---
# Base directory for all outputs
BASE_OUTPUT_DIR = "/home/ec2-user/matmul_outputs"
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
        """Evaluate an expression that may contain chained +/- operations"""
        expression = expression.strip()
        
        # Simple assignment (no operators) - check for variable names or numbers only
        # Must be: variable name, matrix element, or number (no spaces around operators)
        assign_match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*(?:\[[0-9,\s]*\])?|\-?\d+(?:\.\d+)?)\s*$", expression)
        if assign_match:
            return self._get_value(assign_match.group(1).strip())
        
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
        
        # Chained operations (e.g., M1 + M4 - M5 + M7)
        # Split by + and - while preserving operators, but be careful with negative numbers
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
            
            if operator == '+': result += value
            elif operator == '-': result -= value
            else: raise ValueError(f"Unsupported operator '{operator}' in chained expression: '{expression}'")
            
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

def _generate_random_2x2_matrix_for_inference(low=-99, high=99):
    return [[random.randint(low, high) for _ in range(2)] for _ in range(2)]

A_INFERENCE_MATRIX = _generate_random_2x2_matrix_for_inference()
B_INFERENCE_MATRIX = _generate_random_2x2_matrix_for_inference()
C_EXPECTED_INFERENCE_RESULT = manual_matrix_multiply_2x2(A_INFERENCE_MATRIX, B_INFERENCE_MATRIX)

# --- 3. GRPO Configuration and System Prompt ---
BASE_MODEL_NAME_FOR_FINETUNING = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TRAINED_MODEL_DIR_NAME = f"{BASE_MODEL_NAME_FOR_FINETUNING.split('/')[-1]}-GRPO-MatMulDSL-JSONL"
LOCAL_TRAINED_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, TRAINED_MODEL_DIR_NAME)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Configuration for checkpoint loading and fresh training
CHECKPOINT_PATH = "/home/ec2-user/previous_checkpoint"  # Update this path
NEW_LEARNING_RATE = 2e-5
LOAD_FROM_CHECKPOINT = True

# Training configuration - Auto-detect distributed vs single GPU
EPOCHS = 1

# Auto-configure based on distributed environment
if 'WORLD_SIZE' in os.environ:
    # Multi-GPU distributed training
    NUM_GPUS = int(os.environ['WORLD_SIZE'])
    BATCH_SIZE_PER_GPU = 2  # Reduced for multi-GPU to fit in T4 memory
    GRAD_ACC_STEPS = 4      # Reduced since we have more GPUs
    print(f"[CONFIG] Multi-GPU mode detected: {NUM_GPUS} GPUs")
else:
    # Single GPU training
    NUM_GPUS = 1
    BATCH_SIZE_PER_GPU = 4  # Increased batch size for single GPU
    GRAD_ACC_STEPS = 8      # Increased gradient accumulation
    print(f"[CONFIG] Single GPU mode")

# Calculate total batch size
TOTAL_BATCH_SIZE = BATCH_SIZE_PER_GPU * NUM_GPUS * GRAD_ACC_STEPS

# Memory optimization settings for T4 GPUs
TORCH_DTYPE = torch.float16  # Use FP16 for T4 GPUs
USE_GRADIENT_CHECKPOINTING = True
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.1

model_config_desc = f"lr{NEW_LEARNING_RATE}_epochs{EPOCHS}_batch{TOTAL_BATCH_SIZE}_gradacc{GRAD_ACC_STEPS}_gpu{NUM_GPUS}_t4"
model_name = f"{TRAINED_MODEL_DIR_NAME}_{model_config_desc}_{timestamp}"
tensorboard_name = f"runs_{model_config_desc}_{timestamp}"

FINAL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, model_name)
FINAL_TENSORBOARD_PATH = os.path.join(TENSORBOARD_LOGS_DIR, tensorboard_name)

print(f"[SAVE CONFIG] Model will be saved to: {FINAL_MODEL_PATH}")
print(f"[SAVE CONFIG] TensorBoard logs will be saved to: {FINAL_TENSORBOARD_PATH}")
print(f"[GPU CONFIG] Using {NUM_GPUS} Tesla T4 GPU with {BATCH_SIZE_PER_GPU} batch size per GPU")
print(f"[MEMORY CONFIG] Using {TORCH_DTYPE} precision with gradient checkpointing")
print(f"[TRAINING CONFIG] Effective batch size: {TOTAL_BATCH_SIZE} (batch_size={BATCH_SIZE_PER_GPU} × grad_acc={GRAD_ACC_STEPS})")

SYSTEM_MESSAGE = """You are an AI assistant specialized in generating Domain Specific Language (DSL) scripts for 2x2 matrix multiplication. You can provide explanations, but must wrap your DSL code in <DSL></DSL> tags.
  EXAMPLE DSL OUTPUT FORMAT: For matrices A=[[1,2],[3,4]] and B=[[5,6],[7,8]], a valid response would be:  I'll generate the DSL script for matrix multiplication:  
<DSL> M1 = A[0,0] * B[0,0] M2 = A[0,1] * B[1,0] S1 = M1 + M2 C[0,0] = S1
M3 = A[0,0] * B[0,1]
M4 = A[0,1] * B[1,1]
S2 = M3 + M4
C[0,1] = S2

M5 = A[1,0] * B[0,0]
M6 = A[1,1] * B[1,0]
S3 = M5 + M6
C[1,0] = S3

M7 = A[1,1] * B[1,1]
S4 = M7 - M6
C[1,1] = S4
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
print(f"Loading base model: {BASE_MODEL_NAME_FOR_FINETUNING}")

# Configure device map for distributed vs single GPU
if 'RANK' in os.environ:
    # Distributed training - don't use device_map="auto"
    device_map = None
    print("[DISTRIBUTED] Loading model without device_map for distributed training")
else:
    # Single GPU - use device_map="auto"
    device_map = "auto"
    print("[SINGLE GPU] Loading model with device_map='auto'")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME_FOR_FINETUNING,
    torch_dtype=TORCH_DTYPE,  # Use FP16 for T4 GPUs
    device_map=device_map,
    trust_remote_code=True
)

# Configure model for gradient checkpointing
base_model.config.use_cache = False
if USE_GRADIENT_CHECKPOINTING:
    base_model.gradient_checkpointing_enable()
    print("[INFO] Enabled gradient checkpointing for memory efficiency")

# Load previous LoRA checkpoint or create fresh LoRA
if LOAD_FROM_CHECKPOINT:
    print(f"Loading previous LoRA checkpoint from: {CHECKPOINT_PATH}")
    try:
        model_peft = PeftModel.from_pretrained(
            base_model, 
            CHECKPOINT_PATH,
            torch_dtype=TORCH_DTYPE  # Ensure consistent dtype
        )
        print("[SUCCESS] Successfully loaded previous LoRA checkpoint")
        
        # Enable gradients for LoRA parameters
        for name, param in model_peft.named_parameters():
            if 'lora_' in name.lower():
                param.requires_grad = True
        print("[FIX] Enabled gradients for LoRA parameters")
        
    except Exception as e:
        print(f"[WARNING] Could not load checkpoint ({e}), creating fresh LoRA...")
        LOAD_FROM_CHECKPOINT = False

if not LOAD_FROM_CHECKPOINT:
    print("Creating fresh LoRA configuration...")
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    model_peft = get_peft_model(base_model, lora_config)

model_peft.print_trainable_parameters()

# Fresh optimizer with new learning rate and gradient clipping
from torch.optim import AdamW
optimizer = AdamW(
    model_peft.parameters(),
    lr=NEW_LEARNING_RATE,
    weight_decay=0.01,  # Added weight decay for better regularization
    eps=1e-8  # Increased epsilon for numerical stability with FP16
)
print(f"[SUCCESS] Created fresh AdamW optimizer with lr={NEW_LEARNING_RATE}")

# Reset value head parameters if it exists
try:
    if hasattr(model_peft, 'v_head') and model_peft.v_head is not None:
        model_peft.v_head.reset_parameters()
        print("[SUCCESS] Reset value head parameters")
    else:
        print("[INFO] No value head found to reset")
except Exception as e:
    print(f"[WARNING] Could not reset value head: {e}")

tokenizer_for_training = AutoTokenizer.from_pretrained(BASE_MODEL_NAME_FOR_FINETUNING, trust_remote_code=True)
if tokenizer_for_training.pad_token is None:
    tokenizer_for_training.pad_token = tokenizer_for_training.eos_token
    model_peft.config.pad_token_id = tokenizer_for_training.eos_token_id
if tokenizer_for_training.padding_side == 'right':
    tokenizer_for_training.padding_side = 'left'

# --- 6. Reward Function for GRPO ---
_num_generations_per_prompt_for_reward = 2
_reward_call_count = 0
_best_n_mults = float('inf')  # Track the best number of multiplications found so far

# New reward hyperparameters
CORRECT_7_MULT_BONUS = 5.0  # +5 for correct 7 multiplication solutions
NEAR_MISS_PENALTY = -15.0   # -1 for near-miss (8×, ≤6×) 
WEIRD_ANSWER_PENALTY = -20.0   # -10 for weird/incorrect answers
TAG_BONUS = 0.1  # +0.1 for DSL tags

# Exploration formula: -1.06×10^-8 * ||AB-C||^2 + 6
# Ranges ≈[-11, 6], so best exploration > correct (5), worst < weird (-10)
EXPLORATION_SCALE = -10.0 / 1.59936e17
EXPLORATION_OFFSET = 6.0      # Offset to ensure best exploration > correct answers

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
log_discovery(f"Exploration-Prioritized Scoring: -1.06e-8*||AB-C||²+6 for exploration (capped at -4 for 7-mult), +5 for correct 7-mult, -15 for near-miss, -10 for weird answers, +0.1 for DSL tags, min -19 for solvable DSL")

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
            # Count multiplications
            for line in final_dsl_script.split('\n'):
                if re.search(r"=\s*[\w\[\],\s\.\d\-]+\s*\*\s*[\w\[\],\s\.\d\-]+", line.strip()):
                    num_multiplications += 1
            
            # Execute DSL
            executor = DSLExecutor(A, B)
            C_dsl = executor.run_dsl_and_get_c(final_dsl_script)

            # NEW EXPLORATION-PRIORITIZED SCORING SYSTEM
            # Calculate L2 squared distance for all cases
            l2_sq_distance = 0.0
            for r in range(2):
                for c in range(2):
                    diff = C_dsl[r][c] - expected_C[r][c]
                    l2_sq_distance += diff * diff
            
            if num_multiplications == 7:
                # PRIORITIZED: 7-multiplication solutions get exploration formula with L2 distance capped at max -10
                # Formula: -1.06×10^-8 * ||AB-C||^2 + 6, but capped at minimum of -4 (6 - 10)
                base_reward = EXPLORATION_SCALE * l2_sq_distance + EXPLORATION_OFFSET
                reward = max(base_reward, EXPLORATION_OFFSET - 10.0)  # Cap L2 penalty at max -10
                
                if C_dsl == expected_C:
                    print(f"  Completion {i}: **PERFECT** Correct 7-multiplication solution! (L2²={l2_sq_distance:.0f}, reward={reward:.3f})")
                    
                    # Log the perfect solution
                    log_discovery(f"PERFECT 7-MULT SOLUTION! Score: {reward:.3f}", final_dsl_script)
                    log_discovery(f"Test matrices: A={A}, B={B}, Expected C={expected_C}")
                    
                    if num_multiplications < _best_n_mults:
                        _best_n_mults = num_multiplications
                else:
                    print(f"  Completion {i}: **7-MULT EXPLORATION** L2²={l2_sq_distance:.0f}, reward={reward:.3f} (capped)")
                    print(f"  Completion {i}: Expected: {expected_C}, Got: {C_dsl}")
                    
            elif C_dsl == expected_C:
                # CORRECT DSL but not 7-multiplication - use fixed penalty for near-miss
                reward = NEAR_MISS_PENALTY  # -15 for near-miss (correct but not 7-mult)
                print(f"  Completion {i}: **CORRECT** {num_multiplications}-mult (near-miss penalty: {NEAR_MISS_PENALTY})")
                
                if num_multiplications < _best_n_mults:
                    _best_n_mults = num_multiplications
                    log_discovery(f"NEW BEST SOLUTION! {num_multiplications} multiplications", final_dsl_script)
                    
            else:
                # INCORRECT non-7-multiplication solutions - use exploration formula but ensure lower priority
                base_exploration_reward = EXPLORATION_SCALE * l2_sq_distance + EXPLORATION_OFFSET
                
                # Ensure non-7-mult incorrect solutions are always worse than weird answers threshold
                # by capping them at slightly above weird answer penalty
                reward = min(base_exploration_reward, WEIRD_ANSWER_PENALTY + 1.0)
                
                print(f"  Completion {i}: **INCORRECT** {num_multiplications}-mul attempt. L2²={l2_sq_distance:.0f}, reward={reward:.3f}")
                print(f"  Completion {i}: Expected: {expected_C}, Got: {C_dsl}")
            
            # Ensure no solvable DSL gets reward less than -19
            reward = max(reward, -19.0)

        except Exception as e:
            # FAILED EXECUTION - weird answer penalty
            reward = WEIRD_ANSWER_PENALTY
            print(f"  Completion {i}: **EXECUTION FAILED**: {str(e)[:100]}...")
            print(f"  Completion {i}: Weird answer penalty: {reward:.1f}")
        
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
use_bf16 = False  # T4s don't support bfloat16
use_fp16 = True   # Use FP16 for T4 GPUs

# Configure distributed training arguments
distributed_args = {}
if 'RANK' in os.environ:
    distributed_args.update({
        # CRITICAL: Disable problematic distributed features
        "ddp_find_unused_parameters": False,
        "ddp_broadcast_buffers": False,
        "dataloader_pin_memory": False,
        # Note: world_size, process_index, local_rank are handled automatically by torch.distributed.launch
    })
    print(f"[DISTRIBUTED] Configured for distributed training")
    print(f"[DISTRIBUTED] Environment: RANK={os.environ.get('RANK')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}")
else:
    distributed_args.update({
        "ddp_find_unused_parameters": False,
        "dataloader_pin_memory": False,
    })
    print(f"[SINGLE GPU] Configured for single GPU training")

training_args_grpo = GRPOConfig(
    output_dir=FINAL_MODEL_PATH,
    learning_rate=NEW_LEARNING_RATE,
    remove_unused_columns=False,
    gradient_accumulation_steps=GRAD_ACC_STEPS,
    num_train_epochs=EPOCHS,
    bf16=use_bf16,
    fp16=use_fp16,
    per_device_train_batch_size=BATCH_SIZE_PER_GPU,
    max_completion_length=8000,  # Reduced from 16000 for memory efficiency
    num_generations=_num_generations_per_prompt_for_reward,
    max_prompt_length=1000,
    logging_steps=5,
    save_strategy="steps",
    save_steps=100,
    logging_dir=FINAL_TENSORBOARD_PATH,
    report_to="tensorboard",
    push_to_hub=False,
    dataloader_drop_last=True,  # CRITICAL for distributed training
    warmup_steps=5,
    # Data loading configuration for distributed training
    dataloader_num_workers=0,  # Avoid multiprocessing issues in distributed mode
    # Memory optimization settings
    gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
    max_grad_norm=MAX_GRAD_NORM,
    warmup_ratio=WARMUP_RATIO,
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
    reward_funcs=[matrix_dsl_reward],
    args=training_args_grpo,
    train_dataset=train_dataset_for_grpo,
    optimizers=(optimizer, None),
)

# --- 8. Training Execution ---
print("Starting GRPO training...")
print(f"  - Number of GPUs: {NUM_GPUS}")
print(f"  - Batch size per GPU: {BATCH_SIZE_PER_GPU}")
print(f"  - Gradient accumulation steps: {GRAD_ACC_STEPS}")
print(f"  - Total effective batch size: {TOTAL_BATCH_SIZE}")
print(f"  - Learning rate: {training_args_grpo.learning_rate}")
print(f"  - Epochs: {training_args_grpo.num_train_epochs}")

# Show training mode
if 'RANK' in os.environ:
    print(f"  - Training mode: Distributed ({NUM_GPUS} GPUs)")
    print(f"  - Current rank: {os.environ.get('RANK')}/{os.environ.get('WORLD_SIZE')}")
else:
    print(f"  - Training mode: Single GPU")

print(f"TensorBoard logs will be saved to: {FINAL_TENSORBOARD_PATH}")

trainer_grpo.train()
print("GRPO Training finished.")

# Log final discovery summary
final_best = _best_n_mults if _best_n_mults != float('inf') else 'None found'
log_discovery(f"TRAINING COMPLETED - Final best: {final_best} multiplications")
log_discovery(f"Discoveries log saved to: {DISCOVERIES_LOG_FILE}")

# --- 9. Save Model ---
print(f"Saving fine-tuned model to {FINAL_MODEL_PATH}...")
trainer_grpo.save_model(FINAL_MODEL_PATH)

# --- 10. Inference and Verification ---
print("\n--- Inference with GRPO Fine-tuned Model ---")
try:
    print(f"Loading fine-tuned model from: {FINAL_MODEL_PATH}")
    inference_base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME_FOR_FINETUNING, torch_dtype=TORCH_DTYPE, trust_remote_code=True)
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
            BASE_MODEL_NAME_FOR_FINETUNING, torch_dtype=TORCH_DTYPE, trust_remote_code=True)
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

print(f"\n*** SCRIPT COMPLETED SUCCESSFULLY ***")
print(f"[SAVED] Model saved to: {FINAL_MODEL_PATH}")
print(f"[LOGS] TensorBoard logs: {FINAL_TENSORBOARD_PATH}")
print(f"[DISCOVERIES] Log: {DISCOVERIES_LOG_FILE}")
final_best_summary = _best_n_mults if _best_n_mults != float('inf') else 'None found'
print(f"[BEST SOLUTION] {final_best_summary} multiplications")
print(f"[TIP] To view TensorBoard: tensorboard --logdir {FINAL_TENSORBOARD_PATH}") 