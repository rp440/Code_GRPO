"""
Configuration Constants and System Prompts

Contains all configuration parameters, paths, and system prompts for the GRPO training.
"""

import os
from datetime import datetime

# --- Model Configuration ---
BASE_MODEL_NAME_FOR_FINETUNING = "Qwen/Qwen2.5-1.5B"
TRAINED_MODEL_DIR_NAME = f"{BASE_MODEL_NAME_FOR_FINETUNING.split('/')[-1]}-GRPO-MatMulDSL-JSONL"
LOCAL_TRAINED_MODEL_PATH = f"/content/{TRAINED_MODEL_DIR_NAME}"

# --- Checkpoint Configuration ---
CHECKPOINT_PATH = "/Qwen2.5-1.5B-GRPO-MatMulDSL-JSONL_lr2e-05_epochs2_batch5_gradacc12_20250608_202137"
NEW_LEARNING_RATE = 2e-5
LOAD_FROM_CHECKPOINT = True

# --- Training Hyperparameters ---
EPOCHS = 2
BATCH_SIZE = 5
GRAD_ACC_STEPS = 12

# --- Paths ---
DRIVE_MOUNT_PATH = '/content/drive'
MODEL_SAVE_PARENT_DIR_DRIVE = os.path.join(DRIVE_MOUNT_PATH, "MyDrive", "Matmul_GPRO_Finetuned_JSONL")
TENSORBOARD_LOGS_DRIVE = os.path.join(DRIVE_MOUNT_PATH, "MyDrive", "Matmul_GPRO_TensorBoard_Logs")
DATASET_PATH = "/content/matrix_io_data_for_grpo.jsonl"

# --- Reward Function Parameters ---
CORRECT_7_MULT_BONUS = 5.0
NEAR_MISS_PENALTY = -15.0
WEIRD_ANSWER_PENALTY = -20.0
TAG_BONUS = 0.1
EXPLORATION_SCALE = -10.0 / 1.59936e17
EXPLORATION_OFFSET = 6.0

# --- System Message ---
SYSTEM_MESSAGE = """You are an AI assistant specialized in generating Domain Specific Language (DSL) scripts for 2x2 matrix multiplication. You can provide explanations, but must wrap your DSL code in <DSL></DSL> tags.

EXAMPLE DSL OUTPUT FORMAT:
For matrices A=[[1,2],[3,4]] and B=[[5,6],[7,8]], a valid response would be:

I'll generate the DSL script for matrix multiplication:

<DSL> 
M1 = A[0,0] * B[0,0]
M2 = A[0,1] * B[1,0] 
S1 = M1 + M2 
C[0,0] = S1
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



This uses 7 multiplications, but can be optimized using techniques like Strassen's algorithm.

YOUR TASK:
Generate a DSL script that performs 2x2 matrix multiplication using 7 or fewer multiplications. You may provide explanations outside the DSL tags, but the actual code must be within <DSL></DSL> tags.

DSL SYNTAX RULES:
- M variables: Store multiplication results (e.g., M1 = A[0,0] * B[0,0])
- S variables: Store addition/subtraction results (e.g., S1 = M1 + M2)
- Matrix elements: A[row,col] and B[row,col] where row,col ∈ {0,1}
- Final output: C[row,col] = result
- Operations: + (addition), * (multiplication), - (subtraction)
- Variable assignment: VAR = expression

REQUIREMENTS:
- Use ≤7 multiplications total within the <DSL></DSL> tags
- Compute all four elements: C[0,0], C[0,1], C[1,0], C[1,1]
- Wrap DSL code in <DSL></DSL> tags
- You may add explanations outside the tags

If you cannot determine a valid sequence, output: Error: Cannot determine full sequence."""

DEFAULT_USER_PROMPT_FOR_DSL_GENERATION = "Generate the DSL script to calculate C = A * B for the given 2x2 matrices, using 7 or fewer multiplications."


def get_timestamp():
    """Get current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_model_config_desc():
    """Get model configuration description string"""
    return f"lr{NEW_LEARNING_RATE}_epochs{EPOCHS}_batch{BATCH_SIZE}_gradacc{GRAD_ACC_STEPS}"


def get_drive_paths():
    """Get drive paths for model and logs"""
    timestamp = get_timestamp()
    model_config_desc = get_model_config_desc()
    drive_model_name = f"{TRAINED_MODEL_DIR_NAME}_{model_config_desc}_{timestamp}"
    drive_logs_name = f"runs_{model_config_desc}_{timestamp}"
    
    drive_model_path = os.path.join(MODEL_SAVE_PARENT_DIR_DRIVE, drive_model_name)
    drive_logs_path = os.path.join(TENSORBOARD_LOGS_DRIVE, drive_logs_name)
    
    return drive_model_path, drive_logs_path 