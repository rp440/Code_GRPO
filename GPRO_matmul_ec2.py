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

# --- 1. Paths Configuration ---
# Base directory for all outputs
BASE_OUTPUT_DIR = "/home/ec2-user/matmul_outputs"
MODEL_SAVE_DIR = os.path.join(BASE_OUTPUT_DIR, "models")
TENSORBOARD_LOGS_DIR = os.path.join(BASE_OUTPUT_DIR, "tensorboard_logs")
DATASET_PATH = "/home/ec2-user/matrix_io_data_for_grpo.jsonl"

# Create necessary directories
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOGS_DIR, exist_ok=True)

# --- 2. Core DSL Logic and Matrix Operations ---
[Previous DSLExecutor class and matrix operations code remains unchanged]

# --- 3. GRPO Configuration and System Prompt ---
BASE_MODEL_NAME_FOR_FINETUNING = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TRAINED_MODEL_DIR_NAME = f"{BASE_MODEL_NAME_FOR_FINETUNING.split('/')[-1]}-GRPO-MatMulDSL-JSONL"
LOCAL_TRAINED_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, TRAINED_MODEL_DIR_NAME)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Configuration for checkpoint loading and fresh training
CHECKPOINT_PATH = "/home/ec2-user/previous_checkpoint"  # Update this path
NEW_LEARNING_RATE = 2e-5
LOAD_FROM_CHECKPOINT = True

# Training configuration for 4 Tesla T4 GPUs (15GB VRAM each)
EPOCHS = 1
BATCH_SIZE_PER_GPU = 2  # Increased from 1 to 2 since T4s have 15GB VRAM
GRAD_ACC_STEPS = 4
NUM_GPUS = 4  # Number of Tesla T4 GPUs

# Calculate total batch size across all GPUs
TOTAL_BATCH_SIZE = BATCH_SIZE_PER_GPU * NUM_GPUS * GRAD_ACC_STEPS

# Memory optimization settings for T4 GPUs
TORCH_DTYPE = torch.float16  # Use FP16 for T4 GPUs
USE_GRADIENT_CHECKPOINTING = True
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.1

model_config_desc = f"lr{NEW_LEARNING_RATE}_epochs{EPOCHS}_batch{TOTAL_BATCH_SIZE}_gradacc{GRAD_ACC_STEPS}_gpus{NUM_GPUS}_t4"
model_name = f"{TRAINED_MODEL_DIR_NAME}_{model_config_desc}_{timestamp}"
tensorboard_name = f"runs_{model_config_desc}_{timestamp}"

FINAL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, model_name)
FINAL_TENSORBOARD_PATH = os.path.join(TENSORBOARD_LOGS_DIR, tensorboard_name)

print(f"[SAVE CONFIG] Model will be saved to: {FINAL_MODEL_PATH}")
print(f"[SAVE CONFIG] TensorBoard logs will be saved to: {FINAL_TENSORBOARD_PATH}")
print(f"[GPU CONFIG] Using {NUM_GPUS} Tesla T4 GPUs with {BATCH_SIZE_PER_GPU} batch size per GPU")
print(f"[MEMORY CONFIG] Using {TORCH_DTYPE} precision with gradient checkpointing")

[Previous SYSTEM_MESSAGE and DEFAULT_USER_PROMPT_FOR_DSL_GENERATION remain unchanged]

# --- 4. Dataset Preparation from JSONL ---
[Previous dataset preparation code remains unchanged]

# --- 5. Model Loading and PEFT Setup ---
print(f"Loading base model: {BASE_MODEL_NAME_FOR_FINETUNING}")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME_FOR_FINETUNING,
    torch_dtype=TORCH_DTYPE,  # Use FP16 for T4 GPUs
    device_map="auto",
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
[Previous reward function code remains unchanged]

# --- 7. Training Arguments and GRPOTrainer ---
print("Configuring training arguments for GRPO...")
use_bf16 = False  # T4s don't support bfloat16
use_fp16 = True   # Use FP16 for T4 GPUs

training_args_grpo = GRPOConfig(
    output_dir=FINAL_MODEL_PATH,
    learning_rate=NEW_LEARNING_RATE,
    remove_unused_columns=False,
    gradient_accumulation_steps=GRAD_ACC_STEPS,
    num_train_epochs=EPOCHS,
    bf16=use_bf16,
    fp16=use_fp16,
    per_device_train_batch_size=BATCH_SIZE_PER_GPU,
    max_completion_length=8000,
    num_generations=_num_generations_per_prompt_for_reward,
    max_prompt_length=1000,
    logging_steps=5,
    save_strategy="steps",
    save_steps=100,
    logging_dir=FINAL_TENSORBOARD_PATH,
    report_to="tensorboard",
    push_to_hub=False,
    dataloader_drop_last=True,
    warmup_steps=5,
    # Distributed training settings
    ddp_find_unused_parameters=False,
    ddp_backend="nccl",
    local_rank=-1,  # Will be set by torch.distributed.launch
    # Memory optimization settings
    gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
    max_grad_norm=MAX_GRAD_NORM,
    warmup_ratio=WARMUP_RATIO,
    # T4-specific optimizations
    fp16_opt_level="O1",  # Mixed precision optimization level
    fp16_backend="auto",  # Let PyTorch choose the best backend
)

model_peft.config.pad_token_id = tokenizer_for_training.pad_token_id

# Validate dataset before training
print(f"Dataset validation:")
print(f"  - Dataset size: {len(train_dataset_for_grpo)}")
print(f"  - Dataset features: {train_dataset_for_grpo.features}")
if len(train_dataset_for_grpo) == 0:
    print("[ERROR] Dataset is empty!")
    exit()

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
[Previous inference code remains unchanged]

print(f"\n*** SCRIPT COMPLETED SUCCESSFULLY ***")
print(f"[SAVED] Model saved to: {FINAL_MODEL_PATH}")
print(f"[LOGS] TensorBoard logs: {FINAL_TENSORBOARD_PATH}")
print(f"[DISCOVERIES] Log: {DISCOVERIES_LOG_FILE}")
final_best_summary = _best_n_mults if _best_n_mults != float('inf') else 'None found'
print(f"[BEST SOLUTION] {final_best_summary} multiplications")
print(f"[TIP] To view TensorBoard: tensorboard --logdir {FINAL_TENSORBOARD_PATH}") 