"""
Model Utilities for GRPO Training

Contains model loading, PEFT setup, training configuration, and related utilities.
"""

import os
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from torch.optim import AdamW
from google.colab import drive
from huggingface_hub import login

from config import (
    BASE_MODEL_NAME_FOR_FINETUNING, LOAD_FROM_CHECKPOINT, CHECKPOINT_PATH,
    NEW_LEARNING_RATE, DRIVE_MOUNT_PATH, MODEL_SAVE_PARENT_DIR_DRIVE,
    TENSORBOARD_LOGS_DRIVE, LOCAL_TRAINED_MODEL_PATH
)


def setup_huggingface_login():
    """Setup Hugging Face Hub login"""
    try:
        login_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if login_token:
            login(token=login_token, add_to_git_credential=False)
            print("[SUCCESS] Hugging Face Hub login successful.")
        else:
            print("[WARNING] HUGGING_FACE_HUB_TOKEN not found.")
    except Exception as e:
        print(f"[ERROR] Hugging Face Hub login attempt issue: {e}")


def setup_google_drive():
    """Mount Google Drive and setup directories"""
    try:
        print("Mounting Google Drive...")
        drive.mount(DRIVE_MOUNT_PATH, force_remount=True)
        print(f"Google Drive mounted.")
        os.makedirs(MODEL_SAVE_PARENT_DIR_DRIVE, exist_ok=True)
        os.makedirs(TENSORBOARD_LOGS_DRIVE, exist_ok=True)
        print(f"Ensured save directories: {MODEL_SAVE_PARENT_DIR_DRIVE}, {TENSORBOARD_LOGS_DRIVE}")
        return True
    except Exception as e:
        print(f"Could not mount Google Drive: {e}")
        return False


def load_base_model_and_tokenizer():
    """Load base model and tokenizer"""
    print(f"Loading base model: {BASE_MODEL_NAME_FOR_FINETUNING}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME_FOR_FINETUNING,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME_FOR_FINETUNING, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.padding_side == 'right':
        tokenizer.padding_side = 'left'
    
    return base_model, tokenizer


def setup_peft_model(base_model):
    """Setup PEFT model with LoRA configuration"""
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
            model_peft = _create_fresh_lora_model(base_model)
    else:
        print("Creating fresh LoRA configuration...")
        model_peft = _create_fresh_lora_model(base_model)

    model_peft.print_trainable_parameters()
    return model_peft


def _create_fresh_lora_model(base_model):
    """Create a fresh LoRA model configuration"""
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    return get_peft_model(base_model, lora_config)


def setup_optimizer_and_reset_value_head(model_peft):
    """Setup fresh optimizer and reset value head parameters"""
    # Fresh optimizer with new learning rate
    optimizer = AdamW(model_peft.parameters(), lr=NEW_LEARNING_RATE)
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
    
    return optimizer


def setup_model_config(model_peft, tokenizer):
    """Setup model configuration for training"""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model_peft.config.pad_token_id = tokenizer.eos_token_id
    if tokenizer.padding_side == 'right':
        tokenizer.padding_side = 'left'
    
    model_peft.config.pad_token_id = tokenizer.pad_token_id


def save_model_to_drive(trainer, drive_model_path, drive_logs_path, local_tensorboard_dir):
    """Save model and logs to Google Drive"""
    print(f"Saving fine-tuned model to {LOCAL_TRAINED_MODEL_PATH}...")
    trainer.save_model(LOCAL_TRAINED_MODEL_PATH)

    if drive_model_path:
        print(f"\n=== SAVING TO GOOGLE DRIVE ===")
        print(f"Drive Model Path: {drive_model_path}")
        
        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(drive_model_path), exist_ok=True)
            
            if os.path.exists(drive_model_path):
                print(f"[INFO] Removing existing model at {drive_model_path}")
                shutil.rmtree(drive_model_path)
            
            print(f"[COPYING] Copying model to Google Drive...")
            shutil.copytree(LOCAL_TRAINED_MODEL_PATH, drive_model_path)
            
            # Verify the copy was successful
            if os.path.exists(drive_model_path):
                saved_files = os.listdir(drive_model_path)
                print(f"[SUCCESS] Model saved to Drive with {len(saved_files)} files")
                print(f"[SUCCESS] Final model path: {drive_model_path}")
            else:
                print(f"[ERROR] Model directory not found after copy operation")
                
        except Exception as e:
            print(f"[ERROR] Failed to copy model to Drive: {e}")
            print(f"[FALLBACK] Model remains available locally at: {LOCAL_TRAINED_MODEL_PATH}")
        
        # Copy TensorBoard logs to Drive
        if drive_logs_path and os.path.exists(local_tensorboard_dir):
            print(f"\n[COPYING] TensorBoard logs to Google Drive: {drive_logs_path}")
            try:
                os.makedirs(os.path.dirname(drive_logs_path), exist_ok=True)
                if os.path.exists(drive_logs_path):
                    shutil.rmtree(drive_logs_path)
                shutil.copytree(local_tensorboard_dir, drive_logs_path)
                print(f"[SUCCESS] TensorBoard logs copied to: {drive_logs_path}")
            except Exception as e:
                print(f"[ERROR] Error copying TensorBoard logs to Drive: {e}")
                print(f"[FALLBACK] TensorBoard logs remain available locally at: {local_tensorboard_dir}")
    else:
        print(f"[WARNING] Drive saving disabled - Model saved locally only at: {LOCAL_TRAINED_MODEL_PATH}")
        print(f"[WARNING] To enable Drive saving, ensure Google Drive is mounted and USE_DRIVE_FOR_SAVING=True")


def setup_tensorboard(actual_tensorboard_dir):
    """Setup TensorBoard in Colab environment"""
    try:
        # Load TensorBoard extension and start it
        get_ipython().run_line_magic('load_ext', 'tensorboard')
        get_ipython().run_line_magic('tensorboard', f'--logdir {actual_tensorboard_dir}')
        print(f"TensorBoard started successfully! Logs directory: {actual_tensorboard_dir}")
    except:
        print("Note: TensorBoard extension not loaded (not in notebook environment)")
        print(f"To manually start TensorBoard, run: %tensorboard --logdir {actual_tensorboard_dir}")


def get_tensorboard_directory():
    """Get TensorBoard logging directory"""
    local_tensorboard_dir = os.path.join(LOCAL_TRAINED_MODEL_PATH, "runs")
    os.makedirs(local_tensorboard_dir, exist_ok=True)
    return local_tensorboard_dir 