# -*- coding: utf-8 -*-
# GPRO_3b_ec2_multi_gpu_optimized.py

# Installation instructions for EC2 (run these in your environment before executing the script):
# pip install -U trl peft math_verify transformers datasets huggingface_hub accelerate torch bitsandbytes
# Ensure CUDA is available and compatible with your torch version if using GPU.
# You will also need to configure Accelerate: `accelerate config`

import os
import re
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, PeftModel
from math_verify import LatexExtractionConfig, parse, verify
from trl import GRPOConfig, GRPOTrainer
import bitsandbytes as bnb # For potential 4-bit or 8-bit loading

# --- Configuration ---
try:
    login_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if login_token:
        login(token=login_token, add_to_git_credential=False)
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print("Hugging Face Hub login successful using HUGGING_FACE_HUB_TOKEN.")
    else:
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print("HUGGING_FACE_HUB_TOKEN not found. Attempting interactive login if needed by model download.")
except Exception as e:
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(f"Hugging Face Hub login attempt issue. Ensure you are logged in if private models are used. Error: {e}")

# Dataset and Model IDs
dataset_id = "AI-MO/NuminaMath-TIR"
model_id = "Qwen/Qwen2-3B-Instruct" # CHANGED TO 3B MODEL
TRAINED_MODEL_DIR = "Qwen2-3B-GRPO-NuminaMath-Finetuned" # Updated directory name

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

# --- Dataset Preparation ---
if os.environ.get("LOCAL_RANK", "0") == "0":
    print(f"Loading dataset: {dataset_id}")

# MODIFICATION: Use the full train split for better throughput.
# For initial testing, you might use "train[:10%]" or "train[:5%]" but for full training, use "train".
train_dataset_raw = load_dataset(dataset_id, split="train") # Use full dataset for better utilization

if os.environ.get("LOCAL_RANK", "0") == "0":
    print(f"Loaded raw train dataset: {train_dataset_raw}")

def make_conversation_and_prepare_columns(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
        "solution": example["solution"]
    }

# Using batched=True can help with large datasets and faster processing if map function is optimized.
# However, for 'map' with a single example per call, default is fine.
train_dataset = train_dataset_raw.map(make_conversation_and_prepare_columns, batched=False)
train_dataset = train_dataset.remove_columns(["problem", "messages"])

if os.environ.get("LOCAL_RANK", "0") == "0":
    print(f"Processed train dataset: {train_dataset}")
    print(f"Train dataset features: {train_dataset.features}")
    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty. Check data loading and slicing.")


# --- Model Loading and PEFT Setup ---
if os.environ.get("LOCAL_RANK", "0") == "0":
    print(f"Loading base model: {model_id}")

# MODIFICATION: Consider loading in 4-bit or 8-bit if you encounter OOM.
# For 3B on T4 (15GB), FP16 for the base model + sharded optimizer states
# for PEFT parameters should generally fit, but it can be tight.
# If you experience OOM, uncomment and use load_in_4bit=True or load_in_8bit=True
# load_in_4bit = False # Set to True if you face OOM
# load_in_8bit = False # Set to True if you face OOM

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto", # This should correctly pick float16 on T4 for model weights.
    # device_map="auto", # REMOVED for distributed training handled by Accelerate
    # load_in_4bit=load_in_4bit, # Uncomment if needed
    # load_in_8bit=load_in_8bit, # Uncomment if needed
)

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Expanded target modules for better performance
    # Typically, for Qwen2, you'd want to include more modules beyond just q_proj and v_proj for instruct models.
    # Common ones: k_proj, o_proj, gate_proj, up_proj, down_proj.
)

model = get_peft_model(model, lora_config)
if os.environ.get("LOCAL_RANK", "0") == "0":
    print("Trainable parameters after PEFT setup:")
    model.print_trainable_parameters()

# --- Reward Functions ---
def format_reward(completions, **kwargs):
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [comp[0]["content"] for comp in completions if comp and comp[0] and "content" in comp[0]]
    rewards_list = [1.0 if re.match(pattern, content, re.DOTALL) else 0.0 for content in completion_contents]
    return rewards_list

trainer_args_global_ref = None # Global placeholder for training_args

def accuracy_reward(completions, **kwargs):
    solutions = kwargs["solution"]
    num_gens_per_prompt = 1
    if trainer_args_global_ref is not None:
        num_gens_per_prompt = trainer_args_global_ref.num_generations
    else:
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print("Warning: trainer_args_global_ref not available in accuracy_reward, num_generations might be default.")

    expanded_solutions = []
    for sol in solutions:
        expanded_solutions.extend([sol] * num_gens_per_prompt)

    completion_contents = [comp[0]["content"] for comp in completions if comp and comp[0] and "content" in comp[0]]
    rewards = []

    for content, solution_text in zip(completion_contents, expanded_solutions):
        gold_parsed = parse(solution_text, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            answer_parsed = parse(answer_text, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) != 0 and len(answer_parsed) != 0:
                try:
                    rewards.append(float(verify(answer_parsed, gold_parsed)))
                except Exception:
                    rewards.append(0.0)
            elif len(gold_parsed) == 0 and len(answer_parsed) == 0 :
                 rewards.append(1.0 if answer_text == solution_text else 0.0)
            elif len(gold_parsed) == 0 and len(answer_parsed) != 0 :
                rewards.append(0.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards

# --- Training Arguments and Trainer ---
if os.environ.get("LOCAL_RANK", "0") == "0":
    print("Configuring training arguments...")

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
use_fp16 = torch.cuda.is_available() and not use_bf16 # T4 supports FP16 but not BF16

# MODIFICATIONS FOR OPTIMIZATION:
training_args = GRPOConfig(
    output_dir=TRAINED_MODEL_DIR,
    learning_rate=1e-5,
    remove_unused_columns=False,

    # Batch size and accumulation strategy:
    # Try to increase per_device_train_batch_size while keeping grad_acc_steps manageable.
    # For 3B on 15GB T4s, this is the most crucial part for VRAM.
    # Start conservatively (e.g., 1) and increase if it fits.
    # Effective batch size = per_device_train_batch_size * num_gpus * gradient_accumulation_steps
    per_device_train_batch_size=1,  # Start with 1 for 3B model on T4s, increase if OOM doesn't occur.
    gradient_accumulation_steps=16, # Effective batch size: 1 * 4 * 16 = 64. A good starting point.
                                    # Can reduce to 8 (eff=32) if memory allows and you want more frequent updates.

    num_train_epochs=10,
    bf16=use_bf16,
    fp16=use_fp16, # This should be True for T4s
    
    # MODIFICATION: Increase dataloader_num_workers to utilize CPU cores for data loading and preprocessing.
    # This is KEY for keeping GPUs busy. Use os.cpu_count() or a sensible number (e.g., 8-16).
    dataloader_num_workers=os.cpu_count(), # Maximize CPU core usage for data loading.

    # Parameters controlling data preprocessing and generation during training
    max_completion_length=128,
    num_generations=2,
    max_prompt_length=256,

    # Reporting and saving
    report_to="tensorboard",
    logging_steps=10,
    save_strategy="epoch",
    push_to_hub=False,
)
trainer_args_global_ref = training_args # Assign to global ref

tokenizer_for_training = AutoTokenizer.from_pretrained(model_id)
if tokenizer_for_training.pad_token_id is None:
    tokenizer_for_training.pad_token_id = tokenizer_for_training.eos_token_id
    if model.config.pad_token_id is None:
         model.config.pad_token_id = tokenizer_for_training.eos_token_id

trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer_for_training,
    reward_funcs=[format_reward, accuracy_reward],
    config=training_args, # Corrected argument name
    train_dataset=train_dataset,
)

if os.environ.get("LOCAL_RANK", "0") == "0":
    print("Starting training...")
trainer.train()
if os.environ.get("LOCAL_RANK", "0") == "0":
    print("Training finished.")

# --- Save Model ---
if trainer.is_world_process_zero():
    print(f"Saving fine-tuned model (adapter) and tokenizer to {TRAINED_MODEL_DIR}...")
    trainer.save_model(TRAINED_MODEL_DIR)
    print(f"Model and tokenizer saved to {TRAINED_MODEL_DIR}.")

# --- Inference with the Fine-tuned Model ---
if trainer.is_world_process_zero():
    print("\n--- Starting Inference with the Fine-tuned Model (on main process) ---")

    print(f"Loading fine-tuned model from {TRAINED_MODEL_DIR} for inference...")
    try:
        base_model_for_inference = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            # load_in_4bit=load_in_4bit, # Uncomment if needed
            # load_in_8bit=load_in_8bit, # Uncomment if needed
        )
        inference_model = PeftModel.from_pretrained(base_model_for_inference, TRAINED_MODEL_DIR)
        inference_tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_DIR)
        print("Successfully loaded fine-tuned PEFT model and tokenizer for inference.")

    except Exception as e:
        print(f"Error loading PEFT model for inference: {e}")
        print("Ensure TRAINED_MODEL_DIR contains valid adapter weights and config.")
        exit()

    if inference_tokenizer.pad_token_id is None:
        inference_tokenizer.pad_token_id = inference_tokenizer.eos_token_id

    gen_pipeline = pipeline(
        'text-generation',
        model=inference_model,
        tokenizer=inference_tokenizer,
    )

    prompts_for_testing = [
        "Explain Group Relative Policy Optimization in simple terms.",
        "In 1988, a person's age was equal to the sum of the digits of their birth year. How old was this person?"
    ]

    for user_prompt_text in prompts_for_testing:
        print(f"\nUser Prompt: {user_prompt_text}")
        messages_for_inference = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt_text},
        ]

        if hasattr(inference_tokenizer, "apply_chat_template") and inference_tokenizer.chat_template:
            formatted_model_input = inference_tokenizer.apply_chat_template(
                messages_for_inference,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            print("Warning: Tokenizer does not have a chat template. Using basic formatting.")
            formatted_model_input = f"{SYSTEM_PROMPT}\nUser: {user_prompt_text}\nAssistant:"

        print(f"--- Formatted Input to Model ---\n{formatted_model_input}\n-------------------------------")

        outputs = gen_pipeline(
            formatted_model_input,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        generated_full_text = outputs[0]['generated_text']
        assistant_reply = generated_full_text
        if generated_full_text.startswith(formatted_model_input):
            assistant_reply = generated_full_text[len(formatted_model_input):].strip()
        else:
            assistant_marker = None
            if "qwen2" in inference_tokenizer.name_or_path.lower():
                 assistant_marker = "<|im_start|>assistant"
            if assistant_marker:
                last_occurrence_idx = generated_full_text.rfind(assistant_marker)
                if last_occurrence_idx != -1:
                    end_of_marker_idx = generated_full_text.find("\n", last_occurrence_idx)
                    if end_of_marker_idx != -1:
                         assistant_reply = generated_full_text[end_of_marker_idx+1:].strip()

        if assistant_reply.endswith(inference_tokenizer.eos_token) or \
           (hasattr(inference_tokenizer, 'special_tokens_map') and \
            'im_end_token' in inference_tokenizer.special_tokens_map and \
            assistant_reply.endswith(inference_tokenizer.special_tokens_map['im_end_token'])):
            
            if assistant_reply.endswith(inference_tokenizer.eos_token):
                assistant_reply = assistant_reply[:-len(inference_tokenizer.eos_token)].strip()
            
            if hasattr(inference_tokenizer, 'special_tokens_map') and \
               'im_end_token' in inference_tokenizer.special_tokens_map and \
               assistant_reply.endswith(inference_tokenizer.special_tokens_map['im_end_token']):
               assistant_reply = assistant_reply[:-len(inference_tokenizer.special_tokens_map['im_end_token'])].strip()

        print(f"--- Assistant's Reply ---\n{assistant_reply}\n-------------------------")

if os.environ.get("LOCAL_RANK", "0") == "0":
    print("\nScript finished.")