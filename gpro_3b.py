# -*- coding: utf-8 -*-
# GPRO_3b_ec2.py

# Installation instructions for EC2 (run these in your environment before executing the script):
# pip install -U trl peft math_verify transformers datasets huggingface_hub accelerate torch
# Ensure CUDA is available and compatible with your torch version if using GPU.

import os
import re
import shutil # For managing directories if needed

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, PeftModel
from math_verify import LatexExtractionConfig, parse, verify
from trl import GRPOConfig, GRPOTrainer

# --- Configuration ---
# Hugging Face Login (recommended: set HUGGING_FACE_HUB_TOKEN environment variable)
try:
    login_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if login_token:
        login(token=login_token, add_to_git_credential=False) # add_to_git_credential=False for non-interactive
        print("Hugging Face Hub login successful using HUGGING_FACE_HUB_TOKEN.")
    else:
        print("HUGGING_FACE_HUB_TOKEN not found. Attempting interactive login if needed by model download.")
        # login() # Or just let operations that need it fail/prompt.
except Exception as e:
    print(f"Hugging Face Hub login attempt issue. Ensure you are logged in if private models are used. Error: {e}")

# Dataset and Model IDs
dataset_id = "AI-MO/NuminaMath-TIR"
model_id = "Qwen/Qwen2-0.5B-Instruct" # Base model for fine-tuning

# Output directory for the trained model
TRAINED_MODEL_DIR = "Qwen2-0.5B-GRPO-NuminaMath-Finetuned"

# System prompt for the task
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

# --- Dataset Preparation ---
print(f"Loading dataset: {dataset_id}")
# Using a smaller portion for demonstration. Adjust or remove slicing for full training.
# E.g., split=["train", "test"] for full dataset.
# For EC2, manage data size based on instance capabilities and time.
train_dataset_raw = load_dataset(dataset_id, split="train[:1%]") # Use a small slice for faster run
# test_dataset_raw = load_dataset(dataset_id, split="test[:1%]") # Test set not directly used by GRPOTrainer.train, but good for eval

print(f"Loaded raw train dataset: {train_dataset_raw}")

def make_conversation_and_prepare_columns(example):
    """Formats the problem into a chat prompt and ensures 'solution' is available."""
    return {
        "prompt": [ # This will be the main input for the GRPOTrainer
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
        "solution": example["solution"] # Keep solution for the accuracy reward function
    }

train_dataset = train_dataset_raw.map(make_conversation_and_prepare_columns)

# Remove original columns that are now incorporated into 'prompt' or are not needed
# 'problem' is in 'prompt', 'solution' is kept, 'messages' (original format) is not needed.
train_dataset = train_dataset.remove_columns(["problem", "messages"])

print(f"Processed train dataset: {train_dataset}")
print(f"Train dataset features: {train_dataset.features}")
if len(train_dataset) == 0:
    raise ValueError("Train dataset is empty. Check data loading and slicing.")


# --- Model Loading and PEFT Setup ---
print(f"Loading base model: {model_id}")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",  # Automatically selects precision (bfloat16, float16, or float32)
    device_map="auto",   # Automatically distributes model across available devices (GPU/CPU)
)

# LoRA Configuration
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"], # Modules to apply LoRA to. Check model architecture if changing.
)

model = get_peft_model(model, lora_config)
print("Trainable parameters after PEFT setup:")
model.print_trainable_parameters()

# --- Reward Functions ---
def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has the specified <think>...</think><answer>...</answer> format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    # Completions is a list of lists of dicts: [[{"role": "assistant", "content": "..."}], ...]
    # We need the content string.
    completion_contents = [comp[0]["content"] for comp in completions if comp and comp[0] and "content" in comp[0]]
    rewards_list = [1.0 if re.match(pattern, content, re.DOTALL) else 0.0 for content in completion_contents]
    return rewards_list

def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the answer in the completion matches the ground truth solution."""
    # 'solution' column from the dataset is passed via kwargs by GRPOTrainer if remove_unused_columns=False
    solutions = kwargs["solution"] # This will be a list of solutions, one for each prompt in the batch
    
    # Completions is List[List[Dict[str, str]]], where outer list corresponds to prompts,
    # inner list corresponds to `num_generations` per prompt.
    # We need to align generated completions with their original solutions.
    # GRPOTrainer calls reward_funcs with `completions` shaped (batch_size * num_generations, 1, dict)
    # and `solution` will be of length `batch_size`. We need to repeat/align `solution`.
    # The GRPOTrainer's documentation/source should clarify how kwargs are batched.
    # Assuming `solution` corresponds to the prompts, and we have `num_generations` for each.
    # Let's assume for now `completions` and `solutions` are aligned by GRPOTrainer or this needs adjustment.
    # The provided example implies completions and solutions are already aligned or broadcasted.
    # `completions` is (batch_size * num_generations), `solutions` is (batch_size)
    # Let `num_generations` be from `training_args`.
    
    num_gens_per_prompt = kwargs.get("num_generations", 1) # Get from training_args if possible or infer
    if "num_generations" not in kwargs and trainer_args: # Access from global if defined
        num_gens_per_prompt = trainer_args.num_generations

    expanded_solutions = []
    for sol in solutions:
        expanded_solutions.extend([sol] * num_gens_per_prompt)

    completion_contents = [comp[0]["content"] for comp in completions if comp and comp[0] and "content" in comp[0]]
    rewards = []

    for content, solution_text in zip(completion_contents, expanded_solutions):
        gold_parsed = parse(solution_text, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        # Extract from <answer> tag in generated content
        answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            answer_parsed = parse(answer_text, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) != 0 and len(answer_parsed) != 0:
                try:
                    rewards.append(float(verify(answer_parsed, gold_parsed)))
                except Exception:
                    rewards.append(0.0) # Error during verification
            elif len(gold_parsed) == 0 and len(answer_parsed) == 0 : # Both empty (e.g. non-math textual answer)
                 rewards.append(1.0 if answer_text == solution_text else 0.0) # Simple string match for non-parsable
            elif len(gold_parsed) == 0 and len(answer_parsed) != 0 : # Gold is empty, answer is not
                rewards.append(0.0)
            else: # Gold is not empty, answer is empty or unparsable
                rewards.append(0.0)
        else:
            rewards.append(0.0) # No <answer> tag
    return rewards

# --- Training Arguments and Trainer ---
print("Configuring training arguments...")
# Conditional mixed precision based on hardware support
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
use_fp16 = torch.cuda.is_available() and not use_bf16

# Global placeholder for training_args to be accessible by reward function if needed for num_generations
trainer_args = None

training_args = GRPOConfig(
    output_dir=TRAINED_MODEL_DIR,
    learning_rate=1e-5,       # Typical learning rate for fine-tuning
    remove_unused_columns=False, # Crucial for accessing 'solution' in accuracy_reward
    gradient_accumulation_steps=4, # Adjust based on VRAM. Original: 16. Reduced for wider compatibility.
    num_train_epochs=1,        # Number of training epochs
    bf16=use_bf16,             # Use bfloat16 if supported
    fp16=use_fp16,             # Use float16 if bfloat16 not supported but CUDA is available
    
    # per_device_train_batch_size: From TrainingArguments, defaults to 8.
    # This is the number of prompts processed for loss computation by one device.
    per_device_train_batch_size=2, # Adjust based on VRAM. Original: 8 (default). Reduced.

    # batch_size (for GRPO generation step): From GRPOConfig, defaults to 1.
    # Number of prompts to generate completions for simultaneously.
    # batch_size = 1, # Default in GRPOConfig

    # Parameters controlling data preprocessing and generation during training
    max_completion_length=128, # Max length of generated completions. Original: 64. Increased for math.
    num_generations=2,         # Number of completions to generate per prompt. Original: 4. Reduced.
    max_prompt_length=256,     # Max length of prompts. Original: 128. Increased.

    # Reporting and saving
    report_to="tensorboard",   # For EC2, consider "none" or ensure TensorBoard is set up.
    logging_steps=10,
    save_strategy="epoch",     # Save at the end of each epoch. Original: "steps"
    # save_steps=10,           # Used if save_strategy="steps". Original: 10
    # save_total_limit=1,      # Optional: limit the number of saved checkpoints
    push_to_hub=False,         # Set to True to push to Hugging Face Hub
    # hub_model_id="your-username/your-model-name", # Required if push_to_hub=True
)
trainer_args = training_args # Make it accessible globally for reward function

tokenizer_for_training = AutoTokenizer.from_pretrained(model_id)
if tokenizer_for_training.pad_token_id is None:
    tokenizer_for_training.pad_token_id = tokenizer_for_training.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id # Ensure model config also updated

trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer_for_training, # Pass tokenizer to GRPOTrainer
    reward_funcs=[format_reward, accuracy_reward],
    args=training_args,
    train_dataset=train_dataset,
)

print("Starting training...")
trainer.train()
print("Training finished.")

# --- Save Model ---
print(f"Saving fine-tuned model (adapter) and tokenizer to {TRAINED_MODEL_DIR}...")
trainer.save_model(TRAINED_MODEL_DIR) # Saves adapter & tokenizer
# tokenizer_for_training.save_pretrained(TRAINED_MODEL_DIR) # Trainer should do this
print(f"Model and tokenizer saved to {TRAINED_MODEL_DIR}.")

# Optional: Zip the model directory if needed for transfer
# shutil.make_archive(f"{TRAINED_MODEL_DIR}_archive", "zip", TRAINED_MODEL_DIR)
# print(f"Model archived to {TRAINED_MODEL_DIR}_archive.zip")


# --- Inference with the Fine-tuned Model ---
print("\n--- Starting Inference with the Fine-tuned Model ---")

# Load the fine-tuned model (base model + adapter) and tokenizer
# AutoModelForCausalLM.from_pretrained on a PEFT-saved directory
# should load the base model and apply the adapter.
print(f"Loading fine-tuned model from {TRAINED_MODEL_DIR} for inference...")
try:
    inference_model = AutoModelForCausalLM.from_pretrained(
        TRAINED_MODEL_DIR,
        torch_dtype="auto", # Use appropriate dtype
        device_map="auto",  # Load on available device
    )
    # Tokenizer should have been saved by trainer.save_model()
    inference_tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_DIR)
    print("Successfully loaded fine-tuned model and tokenizer for inference.")
except Exception as e:
    print(f"Error loading model directly from {TRAINED_MODEL_DIR}: {e}")
    print("Attempting fallback: loading base model and then attaching adapter...")
    base_model_for_inference = AutoModelForCausalLM.from_pretrained(
        model_id, # Base model ID
        torch_dtype="auto",
        device_map="auto",
    )
    inference_model = PeftModel.from_pretrained(base_model_for_inference, TRAINED_MODEL_DIR)
    # Optional: merge LoRA weights for potentially faster inference if not done automatically.
    # This replaces the model with the merged version.
    # inference_model = inference_model.merge_and_unload()
    inference_tokenizer = AutoTokenizer.from_pretrained(model_id) # Fallback to base tokenizer
    print("Fallback loading complete.")

# Ensure pad_token_id is set for the inference tokenizer
if inference_tokenizer.pad_token_id is None:
    inference_tokenizer.pad_token_id = inference_tokenizer.eos_token_id

# Determine device for pipeline (GPU if available, else CPU)
pipeline_device = 0 if torch.cuda.is_available() else -1

print(f"Creating text-generation pipeline on device: {'cuda:0' if pipeline_device == 0 else 'cpu'}")
gen_pipeline = pipeline(
    'text-generation',
    model=inference_model,
    tokenizer=inference_tokenizer,
    device=pipeline_device
)

# Test prompts
prompts_for_testing = [
    "Explain Group Relative Policy Optimization in simple terms.",
    "In 1988, a person's age was equal to the sum of the digits of their birth year. How old was this person?"
]

for user_prompt_text in prompts_for_testing:
    print(f"\nUser Prompt: {user_prompt_text}")

    # Format the prompt using the chat template, including the system prompt used during training
    messages_for_inference = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt_text},
    ]

    if inference_tokenizer.chat_template:
        formatted_model_input = inference_tokenizer.apply_chat_template(
            messages_for_inference,
            tokenize=False,
            add_generation_prompt=True  # Crucial for instruct models to know it's their turn
        )
    else:
        # Fallback if no chat template (should not happen for Qwen2-Instruct)
        print("Warning: Tokenizer does not have a chat template. Using basic formatting.")
        formatted_model_input = f"{SYSTEM_PROMPT}\nUser: {user_prompt_text}\nAssistant:"

    print(f"--- Formatted Input to Model ---\n{formatted_model_input}\n-------------------------------")

    # Generate text
    outputs = gen_pipeline(
        formatted_model_input,
        max_new_tokens=512,  # Max tokens for the generated response
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        #eos_token_id=inference_tokenizer.eos_token_id, # Pipeline usually handles this
        #pad_token_id=inference_tokenizer.pad_token_id # Pipeline usually handles this
    )

    generated_full_text = outputs[0]['generated_text']
    
    # Extract only the newly generated assistant's reply
    # The `formatted_model_input` is what we sent.
    # The `generated_full_text` contains this input + the model's generation.
    assistant_reply = generated_full_text
    if generated_full_text.startswith(formatted_model_input):
        assistant_reply = generated_full_text[len(formatted_model_input):].strip()
    else:
        # If the pipeline behaves differently or input string had subtle changes (e.g. whitespace)
        # try to find the last part based on assistant markers if possible.
        # For Qwen2, this would be after "<|im_start|>assistant\n"
        # This is a heuristic if the direct prefix strip fails.
        assistant_marker = None
        if "qwen2" in inference_tokenizer.name_or_path.lower():
             assistant_marker = "<|im_start|>assistant" # Don't include \n as it might vary slightly

        if assistant_marker:
            last_occurrence_idx = generated_full_text.rfind(assistant_marker)
            if last_occurrence_idx != -1:
                # Find the end of the marker (e.g., after the newline)
                end_of_marker_idx = generated_full_text.find("\n", last_occurrence_idx)
                if end_of_marker_idx != -1:
                     assistant_reply = generated_full_text[end_of_marker_idx+1:].strip()

    # Clean up trailing special tokens like <|im_end|>
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

print("\nScript finished.")