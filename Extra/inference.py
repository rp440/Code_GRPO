import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import os

# --- Configuration ---
BASE_MODEL_ID = "Qwen/Qwen2-0.5B-Instruct" # The original base model ID
# Path to the directory where your fine-tuned adapter and tokenizer were saved
FINETUNED_ADAPTER_DIR = "Qwen2-0.5B-GRPO-NuminaMath-Finetuned" # Or your actual TRAINED_MODEL_DIR

# System prompt used during fine-tuning (important for consistency)
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

# Optional: 4-bit loading configuration if you trained with it or want to save VRAM during inference
# from transformers import BitsAndBytesConfig
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16 # or torch.float16
# )

# --- Load Model and Tokenizer ---
print(f"Loading base model: {BASE_MODEL_ID}")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype="auto",  # Or torch.float16 explicitly for T4
    device_map="auto",   # Automatically uses available GPUs, or a single GPU, or CPU
    # quantization_config=bnb_config, # Uncomment if using 4-bit loading for the base model
)
print(f"Base model loaded on: {base_model.device}")


print(f"Loading fine-tuned LoRA adapter from: {FINETUNED_ADAPTER_DIR}")
# Load the LoRA adapter and apply it to the base model
inference_model = PeftModel.from_pretrained(base_model, FINETUNED_ADAPTER_DIR)
print("LoRA adapter loaded.")

# Optional: Merge LoRA weights into the base model for potentially faster inference.
# This replaces the PeftModel with a standard CausalLM model with merged weights.
# You'll need enough VRAM to hold the full merged model.
# print("Merging LoRA adapter into the base model...")
# inference_model = inference_model.merge_and_unload()
# print("Adapter merged.")

print(f"Loading tokenizer from: {FINETUNED_ADAPTER_DIR}")
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_ADAPTER_DIR)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# --- Create Generation Pipeline ---
# The device for the pipeline will be handled by the model's device_map
print("Creating text-generation pipeline...")
gen_pipeline = pipeline(
    'text-generation',
    model=inference_model,
    tokenizer=tokenizer,
    # device=0 # Not strictly necessary if device_map="auto" is used and works
)
print("Pipeline created.")

# --- Inference ---
prompts_for_testing = [
    "What is the capital of France?",
    "In 1988, a person's age was equal to the sum of the digits of their birth year. How old was this person?",
    "Write a short story about a robot who dreams of becoming a chef."
]

for user_prompt_text in prompts_for_testing:
    print(f"\nUser Prompt: {user_prompt_text}")

    messages_for_inference = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt_text},
    ]

    # Apply the chat template
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        formatted_model_input = tokenizer.apply_chat_template(
            messages_for_inference,
            tokenize=False,
            add_generation_prompt=True  # Crucial for instruct models
        )
    else:
        # Fallback if no chat template (should not happen for Qwen2-Instruct)
        print("Warning: Tokenizer does not have a chat template. Using basic formatting.")
        formatted_model_input = f"{SYSTEM_PROMPT}\nUser: {user_prompt_text}\nAssistant:"

    print(f"--- Formatted Input to Model ---\n{formatted_model_input}\n-------------------------------")

    # Generate text
    outputs = gen_pipeline(
        formatted_model_input,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        # eos_token_id=tokenizer.eos_token_id, # Pipeline usually handles this
        # pad_token_id=tokenizer.pad_token_id # Pipeline usually handles this
    )

    generated_full_text = outputs[0]['generated_text']
    
    # Extract only the newly generated assistant's reply
    assistant_reply = generated_full_text
    if generated_full_text.startswith(formatted_model_input):
        assistant_reply = generated_full_text[len(formatted_model_input):].strip()
    else:
        # Fallback to find assistant's reply if direct stripping fails
        assistant_marker = None
        if "qwen2" in tokenizer.name_or_path.lower() or "qwen2" in BASE_MODEL_ID.lower() :
             assistant_marker = "<|im_start|>assistant"

        if assistant_marker:
            last_occurrence_idx = generated_full_text.rfind(assistant_marker)
            if last_occurrence_idx != -1:
                # Find the end of the marker (e.g., after the newline)
                end_of_marker_idx = generated_full_text.find("\n", last_occurrence_idx)
                if end_of_marker_idx != -1:
                     assistant_reply = generated_full_text[end_of_marker_idx+1:].strip()
    
    # Clean up trailing special tokens like <|im_end|> or <|endoftext|>
    final_eos_tokens = [tokenizer.eos_token]
    if hasattr(tokenizer, 'special_tokens_map') and 'im_end_token' in tokenizer.special_tokens_map:
        # For Qwen models, im_end_token is often the one to strip
        final_eos_tokens.append(tokenizer.special_tokens_map['im_end_token'])
    
    for tok in final_eos_tokens:
        if tok and assistant_reply.endswith(tok): # Check if tok is not None or empty
            assistant_reply = assistant_reply[:-len(tok)].strip()


    print(f"--- Assistant's Reply ---\n{assistant_reply}\n-------------------------")

print("\nInference script finished.")