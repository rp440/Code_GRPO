import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel, LoraConfig, get_peft_model
from google.colab import drive
import os
import shutil
import ast
from datetime import datetime
from datasets import load_dataset # Changed from Dataset, IterableDataset
from huggingface_hub import login
from trl import GRPOConfig, GRPOTrainer
import random

# --- 0. Hugging Face Login ---
try:
    login_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if login_token:
        login(token=login_token, add_to_git_credential=False)
        print("Hugging Face Hub login successful.")
    else:
        print("HUGGING_FACE_HUB_TOKEN not found.")
except Exception as e:
    print(f"Hugging Face Hub login attempt issue: {e}")

# --- 1. Mount Google Drive & Paths ---
DRIVE_MOUNT_PATH = '/content/drive'
MODEL_SAVE_PARENT_DIR_DRIVE = os.path.join(DRIVE_MOUNT_PATH, "MyDrive", "Matmul_GPRO_Finetuned_JSONL")
DATASET_PATH = "/content/matrix_io_data_for_grpo.jsonl" # Your specified dataset path

try:
    print("Mounting Google Drive...")
    drive.mount(DRIVE_MOUNT_PATH, force_remount=True)
    print(f"Google Drive mounted.")
    os.makedirs(MODEL_SAVE_PARENT_DIR_DRIVE, exist_ok=True)
    print(f"Ensured save directory: {MODEL_SAVE_PARENT_DIR_DRIVE}")
    USE_DRIVE_FOR_SAVING = True
except Exception as e:
    print(f"Could not mount Google Drive: {e}")
    USE_DRIVE_FOR_SAVING = False

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
        op_match = re.match(r"^\s*([\w\[\],\s\.\d\-]+)\s*([*+])\s*([\w\[\],\s\.\d\-]+)\s*$", expression)
        assign_match = re.match(r"^\s*([\w\[\],\s\.\d\-]+)\s*$", expression)

        if op_match:
            op1_name = op_match.group(1).strip()
            operator = op_match.group(2).strip()
            op2_name = op_match.group(3).strip()
            val1 = self._get_value(op1_name)
            val2 = self._get_value(op2_name)
            if operator == '+': result = val1 + val2
            elif operator == '*': result = val1 * val2
            else: raise ValueError(f"Unsupported operator '{operator}' in expression: '{expression}'")
        elif assign_match:
            result = self._get_value(assign_match.group(1).strip())
        else:
            raise ValueError(f"Malformed expression: '{expression}' in DSL line: '{original_step_line}'")
        self.variables[target_var] = result

    def run_dsl_and_get_c(self, dsl_script_string):
        steps = dsl_script_string.strip().split('\n')
        for step in steps:
            clean_step = step.strip()
            if clean_step: self.execute_step(clean_step)
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

def _generate_random_2x2_matrix_for_inference(low=-99, high=99): # Simpler range for inference example
    return [[random.randint(low, high) for _ in range(2)] for _ in range(2)]

A_INFERENCE_MATRIX = _generate_random_2x2_matrix_for_inference()
B_INFERENCE_MATRIX = _generate_random_2x2_matrix_for_inference()
C_EXPECTED_INFERENCE_RESULT = manual_matrix_multiply_2x2(A_INFERENCE_MATRIX, B_INFERENCE_MATRIX)

# --- 3. GRPO Configuration and System Prompt ---
BASE_MODEL_NAME_FOR_FINETUNING = "Qwen/Qwen2.5-1.5B" # Changed to a more recent Qwen model
ADAPTER_PATH = "/content/gemma-text-to-sql/checkpoint-171" # Example path, adjust if needed
TRAINED_MODEL_DIR_NAME = f"{BASE_MODEL_NAME_FOR_FINETUNING.split('/')[-1]}-GRPO-MatMulDSL-JSONL"
LOCAL_TRAINED_MODEL_PATH = f"./{TRAINED_MODEL_DIR_NAME}"
if USE_DRIVE_FOR_SAVING:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    DRIVE_TRAINED_MODEL_PATH = os.path.join(MODEL_SAVE_PARENT_DIR_DRIVE, f"{TRAINED_MODEL_DIR_NAME}_{timestamp}")
else:
    DRIVE_TRAINED_MODEL_PATH = None

SYSTEM_MESSAGE = """You are an AI assistant. Your ONLY task is to output the COMPLETE sequence of VALID DSL lines to perform a 2x2 matrix multiplication (C = A * B), with each DSL line on a new line. The solution MUST use a total of 7 multiplications or fewer. The input matrices A and B will have integer elements.

STRICT RULES:

Adhere strictly to the DSL formats defined below for each line in the sequence.
NO PYTHON code. NO EXPLANATIONS. Output only the DSL lines.
The generated sequence must correctly implement a 2x2 matrix multiplication algorithm (C = A * B) using 7 or fewer multiplications.
Use M-variables for intermediate products/terms and S-variables for sums of products before C assignment.
If you cannot determine a valid full DSL sequence, output ONLY: Error: Cannot determine full sequence.

DSL FORMAT DEFINITIONS:
Intermediate Vars (M, S): <VAR_TYPE><id> = <operand1> {+/*} <operand2> OR <VAR_TYPE><id> = <operand>
Output Matrix: C[<row>,<col>] = <S_or_M_operand>
Operands: A[r,c], B[r,c], M<id>, S<id>. Indices 0 or 1.
Goal: Calculate C = A * B using <= 7 multiplications."""

# This will be complemented by specific matrix info if available in the dataset's 'user_query' field
DEFAULT_USER_PROMPT_FOR_DSL_GENERATION = "Generate the DSL script to calculate C = A * B for the given 2x2 matrices, using 7 or fewer multiplications."

# --- 4. Dataset Preparation from JSONL ---
def preprocess_jsonl_data(item):
    """
    Prepares a single item from the JSONL dataset.
    Assumes item might have 'matrix_A_list', 'matrix_B_list' or 'A_matrix_str', 'B_matrix_str'.
    Also looks for 'user_query'.
    """
    matrix_a = None
    matrix_b = None

    # Try to get matrices as lists first
    if "matrix_A_list" in item and "matrix_B_list" in item:
        try:
            matrix_a = item["matrix_A_list"]
            matrix_b = item["matrix_B_list"]
            if not (isinstance(matrix_a, list) and isinstance(matrix_b, list)): # Basic check
                matrix_a, matrix_b = None, None # Fallback if not list
        except: # Catch any error during access/conversion
            matrix_a, matrix_b = None, None
            
    # Fallback: Try to get matrices as strings and parse them
    if matrix_a is None and "A_matrix_str" in item:
        try:
            matrix_a = ast.literal_eval(item["A_matrix_str"])
        except:
            print(f"Warning: Could not parse A_matrix_str: {item.get('A_matrix_str')}")
            matrix_a = [[0,0],[0,0]] # Default to avoid crash, but this data point will be poor.
    if matrix_b is None and "B_matrix_str" in item:
        try:
            matrix_b = ast.literal_eval(item["B_matrix_str"])
        except:
            print(f"Warning: Could not parse B_matrix_str: {item.get('B_matrix_str')}")
            matrix_b = [[0,0],[0,0]]

    # If matrices couldn't be loaded, this data point is problematic for the current reward
    if matrix_a is None or matrix_b is None:
        print(f"Error: Could not determine matrix A or B for dataset item: {item}. Using placeholder matrices.")
        # Provide some default so it doesn't crash, but reward will be meaningless for this sample
        matrix_a = [[1,1],[1,1]]
        matrix_b = [[1,1],[1,1]]
        # Consider raising an error or filtering these items if they are common

    # Calculate expected C
    try:
        expected_c = manual_matrix_multiply_2x2(matrix_a, matrix_b)
    except Exception as e:
        print(f"Error calculating expected_C for A={matrix_a}, B={matrix_b}: {e}. Using placeholder C.")
        expected_c = [[0,0],[0,0]]


    user_content = item.get("user_query", DEFAULT_USER_PROMPT_FOR_DSL_GENERATION)
    # Optionally, append matrix info to user prompt if not already there, for LLM context
    # if "A=" not in user_content and "B=" not in user_content: # Basic check
    #    user_content += f" (A={str(matrix_a)}, B={str(matrix_b)})"


    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_content},
        ],
        "A_matrix_str": str(matrix_a),
        "B_matrix_str": str(matrix_b),
        "expected_C_str": str(expected_c),
        # "target_dsl_solution": item.get("target_dsl_solution", "") # If you have target DSLs
    }

print(f"Loading dataset from: {DATASET_PATH}")
try:
    # Check if dataset file actually exists
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset file not found: {DATASET_PATH}")
        print("Creating a minimal sample dataset for testing purposes...")
        
        # Create a simple sample dataset
        sample_data = [
            {
                "matrix_A_list": [[1, 2], [3, 4]],
                "matrix_B_list": [[5, 6], [7, 8]],
                "user_query": "Generate the DSL script to calculate C = A * B for the given 2x2 matrices, using 7 or fewer multiplications."
            },
            {
                "matrix_A_list": [[2, 1], [1, 2]],
                "matrix_B_list": [[3, 0], [1, 3]],
                "user_query": "Generate the DSL script to calculate C = A * B for the given 2x2 matrices, using 7 or fewer multiplications."
            }
        ]
        
        import json
        with open(DATASET_PATH, 'w') as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\n')
        print(f"Created sample dataset at: {DATASET_PATH}")
    
    raw_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    # Filter out problematic data if necessary, e.g., if matrices couldn't be parsed.
    # For now, preprocess_jsonl_data uses defaults if parsing fails.
    train_dataset_for_grpo = raw_dataset.map(
        preprocess_jsonl_data,
        # batched=True, # Can be faster if preprocess_jsonl_data handles batches
        # num_proc=4 # If dataset is large
    )
    # Remove columns not needed by GRPOTrainer or custom reward function, if any were left from raw load
    # GRPOTrainer with remove_unused_columns=False will keep A_matrix_str etc.
    # columns_to_keep = ["prompt", "A_matrix_str", "B_matrix_str", "expected_C_str"]
    # columns_to_remove = [col for col in train_dataset_for_grpo.column_names if col not in columns_to_keep]
    # if columns_to_remove:
    #    train_dataset_for_grpo = train_dataset_for_grpo.remove_columns(columns_to_remove)

    print(f"Processed dataset. Number of samples: {len(train_dataset_for_grpo)}")
    if len(train_dataset_for_grpo) > 0:
        print(f"First processed sample: {train_dataset_for_grpo[0]}")
    else:
        raise ValueError("Dataset is empty after processing. Check JSONL path and content.")
except Exception as e:
    print(f"Failed to load or process dataset from {DATASET_PATH}: {e}")
    print("Exiting due to dataset error.")
    exit()


# --- 5. Model Loading and PEFT Setup ---
print(f"Loading base model for fine-tuning: {BASE_MODEL_NAME_FOR_FINETUNING}")
model_for_finetuning = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME_FOR_FINETUNING,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

# Load existing finetuned adapter instead of creating fresh LoRA
print(f"Loading existing adapter from: {ADAPTER_PATH}")
try:
    model_peft = PeftModel.from_pretrained(model_for_finetuning, ADAPTER_PATH)
    print("Existing adapter loaded successfully.")
except Exception as e:
    print(f"Error loading adapter from {ADAPTER_PATH}: {e}")
    print("Falling back to fresh LoRA configuration...")
    # Fallback to fresh LoRA if adapter loading fails
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Updated for Qwen2.5 architecture
        bias="none"
    )
    model_peft = get_peft_model(model_for_finetuning, lora_config)

model_peft.print_trainable_parameters()

tokenizer_for_training = AutoTokenizer.from_pretrained(BASE_MODEL_NAME_FOR_FINETUNING, trust_remote_code=True)
if tokenizer_for_training.pad_token is None:
    tokenizer_for_training.pad_token = tokenizer_for_training.eos_token
    model_peft.config.pad_token_id = tokenizer_for_training.eos_token_id
if tokenizer_for_training.padding_side == 'right':
    tokenizer_for_training.padding_side = 'left'

# --- 6. Reward Function for GRPO ---
_num_generations_per_prompt_for_reward = 2

def matrix_dsl_reward(completions, prompts=None, completion_ids=None, **kwargs):
    global _num_generations_per_prompt_for_reward
    A_matrix_str_list = kwargs["A_matrix_str"]
    B_matrix_str_list = kwargs["B_matrix_str"]
    expected_C_str_list = kwargs["expected_C_str"]
    rewards = []

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
            print(f"Reward function: Error parsing matrices for completion {i}: {e}")
            rewards.append(-20.0) # Severe penalty for data error in reward
            continue

        # Ensure dsl_script_raw_content is a string - handle chat format from GRPO
        if isinstance(dsl_script_raw_content, list):
            if len(dsl_script_raw_content) == 1:
                item = dsl_script_raw_content[0]
                if isinstance(item, dict) and 'role' in item and 'content' in item:
                    # Chat format: [{'role': 'assistant', 'content': '...'}]
                    processed_dsl_script = item['content']
                elif isinstance(item, str):
                    # List containing a single string
                    processed_dsl_script = item
                else:
                    print(f"Warning: Reward function received unhandled list item at index {i}: {item}. Skipping.")
                    rewards.append(-50.0)
                    continue
            else:
                print(f"Warning: Reward function received list with {len(dsl_script_raw_content)} items at index {i}. Skipping.")
                rewards.append(-50.0)
                continue
        elif isinstance(dsl_script_raw_content, dict) and 'role' in dsl_script_raw_content and 'content' in dsl_script_raw_content:
            # Single chat format: {'role': 'assistant', 'content': '...'}
            processed_dsl_script = dsl_script_raw_content['content']
        elif isinstance(dsl_script_raw_content, str):
            # Already a string
            processed_dsl_script = dsl_script_raw_content
        else:
            print(f"Warning: Reward function received unhandled completion type at index {i}: {type(dsl_script_raw_content)}. Skipping.")
            rewards.append(-50.0)
            continue

        # Robust construction of tokens to remove
        temp_tokens_to_remove = ["<|im_end|>", "<|endoftext|>", "<|file_separator|>"]
        if hasattr(tokenizer_for_training, 'all_special_tokens') and tokenizer_for_training.all_special_tokens is not None:
            valid_special_tokens = [
                str(t) for t in tokenizer_for_training.all_special_tokens
                if t is not None and isinstance(t, str) and t # Ensure t is a non-empty string
            ]
            temp_tokens_to_remove.extend(valid_special_tokens)
        
        # Add eos_token if not already covered and is a string
        if tokenizer_for_training.eos_token and isinstance(tokenizer_for_training.eos_token, str):
            temp_tokens_to_remove.append(tokenizer_for_training.eos_token)

        unique_tokens_to_remove = list(set(t for t in temp_tokens_to_remove if t)) # Ensure unique and non-empty

        # At this point, processed_dsl_script is guaranteed to be a string.
        # And unique_tokens_to_remove contains only non-empty strings.
        for token_str in unique_tokens_to_remove:
            processed_dsl_script = processed_dsl_script.replace(token_str, "")

        lines = processed_dsl_script.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        final_dsl_script = "\n".join(cleaned_lines)

        if not final_dsl_script or final_dsl_script.strip().lower() == "error: cannot determine full sequence.":
            rewards.append(-10.0)
            continue
            
        base_score = 0
        num_multiplications = 0
        try:
            for line in final_dsl_script.split('\n'):
                if re.search(r"=\s*[\w\[\],\s\.\d\-]+\s*\*\s*[\w\[\],\s\.\d\-]+", line.strip()):
                    num_multiplications += 1
            
            # Re-initialize DSLExecutor for each script with the correct A and B
            executor = DSLExecutor(A, B)
            C_dsl = executor.run_dsl_and_get_c(final_dsl_script)

            if C_dsl == expected_C:
                base_score = 10.0 # Positive reward for correctness
            else:
                base_score = -5.0 # Penalty for incorrect result
        except ValueError as e: # Catch DSL parsing/execution errors
            # print(f"Reward function: DSL execution error for script:\n---\n{final_dsl_script}\n---\nError: {e}")
            base_score = -7.0 # Penalty for invalid DSL structure
        except Exception as e: # Catch any other unexpected error during execution
            # print(f"Reward function: Unexpected error during DSL execution: {e} for script:\n---\n{final_dsl_script}\n---")
            base_score = -8.0
            num_multiplications += 5 # Penalize unexpected errors more
        
        # Reward calculation: correctness minus complexity (number of multiplications)
        # We want fewer multiplications.
        reward_value = base_score - float(num_multiplications)
        rewards.append(reward_value)
        
    return rewards

# --- 7. Training Arguments and GRPOTrainer ---
print("Configuring training arguments for GRPO...")
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
use_fp16 = torch.cuda.is_available() and not use_bf16

training_args_grpo = GRPOConfig(
    output_dir=LOCAL_TRAINED_MODEL_PATH,
    learning_rate=2e-5, # Tunable
    remove_unused_columns=False, # CRITICAL for reward function to access dataset columns
    gradient_accumulation_steps=4,
    num_train_epochs=1, # Adjust based on dataset size and desired training
    bf16=use_bf16, fp16=use_fp16,
    per_device_train_batch_size=2, # Number of prompts for loss computation
    max_completion_length=350, # Max length of generated DSL
    num_generations=_num_generations_per_prompt_for_reward,
    max_prompt_length=600, # System prompt + user query (with matrices) can be long
    logging_steps=10, # Log more frequently
    save_strategy="epoch",
    report_to="tensorboard", # Enable TensorBoard
    push_to_hub=False,
    # chat_template=tokenizer_for_training.chat_template or tokenizer_for_training.default_chat_template,
    # GRPOTrainer will handle tokenization using the passed tokenizer.
)

model_peft.config.pad_token_id = tokenizer_for_training.pad_token_id

trainer_grpo = GRPOTrainer(
    model=model_peft,
    # tokenizer=tokenizer_for_training,  # Re-enable tokenizer - required for GRPO
    reward_funcs=[matrix_dsl_reward],
    args=training_args_grpo,
    train_dataset=train_dataset_for_grpo,
    # eval_dataset=eval_dataset_for_grpo, # Optional: if you have an eval set
)

print("Starting GRPO training...")
print(f"TensorBoard logs will be saved in: {LOCAL_TRAINED_MODEL_PATH}/runs")
print(f"To view TensorBoard, run: tensorboard --logdir={os.path.join(LOCAL_TRAINED_MODEL_PATH, 'runs')}") # Corrected path for launch command
trainer_grpo.train()
print("GRPO Training finished.")

# --- 8. Save Model ---
print(f"Saving fine-tuned model to {LOCAL_TRAINED_MODEL_PATH}...")
trainer_grpo.save_model(LOCAL_TRAINED_MODEL_PATH)
if USE_DRIVE_FOR_SAVING and DRIVE_TRAINED_MODEL_PATH:
    print(f"Copying model to Google Drive: {DRIVE_TRAINED_MODEL_PATH}...")
    try:
        if os.path.exists(DRIVE_TRAINED_MODEL_PATH): # shutil.copytree fails if dst exists
            shutil.rmtree(DRIVE_TRAINED_MODEL_PATH)
        shutil.copytree(LOCAL_TRAINED_MODEL_PATH, DRIVE_TRAINED_MODEL_PATH)
        print(f"Model copied to {DRIVE_TRAINED_MODEL_PATH}")
    except Exception as e:
        print(f"Error copying model to Drive: {e}")
else:
    print(f"Model saved locally at: {LOCAL_TRAINED_MODEL_PATH}")

# --- 9. Inference and Verification ---
print("\n--- Inference with GRPO Fine-tuned Model ---")
try:
    print(f"Loading fine-tuned model from: {LOCAL_TRAINED_MODEL_PATH}")
    inference_base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME_FOR_FINETUNING, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    inference_model = PeftModel.from_pretrained(inference_base_model, LOCAL_TRAINED_MODEL_PATH)
    inference_model = inference_model.merge_and_unload() # Merge for potentially faster inference
    
    # Load tokenizer from base model, not from LOCAL_TRAINED_MODEL_PATH (which might not have tokenizer files)
    inference_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME_FOR_FINETUNING, trust_remote_code=True)
    if inference_tokenizer.pad_token is None: inference_tokenizer.pad_token = inference_tokenizer.eos_token
    if inference_tokenizer.padding_side == 'right': inference_tokenizer.padding_side = 'left'
    print("Fine-tuned model loaded for inference.")
except Exception as e:
    print(f"Error loading fine-tuned model: {e}")
    print("Falling back to base model for inference...")
    try:
        inference_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME_FOR_FINETUNING, torch_dtype="auto", device_map="auto", trust_remote_code=True)
        inference_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME_FOR_FINETUNING, trust_remote_code=True)
        if inference_tokenizer.pad_token is None: inference_tokenizer.pad_token = inference_tokenizer.eos_token
        if inference_tokenizer.padding_side == 'right': inference_tokenizer.padding_side = 'left'
        print("Using base model for inference.")
    except Exception as fallback_e:
        print(f"Error loading base model for inference: {fallback_e}. Exiting.")
        exit()

text_gen_pipeline = pipeline("text-generation", model=inference_model, tokenizer=inference_tokenizer, device=0 if torch.cuda.is_available() else -1)

# Use the predefined A_INFERENCE_MATRIX and B_INFERENCE_MATRIX for testing
user_query_for_inference = DEFAULT_USER_PROMPT_FOR_DSL_GENERATION # Generic prompt
# Append specific matrix info to the prompt for the LLM, as it might not be in the generic prompt
user_query_for_inference += f" (Using A={str(A_INFERENCE_MATRIX)}, B={str(B_INFERENCE_MATRIX)})"


inference_chat_messages = [
    {"role": "system", "content": SYSTEM_MESSAGE},
    {"role": "user", "content": user_query_for_inference}
]
formatted_inference_prompt = inference_tokenizer.apply_chat_template(
    inference_chat_messages, tokenize=False, add_generation_prompt=True)

print(f"\nGenerating DSL for A={A_INFERENCE_MATRIX}, B={B_INFERENCE_MATRIX}")
print(f"Input prompt to model:\n{formatted_inference_prompt[:500]}...") # Print start of prompt

outputs = text_gen_pipeline(
    formatted_inference_prompt, max_new_tokens=350, do_sample=False, temperature=0.1, top_p=0.9,
    pad_token_id=inference_tokenizer.pad_token_id, eos_token_id=inference_tokenizer.eos_token_id)

generated_full_text = outputs[0]['generated_text']
assistant_reply_raw = generated_full_text
if generated_full_text.startswith(formatted_inference_prompt):
    assistant_reply_raw = generated_full_text[len(formatted_inference_prompt):].strip()
else: # Fallback for pipeline differences
    assistant_marker = "<|im_start|>assistant"
    last_occurrence_idx = generated_full_text.rfind(assistant_marker)
    if last_occurrence_idx != -1:
        start_of_reply_idx = generated_full_text.find("\n", last_occurrence_idx)
        if start_of_reply_idx != -1:
            assistant_reply_raw = generated_full_text[start_of_reply_idx+1:].strip()

tokens_to_clean = ["<|im_end|>", "<|endoftext|>"] + ([inference_tokenizer.eos_token] if inference_tokenizer.eos_token else [])
unique_tokens_to_clean = list(set(t for t in tokens_to_clean if t)) # Ensure unique and not None
for token in unique_tokens_to_clean:
    if assistant_reply_raw.endswith(token):
        assistant_reply_raw = assistant_reply_raw[:-len(token)].strip()

print(f"\n--- Raw Assistant's Reply (DSL Script) ---\n{assistant_reply_raw}\n------------------------------------")
print("\n--- Verifying Generated DSL ---")
lines = assistant_reply_raw.split('\n')
cleaned_lines = [line.strip() for line in lines if line.strip()]
final_generated_dsl = "\n".join(cleaned_lines)

if not final_generated_dsl or final_generated_dsl.strip().lower() == "error: cannot determine full sequence.":
    print("❌ FAILED: Model did not generate a valid DSL script or explicitly errored.")
else:
    try:
        executor = DSLExecutor(A_INFERENCE_MATRIX, B_INFERENCE_MATRIX) # Use the inference matrices
        C_generated = executor.run_dsl_and_get_c(final_generated_dsl)
        num_mults_generated = sum(1 for line in final_generated_dsl.split('\n') if re.search(r"=\s*[\w\[\],\s\.\d\-]+\s*\*\s*[\w\[\],\s\.\d\-]+", line.strip()))

        print(f"  Generated DSL executed. Multiplications: {num_mults_generated}")
        print(f"  Resulting C: {C_generated}")
        print(f"  Expected C:  {C_EXPECTED_INFERENCE_RESULT}")

        if C_generated == C_EXPECTED_INFERENCE_RESULT:
            print(f"✅ PASSED: Algorithmically correct.")
            if num_mults_generated <= 7: print(f"    Efficient: {num_mults_generated} multiplications (<= 7).")
            else: print(f"    Suboptimal: {num_mults_generated} multiplications (> 7).")
        else:
            print("❌ FAILED: Algorithmically INCORRECT.")
    except ValueError as e: print(f"❌ FAILED: Invalid DSL or execution error: {e}")
    except Exception as e: print(f"❌ FAILED: Unexpected verification error: {e}"); import traceback; traceback.print_exc()

print("\nScript finished.")