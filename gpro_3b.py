# -*- coding: utf-8 -*-
"""
gpro_ec2_train_py311.py

Script to train Qwen2-0.5B with GRPO on an EC2 instance with multiple GPUs.
This script is compatible with Python 3.11.x environments.
Ensure all dependencies (transformers, peft, trl, datasets, torch, accelerate, math-verify)
are installed in your Python 3.11 environment and are compatible with your CUDA version.
"""

import os
import re
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from math_verify import LatexExtractionConfig, parse, verify # Ensure 'math-verify' is installed
from trl import GRPOConfig, GRPOTrainer

# Helper function to print only on the main process in DDP
def print_on_main_process(message):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(message)

def main():
    # --- Configuration ---
    dataset_id = "AI-MO/NuminaMath-TIR"
    model_id = "Qwen/Qwen2-0.5B-Instruct"
    
    # Training parameters - adjust as needed for your EC2 instance capabilities
    # These are smaller for a quicker demonstration. Increase for full training.
    TRAIN_DATASET_SLICE = "train[:1%]" # e.g., "train[:5%]" or "train" for full
    TEST_DATASET_SLICE = "test[:1%]"   # e.g., "test[:5%]" or "test" for full
    NUM_TRAIN_EPOCHS_RUN1 = 1
    NUM_TRAIN_EPOCHS_RUN2 = 1
    PER_DEVICE_BATCH_SIZE_RUN1 = 1
    GRAD_ACCUM_STEPS_RUN1 = 4
    PER_DEVICE_BATCH_SIZE_RUN2 = 2 
    GRAD_ACCUM_STEPS_RUN2 = 8      

    # --- 1. Load Dataset ---
    print_on_main_process("Loading dataset...")
    try:
        train_dataset, test_dataset = load_dataset(dataset_id, split=[TRAIN_DATASET_SLICE, TEST_DATASET_SLICE])
    except Exception as e:
        print_on_main_process(f"Failed to load specified dataset slice, trying with a smaller one: {e}")
        train_dataset, test_dataset = load_dataset(dataset_id, split=["train[:100]", "test[:100]"])

    print_on_main_process(f"Train dataset size: {len(train_dataset)}")
    print_on_main_process(f"Test dataset size: {len(test_dataset)}")

    # --- 2. Define System Prompt and Conversation Formatting ---
    SYSTEM_PROMPT = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
        "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think><answer> answer here </answer>"
    )

    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
            "solution": example["solution"] # Keep solution for reward calculation
        }

    # Inspect columns before mapping
    print_on_main_process(f"Columns in train_dataset before map: {train_dataset.column_names}")
    print_on_main_process(f"Columns in test_dataset before map: {test_dataset.column_names}")

    # The make_conversation function uses 'problem' and 'solution'.
    # 'messages' is the only original column not directly used by make_conversation and can be removed.
    # 'grade', 'type', 'source' are NOT top-level columns in this specific dataset.
    columns_to_remove_from_original = ["messages"] 

    train_dataset = train_dataset.map(
        make_conversation,
        batched=True, 
        remove_columns=columns_to_remove_from_original
    )
    test_dataset = test_dataset.map(
        make_conversation,
        batched=True,
        remove_columns=columns_to_remove_from_original
    )
    
    # Inspect columns after mapping
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print_on_main_process(f"\nProcessed train_dataset sample (after map):")
        print_on_main_process(train_dataset[0])
        print_on_main_process(f"Columns in train_dataset after map: {train_dataset.column_names}")
        print_on_main_process(f"Columns in test_dataset after map: {test_dataset.column_names}")


    # --- 3. Load Tokenizer and Model (for DDP, load on CPU first) ---
    print_on_main_process(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
        print_on_main_process("Tokenizer pad_token set to eos_token.")

    print_on_main_process(f"Loading model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto", 
    )

    # --- 4. Configure PEFT (LoRA) ---
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"], # Common for Qwen models, verify if different for 0.5B
    )
    model = get_peft_model(model, lora_config)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        model.print_trainable_parameters()

    # --- 5. Define Reward Functions ---
    def format_reward(completions, **kwargs):
        pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
        # Ensure completions are strings and handle None
        completion_contents = [str(comp[0]["content"]) if comp and len(comp) > 0 and comp[0].get("content") is not None else "" for comp in completions]
        matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    def accuracy_reward(completions, **kwargs):
        solutions = kwargs["solution"] # This comes from the dataset items
        completion_contents = [str(comp[0]["content"]) if comp and len(comp) > 0 and comp[0].get("content") is not None else "" for comp in completions]
        rewards = []
        for content, solution_text in zip(completion_contents, solutions):
            gold_parsed = parse(str(solution_text), extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            answer_parsed = parse(str(content), extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) != 0:
                try:
                    rewards.append(float(verify(answer_parsed, gold_parsed)))
                except Exception:
                    rewards.append(0.0)
            else: # If gold solution is empty/unparsable
                if len(answer_parsed) == 0 : # If answer is also empty/unparsable
                     rewards.append(1.0)
                else: # Gold empty, answer not
                     rewards.append(0.0)
        return rewards

    # --- 6. First Training Configuration and Run ---
    print_on_main_process("\n--- Configuring First Training Run ---")
    output_dir_run1 = "Qwen2-0.5B-GRPO-EC2-run1"
    training_args_1 = GRPOConfig(
        output_dir=output_dir_run1,
        learning_rate=1e-5,
        remove_unused_columns=False, # Important: GRPOTrainer needs 'solution' from dataset for accuracy_reward
        gradient_accumulation_steps=GRAD_ACCUM_STEPS_RUN1,
        num_train_epochs=NUM_TRAIN_EPOCHS_RUN1,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE_RUN1,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        max_completion_length=64,
        num_generations=2,
        max_prompt_length=128,
        report_to=["tensorboard"] if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) else "none",
        logging_steps=10, 
        push_to_hub=False,
        save_strategy="steps",
        save_steps=50, 
        tokenizer=tokenizer,
        ddp_find_unused_parameters=False 
    )

    trainer_1 = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, accuracy_reward],
        args=training_args_1,
        train_dataset=train_dataset,
        # GRPOTrainer gets tokenizer from args if not passed directly
    )
    
    print_on_main_process("--- Starting First Training Run ---")
    trainer_1.train()
    print_on_main_process("--- First Training Run Completed ---")
    
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        trainer_1.save_model(output_dir_run1)
        print_on_main_process(f"Model from first run saved to {output_dir_run1}")


    # --- 7. Second Training Configuration (Optimized) and Run ---
    print_on_main_process("\n--- Configuring Second Training Run (Optimized) ---")
    output_dir_run2 = "Qwen2-0.5B-GRPO-EC2-run2-final"
    training_args_2 = GRPOConfig(
        output_dir=output_dir_run2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE_RUN2,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS_RUN2,
        num_generations=2, 
        max_prompt_length=128,
        max_completion_length=64,
        logging_steps=20, 
        save_strategy="epoch",
        learning_rate=1e-5, 
        num_train_epochs=NUM_TRAIN_EPOCHS_RUN2,
        remove_unused_columns=False, # Keep 'solution'
        report_to=["tensorboard"] if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) else "none",
        push_to_hub=False,
        tokenizer=tokenizer,
        ddp_find_unused_parameters=False
    )

    trainer_2 = GRPOTrainer(
        model=model, 
        reward_funcs=[format_reward, accuracy_reward],
        args=training_args_2,
        train_dataset=train_dataset, 
    )

    print_on_main_process("--- Starting Second Training Run ---")
    trainer_2.train()
    print_on_main_process("--- Second Training Run Completed ---")

    # --- 8. Save the final model ---
    final_output_dir = training_args_2.output_dir
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print_on_main_process(f"\nSaving final model to {final_output_dir}...")
        trainer_2.save_model(final_output_dir)
        print_on_main_process(f"Model saved to {final_output_dir}")

        # --- 9. Zip up the model directory (optional, for easier transfer) ---
        archive_name = "my_final_qwen2_model_ec2"
        shutil.make_archive(archive_name, "zip", final_output_dir)
        print_on_main_process(f"Model archived to {archive_name}.zip. You can transfer this file (e.g., to S3).")

    # --- 10. Reloading the model and Inference Example (on main process or single GPU after training) ---
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print_on_main_process("\n--- Reloading and Inference Example (on main process) ---")
        model_to_load_path = final_output_dir

        try:
            print_on_main_process(f"Loading tokenizer from: {model_to_load_path}")
            reloaded_tokenizer = AutoTokenizer.from_pretrained(model_to_load_path)
            if reloaded_tokenizer.pad_token is None: # Ensure pad token for generation
                reloaded_tokenizer.pad_token = reloaded_tokenizer.eos_token

            print_on_main_process(f"Loading model from: {model_to_load_path}")
            reloaded_model = AutoModelForCausalLM.from_pretrained(
                model_to_load_path,
                device_map="auto", 
                torch_dtype='auto'
            )
            print_on_main_process("Model and tokenizer reloaded successfully for inference.")

            gen_pipeline = pipeline(
                'text-generation',
                model=reloaded_model,
                tokenizer=reloaded_tokenizer,
            )

            prompts_for_inference = [
                "Explain Group Relative Policy Optimization in simple terms:",
                "In 1988, a person's age was equal to the sum of the digits of their birth year. How old was this person?"
            ]

            for i, prompt_text in enumerate(prompts_for_inference):
                print_on_main_process(f"\n--- Running prompt {i+1} ---")
                
                # Format prompt for chat model
                # The make_conversation was used to structure the training data.
                # For inference, we create a similar structure for the prompt.
                if "1988" in prompt_text: 
                    conversation_input_for_inference = make_conversation({"problem": prompt_text, "solution": ""})["prompt"]
                else: 
                    conversation_input_for_inference = [{"role": "user", "content": prompt_text}]
                
                formatted_prompt_for_inference = reloaded_tokenizer.apply_chat_template(
                    conversation_input_for_inference,
                    tokenize=False,
                    add_generation_prompt=True # Important for instruction-tuned models
                )
                print_on_main_process(f"Formatted Inference Prompt:\n{formatted_prompt_for_inference}")

                outputs = gen_pipeline(
                    formatted_prompt_for_inference, 
                    max_new_tokens=250, # Increased for potentially longer math solutions
                    do_sample=True, 
                    temperature=0.7, 
                    pad_token_id=reloaded_tokenizer.eos_token_id
                )
                
                print_on_main_process("\nGenerated Output:")
                full_generated_text = outputs[0]['generated_text']
                print_on_main_process(full_generated_text)

        except Exception as e:
            print_on_main_process(f"Error during model reloading or inference: {e}")
            print_on_main_process("Please ensure the model was saved correctly and the path is accessible.")

    print_on_main_process("\n--- Script execution finished ---")

if __name__ == "__main__":
    # This will be run by torchrun, which handles DDP setup
    main()