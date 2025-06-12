"""
Main GRPO Training Script

Orchestrates the complete GRPO training pipeline for matrix multiplication DSL learning.
Integrates all modules: model loading, dataset preprocessing, reward system, and training.
"""

import os
from trl import GRPOConfig, GRPOTrainer

# Import our custom modules
from config import (
    NEW_LEARNING_RATE, EPOCHS, BATCH_SIZE, GRAD_ACC_STEPS,
    get_drive_paths
)
from model_utils import (
    setup_huggingface_login, setup_google_drive, load_base_model_and_tokenizer,
    setup_peft_model, setup_optimizer_and_reset_value_head, setup_model_config,
    save_model_to_drive, setup_tensorboard, get_tensorboard_directory
)
from dataset_utils import load_and_preprocess_dataset
from reward_system import setup_discovery_logging, matrix_dsl_reward, get_discovery_summary
from matrix_ops import generate_random_2x2_matrix, manual_matrix_multiply_2x2
from inference import load_inference_model, run_inference_test


def main():
    """Main training function"""
    print("="*60)
    print("GPRO Matrix Multiplication DSL Training Script")
    print("="*60)
    
    # --- 1. Setup Environment ---
    print("\n--- 1. Environment Setup ---")
    setup_huggingface_login()
    use_drive_for_saving = setup_google_drive()
    
    # Setup paths
    drive_model_path, drive_logs_path = get_drive_paths()
    if use_drive_for_saving:
        print(f"[SAVE CONFIG] Model will be saved to: {drive_model_path}")
        print(f"[SAVE CONFIG] TensorBoard logs will be saved to: {drive_logs_path}")
    else:
        drive_model_path, drive_logs_path = None, None
        print("[WARNING] Drive saving is disabled - models will only be saved locally")
    
    # Setup discovery logging
    setup_discovery_logging(use_drive_for_saving)
    
    # --- 2. Load Dataset ---
    print("\n--- 2. Dataset Loading ---")
    train_dataset = load_and_preprocess_dataset()
    
    # --- 3. Model Setup ---
    print("\n--- 3. Model Setup ---")
    base_model, tokenizer = load_base_model_and_tokenizer()
    model_peft = setup_peft_model(base_model)
    optimizer = setup_optimizer_and_reset_value_head(model_peft)
    setup_model_config(model_peft, tokenizer)
    
    # Make tokenizer globally available for reward function
    global tokenizer_for_training
    tokenizer_for_training = tokenizer
    
    # --- 4. Training Configuration ---
    print("\n--- 4. Training Configuration ---")
    
    # Setup TensorBoard logging directory
    local_tensorboard_dir = get_tensorboard_directory()
    actual_tensorboard_dir = local_tensorboard_dir
    
    # Training arguments
    use_bf16 = False  # Disable bf16 to avoid device-specific optimizations
    use_fp16 = False  # Disable fp16 to avoid device-specific optimizations
    
    training_args_grpo = GRPOConfig(
        output_dir=drive_model_path if use_drive_for_saving else f"/content/local_model_output",
        learning_rate=NEW_LEARNING_RATE,
        remove_unused_columns=False,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        num_train_epochs=EPOCHS,
        bf16=use_bf16, 
        fp16=use_fp16,
        per_device_train_batch_size=BATCH_SIZE,
        max_completion_length=500,
        num_generations=10,  # Matching reward function expectation
        max_prompt_length=1000,
        logging_steps=5,
        save_strategy="steps",
        save_steps=100,
        logging_dir=actual_tensorboard_dir,
        report_to="tensorboard",
        push_to_hub=False,
        dataloader_drop_last=True,
        warmup_steps=5,
        dataloader_num_workers=0,
    )

    # Validate dataset before training
    print(f"Dataset validation:")
    print(f"  - Dataset size: {len(train_dataset)}")
    print(f"  - Dataset features: {train_dataset.features}")
    if len(train_dataset) == 0:
        print("[ERROR] Dataset is empty!")
        return

    # --- 5. GRPO Trainer Setup ---
    print("\n--- 5. GRPO Trainer Setup ---")
    trainer_grpo = GRPOTrainer(
        model=model_peft,
        reward_funcs=[matrix_dsl_reward],
        args=training_args_grpo,
        train_dataset=train_dataset,
        optimizers=(optimizer, None),  # Use our custom optimizer, no scheduler
    )

    print("Training configuration:")
    print(f"  - Per-device batch size: {training_args_grpo.per_device_train_batch_size}")
    print(f"  - Gradient accumulation steps: {training_args_grpo.gradient_accumulation_steps}")
    print(f"  - Total effective batch size: {training_args_grpo.per_device_train_batch_size * training_args_grpo.gradient_accumulation_steps}")
    print(f"  - Learning rate: {training_args_grpo.learning_rate}")
    print(f"  - Epochs: {training_args_grpo.num_train_epochs}")
    print(f"TensorBoard logs: {actual_tensorboard_dir}")
    if use_drive_for_saving and drive_logs_path:
        print(f"Logs will be copied to Drive after training: {drive_logs_path}")
    
    # Setup TensorBoard
    setup_tensorboard(actual_tensorboard_dir)
    
    # --- 6. Training ---
    print("\n--- 6. Starting GRPO Training ---")
    print("="*50)
    trainer_grpo.train()
    print("="*50)
    print("GRPO Training finished.")
    
    # --- 7. Save Model ---
    print("\n--- 7. Model Saving ---")
    save_model_to_drive(trainer_grpo, drive_model_path, drive_logs_path, local_tensorboard_dir)
    
    # --- 8. Discovery Summary ---
    print("\n--- 8. Training Summary ---")
    discovery_summary = get_discovery_summary()
    
    # --- 9. Inference Test ---
    print("\n--- 9. Inference Verification ---")
    try:
        # Generate test matrices
        A_inference = generate_random_2x2_matrix()
        B_inference = generate_random_2x2_matrix()
        expected_inference = manual_matrix_multiply_2x2(A_inference, B_inference)
        
        print(f"Test matrices: A={A_inference}, B={B_inference}")
        print(f"Expected result: {expected_inference}")
        
        # Load trained model for inference
        inference_model, inference_tokenizer = load_inference_model()
        
        # Run inference test
        test_result = run_inference_test(
            inference_model, 
            inference_tokenizer, 
            test_matrices=(A_inference, B_inference)
        )
        
        print(f"\nInference Test Results:")
        print(f"  - Correct: {test_result['verification']['correct']}")
        print(f"  - Efficient: {test_result['verification']['efficient']}")
        print(f"  - Multiplications: {test_result['verification']['multiplications']}")
        
    except Exception as e:
        print(f"[WARNING] Inference test failed: {e}")
    
    # --- 10. Final Summary ---
    print("\n" + "="*60)
    print("*** SCRIPT COMPLETED SUCCESSFULLY ***")
    print("="*60)
    
    final_paths = {
        'model': drive_model_path if use_drive_for_saving else "/content/local_model_output",
        'tensorboard_local': actual_tensorboard_dir,
        'tensorboard_drive': drive_logs_path if use_drive_for_saving else None,
        'discoveries_local': discovery_summary['local_log'],
        'discoveries_drive': discovery_summary['drive_log'],
        'best_solution': discovery_summary['best_multiplications']
    }
    
    for key, path in final_paths.items():
        if path:
            print(f"[{key.upper()}] {path}")
    
    print(f"[TIP] To restart TensorBoard: %tensorboard --logdir {actual_tensorboard_dir}")
    
    return final_paths


if __name__ == "__main__":
    main() 