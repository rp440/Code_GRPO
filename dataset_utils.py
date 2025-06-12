"""
Dataset Utilities for GRPO Training

Handles loading and preprocessing of JSONL dataset for matrix multiplication DSL training.
"""

import os
import ast
from datasets import load_dataset
from config import DATASET_PATH, DEFAULT_USER_PROMPT_FOR_DSL_GENERATION, SYSTEM_MESSAGE
from matrix_ops import manual_matrix_multiply_2x2


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


def load_and_preprocess_dataset():
    """Load and preprocess the JSONL dataset for training"""
    print(f"Loading dataset from: {DATASET_PATH}")
    print("*** IMPORTANT: Make sure you have run dataset.py to generate the dataset file first! ***")

    try:
        # Check if dataset file exists
        if not os.path.exists(DATASET_PATH):
            print(f"[ERROR] Dataset file not found at {DATASET_PATH}")
            print("[REQUIRED] Please run dataset.py first to generate the training dataset!")
            print("[INFO] The dataset.py script should create the JSONL file with matrix multiplication examples.")
            exit(1)
        
        print(f"[SUCCESS] Found dataset file at: {DATASET_PATH}")
        
        # Check file size
        file_size = os.path.getsize(DATASET_PATH)
        print(f"[INFO] Dataset file size: {file_size} bytes")
        
        if file_size == 0:
            print("[ERROR] Dataset file is empty!")
            print("[REQUIRED] Please run dataset.py to generate proper training data.")
            exit(1)
        
        # Load and process dataset
        raw_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
        train_dataset = raw_dataset.map(preprocess_jsonl_data)

        print(f"[SUCCESS] Processed dataset. Number of samples: {len(train_dataset)}")
        
        if len(train_dataset) == 0:
            print("[ERROR] No valid samples found after processing!")
            print("[CHECK] Check the format of your dataset.py generated file.")
            exit(1)
        
        print(f"[SAMPLE] First processed sample keys: {list(train_dataset[0].keys())}")
        print(f"[SAMPLE] Sample A matrix: {train_dataset[0]['A_matrix_str']}")
        print(f"[SAMPLE] Sample B matrix: {train_dataset[0]['B_matrix_str']}")
        
        return train_dataset
        
    except Exception as e:
        print(f"[ERROR] Failed to load or process dataset from {DATASET_PATH}: {e}")
        print("[TIP] Make sure dataset.py has been run and generated a valid JSONL file.")
        print("[FORMAT] Expected format: Each line should be a JSON object with matrix data.")
        exit(1) 