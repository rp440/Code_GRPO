import random
import json
import os

def manual_matrix_multiply_2x2(A, B):
    """Standard 2x2 matrix multiplication."""
    C = [[0,0], [0,0]]
    if not (len(A) == 2 and len(A[0]) == 2 and len(B) == 2 and len(B[0]) == 2):
        raise ValueError("Matrices must be 2x2")
    C[0][0] = A[0][0] * B[0][0] + A[0][1] * B[1][0]
    C[0][1] = A[0][0] * B[0][1] + A[0][1] * B[1][1]
    C[1][0] = A[1][0] * B[0][0] + A[1][1] * B[1][0]
    C[1][1] = A[1][0] * B[0][1] + A[1][1] * B[1][1]
    return C

def generate_random_matrix(low=-9999, high=9999):
    """Generates a 2x2 matrix with random integer elements."""
    return [[random.randint(low, high) for _ in range(2)] for _ in range(2)]

def generate_matrix_io_dataset(num_examples):
    """
    Generates a dataset containing input matrices A, B and the expected output matrix C.
    This format is suitable for tasks where the LLM might be asked to directly predict
    the C matrix from A and B, or for the GRPO script's reward function data.

    Args:
        num_examples (int): Number of examples to generate.
    """
    dataset_items = []

    for i in range(num_examples):
        # Generate new A and B matrices for each example
        if i == 0:
            current_A_val, current_B_val = [[1,0],[0,1]], generate_random_matrix() # Identity A
        elif i == 1:
            current_A_val, current_B_val = generate_random_matrix(), [[1,0],[0,1]] # Identity B
        elif i == 2:
            current_A_val, current_B_val = [[0,0],[0,0]], generate_random_matrix() # Zero A
        elif i == 3:
            current_A_val, current_B_val = generate_random_matrix(), [[0,0],[0,0]] # Zero B
        elif i == 4:
            current_A_val, current_B_val = [[-2, 3],[1, -4]], [[5, -1],[-2, 6]]   # Mixed signs, specific
        elif i == 5:
            current_A_val, current_B_val = generate_random_matrix(low=0, high=9), \
                                           generate_random_matrix(low=0, high=9) # Small positive
        elif i == 6:
            current_A_val, current_B_val = generate_random_matrix(low=-9, high=0), \
                                           generate_random_matrix(low=-9, high=0) # Small negative
        else:
            current_A_val, current_B_val = generate_random_matrix(), generate_random_matrix()

        # Calculate the expected C matrix
        try:
            expected_C_val = manual_matrix_multiply_2x2(current_A_val, current_B_val)
        except ValueError as e:
            print(f"Skipping example due to matrix dimension error: {e}") # Should not happen with 2x2 generation
            continue

        # User query for context (optional, but can be useful)
        # This query does NOT ask for DSL, just implies a matrix multiplication context.
        user_query_content = (
            f"Given matrices A = {str(current_A_val).replace(' ', '')} and "
            f"B = {str(current_B_val).replace(' ', '')}, what is the resulting matrix C = A * B?"
        )

        data_item = {
            "user_query": user_query_content, # Contextual query
            "matrix_A_list": current_A_val,   # Input matrix A as list of lists
            "matrix_B_list": current_B_val,   # Input matrix B as list of lists
            "expected_C_list": expected_C_val # Expected output matrix C as list of lists
        }
        dataset_items.append(data_item)

    return dataset_items

if __name__ == '__main__':
    # --- Configuration ---
    # CRITICAL: Need sufficient samples for distributed training (4 GPUs)
    # Minimum: batch_size_per_gpu * num_gpus * grad_acc_steps * 10
    # For 4 GPUs with batch_size=2, grad_acc=4: 2 * 4 * 4 * 10 = 320 minimum
    NUM_EXAMPLES_TO_GENERATE = 1000  # Increased from 10 to 1000 for distributed training
    # Output filename suitable for the GRPO script, assuming it needs A, B, and C.
    # The GRPO script's `preprocess_jsonl_data` function expects `A_matrix_str`, `B_matrix_str`,
    # and calculates `expected_C_str` from them.
    # This script generates `matrix_A_list`, `matrix_B_list`, `expected_C_list`.
    # The GRPO preprocessor will need to be adapted if it strictly expects string versions,
    # or this script could output string versions directly.
    # For now, let's stick to list versions and note the GRPO script's preprocessor might need a tweak.
    OUTPUT_JSONL_PATH = "matrix_io_data_for_grpo.jsonl"
    # Alternative: Save to the name the GRPO script expects, but ensure its preprocessor handles lists.
    # OUTPUT_JSONL_PATH = "dsl_finetune_data_standard_matmul_full_sequence.jsonl"


    print(f"Generating {NUM_EXAMPLES_TO_GENERATE} matrix input-output examples.")
    print(f"Matrix elements will range from -9999 to 9999.")

    dataset_jsonl_items = generate_matrix_io_dataset(
        num_examples=NUM_EXAMPLES_TO_GENERATE,
    )

    print(f"\nGenerated {len(dataset_jsonl_items)} JSONL items.")
    print(f"Each item contains 'matrix_A_list', 'matrix_B_list', and 'expected_C_list'.")

    print("\n----- Example JSONL Data (First item) -----")
    if dataset_jsonl_items:
        print(json.dumps(dataset_jsonl_items[0], indent=2))
        print("--------------------------------------------------")

    with open(OUTPUT_JSONL_PATH, 'w') as f:
        for item in dataset_jsonl_items:
            f.write(json.dumps(item) + "\n")

    print(f"\nMatrix input-output data saved to {OUTPUT_JSONL_PATH}")
    print(f"If using with the provided GRPO script, ensure its 'preprocess_jsonl_data' function can handle")
    print(f"'matrix_A_list' and 'matrix_B_list' directly to calculate 'expected_C_str', or modify this script")
    print(f"to output 'A_matrix_str' and 'B_matrix_str' instead of list versions if the GRPO script requires strings.")
    print(f"The current GRPO script's preprocessor expects 'matrix_A_list' and 'matrix_B_list' or 'A_matrix_str', 'B_matrix_str'.")
    print(f"This generated format with '_list' suffixes should be compatible if the GRPO script's `preprocess_jsonl_data` is robust.")