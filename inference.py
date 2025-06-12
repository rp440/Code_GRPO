"""
Inference and Verification Module

Contains inference logic for testing the trained GRPO model and verifying DSL generation.
"""

import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from dsl_executor import DSLExecutor
from matrix_ops import generate_random_2x2_matrix, manual_matrix_multiply_2x2, count_multiplications_in_dsl
from config import BASE_MODEL_NAME_FOR_FINETUNING, LOCAL_TRAINED_MODEL_PATH, DEFAULT_USER_PROMPT_FOR_DSL_GENERATION, SYSTEM_MESSAGE


def load_inference_model(model_path=None):
    """Load the fine-tuned model for inference"""
    if model_path is None:
        model_path = LOCAL_TRAINED_MODEL_PATH
    
    print(f"Loading fine-tuned model from: {model_path}")
    try:
        inference_base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME_FOR_FINETUNING, torch_dtype="auto", trust_remote_code=True
        )
        inference_model = PeftModel.from_pretrained(inference_base_model, model_path)
        inference_model = inference_model.merge_and_unload()
        
        inference_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME_FOR_FINETUNING, trust_remote_code=True)
        if inference_tokenizer.pad_token is None: 
            inference_tokenizer.pad_token = inference_tokenizer.eos_token
        if inference_tokenizer.padding_side == 'right': 
            inference_tokenizer.padding_side = 'left'
        
        print("[SUCCESS] Fine-tuned model loaded for inference.")
        return inference_model, inference_tokenizer
        
    except Exception as e:
        print(f"[ERROR] Error loading fine-tuned model: {e}")
        print("[FALLBACK] Falling back to base model for inference...")
        try:
            inference_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME_FOR_FINETUNING, torch_dtype="auto", trust_remote_code=True
            )
            inference_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME_FOR_FINETUNING, trust_remote_code=True)
            if inference_tokenizer.pad_token is None: 
                inference_tokenizer.pad_token = inference_tokenizer.eos_token
            if inference_tokenizer.padding_side == 'right': 
                inference_tokenizer.padding_side = 'left'
            
            print("[FALLBACK] Using base model for inference.")
            return inference_model, inference_tokenizer
            
        except Exception as fallback_e:
            print(f"[ERROR] Error loading base model for inference: {fallback_e}. Exiting.")
            raise fallback_e


def run_inference_test(inference_model, inference_tokenizer, test_matrices=None):
    """Run inference test on the model"""
    # Generate test matrices if not provided
    if test_matrices is None:
        A_matrix = generate_random_2x2_matrix()
        B_matrix = generate_random_2x2_matrix()
    else:
        A_matrix, B_matrix = test_matrices
    
    expected_result = manual_matrix_multiply_2x2(A_matrix, B_matrix)
    
    # Create text generation pipeline
    text_gen_pipeline = pipeline(
        "text-generation", 
        model=inference_model, 
        tokenizer=inference_tokenizer
    )

    # Prepare inference prompt
    user_query = DEFAULT_USER_PROMPT_FOR_DSL_GENERATION
    user_query += f" (Using A={str(A_matrix)}, B={str(B_matrix)})"

    inference_chat_messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_query}
    ]
    
    formatted_prompt = inference_tokenizer.apply_chat_template(
        inference_chat_messages, tokenize=False, add_generation_prompt=True
    )

    print(f"\nGenerating DSL for A={A_matrix}, B={B_matrix}")
    print(f"Expected result: {expected_result}")

    # Generate response
    outputs = text_gen_pipeline(
        formatted_prompt, 
        max_new_tokens=350, 
        do_sample=False, 
        temperature=0.1, 
        top_p=0.9,
        pad_token_id=inference_tokenizer.pad_token_id, 
        eos_token_id=inference_tokenizer.eos_token_id
    )

    # Extract assistant's reply
    generated_full_text = outputs[0]['generated_text']
    assistant_reply = extract_assistant_reply(generated_full_text, formatted_prompt)
    
    print(f"\n--- Raw Assistant's Reply (DSL Script) ---\n{assistant_reply}\n------------------------------------")
    
    # Verify generated DSL
    verification_result = verify_generated_dsl(assistant_reply, A_matrix, B_matrix, expected_result)
    
    return {
        'test_matrices': (A_matrix, B_matrix),
        'expected_result': expected_result,
        'generated_dsl': assistant_reply,
        'verification': verification_result
    }


def extract_assistant_reply(generated_full_text, formatted_prompt):
    """Extract the assistant's reply from the generated text"""
    assistant_reply_raw = generated_full_text
    
    if generated_full_text.startswith(formatted_prompt):
        assistant_reply_raw = generated_full_text[len(formatted_prompt):].strip()
    else:
        assistant_marker = "<|im_start|>assistant"
        last_occurrence_idx = generated_full_text.rfind(assistant_marker)
        if last_occurrence_idx != -1:
            start_of_reply_idx = generated_full_text.find("\n", last_occurrence_idx)
            if start_of_reply_idx != -1:
                assistant_reply_raw = generated_full_text[start_of_reply_idx+1:].strip()

    # Clean tokens
    tokens_to_clean = ["<|im_end|>", "<|endoftext|>"]
    for token in tokens_to_clean:
        if assistant_reply_raw.endswith(token):
            assistant_reply_raw = assistant_reply_raw[:-len(token)].strip()

    return assistant_reply_raw


def verify_generated_dsl(assistant_reply, A_matrix, B_matrix, expected_result):
    """Verify the generated DSL script"""
    print("\n--- Verifying Generated DSL ---")
    
    lines = assistant_reply.split('\n')
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    final_generated_dsl = "\n".join(cleaned_lines)

    if not final_generated_dsl or final_generated_dsl.strip().lower() == "error: cannot determine full sequence.":
        return {
            'status': 'FAILED',
            'reason': 'Model did not generate a valid DSL script or explicitly errored.',
            'correct': False,
            'efficient': False,
            'multiplications': 0
        }

    try:
        executor = DSLExecutor(A_matrix, B_matrix)
        C_generated = executor.run_dsl_and_get_c(final_generated_dsl)
        num_mults_generated = count_multiplications_in_dsl(final_generated_dsl)

        print(f"  Generated DSL executed. Multiplications: {num_mults_generated}")
        print(f"  Resulting C: {C_generated}")
        print(f"  Expected C:  {expected_result}")

        is_correct = C_generated == expected_result
        is_efficient = num_mults_generated <= 7

        if is_correct:
            print(f"[PASSED] Algorithmically correct.")
            if is_efficient: 
                print(f"    [EFFICIENT] {num_mults_generated} multiplications (<= 7).")
            else: 
                print(f"    [SUBOPTIMAL] {num_mults_generated} multiplications (> 7).")
        else:
            print("[FAILED] Algorithmically INCORRECT.")
        
        return {
            'status': 'SUCCESS',
            'correct': is_correct,
            'efficient': is_efficient,
            'multiplications': num_mults_generated,
            'generated_result': C_generated,
            'dsl_script': final_generated_dsl
        }

    except ValueError as e: 
        print(f"[FAILED] Invalid DSL or execution error: {e}")
        return {
            'status': 'FAILED',
            'reason': f'DSL execution error: {e}',
            'correct': False,
            'efficient': False,
            'multiplications': 0
        }
    except Exception as e: 
        print(f"[FAILED] Unexpected verification error: {e}")
        return {
            'status': 'FAILED',
            'reason': f'Unexpected error: {e}',
            'correct': False,
            'efficient': False,
            'multiplications': 0
        }


def run_multiple_inference_tests(inference_model, inference_tokenizer, num_tests=5):
    """Run multiple inference tests"""
    print(f"\n=== Running {num_tests} Inference Tests ===")
    
    results = []
    correct_count = 0
    efficient_count = 0
    
    for i in range(num_tests):
        print(f"\n--- Test {i+1}/{num_tests} ---")
        result = run_inference_test(inference_model, inference_tokenizer)
        results.append(result)
        
        if result['verification']['correct']:
            correct_count += 1
        if result['verification']['efficient']:
            efficient_count += 1
    
    print(f"\n=== Summary of {num_tests} Tests ===")
    print(f"Correct: {correct_count}/{num_tests} ({100*correct_count/num_tests:.1f}%)")
    print(f"Efficient (≤7 mults): {efficient_count}/{num_tests} ({100*efficient_count/num_tests:.1f}%)")
    
    return results


if __name__ == "__main__":
    # Load model and run tests
    model, tokenizer = load_inference_model()
    run_multiple_inference_tests(model, tokenizer, num_tests=3) 