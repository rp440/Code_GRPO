"""
Reward System for GRPO Training

Contains the reward function and discovery logging system for matrix multiplication DSL training.
"""

import os
import re
import ast
import shutil
from datetime import datetime
from dsl_executor import DSLExecutor
from matrix_ops import count_multiplications_in_dsl
from config import (
    CORRECT_7_MULT_BONUS, NEAR_MISS_PENALTY, WEIRD_ANSWER_PENALTY, TAG_BONUS,
    EXPLORATION_SCALE, EXPLORATION_OFFSET, MODEL_SAVE_PARENT_DIR_DRIVE
)

# Global variables for reward tracking
_num_generations_per_prompt_for_reward = 10
_reward_call_count = 0
_best_n_mults = float('inf')

# Discovery logging setup
timestamp_for_discoveries = datetime.now().strftime("%Y%m%d_%H%M%S")
DISCOVERIES_LOG_FILE = f"/content/dsl_discoveries_{timestamp_for_discoveries}.txt"
DISCOVERIES_DRIVE_FILE = None


def setup_discovery_logging(use_drive_for_saving):
    """Setup discovery logging paths"""
    global DISCOVERIES_DRIVE_FILE
    if use_drive_for_saving:
        DISCOVERIES_DRIVE_FILE = os.path.join(MODEL_SAVE_PARENT_DIR_DRIVE, f"dsl_discoveries_{timestamp_for_discoveries}.txt")
    
    # Initialize discovery log
    log_discovery(f"DISCOVERY LOG STARTED - Training Session {timestamp_for_discoveries}")
    log_discovery(f"Exploration-Prioritized Scoring: -1.06e-8*||AB-C||²+6 for exploration (capped at -4 for 7-mult), +5 for correct 7-mult, -15 for near-miss, -10 for weird answers, +0.1 for DSL tags, min -19 for solvable DSL")


def log_discovery(message, dsl_script=None):
    """Log discoveries to both file and console"""
    full_message = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
    print(f"**DISCOVERY** {full_message}")
    
    # Write to local file
    with open(DISCOVERIES_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{full_message}\n")
        if dsl_script:
            f.write(f"DSL Script:\n{dsl_script}\n")
            f.write("-" * 50 + "\n")
    
    # Copy to Drive if enabled
    if DISCOVERIES_DRIVE_FILE:
        try:
            shutil.copy2(DISCOVERIES_LOG_FILE, DISCOVERIES_DRIVE_FILE)
        except Exception as e:
            print(f"[WARNING] Could not copy discoveries log to Drive: {e}")


def matrix_dsl_reward(completions, prompts=None, completion_ids=None, **kwargs):
    """
    Reward function for GRPO training with exploration-prioritized scoring.
    """
    global _num_generations_per_prompt_for_reward, _reward_call_count, _best_n_mults
    _reward_call_count += 1
    
    A_matrix_str_list = kwargs["A_matrix_str"]
    B_matrix_str_list = kwargs["B_matrix_str"]
    expected_C_str_list = kwargs["expected_C_str"]
    rewards = []

    print(f"\n=== NEW REWARD CALCULATION CALL #{_reward_call_count} ===")
    print(f"Processing {len(completions)} completions...")
    print(f"Current best multiplications: {_best_n_mults if _best_n_mults != float('inf') else 'None found yet'}")

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
            print(f"  Completion {i}: Error parsing matrices: {e}")
            rewards.append(WEIRD_ANSWER_PENALTY + TAG_BONUS)
            continue

        # Extract and clean DSL content
        processed_content = _extract_content(dsl_script_raw_content)
        dsl_content, tag_bonus = _extract_dsl_content(processed_content)
        final_dsl_script = _clean_dsl_script(dsl_content)

        if not final_dsl_script or final_dsl_script.strip().lower() == "error: cannot determine full sequence.":
            print(f"  Completion {i}: Empty or error DSL script. Reward: {WEIRD_ANSWER_PENALTY + tag_bonus:.1f}")
            rewards.append(WEIRD_ANSWER_PENALTY + tag_bonus)
            continue
            
        # Evaluate DSL and assign reward
        reward = _evaluate_dsl_and_assign_reward(i, final_dsl_script, A, B, expected_C)
        
        # Ensure no solvable DSL gets reward less than -19
        reward = max(reward, -19.0)
        
        # Final reward with tag bonus
        final_reward = reward + tag_bonus
        rewards.append(final_reward)
        
        if tag_bonus > 0:
            print(f"  Completion {i}: Final reward: {final_reward:.1f} (base: {reward:.1f}, tag bonus: +{tag_bonus:.1f})")
        else:
            print(f"  Completion {i}: Final reward: {final_reward:.1f}")
        
    print(f"Batch average reward: {sum(rewards)/len(rewards):.2f}")
    print(f"Current global best: {_best_n_mults if _best_n_mults != float('inf') else 'None'} multiplications")
    print("=" * 50)
    return rewards


def _extract_content(dsl_script_raw_content):
    """Extract content from various completion formats"""
    if isinstance(dsl_script_raw_content, list):
        if len(dsl_script_raw_content) == 1:
            item = dsl_script_raw_content[0]
            if isinstance(item, dict) and 'role' in item and 'content' in item:
                return item['content']
            elif isinstance(item, str):
                return item
            else:
                return str(item)
        else:
            return str(dsl_script_raw_content)
    elif isinstance(dsl_script_raw_content, dict) and 'role' in dsl_script_raw_content and 'content' in dsl_script_raw_content:
        return dsl_script_raw_content['content']
    elif isinstance(dsl_script_raw_content, str):
        return dsl_script_raw_content
    else:
        return str(dsl_script_raw_content)


def _extract_dsl_content(processed_content):
    """Extract DSL content and determine tag bonus"""
    dsl_match = re.search(r'<DSL>(.*?)</DSL>', processed_content, re.DOTALL | re.IGNORECASE)
    if dsl_match:
        return dsl_match.group(1).strip(), TAG_BONUS
    else:
        return processed_content, 0.0


def _clean_dsl_script(dsl_content):
    """Clean special tokens from DSL content"""
    # Import tokenizer if available (from training context)
    try:
        from __main__ import tokenizer_for_training
        temp_tokens_to_remove = ["<|im_end|>", "<|endoftext|>", "<|file_separator|>"]
        if hasattr(tokenizer_for_training, 'all_special_tokens') and tokenizer_for_training.all_special_tokens is not None:
            valid_special_tokens = [
                str(t) for t in tokenizer_for_training.all_special_tokens
                if t is not None and isinstance(t, str) and t
            ]
            temp_tokens_to_remove.extend(valid_special_tokens)
        
        if tokenizer_for_training.eos_token and isinstance(tokenizer_for_training.eos_token, str):
            temp_tokens_to_remove.append(tokenizer_for_training.eos_token)

        unique_tokens_to_remove = list(set(t for t in temp_tokens_to_remove if t))

        for token_str in unique_tokens_to_remove:
            dsl_content = dsl_content.replace(token_str, "")
    except:
        # Fallback cleaning if tokenizer not available
        common_tokens = ["<|im_end|>", "<|endoftext|>", "<|file_separator|>"]
        for token in common_tokens:
            dsl_content = dsl_content.replace(token, "")

    lines = dsl_content.split('\n')
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    return "\n".join(cleaned_lines)


def _evaluate_dsl_and_assign_reward(completion_idx, final_dsl_script, A, B, expected_C):
    """Evaluate DSL script and assign appropriate reward"""
    global _best_n_mults
    
    try:
        # Count multiplications and execute DSL
        num_multiplications = count_multiplications_in_dsl(final_dsl_script)
        executor = DSLExecutor(A, B)
        C_dsl = executor.run_dsl_and_get_c(final_dsl_script)

        # Calculate L2 squared distance for all cases
        l2_sq_distance = 0.0
        for r in range(2):
            for c in range(2):
                diff = C_dsl[r][c] - expected_C[r][c]
                l2_sq_distance += diff * diff
        
        if num_multiplications == 7:
            # PRIORITIZED: 7-multiplication solutions get exploration formula with L2 distance capped at max -10
            # Formula: -1.06×10^-8 * ||AB-C||^2 + 6, but capped at minimum of -4 (6 - 10)
            base_reward = EXPLORATION_SCALE * l2_sq_distance + EXPLORATION_OFFSET
            reward = max(base_reward, EXPLORATION_OFFSET - 10.0)  # Cap L2 penalty at max -10
            
            if C_dsl == expected_C:
                print(f"  Completion {completion_idx}: **PERFECT** Correct 7-multiplication solution! (L2²={l2_sq_distance:.0f}, reward={reward:.3f})")
                
                # Log the perfect solution
                log_discovery(f"PERFECT 7-MULT SOLUTION! Score: {reward:.3f}", final_dsl_script)
                log_discovery(f"Test matrices: A={A}, B={B}, Expected C={expected_C}")
                
                if num_multiplications < _best_n_mults:
                    _best_n_mults = num_multiplications
            else:
                print(f"  Completion {completion_idx}: **7-MULT EXPLORATION** L2²={l2_sq_distance:.0f}, reward={reward:.3f} (capped)")
                print(f"  Completion {completion_idx}: Expected: {expected_C}, Got: {C_dsl}")
                
        elif C_dsl == expected_C:
            # CORRECT DSL but not 7-multiplication - use fixed penalty for near-miss
            reward = NEAR_MISS_PENALTY  # -15 for near-miss (correct but not 7-mult)
            print(f"  Completion {completion_idx}: **CORRECT** {num_multiplications}-mult (near-miss penalty: {NEAR_MISS_PENALTY})")
            
            if num_multiplications < _best_n_mults:
                _best_n_mults = num_multiplications
                log_discovery(f"NEW BEST SOLUTION! {num_multiplications} multiplications", final_dsl_script)
                
        else:
            # INCORRECT non-7-multiplication solutions - use exploration formula but ensure lower priority
            base_exploration_reward = EXPLORATION_SCALE * l2_sq_distance + EXPLORATION_OFFSET
            
            # Ensure non-7-mult incorrect solutions are always worse than weird answers threshold
            # by capping them at slightly above weird answer penalty
            reward = min(base_exploration_reward, WEIRD_ANSWER_PENALTY + 1.0)
            
            print(f"  Completion {completion_idx}: **INCORRECT** {num_multiplications}-mul attempt. L2²={l2_sq_distance:.0f}, reward={reward:.3f}")
            print(f"  Completion {completion_idx}: Expected: {expected_C}, Got: {C_dsl}")

        return reward

    except Exception as e:
        # FAILED EXECUTION - weird answer penalty
        reward = WEIRD_ANSWER_PENALTY
        print(f"  Completion {completion_idx}: **EXECUTION FAILED**: {str(e)[:100]}...")
        print(f"  Completion {completion_idx}: Weird answer penalty: {reward:.1f}")
        return reward


def get_discovery_summary():
    """Get final discovery summary"""
    final_best = _best_n_mults if _best_n_mults != float('inf') else 'None found'
    log_discovery(f"TRAINING COMPLETED - Final best: {final_best} multiplications")
    log_discovery(f"Discoveries log saved to: {DISCOVERIES_LOG_FILE}")
    if DISCOVERIES_DRIVE_FILE:
        log_discovery(f"Discoveries log copied to Drive: {DISCOVERIES_DRIVE_FILE}")
    
    return {
        'best_multiplications': final_best,
        'local_log': DISCOVERIES_LOG_FILE,
        'drive_log': DISCOVERIES_DRIVE_FILE
    } 