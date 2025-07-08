"""
EC2 Inference and Chat Interface for GRPO Matrix Multiplication DSL

Optimized for EC2 deployment with streaming chat capabilities and multiple model support.
"""

import re
import os
import sys
import torch
import ast
import random
import threading
import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextIteratorStreamer
from peft import PeftModel

# Configure logging for EC2
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference_ec2.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==============================================
#  INFERENCE CONFIGURATION
# ==============================================
@dataclass
class InferenceConfig:
    """Inference configuration parameters for GRPO model."""
    # --- sequence lengths ---
    max_completion_length: int = 750
    max_prompt_length: int = 256

    # --- generation / exploration ---
    temperature: float = 1                 # higher entropy for exploration
    top_k: int | None = None                 # allow disabling top-k sampling
    top_p: float = 0.95                      # nucleus sampling threshold

    # --- generation parameters ---
    do_sample: bool = True                   # enable sampling
    num_beams: int = 1                       # beam search disabled for exploration
    early_stopping: bool = False             # disable early stopping
    
    # --- batch processing ---
    batch_size: int = 1                      # inference batch size
    
    # --- verbose logging helpers ---
    log_completions: bool = True
    num_completions_to_print: int = 2

# Global inference configuration
INFERENCE_CFG = InferenceConfig()

# EC2 Configuration Constants
BASE_MODEL_NAME_FOR_FINETUNING = "Qwen/Qwen2-1.5B"
MODEL_SAVE_DIR = "./models"
TRAINED_MODEL_DIR_NAME = f"{BASE_MODEL_NAME_FOR_FINETUNING.split('/')[-1]}-GRPO-MatMulDSL-JSONL"
LOCAL_TRAINED_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, TRAINED_MODEL_DIR_NAME)

# Final adapter path (from Google Drive download)
FINAL_ADAPTER_PATH = "./final_adapter"

# DSL Configuration
DEFAULT_USER_PROMPT_FOR_DSL_GENERATION = """Generate DSL code to multiply two 2x2 matrices A and B to get result matrix C. 
Use intermediate variables efficiently to minimize multiplications. Each line should have the format: variable = expression.
Valid operations: +, -, *. Matrix elements are accessed as A[i,j] and B[i,j] where i,j are 0 or 1."""

SYSTEM_MESSAGE = """You are an expert in matrix multiplication optimization using Domain-Specific Language (DSL). 
Generate efficient DSL code for 2x2 matrix multiplication that minimizes the number of multiplication operations."""

# --- Enhanced DSL Executor for EC2 ---
class DSLExecutor:
    """Enhanced DSL Executor with better error handling and logging for EC2"""
    
    def __init__(self, matrix_a: List[List[float]], matrix_b: List[List[float]]):
        self.variables = {}
        self._validate_matrices(matrix_a, matrix_b)
        self._initialize_matrices(matrix_a, matrix_b)
        logger.debug(f"DSLExecutor initialized with A={matrix_a}, B={matrix_b}")

    def _validate_matrices(self, matrix_a: List[List[float]], matrix_b: List[List[float]]) -> None:
        """Validate input matrices"""
        for name, matrix in [("A", matrix_a), ("B", matrix_b)]:
            if not (isinstance(matrix, list) and len(matrix) == 2 and
                    isinstance(matrix[0], list) and len(matrix[0]) == 2 and
                    all(isinstance(el, (int, float)) for row in matrix for el in row)):
                raise ValueError(f"Matrix {name} must be a 2x2 list of numbers.")

    def _initialize_matrices(self, matrix_a: List[List[float]], matrix_b: List[List[float]]) -> None:
        """Initialize matrix variables"""
        for r_idx in range(2):
            for c_idx in range(2):
                self.variables[f'A[{r_idx},{c_idx}]'] = matrix_a[r_idx][c_idx]
                self.variables[f'B[{r_idx},{c_idx}]'] = matrix_b[r_idx][c_idx]

    def _get_value(self, var_name: str) -> float:
        """Get value of variable or literal"""
        try:
            return ast.literal_eval(var_name)
        except (ValueError, SyntaxError):
            if var_name not in self.variables:
                raise ValueError(f"Variable '{var_name}' not found.")
            return self.variables[var_name]

    def execute_step(self, step_line: str) -> None:
        """Execute a single DSL step"""
        if not step_line.strip():
            return
        
        if '=' not in step_line:
            raise ValueError(f"Malformed DSL step: '{step_line}'")
        
        target_var, expression = [s.strip() for s in step_line.split('=', 1)]
        
        # Parse operation
        op_match = re.match(r"^([\w\[\],\.\d\-]+)\s*([*+\-])\s*([\w\[\],\.\d\-]+)$", expression)
        if op_match:
            operand_a = self._get_value(op_match.group(1))
            operator = op_match.group(2)
            operand_b = self._get_value(op_match.group(3))
            
            if operator == '+':
                result = operand_a + operand_b
            elif operator == '-':
                result = operand_a - operand_b
            elif operator == '*':
                result = operand_a * operand_b
            else:
                raise ValueError(f"Unknown operator: {operator}")
        else:
            result = self._get_value(expression)
        
        self.variables[target_var] = result
        logger.debug(f"Executed: {target_var} = {result}")

    def run_dsl_and_get_c(self, dsl_script: str) -> List[List[float]]:
        """Execute DSL script and return result matrix C"""
        try:
            for line_num, line in enumerate(dsl_script.strip().split('\n'), 1):
                try:
                    self.execute_step(line)
                except Exception as e:
                    raise ValueError(f"Error on line {line_num}: {e}")
            
            # Extract result matrix C
            result = []
            for r in range(2):
                row = []
                for c in range(2):
                    c_var = f'C[{r},{c}]'
                    if c_var not in self.variables:
                        raise ValueError(f"Result variable {c_var} not found")
                    row.append(self.variables[c_var])
                result.append(row)
            
            return result
            
        except Exception as e:
            logger.error(f"DSL execution failed: {e}")
            raise

# --- Enhanced Matrix Operations ---
def generate_random_2x2_matrix() -> List[List[int]]:
    """Generate a random 2x2 matrix with integer values"""
    return [[random.randint(1, 10) for _ in range(2)] for _ in range(2)]

def manual_matrix_multiply_2x2(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Manual 2x2 matrix multiplication for verification"""
    return [
        [A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
        [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]]
    ]

def count_multiplications_in_dsl(dsl_script: str) -> int:
    """Count multiplication operations in DSL script"""
    return len(re.findall(r'\*', dsl_script))

# --- Enhanced Model Management for EC2 ---
class EC2ModelManager:
    """Enhanced model manager optimized for EC2 with resource management"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Monitor GPU memory if available
        if torch.cuda.is_available():
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    def load_model(self, model_name: str, adapter_path: Optional[str] = None) -> Tuple[Any, Any]:
        """Load model with optional adapter, optimized for EC2"""
        if model_name in self.models:
            logger.info(f"Model {model_name} already loaded")
            return self.models[model_name], self.tokenizers[model_name]
        
        logger.info(f"Loading model: {model_name}")
        start_time = time.time()
        
        try:
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map='auto' if self.device == 'cuda' else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Load adapter if provided
            if adapter_path and os.path.exists(adapter_path):
                logger.info(f"Loading adapter from: {adapter_path}")
                try:
                    adapter_model = PeftModel.from_pretrained(base_model, adapter_path)
                    model = adapter_model.merge_and_unload()
                    logger.info("Adapter loaded and merged successfully")
                except Exception as e:
                    logger.warning(f"Failed to load adapter: {e}, using base model")
                    model = base_model
            else:
                model = base_model
                if adapter_path:
                    logger.warning(f"Adapter path not found: {adapter_path}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.padding_side == 'right':
                tokenizer.padding_side = 'left'
            
            # Move to device if not using device_map
            if self.device == 'cpu':
                model = model.to(self.device)
            
            # Store models
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            load_time = time.time() - start_time
            logger.info(f"Model {model_name} loaded successfully in {load_time:.2f}s")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def unload_model(self, model_name: str) -> None:
        """Unload model to free memory"""
        if model_name in self.models:
            del self.models[model_name]
            del self.tokenizers[model_name]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Model {model_name} unloaded")

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        memory_info = {}
        if torch.cuda.is_available():
            memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / 1e9
            memory_info['gpu_reserved'] = torch.cuda.memory_reserved() / 1e9
        return memory_info

# --- Streaming Chat Interface for EC2 ---
class EC2ChatInterface:
    """Enhanced streaming chat interface optimized for EC2"""
    
    def __init__(self, model_names: List[str], trained_adapter: Optional[str] = None):
        self.model_manager = EC2ModelManager()
        self.model_names = model_names
        self.trained_adapter = trained_adapter
        self.conversation_history = []
        
        # Load models
        for name in model_names:
            try:
                self.model_manager.load_model(name, trained_adapter)
            except Exception as e:
                logger.error(f"Failed to load model {name}: {e}")

    def _parse_matrices_from_input(self, user_input: str) -> Optional[Tuple[List[List[float]], List[List[float]]]]:
        """Parse matrices from user input"""
        try:
            # Look for matrix patterns like [[1,2],[3,4]]
            matrix_pattern = r'\[\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\s*,\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\]'
            matrices = re.findall(matrix_pattern, user_input)
            
            if len(matrices) >= 2:
                A = [[float(matrices[0][0]), float(matrices[0][1])], 
                     [float(matrices[0][2]), float(matrices[0][3])]]
                B = [[float(matrices[1][0]), float(matrices[1][1])], 
                     [float(matrices[1][2]), float(matrices[1][3])]]
                return A, B
        except Exception as e:
            logger.debug(f"Failed to parse matrices: {e}")
        
        return None

    def _stream_response(self, model_name: str, prompt: str, matrices: Optional[Tuple] = None) -> str:
        """Stream response from model"""
        if model_name not in self.model_manager.models:
            logger.error(f"Model {model_name} not loaded")
            return "Model not available"
        
        model = self.model_manager.models[model_name]
        tokenizer = self.model_manager.tokenizers[model_name]
        
        # Create DSL-specific prompt
        if matrices:
            A, B = matrices
            dsl_prompt = f"{DEFAULT_USER_PROMPT_FOR_DSL_GENERATION}\nA = {A}\nB = {B}"
        else:
            dsl_prompt = prompt
        
        # Format as chat
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": dsl_prompt}
        ]
        
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except:
            formatted_prompt = f"System: {SYSTEM_MESSAGE}\nUser: {dsl_prompt}\nAssistant: "
        
        # Setup streaming
        streamer = TextIteratorStreamer(
            tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True,
            timeout=30.0
        )
        
        inputs = tokenizer(formatted_prompt, return_tensors='pt').to(self.model_manager.device)
        
        generation_kwargs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'max_new_tokens': INFERENCE_CFG.max_completion_length,
            'streamer': streamer,
            'do_sample': INFERENCE_CFG.do_sample,
            'temperature': INFERENCE_CFG.temperature,
            'top_p': INFERENCE_CFG.top_p,
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id
        }
        
        # Add top_k if specified
        if INFERENCE_CFG.top_k is not None:
            generation_kwargs['top_k'] = INFERENCE_CFG.top_k
        
        # Start generation in separate thread
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream output
        print(f"\n--- {model_name} Response ---")
        full_response = ""
        try:
            for token in streamer:
                if token:
                    print(token, end='', flush=True)
                    full_response += token
        except Exception as e:
            logger.error(f"Streaming error: {e}")
        
        thread.join(timeout=60)  # Wait for completion
        print("\n" + "="*50)
        
        return full_response

    def verify_dsl_response(self, dsl_response: str, A: List[List[float]], B: List[List[float]]) -> Dict:
        """Verify DSL response"""
        try:
            # Clean the response
            lines = dsl_response.split('\n')
            cleaned_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
            dsl_script = "\n".join(cleaned_lines)
            
            if not dsl_script:
                return {'status': 'FAILED', 'reason': 'No valid DSL found'}
            
            # Execute DSL
            executor = DSLExecutor(A, B)
            result = executor.run_dsl_and_get_c(dsl_script)
            expected = manual_matrix_multiply_2x2(A, B)
            
            # Count multiplications
            mult_count = count_multiplications_in_dsl(dsl_script)
            
            is_correct = result == expected
            is_efficient = mult_count <= 7
            
            return {
                'status': 'SUCCESS',
                'correct': is_correct,
                'efficient': is_efficient,
                'multiplications': mult_count,
                'result': result,
                'expected': expected,
                'dsl_script': dsl_script
            }
            
        except Exception as e:
            return {'status': 'FAILED', 'reason': str(e)}

    def run_batch_test(self, num_tests: int = 5) -> Dict:
        """Run batch tests for evaluation"""
        logger.info(f"Running {num_tests} batch tests")
        results = []
        
        for i in range(num_tests):
            A = generate_random_2x2_matrix()
            B = generate_random_2x2_matrix()
            
            print(f"\n=== Test {i+1}/{num_tests} ===")
            print(f"A = {A}, B = {B}")
            
            for model_name in self.model_names:
                if model_name in self.model_manager.models:
                    response = self._stream_response(model_name, "", (A, B))
                    verification = self.verify_dsl_response(response, A, B)
                    
                    results.append({
                        'test_id': i+1,
                        'model': model_name,
                        'matrices': (A, B),
                        'response': response,
                        'verification': verification
                    })
        
        # Calculate summary stats
        correct_count = sum(1 for r in results if r['verification'].get('correct', False))
        efficient_count = sum(1 for r in results if r['verification'].get('efficient', False))
        total_tests = len(results)
        
        summary = {
            'total_tests': total_tests,
            'correct': correct_count,
            'efficient': efficient_count,
            'accuracy': correct_count / total_tests if total_tests > 0 else 0,
            'efficiency_rate': efficient_count / total_tests if total_tests > 0 else 0,
            'results': results
        }
        
        logger.info(f"Batch test completed: {correct_count}/{total_tests} correct, {efficient_count}/{total_tests} efficient")
        return summary

    def interactive_chat(self) -> None:
        """Run interactive chat interface"""
        print("\n🤖 EC2 Matrix Multiplication DSL Chat Interface")
        print("=" * 60)
        print("Commands:")
        print("  - Type matrices like: [[1,2],[3,4]] and [[5,6],[7,8]]")
        print("  - 'test N' - Run N batch tests")
        print("  - 'memory' - Show memory usage")
        print("  - 'models' - List loaded models")
        print("  - 'exit' - Quit")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("Goodbye! 👋")
                    break
                
                if user_input.lower() == 'memory':
                    memory = self.model_manager.get_memory_usage()
                    print(f"Memory usage: {memory}")
                    continue
                
                if user_input.lower() == 'models':
                    print(f"Loaded models: {list(self.model_manager.models.keys())}")
                    continue
                
                if user_input.lower().startswith('test'):
                    try:
                        parts = user_input.split()
                        num_tests = int(parts[1]) if len(parts) > 1 else 3
                        summary = self.run_batch_test(num_tests)
                        print(f"\nBatch Test Summary:")
                        print(f"Accuracy: {summary['accuracy']:.1%}")
                        print(f"Efficiency: {summary['efficiency_rate']:.1%}")
                    except (ValueError, IndexError):
                        print("Usage: test <number_of_tests>")
                    continue
                
                # Parse matrices or use random ones
                matrices = self._parse_matrices_from_input(user_input)
                if not matrices:
                    matrices = (generate_random_2x2_matrix(), generate_random_2x2_matrix())
                    print(f"Using random matrices: A={matrices[0]}, B={matrices[1]}")
                
                # Generate responses from all models
                for model_name in self.model_names:
                    if model_name in self.model_manager.models:
                        try:
                            response = self._stream_response(model_name, user_input, matrices)
                            
                            # Verify the response
                            verification = self.verify_dsl_response(response, matrices[0], matrices[1])
                            
                            print(f"\n📊 Verification for {model_name}:")
                            if verification['status'] == 'SUCCESS':
                                print(f"  ✅ Correct: {verification['correct']}")
                                print(f"  ⚡ Efficient: {verification['efficient']} ({verification['multiplications']} mults)")
                                if verification['correct']:
                                    print(f"  🎯 Result: {verification['result']}")
                            else:
                                print(f"  ❌ Failed: {verification['reason']}")
                            
                        except Exception as e:
                            logger.error(f"Error with model {model_name}: {e}")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Chat error: {e}")
                print(f"Error: {e}")

# --- Main Execution ---
def main():
    """Main function for EC2 inference"""
    logger.info("Starting EC2 GRPO Matrix Multiplication Inference")
    
    # Log inference configuration
    logger.info("Inference Configuration:")
    logger.info(f"  Max completion length: {INFERENCE_CFG.max_completion_length}")
    logger.info(f"  Temperature: {INFERENCE_CFG.temperature}")
    logger.info(f"  Top-p: {INFERENCE_CFG.top_p}")
    logger.info(f"  Top-k: {INFERENCE_CFG.top_k}")
    logger.info(f"  Do sample: {INFERENCE_CFG.do_sample}")
    
    # Configuration
    MODELS = [
        'Qwen/Qwen2-7B',
        # 'Qwen/Qwen2-1.5B-Instruct'  # Uncomment if needed
    ]
    
    # Use final adapter if available, fallback to local trained model
    if os.path.exists(FINAL_ADAPTER_PATH):
        adapter_path = FINAL_ADAPTER_PATH
        logger.info(f"Using final adapter from: {FINAL_ADAPTER_PATH}")
    elif os.path.exists(LOCAL_TRAINED_MODEL_PATH):
        adapter_path = LOCAL_TRAINED_MODEL_PATH
        logger.info(f"Using local trained model from: {LOCAL_TRAINED_MODEL_PATH}")
    else:
        adapter_path = None
        logger.info("No trained adapter found, using base models only")
    
    try:
        # Initialize chat interface
        chat_interface = EC2ChatInterface(MODELS, adapter_path)
        
        # Run interactive chat
        chat_interface.interactive_chat()
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 