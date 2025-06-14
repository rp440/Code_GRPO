import re
import torch
import ast
import random
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import argparse
from typing import List, Tuple, Dict, Any

# --- DSL Executor (same as training script) ---
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

# --- Configuration ---
SYSTEM_MESSAGE_V1 = """You are a DSL-only generator. Every line you output must be one of:
  1. Intermediate sum/difference:  
     T<k> = <term> + <term>  
     T<k> = <term> - <term>  
  2. Scalar multiplication:  
     M<k> = <term> * <term>  
  3. Output assignment:  
     C[i,j] = <term> + <term>  
     C[i,j] = <term> - <term>  
     C[i,j] = <term>  

Where each `<term>` is exactly one of:
  A[0,0]  A[0,1]  A[1,0]  A[1,1]  
  B[0,0]  B[0,1]  B[1,0]  B[1,1]  
  T<k>     M<k>  

—and `<i>,<j>,<k>` are digits `0` or `1` for this 2×2 task.  
No other variables, operators, or text are allowed. Lines missing "=" or using natural language will be rejected by the evaluator.

OUTPUT ONLY DSL CODE. NO EXPLANATIONS. NO COMMENTARY."""

SYSTEM_MESSAGE_V2 = """STRICT DSL GENERATOR - OUTPUT FORMAT ONLY

You must output EXACTLY this format for 2x2 matrix multiplication:

<DSL>
[DSL_LINES_HERE]
</DSL>

DSL SYNTAX RULES:
- M1 to M7 = multiplication operations (M<k> = <term> * <term>)
- T1 to T9 = intermediate sums/differences (T<k> = <term> + <term> OR T<k> = <term> - <term>)
- C[0,0], C[0,1], C[1,0], C[1,1] = final outputs
- Valid terms: A[0,0], A[0,1], A[1,0], A[1,1], B[0,0], B[0,1], B[1,0], B[1,1], M1-M7, T1-T9

REQUIREMENTS:
- Use EXACTLY 7 multiplication lines (M1 through M7)
- Compute all four C[i,j] values
- NO explanatory text outside <DSL> tags
- NO natural language inside <DSL> tags"""

SYSTEM_MESSAGE_V3 = """Matrix multiplication DSL generator. Generate code using Strassen-like optimizations.

EXAMPLE OUTPUT:
<DSL>
M1 = A[0,0] * B[0,0]
M2 = A[0,1] * B[1,0]
T1 = M1 + M2
C[0,0] = T1
M3 = A[0,0] * B[0,1]
M4 = A[0,1] * B[1,1]
T2 = M3 + M4
C[0,1] = T2
M5 = A[1,0] * B[0,0]
M6 = A[1,1] * B[1,0]
T3 = M5 + M6
C[1,0] = T3
M7 = A[1,0] * B[0,1]
C[1,1] = M7
</DSL>

Generate a 7-multiplication DSL script using efficient algorithms. Use intermediate variables T1, T2, etc. for sums. Output ONLY the <DSL></DSL> block."""

# Original system message for comparison
SYSTEM_MESSAGE = """You are an AI assistant specialized in generating Domain Specific Language (DSL) scripts for 2x2 matrix multiplication. You can provide explanations, but must wrap your DSL code in <DSL></DSL> tags.

EXAMPLE DSL OUTPUT FORMAT:
For matrices A=[[1,2],[3,4]] and B=[[5,6],[7,8]], a valid response would be:

I'll generate the DSL script for matrix multiplication:

<DSL>
M1 = A[0,0] * B[0,0]
M2 = A[0,1] * B[1,0]
S1 = M1 + M2
C[0,0] = S1
M3 = A[0,0] * B[0,1]
M4 = A[0,1] * B[1,1]
S2 = M3 + M4
C[0,1] = S2
M5 = A[1,0] * B[0,0]
M6 = A[1,1] * B[1,0]
S3 = M5 + M6
C[1,0] = S3
M7 = A[1,0] * B[0,1]
M8 = A[1,1] * B[1,1]
S4 = M7 + M8
C[1,1] = S4
</DSL>

This uses 8 multiplications, but can be optimized using techniques like Strassen's algorithm.

YOUR TASK:
Generate a DSL script that performs 2x2 matrix multiplication using 7 or fewer multiplications. You may provide explanations outside the DSL tags, but the actual code must be within <DSL></DSL> tags.

DSL SYNTAX RULES:
- M variables: Store multiplication results (e.g., M1 = A[0,0] * B[0,0])
- S variables: Store addition/subtraction results (e.g., S1 = M1 + M2)
- Matrix elements: A[row,col] and B[row,col] where row,col ∈ {0,1}
- Final output: C[row,col] = result
- Operations: + (addition), * (multiplication), - (subtraction)
- Variable assignment: VAR = expression

REQUIREMENTS:
- Use ≤7 multiplications total within the <DSL></DSL> tags
- Compute all four elements: C[0,0], C[0,1], C[1,0], C[1,1]
- Wrap DSL code in <DSL></DSL> tags
- You may add explanations outside the tags

If you cannot determine a valid sequence, output: Error: Cannot determine full sequence."""

# User prompts for different test approaches
USER_PROMPT_V1 = """Generate a DSL program for multiplying two 2×2 matrices under the strict grammar, using **exactly seven** `M# = ... * ...` lines, and ensure it computes all four `C[i,j]` correctly. Output **only** the DSL code—no commentary or extra text."""

USER_PROMPT_V2 = """Create 2x2 matrix multiplication DSL with exactly 7 multiplications. Use Strassen-like optimization. Output format must be <DSL>[code]</DSL> only."""

USER_PROMPT_V3 = "Generate the DSL script to calculate C = A * B for the given 2x2 matrices, using 7 or fewer multiplications."

DEFAULT_USER_PROMPT = "Generate the DSL script to calculate C = A * B for the given 2x2 matrices, using 7 or fewer multiplications."

# --- Model Loading ---
class GRPOMatMulInference:
    def __init__(self, base_model_name: str, trained_model_path: str):
        self.base_model_name = base_model_name
        self.trained_model_path = trained_model_path
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.load_model()
    
    def load_model(self):
        """Load the GRPO-trained model for inference"""
        import os
        
        print(f"Loading model...")
        print(f"Base model: {self.base_model_name}")
        print(f"Trained adapter path: {self.trained_model_path}")
        
        # Check if adapter path exists
        if not os.path.exists(self.trained_model_path):
            print(f"⚠️  Adapter path '{self.trained_model_path}' not found.")
            print("📦 Loading base model without adapter...")
            self._load_base_model_only()
            return
        
        try:
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load adapter
            self.model = PeftModel.from_pretrained(base_model, self.trained_model_path)
            self.model = self.model.merge_and_unload()  # Merge for faster inference
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name, 
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.padding_side == 'right':
                self.tokenizer.padding_side = 'left'
                
            # Create pipeline (no device when using accelerate)
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )
            
            print("✅ GRPO-trained model with adapter loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading adapter: {e}")
            print("📦 Falling back to base model without adapter...")
            self._load_base_model_only()
    
    def _load_base_model_only(self):
        """Load only the base model without adapter"""
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.padding_side == 'right':
                self.tokenizer.padding_side = 'left'
                
            # Create pipeline (no device when using accelerate)
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )
            print("✅ Base model loaded successfully!")
        except Exception as fallback_e:
            raise Exception(f"Failed to load base model: {fallback_e}")
    
    def generate_dsl(self, matrix_a: List[List[int]], matrix_b: List[List[int]], 
                     num_generations: int = 1, temperature: float = 0.7, 
                     use_batching: bool = True) -> List[str]:
        """Generate DSL scripts for given matrices with optional batching"""
        user_query = f"{DEFAULT_USER_PROMPT} (A={matrix_a}, B={matrix_b})"
        
        chat_messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_query}
        ]
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
        
        if use_batching and num_generations > 1:
            print(f"**BATCH MODE**: Generating {num_generations} completions in batch...")
            # **BATCH PROCESSING** - Generate all completions at once
            outputs = self.pipeline(
                formatted_prompt,
                max_new_tokens=350,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=num_generations,
                batch_size=min(num_generations, 8)  # **RAM OPTIMIZED** batch size
            )
            
            results = []
            for output in outputs:
                generated_text = output['generated_text']
                assistant_reply = self._extract_and_clean_response(generated_text, formatted_prompt)
                results.append(assistant_reply)
                
        else:
            print(f"**SEQUENTIAL MODE**: Generating {num_generations} completions sequentially...")
            # **SEQUENTIAL PROCESSING** - Original method for comparison
            results = []
            for i in range(num_generations):
                outputs = self.pipeline(
                    formatted_prompt,
                    max_new_tokens=350,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
                
                generated_text = outputs[0]['generated_text']
                assistant_reply = self._extract_and_clean_response(generated_text, formatted_prompt)
                results.append(assistant_reply)
        
        return results
    
    def _extract_and_clean_response(self, generated_text: str, formatted_prompt: str) -> str:
        """**HELPER**: Extract and clean assistant response from generated text"""
        # Extract assistant response
        if generated_text.startswith(formatted_prompt):
            assistant_reply = generated_text[len(formatted_prompt):].strip()
        else:
            # Fallback extraction
            assistant_marker = "<|im_start|>assistant"
            last_idx = generated_text.rfind(assistant_marker)
            if last_idx != -1:
                start_idx = generated_text.find("\n", last_idx)
                if start_idx != -1:
                    assistant_reply = generated_text[start_idx+1:].strip()
                else:
                    assistant_reply = generated_text
            else:
                assistant_reply = generated_text
        
        # Clean tokens
        tokens_to_clean = ["<|im_end|>", "<|endoftext|>"]
        if self.tokenizer.eos_token:
            tokens_to_clean.append(self.tokenizer.eos_token)
        
        for token in tokens_to_clean:
            if assistant_reply.endswith(token):
                assistant_reply = assistant_reply[:-len(token)].strip()
        
        # Clean and format DSL
        lines = assistant_reply.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        final_dsl = "\n".join(cleaned_lines)
        
        return final_dsl

def evaluate_dsl_solution(dsl_script: str, matrix_a: List[List[int]], 
                         matrix_b: List[List[int]]) -> Dict[str, Any]:
    """Evaluate a single DSL solution"""
    expected_c = manual_matrix_multiply_2x2(matrix_a, matrix_b)
    
    try:
        # Count multiplications
        num_mults = sum(1 for line in dsl_script.split('\n') 
                       if re.search(r"=\s*[\w\[\],\s\.\d\-]+\s*\*\s*[\w\[\],\s\.\d\-]+", line.strip()))
        
        # Execute DSL
        executor = DSLExecutor(matrix_a, matrix_b)
        result_c = executor.run_dsl_and_get_c(dsl_script)
        
        # Check correctness
        is_correct = result_c == expected_c
        is_efficient = num_mults <= 7
        
        return {
            "success": True,
            "correct": is_correct,
            "efficient": is_efficient,
            "multiplications": num_mults,
            "result": result_c,
            "expected": expected_c,
            "error": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "correct": False,
            "efficient": False,
            "multiplications": 0,
            "result": None,
            "expected": expected_c,
            "error": str(e)
        }

def pass_at_k_evaluation(inference_model: GRPOMatMulInference, 
                        test_cases: List[Tuple[List[List[int]], List[List[int]]]], 
                        k: int = 8) -> Dict[str, Any]:
    """Evaluate Pass@k performance"""
    print(f"\n🔬 Running Pass@{k} Evaluation")
    print(f"Test cases: {len(test_cases)}")
    print("-" * 50)
    
    total_cases = len(test_cases)
    passed_cases = 0
    all_results = []
    
    for i, (matrix_a, matrix_b) in enumerate(test_cases):
        print(f"\nTest case {i+1}/{total_cases}: A={matrix_a}, B={matrix_b}")
        
        # Generate k solutions using **BATCH MODE**
        dsl_solutions = inference_model.generate_dsl(matrix_a, matrix_b, num_generations=k, temperature=0.8, use_batching=True)
        
        # Evaluate each solution
        case_passed = False
        case_results = []
        
        for j, dsl in enumerate(dsl_solutions):
            result = evaluate_dsl_solution(dsl, matrix_a, matrix_b)
            case_results.append(result)
            
            if result["correct"] and result["efficient"]:
                case_passed = True
                print(f"  ✅ Solution {j+1}: PASS (correct & efficient, {result['multiplications']} mults)")
            elif result["correct"]:
                print(f"  ⚠️ Solution {j+1}: Correct but inefficient ({result['multiplications']} mults)")
            else:
                print(f"  ❌ Solution {j+1}: {result['error'] or 'Incorrect result'}")
        
        if case_passed:
            passed_cases += 1
            print(f"  🎯 Case {i+1}: PASSED")
        else:
            print(f"  💔 Case {i+1}: FAILED")
        
        all_results.append({
            "test_case": i+1,
            "matrix_a": matrix_a,
            "matrix_b": matrix_b,
            "passed": case_passed,
            "solutions": case_results
        })
    
    pass_rate = passed_cases / total_cases
    print(f"\n📊 Pass@{k} Results:")
    print(f"Passed: {passed_cases}/{total_cases} ({pass_rate:.2%})")
    
    return {
        "pass_rate": pass_rate,
        "passed_cases": passed_cases,
        "total_cases": total_cases,
        "k": k,
        "detailed_results": all_results
    }

def chat_interface(inference_model: GRPOMatMulInference):
    """Interactive chat interface"""
    print("\n💬 Chat Interface - Matrix Multiplication DSL Generator")
    print("=" * 60)
    print("Commands:")
    print("  - Enter two 2x2 matrices: e.g., '[[1,2],[3,4]] [[5,6],[7,8]]'")
    print("  - 'random' - Generate random matrices")
    print("  - 'quit' or 'exit' - Exit chat")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n🔢 Enter matrices or command: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'random':
                matrix_a = [[random.randint(1, 9) for _ in range(2)] for _ in range(2)]
                matrix_b = [[random.randint(1, 9) for _ in range(2)] for _ in range(2)]
                print(f"🎲 Generated: A={matrix_a}, B={matrix_b}")
            else:
                # Parse matrices
                parts = user_input.split()
                if len(parts) != 2:
                    print("❌ Please enter exactly two matrices")
                    continue
                
                try:
                    matrix_a = ast.literal_eval(parts[0])
                    matrix_b = ast.literal_eval(parts[1])
                    
                    if not (isinstance(matrix_a, list) and len(matrix_a) == 2 and 
                           isinstance(matrix_a[0], list) and len(matrix_a[0]) == 2 and
                           isinstance(matrix_b, list) and len(matrix_b) == 2 and 
                           isinstance(matrix_b[0], list) and len(matrix_b[0]) == 2):
                        raise ValueError("Matrices must be 2x2")
                        
                except Exception as e:
                    print(f"❌ Invalid matrix format: {e}")
                    print("Example: [[1,2],[3,4]] [[5,6],[7,8]]")
                    continue
            
            # Generate DSL using **BATCH MODE**
            print("**GENERATING DSL**...")
            dsl_solutions = inference_model.generate_dsl(matrix_a, matrix_b, num_generations=3, temperature=0.7, use_batching=True)
            
            print(f"\n📋 Generated {len(dsl_solutions)} solutions:")
            
            for i, dsl in enumerate(dsl_solutions):
                print(f"\n--- Solution {i+1} ---")
                print(dsl)
                
                # Evaluate
                result = evaluate_dsl_solution(dsl, matrix_a, matrix_b)
                
                if result["success"]:
                    status = "✅ CORRECT" if result["correct"] else "❌ INCORRECT"
                    efficiency = "EFFICIENT" if result["efficient"] else "INEFFICIENT"
                    print(f"Result: {status}, {efficiency} ({result['multiplications']} multiplications)")
                    print(f"Expected: {result['expected']}")
                    print(f"Got:      {result['result']}")
                else:
                    print(f"❌ Error: {result['error']}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def generate_test_cases(num_cases: int = 10) -> List[Tuple[List[List[int]], List[List[int]]]]:
    """Generate random test cases"""
    test_cases = []
    for _ in range(num_cases):
        matrix_a = [[random.randint(1, 9) for _ in range(2)] for _ in range(2)]
        matrix_b = [[random.randint(1, 9) for _ in range(2)] for _ in range(2)]
        test_cases.append((matrix_a, matrix_b))
    return test_cases

def test_prompts(inference_model: GRPOMatMulInference, 
                matrix_a: List[List[int]], matrix_b: List[List[int]], 
                num_generations: int = 3) -> Dict[str, Any]:
    """Test different prompt configurations"""
    print(f"\n🧪 Testing Prompt Configurations")
    print(f"Test matrices: A={matrix_a}, B={matrix_b}")
    print(f"Generations per prompt: {num_generations}")
    print("=" * 60)
    
    # Prompt configurations to test
    prompt_configs = [
        {
            "name": "V1 - Strict DSL Only",
            "system": SYSTEM_MESSAGE_V1,
            "user": USER_PROMPT_V1
        },
        {
            "name": "V2 - Format Enforced",
            "system": SYSTEM_MESSAGE_V2, 
            "user": USER_PROMPT_V2
        },
        {
            "name": "V3 - Example Driven",
            "system": SYSTEM_MESSAGE_V3,
            "user": USER_PROMPT_V3
        },
        {
            "name": "Original",
            "system": SYSTEM_MESSAGE,
            "user": f"{DEFAULT_USER_PROMPT} (A={matrix_a}, B={matrix_b})"
        }
    ]
    
    results = {}
    
    for i, config in enumerate(prompt_configs):
        print(f"\n🔬 Testing: {config['name']} ({i+1}/{len(prompt_configs)})")
        print("-" * 40)
        
        # Show the prompts being used
        print(f"📝 SYSTEM MESSAGE:")
        print(f"   {config['system'][:150]}...")
        print(f"📝 USER MESSAGE:")
        print(f"   {config['user'][:100]}...")
        print()
        
        # Create custom chat messages for this configuration
        user_content = config["user"]
        # Add matrix info if not already present
        if f"A={matrix_a}" not in user_content:
            user_content = f"{user_content} (A={matrix_a}, B={matrix_b})"
            
        chat_messages = [
            {"role": "system", "content": config["system"]},
            {"role": "user", "content": user_content}
        ]
        
        # Format prompt
        formatted_prompt = inference_model.tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
        
        # Generate solutions
        try:
            outputs = inference_model.pipeline(
                formatted_prompt,
                max_new_tokens=350,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=inference_model.tokenizer.pad_token_id,
                eos_token_id=inference_model.tokenizer.eos_token_id,
                num_return_sequences=num_generations,
                batch_size=min(num_generations, 4)
            )
            
            # Process results
            config_results = []
            for i, output in enumerate(outputs):
                generated_text = output['generated_text']
                assistant_reply = inference_model._extract_and_clean_response(generated_text, formatted_prompt)
                
                print(f"\n--- Generation {i+1} ---")
                print(f"Raw output: {assistant_reply[:200]}...")
                
                # Extract DSL from <DSL> tags if present
                dsl_match = re.search(r'<DSL>(.*?)</DSL>', assistant_reply, re.DOTALL)
                if dsl_match:
                    dsl_script = dsl_match.group(1).strip()
                    print(f"Extracted DSL: ✅")
                else:
                    dsl_script = assistant_reply
                    print(f"No DSL tags found: ⚠️")
                
                # Evaluate
                result = evaluate_dsl_solution(dsl_script, matrix_a, matrix_b)
                
                status_icons = {
                    (True, True): "✅🚀",   # Correct & Efficient
                    (True, False): "✅⚠️",  # Correct but Inefficient  
                    (False, True): "❌🚀",  # Incorrect but Efficient
                    (False, False): "❌❌"   # Incorrect & Inefficient
                }
                
                icon = status_icons.get((result["correct"], result["efficient"]), "❌❌")
                
                if result["success"]:
                    print(f"Result: {icon} ({result['multiplications']} mults)")
                else:
                    print(f"Result: ❌ Error - {result['error'][:100]}...")
                
                config_results.append({
                    "generation": i+1,
                    "raw_output": assistant_reply,
                    "dsl_extracted": dsl_script,
                    "evaluation": result
                })
            
            # Summary for this configuration
            successful = sum(1 for r in config_results if r["evaluation"]["success"])
            correct = sum(1 for r in config_results if r["evaluation"]["correct"])
            efficient = sum(1 for r in config_results if r["evaluation"]["efficient"])
            perfect = sum(1 for r in config_results if r["evaluation"]["correct"] and r["evaluation"]["efficient"])
            
            print(f"\n📊 {config['name']} Summary:")
            print(f"  Successful: {successful}/{num_generations}")
            print(f"  Correct: {correct}/{num_generations}")
            print(f"  Efficient (≤7 mults): {efficient}/{num_generations}")
            print(f"  Perfect (correct + efficient): {perfect}/{num_generations}")
            
            results[config['name']] = {
                "config": config,
                "generations": config_results,
                "summary": {
                    "successful": successful,
                    "correct": correct,
                    "efficient": efficient,
                    "perfect": perfect,
                    "total": num_generations
                }
            }
            
        except Exception as e:
            print(f"❌ Error testing {config['name']}: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            results[config['name']] = {"error": str(e)}
    
    print(f"\n🏆 Overall Prompt Test Results:")
    print("=" * 60)
    
    for name, result in results.items():
        if "error" in result:
            print(f"{name}: ❌ ERROR")
        else:
            summary = result["summary"]
            score = summary["perfect"] / summary["total"]
            print(f"{name}: {summary['perfect']}/{summary['total']} perfect ({score:.1%})")
    
    return results

def main():
    import sys
    
    # Filter out Jupyter kernel arguments
    filtered_args = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg == '-f':
            skip_next = True
            continue
        if arg.startswith('-f=') or 'kernel' in arg:
            continue
        filtered_args.append(arg)
    
    parser = argparse.ArgumentParser(description="GRPO Matrix Multiplication DSL Inference")
    # parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B", 
    #                    help="Base model name")
    parser.add_argument("--base_model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
                       help="Base model name")
    parser.add_argument("--trained_model", default="./Qwen2.5-1.5B-GRPO-MatMulDSL-JSONL", 
                       help="Path to trained model")
    parser.add_argument("--mode", choices=["chat", "pass_at_k", "both", "test_prompts"], default="both",
                       help="Inference mode")
    parser.add_argument("--k", type=int, default=8, 
                       help="K for Pass@k evaluation")
    parser.add_argument("--test_cases", type=int, default=10,
                       help="Number of test cases for Pass@k")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for generation (higher = more RAM usage)")
    parser.add_argument("--disable_batching", action="store_true",
                       help="Disable batch processing (use sequential generation)")
    
    args = parser.parse_args(filtered_args)
    
    print("🚀 GRPO Matrix Multiplication DSL Inference")
    print("=" * 50)
    
    # Load model
    inference_model = GRPOMatMulInference(args.base_model, args.trained_model)
    
    if args.mode == "test_prompts":
        # Test different prompt configurations
        test_matrix_a = [[1, 2], [3, 4]]
        test_matrix_b = [[5, 6], [7, 8]]
        
        results = test_prompts(inference_model, test_matrix_a, test_matrix_b, num_generations=1)
        
        # Save results
        with open("prompt_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Prompt test results saved to prompt_test_results.json")
        return
    
    if args.mode in ["pass_at_k", "both"]:
        # Generate test cases
        test_cases = generate_test_cases(args.test_cases)
        
        # Run Pass@k evaluation
        results = pass_at_k_evaluation(inference_model, test_cases, args.k)
        
        # Save results
        with open(f"pass_at_{args.k}_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to pass_at_{args.k}_results.json")
    
    if args.mode in ["chat", "both"]:
        # Run chat interface
        chat_interface(inference_model)

if __name__ == "__main__":
    main() 