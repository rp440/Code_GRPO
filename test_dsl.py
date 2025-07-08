#!/usr/bin/env python3
"""
Test script to verify the user's DSL implementation for matrix multiplication
"""

import re
import ast
from typing import List, Dict, Any

class DSLExecutor:
    """DSL Executor to test the provided implementation"""
    
    def __init__(self, matrix_a: List[List[float]], matrix_b: List[List[float]]):
        self.variables = {}
        self._validate_matrices(matrix_a, matrix_b)
        self._initialize_matrices(matrix_a, matrix_b)

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
        
        # Handle complex expressions with multiple operations
        if '+' in expression and '*' in expression:
            # Parse expressions like "M1 * B[0,0] + U1 * B[1,1]"
            parts = expression.split('+')
            if len(parts) == 2:
                left_part = parts[0].strip()
                right_part = parts[1].strip()
                
                # Parse each multiplication
                left_val = self._parse_multiplication(left_part)
                right_val = self._parse_multiplication(right_part)
                result = left_val + right_val
            else:
                raise ValueError(f"Complex expression not supported: {expression}")
        elif '-' in expression and len(expression.split('-')) == 3:
            # Handle expressions like "M1 - U3 - M2"
            parts = expression.split('-')
            result = self._get_value(parts[0].strip())
            for part in parts[1:]:
                result -= self._get_value(part.strip())
        else:
            # Parse simple operation
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
        print(f"  {target_var} = {result}")

    def _parse_multiplication(self, expr: str) -> float:
        """Parse a multiplication expression"""
        if '*' in expr:
            parts = expr.split('*')
            if len(parts) == 2:
                return self._get_value(parts[0].strip()) * self._get_value(parts[1].strip())
        return self._get_value(expr)

    def run_dsl_and_get_c(self, dsl_script: str) -> List[List[float]]:
        """Execute DSL script and return result matrix C"""
        print("Executing DSL steps:")
        for line_num, line in enumerate(dsl_script.strip().split('\n'), 1):
            try:
                print(f"Step {line_num}: {line}")
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

def manual_matrix_multiply_2x2(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Manual 2x2 matrix multiplication for verification"""
    return [
        [A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
        [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]]
    ]

def count_multiplications_in_dsl(dsl_script: str) -> int:
    """Count multiplication operations in DSL script"""
    return len(re.findall(r'\*', dsl_script))

def test_user_dsl():
    """Test the user's DSL implementation"""
    
    # User's DSL implementation
    user_dsl = """U1 = A[0,0] + B[0,1]
U2 = A[1,1] - A[0,0]
U3 = B[0,0] + B[1,0]
M1 = A[1,1] * B[1,0]
M2 = A[1,0] * M1
M3 = A[1,1] * U2
M4 = B[1,0] * B[0,1]
M5 = M1 * B[0,0] + U1 * B[1,1]
S1 = U3 - M5
C[0,0] = M1 - U3 - M2
C[0,1] = M2 - M5 - S1
C[1,0] = M2 + U2
C[1,1] = M1 + M4"""

    # Test matrices
    test_cases = [
        ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
        ([[2, 1], [1, 2]], [[3, 1], [2, 3]]),
        ([[1, 0], [0, 1]], [[5, 6], [7, 8]]),  # Identity matrix test
    ]
    
    print("Testing User's DSL Implementation")
    print("=" * 50)
    
    for i, (A, B) in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"A = {A}")
        print(f"B = {B}")
        
        try:
            # Execute user's DSL
            executor = DSLExecutor(A, B)
            dsl_result = executor.run_dsl_and_get_c(user_dsl)
            
            # Calculate expected result
            expected_result = manual_matrix_multiply_2x2(A, B)
            
            # Count multiplications
            mult_count = count_multiplications_in_dsl(user_dsl)
            
            print(f"\nResults:")
            print(f"DSL Result:      {dsl_result}")
            print(f"Expected Result: {expected_result}")
            print(f"Multiplications: {mult_count}")
            print(f"Correct: {'✅ YES' if dsl_result == expected_result else '❌ NO'}")
            print(f"Efficient: {'✅ YES' if mult_count <= 7 else '❌ NO'} (≤7 mults)")
            
        except Exception as e:
            print(f"❌ Error executing DSL: {e}")

if __name__ == "__main__":
    test_user_dsl() 