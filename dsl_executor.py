"""
DSL Executor for 2x2 Matrix Multiplication

Handles parsing and execution of Domain Specific Language (DSL) scripts
for matrix multiplication with support for chained operations.
"""

import re
import ast


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
        if not step_line: 
            return
        if '=' not in step_line:
            raise ValueError(f"Malformed DSL step (missing '='): '{original_step_line}'")

        target_var, expression = [s.strip() for s in step_line.split('=', 1)]
        
        # Handle chained operations (e.g., M1 + M4 - M5 + M7)
        result = self._evaluate_expression(expression, original_step_line)
        self.variables[target_var] = result
    
    def _evaluate_expression(self, expression, original_step_line):
        """Evaluate an expression that may contain chained +/- operations"""
        expression = expression.strip()
        
        # Simple assignment (no operators) - check for variable names or numbers only
        # Must be: variable name, matrix element, or number (no spaces around operators)
        assign_match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*(?:\[[0-9,\s]*\])?|\-?\d+(?:\.\d+)?)\s*$", expression)
        if assign_match:
            return self._get_value(assign_match.group(1).strip())
        
        # Single binary operation (backward compatibility)
        # Improved regex to properly handle matrix elements like A[1,0] * B[0,1]
        # Pattern: (variable_or_number) (operator) (variable_or_number)
        binary_match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*(?:\[[0-9,\s]*\])?|\-?\d+(?:\.\d+)?)\s*([*+\-])\s*([A-Za-z_][A-Za-z0-9_]*(?:\[[0-9,\s]*\])?|\-?\d+(?:\.\d+)?)\s*$", expression)
        if binary_match:
            op1_name = binary_match.group(1).strip()
            operator = binary_match.group(2).strip()
            op2_name = binary_match.group(3).strip()
            val1 = self._get_value(op1_name)
            val2 = self._get_value(op2_name)
            if operator == '+': 
                return val1 + val2
            elif operator == '*': 
                return val1 * val2
            elif operator == '-': 
                return val1 - val2
            else: 
                raise ValueError(f"Unsupported operator '{operator}' in expression: '{expression}'")
        
        # Chained operations (e.g., M1 + M4 - M5 + M7 or A[1,0] * B[0,1] + A[1,1] * B[1,1])
        # Handle complex expressions by evaluating sub-expressions first
        
        # Split by + and - while preserving operators, but be careful with multiplication
        # Use a more sophisticated approach to handle nested expressions
        if '+' in expression or (expression.count('-') > expression.count('* -') and expression.count('-') > expression.count('+ -')):
            # This is a chained addition/subtraction expression
            # Split by + and - while preserving operators
            tokens = re.split(r'(\s*[+-]\s*)', expression)
            
            if len(tokens) == 1:
                raise ValueError(f"Malformed expression: '{expression}' in DSL line: '{original_step_line}'")
            
            # First term (no leading operator) - could be a complex expression
            first_term = tokens[0].strip()
            result = self._evaluate_subexpression(first_term)
            
            # Process remaining terms with their operators
            i = 1
            while i < len(tokens):
                if i + 1 >= len(tokens):
                    break
                operator = tokens[i].strip()
                operand = tokens[i + 1].strip()
                
                # Skip empty tokens from splitting
                if not operator or not operand:
                    i += 2
                    continue
                    
                # Evaluate the operand (could be a complex expression)
                value = self._evaluate_subexpression(operand)
                
                if operator == '+': 
                    result += value
                elif operator == '-': 
                    result -= value
                else: 
                    raise ValueError(f"Unsupported operator '{operator}' in chained expression: '{expression}'")
                
                i += 2
            
            return result
        else:
            # Single complex expression or multiplication - shouldn't reach here normally
            raise ValueError(f"Could not parse complex expression: '{expression}' in DSL line: '{original_step_line}'")
    
    def _evaluate_subexpression(self, subexpr):
        """Evaluate a sub-expression that could be a variable or a binary operation"""
        subexpr = subexpr.strip()
        
        # Check if it's a simple variable/number first
        try:
            return self._get_value(subexpr)
        except ValueError:
            pass
        
        # Check if it's a binary operation
        binary_match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*(?:\[[0-9,\s]*\])?|\-?\d+(?:\.\d+)?)\s*([*+\-])\s*([A-Za-z_][A-Za-z0-9_]*(?:\[[0-9,\s]*\])?|\-?\d+(?:\.\d+)?)\s*$", subexpr)
        if binary_match:
            op1_name = binary_match.group(1).strip()
            operator = binary_match.group(2).strip()
            op2_name = binary_match.group(3).strip()
            val1 = self._get_value(op1_name)
            val2 = self._get_value(op2_name)
            if operator == '+': 
                return val1 + val2
            elif operator == '*': 
                return val1 * val2
            elif operator == '-': 
                return val1 - val2
            else: 
                raise ValueError(f"Unsupported operator '{operator}' in subexpression: '{subexpr}'")
        
        # If we can't parse it, raise an error
        raise ValueError(f"Could not evaluate subexpression: '{subexpr}'")

    def run_dsl_and_get_c(self, dsl_script_string):
        steps = dsl_script_string.strip().split('\n')
        for step in steps:
            clean_step = step.strip()
            if clean_step: 
                self.execute_step(clean_step)
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