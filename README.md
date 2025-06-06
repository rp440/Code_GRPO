# GRPO Matrix Multiplication DSL Project

## Overview

This project implements a **Group Preference Optimization (GRPO)** approach to train language models for generating Domain-Specific Language (DSL) scripts that perform optimized 2x2 matrix multiplication. The goal is to teach AI models to produce matrix multiplication algorithms using **7 or fewer multiplications** (compared to the standard 8 multiplications).

## What is this project about?

### Core Concept
The project trains language models to generate DSL scripts that implement efficient matrix multiplication algorithms, specifically focusing on:
- **2x2 matrix multiplication** (C = A × B)
- **Constraint**: Use ≤ 7 multiplications (implementing algorithms like Strassen's)
- **DSL format**: Structured intermediate variables and operations

### Key Components

#### 1. **DSL (Domain-Specific Language)**
A custom language for expressing matrix multiplication operations:
- **M-variables**: Intermediate multiplication results (`M1 = A[0,0] * B[0,0]`)
- **S-variables**: Sum operations (`S1 = M1 + M2`)
- **Output format**: Final matrix elements (`C[0,0] = S1`)

#### 2. **GRPO Training** (`GPRO_matmul.py`)
- Uses **Group Preference Optimization** to train models
- **Base model**: Qwen2.5-1.5B (configurable)
- **Reward function**: Validates DSL correctness and efficiency
- **Dataset**: JSONL format with matrix pairs and expected DSL outputs

#### 3. **Inference Engine** (`inference_grpo_matmul.py`)
- Loads trained GRPO models
- Generates DSL scripts for given matrix pairs
- Evaluates solution correctness and efficiency
- Supports batch testing and Pass@K evaluation

#### 4. **Additional Variants**
- `Extra/gpro_3b.py`: Configuration for larger 3B model training
- `Extra/inference.py`: Alternative inference implementation

### Technical Approach

#### Training Process
1. **Data Preparation**: Matrix pairs with corresponding optimal DSL scripts
2. **Reward Modeling**: Validates DSL syntax, correctness, and multiplication count
3. **GRPO Training**: Optimizes model to generate better DSL scripts
4. **Evaluation**: Tests on unseen matrix pairs

#### DSL Validation
The system automatically:
- Parses generated DSL scripts
- Executes operations step-by-step
- Verifies mathematical correctness
- Counts multiplication operations
- Compares results with ground truth

#### Optimization Goal
Traditional 2x2 matrix multiplication requires 8 multiplications:
```
C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]  # 2 multiplications
C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]  # 2 multiplications  
C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]  # 2 multiplications
C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]  # 2 multiplications
```

The goal is to find algorithms (like Strassen's) that use only 7 multiplications through clever intermediate variable reuse.

## Project Structure

```
├── GPRO_matmul.py           # Main GRPO training script
├── inference_grpo_matmul.py # Inference and evaluation script  
├── Extra/
│   ├── gpro_3b.py          # 3B model variant training
│   └── inference.py         # Alternative inference implementation
└── README.md               # This file
```

## Use Cases

- **Research**: Studying AI's ability to discover mathematical optimizations
- **Education**: Teaching efficient algorithm generation
- **Optimization**: Exploring automated discovery of mathematical shortcuts
- **Language Models**: Training models for structured mathematical reasoning

## Key Features

- **Automatic DSL Generation**: Models learn to produce valid DSL scripts
- **Efficiency Optimization**: Focus on minimizing multiplication operations
- **Correctness Validation**: Automatic verification of mathematical accuracy
- **Flexible Training**: Support for different model sizes and configurations
- **Comprehensive Evaluation**: Pass@K metrics and batch testing capabilities

This project demonstrates how reinforcement learning techniques like GRPO can be applied to teach language models domain-specific mathematical reasoning and optimization. 