# Hill Climbing Algorithms for Optimization and Machine Learning

This project contains various implementations of hill climbing algorithms and their applications in optimization and machine learning tasks. These algorithms are designed to iteratively search for better solutions by making local adjustments to the current solution.

## Table of Contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Features and Implementations](#features-and-implementations)
4. [How to Run the Code](#how-to-run-the-code)
5. [Usage Examples](#usage-examples)
6. [Outputs](#outputs)
7. [Limitations](#limitations)
8. [Acknowledgements](#acknowledgements)

---

## Introduction
Hill climbing is a heuristic optimization technique often used to solve mathematical and real-world problems. This repository demonstrates its application in:

1. **Mathematical Optimization**: Minimize a function using simple hill climbing.
2. **Resource Allocation**: Optimize resource allocation in a cloud environment to meet task demands.
3. **Feature Selection**: Select the best subset of features for a decision tree classifier using hill climbing.

---

## Requirements
To run this project, you need the following Python packages:

- `numpy`
- `matplotlib`
- `scikit-learn`

Install them using pip if they are not already installed:

```bash
pip install numpy matplotlib scikit-learn
```

---

## Features and Implementations

### 1. Hill Climbing for Function Optimization
- **Objective**: Minimize a quadratic function.
- **File**: `hillclimbing_objective.py`
- **Details**: Demonstrates basic hill climbing to find the minimum of a simple mathematical function.

### 2. Resource Allocation Optimization
- **Objective**: Minimize resource allocation costs while meeting task demands.
- **File**: `hillclimbing_cloud.py`
- **Details**: Optimizes cloud resource allocation using stochastic hill climbing.

### 3. Feature Selection for Decision Trees
#### a. Exhaustive Feature Selection
- **Objective**: Find the best subset of features to maximize decision tree accuracy.
- **File**: `feature_selection_exhaustive.py`
- **Details**: Evaluates all possible subsets of features.

#### b. Stochastic Feature Selection
- **Objective**: Use hill climbing to find the best subset of features without evaluating all subsets.
- **File**: `feature_selection_stochastic.py`
- **Details**: Applies mutation to improve the feature subset iteratively.

### 4. Classification on Synthetic Datasets
- **Objective**: Evaluate decision tree performance on synthetic datasets of varying sizes and complexities.
- **File**: `classification_datasets.py`

---

## How to Run the Code
1. Clone this repository.
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```
2. Run individual Python files to execute specific examples.
   ```bash
   python hillclimbing_objective.py
   ```

---

## Usage Examples

### 1. Function Optimization
To optimize a quadratic function:
```python
# Define range for input
bounds = np.array([[-5.0, 5.0]])
# Perform hill climbing
best, score, solutions = hillclimbing(objective, bounds, n_iterations=1000, step_size=0.1)
print(f"Optimal solution: f({best}) = {score}")
```

### 2. Resource Allocation
To optimize cloud resource allocation:
```python
# Define task demands and resource bounds
demands = np.array([2, 4, 1, 3])
bounds = np.array([[1, 10], [1, 10], [1, 10], [1, 10]])
# Perform optimization
best_allocation, best_cost, evaluations = hill_climbing_cloud(objective, demands, bounds, n_iterations=200, step_size=0.5)
print(f"Optimal allocation: {best_allocation} with cost {best_cost}")
```

### 3. Feature Selection
To find the best subset of features:
```python
# Perform stochastic feature selection
subset, score = hillclimbing(X, y, objective, n_iter=100, p_mut=0.02)
print(f"Selected features: {subset} with score {score}")
```

---

## Outputs
Each script outputs:
- Intermediate progress during iterations.
- Final optimal solutions, allocations, or selected features.
- Visualization of cost reduction or objective function behavior (if applicable).

---

## Limitations
1. **Local Optima**: Hill climbing can get stuck in local optima.
2. **Scalability**: Exhaustive feature selection is computationally expensive for high-dimensional data.
3. **Dependence on Initialization**: Results depend on the starting point.

---

## Acknowledgements
This project leverages concepts from optimization and machine learning and uses `scikit-learn` for dataset generation and evaluation.

---

