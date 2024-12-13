# Pruning Methods Evaluation Project

## Project Overview

The objective of this project is to explore and implement various pruning techniques to optimize deep learning models. Pruning is a process of reducing the number of parameters in a model to achieve better efficiency, lower computational requirements, and reduced ecological impact, while maintaining acceptable levels of accuracy and performance. This project investigates both structured and unstructured pruning methods.

### Implemented Pruning Methods

#### Unstructured Pruning
- **Magnitude Pruning**: Removes parameters with the smallest absolute weights.
- **Wanda Pruning**: Implements a more sophisticated algorithm designed to identify and prune weights based on their contributions to model performance.

#### Structured Pruning
- **Llama-3.2-1b Pruning**: Focuses on pruning entire structures (e.g., neurons or filters) in a pre-trained model, specifically using the Llama-3.2-1b model as a case study.

### Evaluation Framework
The project evaluates pruning methods by comparing models before and after pruning across multiple metrics.

## Project Structure

```
src/
|-- pruning_methods/
|   |-- magnitude.py           # Implementation of magnitude pruning
|   |-- structured.py          # Implementation of structured pruning
|   |-- wanda.py               # Implementation of Wanda pruning
|
|-- data_loading.py            # Functions to load and process datasets
|-- demo.ipynb                 # Demonstration of the implemented pruning methods
|-- evaluation_pruning.py      # Code for evaluating pruning methods
|-- llm_pruning.py             # Core logic for large language model pruning
|-- model_config.py            # Configuration for model and training parameters
|-- plot_functions.py          # Visualization utilities
|
|-- Project Proposal.pdf       # Detailed project proposal
|-- README.md                  # This file
|-- requirements.txt           # Dependencies for the project
```

## Key Functions

### Pruning Evaluation
- **`get_wikitext2_unstructured`**
  Loads and encodes the Wikitext-2 dataset for unstructured pruning experiments. The function supports:
  - Loading training and testing datasets.
  - Tokenizing and preparing datasets for evaluation.

- **`global_evaluation`**
  Evaluates a model using various metrics, providing a comprehensive analysis of the effects of pruning. This function calculates:
  - **Perplexity**: Assesses language model performance.
  - **Model Size**: Measures memory usage before and after pruning.
  - **Inference Time**: Evaluates runtime efficiency (structured models only).
  - **Computational Complexity**: Estimates FLOPs (floating-point operations).
  - **Ecological Impact**: Assesses the environmental cost of model execution.
  - **Generated Text Quality**: Evaluates the quality of text produced by the pruned model.

- **`execute_benchmark`**
  Runs the benchmark for a specified pruning method. The function:
  - Loads the necessary datasets (e.g., C4 and Wikitext-2).
  - Applies the selected pruning method (`magnitude_pruning`, `wanda_pruning`, or `structured_pruning`) with varying ratios.
  - Evaluates the pruned models using `global_evaluation` and collects results.

## Metrics and Results
The evaluation metrics provide insights into:
- **Model Size**: Impact of pruning on memory usage.
- **Perplexity**: Comparison of language modeling capabilities.
- **Inference Time**: Efficiency improvements in structured pruning.
- **Ecological Impact**: Environmental benefits of pruning.
- **Computational Complexity**: Reduction in FLOPs.

## Installation

1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

1. **Run the Benchmark**:
   - Open and run `demo.ipynb`. This notebook loads the benchmark for model pruning and provides a step-by-step demonstration of the implemented pruning methods.

2. **Execute Pruning Benchmark Programmatically**:
   - Use the `execute_benchmark` function in your Python scripts to run the pruning benchmarks. Example usage:
     ```python
     from evaluation_pruning import execute_benchmark
     results = execute_benchmark(model_config, pruning_function="wanda_pruning")
     print(results)
     ```

3. **Evaluate Models**:
   - Use `evaluation_pruning.py` to evaluate the performance of the pruned models.
   - Example:
     ```python
     from evaluation_pruning import global_evaluation
     results = global_evaluation(modelConfig, ratio=0.5, trainloader, testloader)
     print(results)
     ```

## Dependencies

The project requires the following Python packages:
- Transformers
- PyTorch
- Datasets
- NumPy
- Matplotlib
- Evaluate
- scikit-learn

Install all dependencies using `requirements.txt`:
```bash
pip install -r requirements.txt
```


