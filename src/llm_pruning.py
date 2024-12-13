from typing import Dict, Any, List
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import time

TEST_SAMPLES = [
    "Explain the concept of neural networks, including details on their structure, types, and real-world applications. Describe how they are trained, the role of activation functions, and the importance of backpropagation in optimizing the network.",
    "What is the capital of France, and what are some key historical and cultural landmarks or features of the city? Include famous monuments, significant events in history, and the role of the city in the arts and politics.",
    "Translate 'Hello' to Spanish, and provide other common greetings in Spanish along with explanations of their usage contexts, such as formal and informal settings. Include phrases like 'Good morning' and 'How are you?' with pronunciation tips.",
    "Describe the process of photosynthesis in plants, including the stages of light-dependent and light-independent reactions, the role of chlorophyll, and the significance of photosynthesis for the ecosystem. Explain how sunlight, water, and carbon dioxide contribute to the creation of glucose and oxygen.",
    "What are the causes and effects of global warming? Include the role of greenhouse gases, deforestation, and fossil fuels. Discuss both the environmental impacts, such as rising sea levels, and societal impacts, like effects on agriculture and human health.",
    "Summarize the major theories of personality psychology, such as the Big Five personality traits, Freud’s psychoanalytic theory, and Carl Rogers' humanistic approach. Explain how these theories differ and the types of behaviors or characteristics each one emphasizes.",
    "Explain how blockchain technology works, describing the role of cryptographic hashing, consensus mechanisms, and decentralization. Include the difference between public and private blockchains and mention applications beyond cryptocurrency, such as supply chain management.",
    "List the most common data structures in computer science, such as arrays, linked lists, stacks, and queues. Describe their key characteristics, use cases, and how each data structure handles storage and retrieval of data.",
]


"""
    Different Pruning Methods mentioned in the Pruning Large Language Models: A Survey” article: 
    1. Magnitude-based pruning
        Explain:

    2. Structured Pruning
        Structured pruning is a model compression technique that removes entire structural components, 
        such as neurons, channels, or attention heads, from a neural network. 
        This approach results in a more regular and dense structure, which is easier to implement 
        and more efficient for hardware, as it doesn't require special sparse matrix operations. 
        While structured pruning is less flexible than unstructured pruning, which removes individual weights, 
        it offers direct reductions in computational complexity. However, it may lead to a higher impact 
        on model accuracy. Overall, structured pruning is beneficial for simplifying models and improving 
        inference speed without the need for complex deployment strategies.
"""


class LLMPruner:
    def __init__(self, model_name: str, pruning_method: str = "magnitude"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.pruning_method = pruning_method
        self.pruning_stats = {}

    def prune_model(self, pruning_ratio: float) -> Dict[str, Any]:
        if self.pruning_method == "magnitude":
            return self._magnitude_based_pruning(pruning_ratio)
        elif self.pruning_method == "structured":  # TODO: Implement structured pruning
            return self._structured_pruning(pruning_ratio)
        else:
            raise ValueError(f"Unsupported pruning method: {self.pruning_method}")

    def _magnitude_based_pruning(self, pruning_ratio: float) -> Dict[str, Any]:
        """
        Implement magnitude-based pruning by removing weights with smallest absolute values.
        """
        total_params = 0
        pruned_params = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                weights = module.weight.data
                flat_weights = weights.abs().flatten()
                k = int(len(flat_weights) * pruning_ratio)

                # Find threshold using torch.kthvalue
                threshold = (
                    torch.kthvalue(flat_weights, k).values
                    if k > 0
                    else flat_weights.min() - 1
                )

                # Create and apply mask
                mask = torch.abs(weights) > threshold
                weights[~mask] = 0

                total_params += weights.numel()
                pruned_params += (~mask).sum().item()


        return {
            "total_parameters": total_params,
            "pruned_parameters": pruned_params,
            "compression_ratio": pruned_params / total_params,
        }


class ModelBenchmark:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.metrics = {}

    def evaluate_performance(self, test_samples):
        total_time = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            for sample in test_samples:
                for i in range(
                    3
                ):  # Calculating the same query multiples time, since we are only measure the Inference Time
                    start_time = time.time()
                    inputs = self.tokenizer(sample, return_tensors="pt").to(device)
                    outputs = self.model.generate(**inputs, max_length=100)
                    end_time = time.time()

                    total_time += end_time - start_time

        self.metrics = {"avg_inference_time": total_time / (len(test_samples) * 3)}
        return self.metrics


def main():
    model_name = "facebook/opt-350m"

    # Create two instances of the model - one for pruning, one as original
    print("Loading models...")
    pruner = LLMPruner(model_name, pruning_method="magnitude")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    original_benchmark = ModelBenchmark(pruner.model, tokenizer)
    original_metrics = original_benchmark.evaluate_performance(TEST_SAMPLES)

    pruning_stats = pruner.prune_model(pruning_ratio=0.3)
    pruned_benchmark = ModelBenchmark(pruner.model, tokenizer)
    pruned_metrics = pruned_benchmark.evaluate_performance(TEST_SAMPLES)

    print("\n=== Performance Comparison ===")
    print("Original Model Metrics:", original_metrics["avg_inference_time"])
    print("Pruned Model Metrics:", pruned_metrics["avg_inference_time"])

    time_improvement = (
        (original_metrics["avg_inference_time"] - pruned_metrics["avg_inference_time"])
        / original_metrics["avg_inference_time"]
        * 100
    )
    size_reduction = pruning_stats["compression_ratio"]

    print("\n=== Improvements ===")
    print(f"Inference Time Improvement: {time_improvement:.2f}%")
    print(f"Model Size Reduction: {size_reduction:.2f}%")


if __name__ == "__main__":
    main()
