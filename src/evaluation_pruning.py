import os
import torch
import torch.nn as nn

from thop import profile
from data_loading import get_wikitext2
from tqdm import tqdm
import time

from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

from data_loading import get_loaders

import numpy as np

import io

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#################################################################################################
#                                               UTILS                                           #
#################################################################################################


def count_parameters(model):
    total_params = 0
    zero_params = 0

    for name, param in model.named_parameters():
        if "weight" in name:  # Only count weights, not biases
            params = param.numel()
            zeros = (param == 0).sum().item()
            total_params += params
            zero_params += zeros

    return {"zero_params": zero_params, "total_params": total_params}


#################################################################################################
#                                        PERPLEXITY EVALUATION                                  #
#################################################################################################


def get_wikitext2(seq_len, tokenizer):
    """
    Load WikiText-2 dataset.

    Args:
        seq_len (int): Sequence length
        tokenizer: Tokenizer to be used (not directly used in this function)

    Returns:
        tuple: Train and test datasets
    """
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return traindata, testdata


class IndexDataset(Dataset):
    """
    Custom Dataset to handle tokenized data.
    """

    def __init__(self, tensors):
        """
        Initialize the dataset with tensors.

        Args:
            tensors (torch.Tensor): Input tensors
        """
        self.tensors = tensors

    def __getitem__(self, index):
        """
        Get a single item from the dataset.

        Args:
            index (int): Index of the item

        Returns:
            torch.Tensor: Tensor at the given index
        """
        return self.tensors[index]

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Number of items in the dataset
        """
        return len(self.tensors)


def process_data(samples, tokenizer, seq_len, field_name="text"):
    """
    Process data into fixed-length sequences.

    Args:
        samples (dict): Dataset samples
        tokenizer: Tokenizer to use
        seq_len (int): Sequence length
        field_name (str, optional): Name of the text field. Defaults to 'text'.

    Returns:
        IndexDataset: Processed dataset with fixed-length sequences
    """
    # Tokenize the entire dataset
    test_ids = tokenizer(
        "\n\n".join(samples[field_name]), return_tensors="pt"
    ).input_ids[0]

    # Create batches of fixed sequence length
    test_ids_batch = []
    nsamples = test_ids.numel() // seq_len
    for i in range(nsamples):
        batch = test_ids[(i * seq_len) : ((i + 1) * seq_len)]
        test_ids_batch.append(batch)

    # Stack batches into a single tensor
    test_ids_batch = torch.stack(test_ids_batch)
    return IndexDataset(tensors=test_ids_batch)


def get_loaders(name, tokenizer, seq_len=2048, batch_size=8):
    """
    Get data loaders for a specific dataset.

    Args:
        name (str): Name of the dataset
        tokenizer: Tokenizer to use
        seq_len (int, optional): Sequence length. Defaults to 2048.
        batch_size (int, optional): Batch size. Defaults to 8.

    Returns:
        tuple: Train data and test data loader
    """
    if "wikitext2" in name:
        train_data, test_data = get_wikitext2(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, "text")
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_data, test_loader


def llama_eval(model, test_loader, device):
    """
    Evaluate the model and calculate perplexity.

    Args:
        model: Language model to evaluate
        test_loader: DataLoader with test data
        device: Device to run the evaluation on

    Returns:
        float: Perplexity of the model
    """
    model.eval()  # Set model to evaluation mode
    nlls = []

    for batch in tqdm(test_loader, desc="Evaluating"):
        batch = batch.to(device)

        # Forward pass
        with torch.no_grad():
            output = model(batch)
            lm_logits = output.logits

        # Prepare for loss calculation
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        # Calculate loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        nlls.append(loss)

    # Calculate perplexity
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl


def evaluate_model_perplexity(
    model, tokenizer, datasets, device, seq_len=128, batch_size=4
):
    """
    Evaluate perplexity across multiple datasets.

    Args:
        model: Language model to evaluate
        tokenizer: Tokenizer used for preprocessing
        datasets (list): List of dataset names to evaluate
        device: Device to run the evaluation on
        seq_len (int, optional): Sequence length. Defaults to 128.
        batch_size (int, optional): Batch size. Defaults to 4.

    Returns:
        dict: Perplexity scores for each dataset
    """
    metric = {}
    for dataset in datasets:
        try:
            _, test_loader = get_loaders(
                dataset, tokenizer, seq_len=seq_len, batch_size=batch_size
            )
            ppl = llama_eval(model, test_loader, device)
            metric[dataset] = ppl
            print(f"Perplexity for {dataset}: {ppl}")
        except Exception as e:
            print(f"Error evaluating {dataset}: {e}")

    return metric


#################################################################################################
#                                       MEMORY SIZE EVALUATION                                  #
#################################################################################################


def get_memory_usage(model, ratio):
    non_zero_params = sum((p != 0).sum().item() for p in model.parameters())

    memory_usage = non_zero_params * 4  # memory in bytes

    memory_usage_MB = memory_usage / (1024 * 1024)

    return memory_usage_MB


#################################################################################################
#                                       INFERENCE TIME EVALUATION                               #
#################################################################################################


def measure_inference_time(model, nsamples, seed, seqlen, tokenizer):
    # Get data samples
    trainloader, testenc = get_wikitext2(nsamples, seed, seqlen, tokenizer)
    device = next(model.parameters()).device

    # Measure inference time
    start_time = time.time()
    for inp, tar in trainloader:
        inp = inp.to(device)
        model(inp)
    inference_time = (time.time() - start_time) / nsamples

    print(f"Average Inference Time: {inference_time:.4f} seconds")
    return inference_time


#################################################################################################
#                                      PROMPT ANSWER EVALUATION                                 #
#################################################################################################


def generate_text(model, tokenizer, prompt, max_length=50):
    """
    Generate text using the pruned model

    Args:
        model (OPTForCausalLM): Pruned OPT model
        tokenizer (OPTTokenizer): Tokenizer
        prompt (str): Input text prompt
        max_length (int): Maximum length of generated text

    Returns:
        str: Generated text
    """

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text


PROMPT = """
    A young girl named Lila discovers an ancient book in the attic of her family home. 
    The book is said to contain powerful secrets, but it is written in a language no one can understand…
    """

#################################################################################################
#                                    ECOLOGICAL IMPACT EVALUATION                               #
#################################################################################################


def calculate_ecological_impact(model_size_in_Mo, inference_time):
    # Constants for ecological impact calculation (simplified example metrics)
    ENERGY_PER_BYTE = 1e-6  # Energy consumption per byte of model size (Joules)
    ENERGY_PER_SECOND = 0.1  # Energy consumption per second of inference (Joules)
    CO2_PER_JOULE = 0.000233  # Average CO2 emissions per joule (grams)

    # Convert model size from MB to bytes (1 MB = 1e6 bytes)
    model_size_in_bytes = model_size_in_Mo * 1e6

    # Calculate energy consumption for the model size (size in bytes * energy per byte)
    size_energy = model_size_in_bytes * ENERGY_PER_BYTE  # Energy consumption in Joules

    # Calculate energy consumption for inference time
    inference_energy = (
        inference_time * ENERGY_PER_SECOND
    )  # Energy consumption in Joules

    # Total energy consumption
    total_energy = size_energy + inference_energy

    # Calculate CO2 emissions (CO2 per joule of energy consumed)
    co2_emissions = total_energy * CO2_PER_JOULE  # CO2 emissions in grams

    # Return the ecological impact metrics
    return {
        "energy_consumption_joules": total_energy,
        "co2_emissions_grams": co2_emissions,
        "size_energy_joules": size_energy,
        "inference_energy_joules": inference_energy,
    }


#################################################################################################
#                                       FLOPS EVALUATION                                        #
#################################################################################################


def measure_model_flops(model, input_sample):
    """
    Measure the number of FLOPs for the model.

    Args:
        model (torch.nn.Module): The model to evaluate
        input_sample (torch.Tensor): A sample input to the model

    Returns:
        dict: FLOPs and related computational complexity metrics
    """
    # Ensure the model is in evaluation mode
    model.eval()

    try:
        # Use thop's profile to calculate FLOPs
        macs, params = profile(model, inputs=(input_sample,), verbose=False)

        return {
            "total_flops": macs
            * 2,  # MACs are multiplications, so multiply by 2 to get FLOPs
            "total_params": params,
            "computational_complexity": {
                "gflops": (macs * 2) / 1e9,
                "millions_of_flops": (macs * 2) / 1e6,
            },
        }
    except Exception as e:
        print(f"Error measuring FLOPs: {e}")
        return None


#################################################################################################
#                                         GLOBAL EVALUATION                                     #
#################################################################################################


def global_evaluation(
    modelConfig, ratio, trainloader, testloader, is_structured=False, device=device
):
    """
    Evaluate a model across multiple metrics.

    Returns a structured dictionary containing all evaluation metrics.
    """
    # Import Data

    # Inference time evaluation
    # There is no point of caculating the inference time for a unstructured model as it does not reduce the model size
    inference_time = 0
    # if is_structured:
    #     inference_time = measure_inference_time(
    #         modelConfig.model,
    #         modelConfig.nsamples,
    #         modelConfig.seed,
    #         modelConfig.seqlen,
    #         modelConfig.tokenizer,
    #     )

    # Perplexity evaluation
    datasets = ["wikitext2"]
    start_time = time.time()
    ppl_test = evaluate_model_perplexity(
        modelConfig.model, modelConfig.tokenizer, datasets, device
    )["wikitext2"]
    end_time = time.time()

    inference_time = end_time - start_time

    # Memory evaluation
    model_size_in_Mo = get_memory_usage(modelConfig.model, ratio)

    # Text generation
    generated_text = generate_text(modelConfig.model, modelConfig.tokenizer, PROMPT)

    # FLOPs evaluation
    sample_input = next(iter(trainloader))[0][0].unsqueeze(0).to(device)
    flops_metrics = measure_model_flops(modelConfig.model, sample_input)

    # Ecological impact evaluation
    ecological_impact = calculate_ecological_impact(model_size_in_Mo, inference_time)

    # Compile all metrics
    evaluation_results = {
        "model_size": model_size_in_Mo,
        "text_generation": {"generated_text": generated_text},
        "perplexity": ppl_test,
        "computational_complexity": flops_metrics,
        "inference_time": {"average_time": inference_time},
        "ecological_impact": ecological_impact,
        "metadata": {
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
        },
    }

    return evaluation_results
