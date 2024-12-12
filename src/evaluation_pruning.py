import torch
import torch.nn as nn

from thop import profile
from data_loading import get_wikitext2
from tqdm import tqdm
import time

import io

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#################################################################################################
#                                        PERPLEXITY EVALUATION                                  #
#################################################################################################


# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []

    # Loop through each batch
    for i in tqdm(range(0, nsamples, bs), desc="Wikitext Perplexity"):
        # Calculate end index
        j = min(i + bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:, (i * model.seqlen) : (j * model.seqlen)].to(device)
        inputs = inputs.reshape(j - i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)
        )

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j - i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()


def eval_perplexity(model, testloader, device=torch.device("cuda:0")):
    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(model, testloader, 1, device)

    return ppl_test


#################################################################################################
#                                       MEMORY SIZE EVALUATION                                  #
#################################################################################################

def get_model_memory(model):
    """Calculate the memory usage of a model.

    Args:
        model (torch.nn.Module): The model to evaluate.

    Returns:
        int: Size of the model in bytes.
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_in_bytes = buffer.tell()
    buffer.close()
    return size_in_bytes


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

def generate_text(model, tokenizer, prompt, max_length=100):
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
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text

PROMPT = """
    A young girl named Lila discovers an ancient book in the attic of her family home. 
    The book is said to contain powerful secrets, but it is written in a language no one can understand…
    """

#################################################################################################
#                                    ECOLOGICAL IMPACT EVALUATION                               #
#################################################################################################

def calculate_ecological_impact(memory_size, inference_time):
    """
    Calculate ecological impact based on model size and inference time.
    Uses simplified metrics for demonstration.
    
    Args:
        memory_size (int): Size of the model in bytes
        inference_time (float): Average inference time in seconds
    
    Returns:
        dict: Dictionary containing ecological impact metrics
    """
    # Constants for ecological impact calculation (simplified example metrics)
    ENERGY_PER_BYTE = 1e-9  # Energy consumption per byte of model size (kWh)
    ENERGY_PER_SECOND = 1e-4  # Energy consumption per second of inference (kWh)
    CO2_PER_KWH = 0.233  # Average CO2 emissions per kWh (kg)

    # Calculate energy consumption
    size_energy = (memory_size * ENERGY_PER_BYTE)  # kWh
    inference_energy = (inference_time * ENERGY_PER_SECOND)  # kWh
    total_energy = size_energy + inference_energy

    # Calculate CO2 emissions
    co2_emissions = total_energy * CO2_PER_KWH  # kg

    return {
        "energy_consumption_kwh": total_energy,
        "co2_emissions_kg": co2_emissions,
        "details": {
            "size_energy_kwh": size_energy,
            "inference_energy_kwh": inference_energy
        }
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
            "total_flops": macs * 2,  # MACs are multiplications, so multiply by 2 to get FLOPs
            "total_params": params,
            "computational_complexity": {
                "gflops": (macs * 2) / 1e9,
                "millions_of_flops": (macs * 2) / 1e6
            }
        }
    except Exception as e:
        print(f"Error measuring FLOPs: {e}")
        return None

#################################################################################################
#                                         GLOBAL EVALUATION                                     #
#################################################################################################

def global_evaluation(modelConfig, is_structured=False, device=device):
    """
    Evaluate a model across multiple metrics.
    
    Returns a structured dictionary containing all evaluation metrics.
    """
    # Import Data
    trainloader, testloader = get_wikitext2(modelConfig.nsamples, modelConfig.seed, modelConfig.seqlen, modelConfig.tokenizer)

    # Perplexity evaluation
    ppl_test = eval_perplexity(modelConfig.model, testloader, device)
    
    # Memory evaluation
    memory_size = get_model_memory(modelConfig.model)
    
    # Text generation
    generated_text = generate_text(modelConfig.model, modelConfig.tokenizer, PROMPT)

    # Inference time evaluation
    # There is no point of caculating the inference time for a unstructured model as it does not reduce the model size
    inference_time = 0
    if is_structured:
        inference_time = measure_inference_time(modelConfig.model, modelConfig.nsamples, modelConfig.seed, 
                                                modelConfig.seqlen, modelConfig.tokenizer)
        
    # FLOPs evaluation
    sample_input = next(iter(trainloader))[0][0].unsqueeze(0).to(device)
    flops_metrics = measure_model_flops(modelConfig.model, sample_input)

    # Ecological impact evaluation
    ecological_impact = calculate_ecological_impact(memory_size, inference_time)

    # Compile all metrics
    evaluation_results = {
        "memory": {
            "model_size_bytes": memory_size,
        },
        "text_generation": {
            "generated_text": generated_text
        },
        "perplexity": {
            "test_ppl": ppl_test
        },
        "computational_complexity": flops_metrics,
        "inference_time": {
            "average_time": inference_time
        },
        "ecological_impact": ecological_impact,
        "metadata": {
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device)
        }
    }

    return evaluation_results