import torch
import torch.nn as nn

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

    nsamples = 20  # To make it faster

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


def eval_perplexity(args, model, tokenizer, device=torch.device("cuda:0")):
    # Get the test loader
    _, testloader = get_wikitext2(args.nsamples, args.seed, args.seqlen, tokenizer)

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(model, testloader, 1, device)

    return ppl_test

# def eval_perplexity(args, model, tokenizer, device=torch.device("cpu")):
#     # Get the test loader
#     _, testloader = get_wikitext2(args.nsamples, args.seed, args.seqlen, tokenizer)

#     # Evaluate ppl in no grad context to avoid updating the model
#     with torch.no_grad():
#         ppl_test = eval_ppl_wikitext(model, testloader, 1, device)

#     return ppl_test


#################################################################################################
#                                       MEMORY SIZE EVALUATION                                  #
#################################################################################################


def compare_model_memory(model_pruned, model_original):
    """
    Compare the memory usage of a pruned model and a non-pruned model.

    Args:
        model_pruned (torch.nn.Module): The pruned model.
        model_original (torch.nn.Module): The original (non-pruned) model.

    Returns:
        dict: A dictionary containing the memory sizes and the percentage of space saved.
    """

    # def get_model_size(model):
    #     """Calculate the size of a model in bytes."""
    #     # Create a temporary buffer to save the model
    #     buffer = torch.save(model.state_dict(), None)
    #     return len(buffer)

    def get_model_size(model):
        """Calculate the size of a model in bytes."""
        # Create a temporary in-memory buffer
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        size_in_bytes = buffer.tell()  # Get the size of the buffer in bytes
        buffer.close()
        return size_in_bytes

    # Calculate memory size for both models
    size_pruned = get_model_size(model_pruned)
    size_original = get_model_size(model_original)

    # Calculate space saved
    space_saved = size_original - size_pruned
    percentage_saved = (space_saved / size_original) * 100

    # Return results as a dictionary
    model_memory = {
        "Pruned Model Size (bytes)": size_pruned,
        "Original Model Size (bytes)": size_original,
        "Space Saved (bytes)": space_saved,
        "Percentage Saved (%)": percentage_saved,
    }

    print("Model Memory Difference: ", model_memory)

    return model_memory


#################################################################################################
#                                    ECOLOGICAL IMPACT EVALUATION                               #
#################################################################################################


POWER_CONSUMPTION_INFERENCE = 200  # watts (example: power consumption per inference)
POWER_PER_MEMORY_UNIT = 0.01  # watts per MB (example: power for storing model weights)
CARBON_INTENSITY = 0.233  # kg CO2 per kWh (average for many regions)


def calculate_energy_savings_memory(memory_saved_mb):
    energy_saved = POWER_PER_MEMORY_UNIT * memory_saved_mb
    energy_saved_kWh = energy_saved / 1000
    return energy_saved_kWh


# Function to calculate energy savings from reduced computation (for sparse pruning)
def calculate_energy_savings_computation(inference_time_saved, num_inferences):
    energy_saved = (
        POWER_CONSUMPTION_INFERENCE * inference_time_saved * num_inferences
    )  # in watt-seconds (Joules)
    energy_saved_kWh = energy_saved / 3600000  # convert Joules to kWh
    return energy_saved_kWh


# Function to calculate carbon footprint reduction
def calculate_carbon_footprint_reduction(energy_saved_kWh):
    carbon_saved = energy_saved_kWh * CARBON_INTENSITY  # in kg CO₂
    return carbon_saved


#################################################################################################
#                                       INFERENCE TIME EVALUATION                               #
#################################################################################################


def compare_inference_time(
    original_model, pruned_model, nsamples, seed, seqlen, tokenizer
):
    # Get data samples
    trainloader, testenc = get_wikitext2(nsamples, seed, seqlen, tokenizer)


    # Measure inference time for the original model
    start_time = time.time()
    for inp, tar in trainloader:
        original_model(inp)
    inference_time_original = (time.time() - start_time) / nsamples

    # Measure inference time for the pruned model
    start_time = time.time()
    for inp, tar in trainloader:
        pruned_model(inp)
    inference_time_pruned = (time.time() - start_time) / nsamples

    # Compare the average inference times
    print(
        f"Average Inference Time for Original Model: {inference_time_original:.4f} seconds"
    )
    print(
        f"Average Inference Time for Pruned Model: {inference_time_pruned:.4f} seconds"
    )

    return inference_time_original, inference_time_pruned


#################################################################################################
#                                      PROMPT ANSWER EVALUATION                                 #
#################################################################################################

PROMPT = """
    A young girl named Lila discovers an ancient book in the attic of her family home. 
    The book is said to contain powerful secrets, but it is written in a language no one can understand…
    """


def compare_models_for_prompt(original_model, pruned_model, prompt=PROMPT):
    # Generate output from the original model
    original_output = original_model(prompt, max_length=100, num_return_sequences=1)[0][
        "generated_text"
    ]

    # Generate output from the pruned model
    pruned_output = pruned_model(prompt, max_length=100, num_return_sequences=1)[0][
        "generated_text"
    ]

    # Display the results
    print(f"Output from Original Model:\n{original_output}\n")
    print(f"Output from Pruned Model:\n{pruned_output}\n")

    return original_output, pruned_output


#################################################################################################
#                                         GLOBAL EVALUATION                                     #
#################################################################################################


def global_evaluation(modelConfig, original_model, pruned_model, tokenizer, device=device):

    original_model_perplexity = eval_perplexity(modelConfig, original_model, tokenizer, device=device)
    pruned_model_perplexity = eval_perplexity(modelConfig, pruned_model, tokenizer, device=device)

    print("Original Model Perplexity: ", original_model_perplexity)
    print("Pruned Model Perplexity: ", pruned_model_perplexity)

    compare_model_memory(original_model, pruned_model)
    # compare_inference_time(
    #     original_model,
    #     pruned_model,
    #     modelConfig.nsamples,
    #     modelConfig.seed,
    #     modelConfig.seqlen,
    #     tokenizer,
    # )
    compare_models_for_prompt(original_model, pruned_model)
