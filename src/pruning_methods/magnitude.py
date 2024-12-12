import torch
from torch import nn


def magnitude_pruning(model, pruning_ratio: float):
    """
    Implement magnitude-based pruning by removing weights with smallest absolute values.
    """
    total_params = 0
    pruned_params = 0

    for name, module in model.named_modules():
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
