from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import os
from dotenv import load_dotenv

load_dotenv()

class ModelConfig:
    """
    A configuration class for setting up model parameters and managing model behavior.
    """
    def __init__(self, 
                 model_name: str = "facebook/opt-350m", 
                 seed: int = 0, 
                 nsamples: int = 128, 
                 sparsity_ratio: float = 0.5, 
                 cache_dir: str = "llm_weights", 
                 save: str = "out/opt_350m/2-4/wanda/", 
                 save_model: str = None):
        
        self.model_name = model_name
        self.model = None
        self.seed = seed
        self.nsamples = nsamples
        self.sparsity_ratio = sparsity_ratio
        self.cache_dir = cache_dir
        self.save = save
        self.save_model = save_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def update_nsamples(self, new_value: int):
        """
        Update the number of samples for evaluation or training.

        Args:
            new_value (int): The new number of samples to use.
        """
        if new_value <= 0:
            raise ValueError("nsamples must be a positive integer.")
        self.nsamples = new_value

    def load_llm(self):
        """
        Load the language model with the specified configuration.
    
        Returns:
            AutoModelForCausalLM: The loaded language model with modified sequence length.
        """
        print(f"Loading model '{self.model_name}' from cache directory '{self.cache_dir}'...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16, 
            cache_dir=self.cache_dir,
            token = os.getenv("HUGGINGFACE_TOKEN"),
            low_cpu_mem_usage=True, 
            device_map="auto"
        )
        self.model.seqlen = self.model.config.max_position_embeddings 
        return self.model
    

    
