from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import copy


class ModelConfig:
    """
    A configuration class for setting up model parameters and managing model behavior.
    """

    def __init__(
        self,
        model_name: str = "facebook/opt-350m",
        token: str = "",
        seed: int = 0,
        nsamples: int = 128,
        seqlen: int = 128,
        sparsity_ratio: float = 0.5,
        cache_dir: str = ".cache/llm_weights/",
        save: str = "out/opt_350m/2-4/wanda/",
        save_model: str = None,
    ):

        self.model_name = model_name
        self.token = token
        self.model = None
        self.seed = seed
        self.nsamples = nsamples
        self.sparsity_ratio = sparsity_ratio
        self.cache_dir = cache_dir
        self.save = save
        self.save_model = save_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=self.token)
        self.seqlen = seqlen

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
        print(
            f"Loading model '{self.model_name}' from cache directory '{self.cache_dir}'..."
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            cache_dir=self.cache_dir,
            token=self.token,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        self.seqlen = self.model.config.max_position_embeddings
        self.model.seqlen = self.model.config.max_position_embeddings
        return self.model

    def copy_model(self):
        """
        Create a deep copy of the ModelConfig instance with a copied model.

        Returns:
            ModelConfig: A new ModelConfig instance with a deep copy of the loaded model.
        """
        if self.model is None:
            raise ValueError("Model has not been loaded yet. Call 'load_llm()' first.")

        new_config = ModelConfig(
            model_name=self.model_name,
            token=self.token,
            seed=self.seed,
            nsamples=self.nsamples,
            seqlen=self.seqlen,
            sparsity_ratio=self.sparsity_ratio,
            cache_dir=self.cache_dir,
            save=self.save,
            save_model=self.save_model,
        )
        new_config.model = copy.deepcopy(self.model)
        return new_config
