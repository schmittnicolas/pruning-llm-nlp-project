import gc
import random
import argparse

from huggingface_hub import login

# Use your Hugging Face token to log in
login("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")

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention, LlamaMLP

import LLMPruner.torch_pruning as tp 
from LLMPruner.pruner import hf_llama_pruner as llama_pruner
from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.evaluator.ppl import PPLMetric


prompts = [
        """Tweet: "I hate it when my phone battery dies."
Sentiment: Negative
###
Tweet: "My day has been 👍"
Sentiment: Positive
###
Tweet: "This is the link to the article"
Sentiment: Neutral
###
Tweet: "This new music video was incredibile"
Sentiment:""",

]

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def main(args):
    set_random_seed(args.seed)

    logger = LoggerWithDepth(
        env_name="{}".format("llama_prune_perso"), 
        config=args.__dict__,
        root_dir='prune_log',
        setup_sublogger=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    if device != "cpu":
        model.half()
    model.to(device)

   
    max_seq_len = 250
    logger.log("\n==================Generation Results before Pruning================\n")
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(device)

            generation_output = model.generate(
                input_ids=input_ids,
                do_sample=True,
                top_k=50,
                max_length=max_seq_len,
                temperature=1,
            )
            
            result = tokenizer.decode(generation_output[0])
            logger.log(result)

    ppl = PPLMetric(model, tokenizer, ['wikitext2'], max_seq_len, device=device)
    logger.log("PPL before pruning: {}".format(ppl))

    for param in model.parameters():
        param.requires_grad_(True)
    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    forward_prompts = torch.tensor([
        [    1,   306,  4658,   278,  6593,   310,  2834,   338],
        [    1,  3439, 17632,  1925, 29892,   278,  6368,   310],
    ]).to(device) # Only for building the dependency graph. Any input will be fine since the computation result are not taken into consideration.

    pruner_type = 'l2'
    imp = llama_pruner.MagnitudeImportance(p=2)

    logger.log("Use {} pruner...".format(pruner_type))




    kwargs = {
        "importance": imp,
        "global_pruning": False,
        "iterative_steps": 1,
        "ch_sparsity": 0.25, 
        "ignored_layers":[],
        "channel_groups": {
            #layer.self_attn: layer.self_attn.num_heads for layer in model.model.layers
        },
        "customized_pruners": {
            LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
            #LlamaAttention: llama_pruner.hf_attention_pruner,
        },
        "root_module_types": [LlamaRMSNorm, LlamaAttention],
    }

    pruner = tp.pruner.MetaPruner(
        model,
        forward_prompts,
        **kwargs
    )
    model.zero_grad()
    
    logger.log("Start Pruning")
    for i in range(10):
        pruner.step()

        after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.log("After Iter {}/{}, #parameters: {}".format(i+1, 10, after_pruning_parameters))

    # Clean the gradient in the model
    model.zero_grad()
    for name, module in model.named_parameters():
        if 'weight' in name:
            module.grad = None

    # modify inferece-related attributes
    model.config.hidden_size = model.model.embed_tokens.weight.shape[1]
    model.zero_grad()
    
    del pruner
            
    # elif args.layer_wise:
    #     model.model.layers = model.model.layers[:args.layer]
    #     after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.log("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters,  100.0*after_pruning_parameters/before_pruning_parameters))
    
    gc.collect()
    torch.cuda.empty_cache()

    # if args.save_model:
    #     model.half()
    #     torch.save({
    #         'model': model, 
    #         'tokenizer': tokenizer,
    #     }, logger.best_checkpoint_path)
    
    # if args.eval_device != "cpu":
    #     model.half()
    # model.to(args.eval_device)

    model.config.pad_token_id = tokenizer.pad_token_id = 0 
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2


    logger.log("\n==================Generation Results After Pruning================\n")
    
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(device)

            generation_output = model.generate(
                input_ids=input_ids,
                do_sample=True,
                top_k=50,
                max_length=max_seq_len,
                temperature=1,
            )
            
            result = tokenizer.decode(generation_output[0])
            logger.log(result)
    
    logger.log("\n==================Finish================\n")

    ppl = PPLMetric(model, tokenizer, ['wikitext2'], max_seq_len, device=device)
    logger.log("PPL after pruning: {}".format(ppl))
    logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

    parser.add_argument('--seed', type=int, default=42, help='seed')
    args = parser.parse_args()

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    main(args)
