from torch.random import manual_seed
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import Dataset
from typing import Dict


def init_pretrained_model(model_name: str, random_seed: int = 0, **kwargs):
    manual_seed(random_seed)
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generation_config = GenerationConfig.from_pretrained(model_name)
    
    return model, tokenizer, generation_config

def tokenize_function(sample: Dict[str, str], tokenizer):
    return tokenizer(sample["context"], sample["response"], truncation=True, padding="max_length")