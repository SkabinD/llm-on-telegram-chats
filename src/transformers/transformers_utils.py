from torch.random import manual_seed
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import Dataset
from typing import Dict


def init_pretrained_model(config: Dict[str, str], **kwargs):
    manual_seed(config["RANDOM_SEED"])
    model_name = config["MODEL_NAME"]
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generation_config = GenerationConfig.from_pretrained(model_name)
    
    return model, tokenizer, generation_config

def tokenize_function(sample: Dict[str, str], tokenizer):
    return tokenizer(sample["context"], sample["response"], truncation=True, padding="max_length")
    

# if __name__ == "__main__":    
    