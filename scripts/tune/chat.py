import yaml
import json
import os
import sys
import argparse

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

if root_path not in sys.path:
    sys.path.append(root_path)

from src.datasets.dataset_from_tg import data_full_routine
from src.transformers.transformers_utils import init_pretrained_model, tokenize_function


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=False)
args = parser.parse_args()


CONFIG_DIR = "configs/tune"
CONFIG_NAME = args.config
config_path = os.path.join(CONFIG_DIR, CONFIG_NAME)


with open(config_path, "r") as f:
    config = yaml.safe_load(f)
    
random_seed = config["RANDOM_SEED"]
model_name = config["MODEL_NAME"]

data_path = os.path.join(config["DATA_DIR"], config["DATA_NAME"])
with open(data_path, "r") as f:
    raw_data = json.load(f)

name = config["RESPONSE_NAME"]


model, tokenizer, generation_config = init_pretrained_model(model_name, random_seed)
dataset = data_full_routine(raw_data, name)
tokenized_dataset = dataset.map(lambda x: tokenize_function(sample=x, tokenizer=tokenizer), batched=True)