import os
import sys
from datetime import datetime

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

if root_path not in sys.path:
    sys.path.append(root_path)

import yaml
import argparse

import torch
from torch.optim import AdamW
from src.tokenizers.char import CharTokenizer
from src.modules.bigram_language_model import BigramLanguageModel

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=False)
args = parser.parse_args()

CONFIG_DIR = "configs/train"
DATA_DIR = "data"
SAVE_DIR = "artifacts"

# DEFAULT_CONFIG_NAME = "bigram_default.yaml"

config_name = args.config
config_path = os.path.join(CONFIG_DIR, config_name)

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

random_seed = config["RANDOM_SEED"]
batch_size = config["BATCH_SiZE"]
learning_rate = config["LEARNING_RATE"]
context_size = config["CONTEXT_SiZE"]
n_embed = config["N_EMBED"]
n_head = config["N_HEAD"]
n_layer = config["N_LAYER"]
dropout = config["DROPOUT"]
max_iters = config["MAX_ITERS"]
eval_interval = config["EVAL_INTERVAL"]
eval_iters = config["EVAL_ITERS"]
try_gpu = config["TRY_GPU"]
train_split = config["TRAIN_SPLIT"]

if try_gpu:
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = "cpu"

DATA_PATH = os.path.join(DATA_DIR, config["DATA_FILE_NAME"])

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = f.read()
    
tokenizer = CharTokenizer()
tokenizer.fit(data)

data = torch.tensor(tokenizer.encode(data), dtype=torch.long)
n = int(train_split * len(data))
train_data, val_data = data[:n], data[n:]

model = BigramLanguageModel(vocab_size=tokenizer.vocab_size,
                            n_embed=n_embed,
                            context_size=context_size,
                            n_head=n_head,
                            n_layer=n_layer,
                            dropout=dropout,
                            device=device)
m = model.to(device)

optimizer = AdamW(m.parameters(), lr=learning_rate)

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - context_size, (batch_size, )) 
    x = torch.stack([data[i:i+context_size] for i in ix])
    y = torch.stack([data[i+1:i+context_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out 

for iter in range(max_iters + 1):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = get_batch("train")
    
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

SAVE_NAME = str(datetime.now().strftime('%Y%m%d')) + "_" + config_name[:-5]
SAVE_PATH = os.path.join(SAVE_DIR, SAVE_NAME)
os.mkdir(SAVE_PATH)
torch.save(model.state_dict(), os.path.join(SAVE_PATH, "model.pth"))
tokenizer.save(os.path.join(SAVE_PATH, "tokenizer.json"))
with open(os.path.join(SAVE_PATH, "config.yaml"), "w") as f:
    yaml.dump(config, f)

# SAVE_NAME = str(datetime.now().strftime('%Y%m%d')) + "_" + config_name[:-5] + ".pth"
# SAVE_PATH = os.path.join(SAVE_DIR, SAVE_NAME)

# torch.save(model.state_dict(), SAVE_PATH)

# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(tokenizer.decode(m.generate(context, max_new_tokens=500)[0].tolist()))