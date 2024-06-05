
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from toktokenizer import BPETokenizer
from torch import nn
from torch.utils.data import DataLoader, Dataset
# tok = BPETokenizer.from_pretrained("./wikibpe.json")
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from configs import Config, TrainConfig
from model import PicoGPT

tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
ds = load_dataset("karpathy/tiny_shakespeare", split = "train", trust_remote_code=True)
# tok = BPETokenizer.from_pretrained("./wikibpe.json")
# tok.train(ds[0]['text'], 5000)
# assert max(list(tok.encoder.values())) <= 5000

config = Config(
    vocab_size = len(tok), #tok.n_vocab+255,
    block_size = 64,
    n_layer = 4,
    n_embd = 128,
    n_head = 4,
    n_query_groups = 2,
    tie_weights = True,
    theta = 10000,
)

train_config = TrainConfig(
    n_epochs = 10,
    batch_size = 64,
    lr = 1e-03, 
    gradient_accumulation_steps = 1,
    warmup_ratio = 0.0,
    grad_clip = None,
    weight_decay = 0.0,
    log_ratio = 0.0,
)


# dataset 
tokens = tok.encode(ds[0]['text'])
tokens = tokens[:(len(tokens)//config.block_size)*config.block_size] # drop last chunk<block_size
tokens = torch.tensor(tokens).view(-1, config.block_size)

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

ds = CustomDataset(tokens)
dl = DataLoader(ds, batch_size = train_config.batch_size, shuffle = False)


# model
model = PicoGPT(config)
model.train()
print(f"total model params: {model.get_num_params()/1e6} [mil]")

#TODO: configure optimizer & weight decay for model itself
training_steps = len(dl)*train_config.n_epochs
warmup_steps = int(train_config.warmup_ratio*training_steps)
log_steps = max(int(training_steps*train_config.log_ratio),1)
n_epochs = train_config.n_epochs
gradient_accumulation_steps = train_config.gradient_accumulation_steps
grad_clip = train_config.grad_clip

optimizer = model.configure_optimizers(train_config)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, training_steps)

start = time.time()
steps = 0 
for epoch in range(n_epochs):
    for mini_batch in dl:
        logits = model(mini_batch)[:,:-1]
        targets = mini_batch[:,1:]

        logits = logits.reshape(-1, logits.shape[-1])
        targets = targets.reshape((targets.shape[0]*targets.shape[1]))

        loss = F.cross_entropy(logits, targets, reduction='mean')
        loss = loss/gradient_accumulation_steps
        loss.backward()

        #step 
        if steps%gradient_accumulation_steps == 0 or steps == training_steps-1:
            # grad clip 
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

        #step scheduler
        scheduler.step()

        steps+=1 
        if steps%log_steps == 0:
            lossf = gradient_accumulation_steps*loss.item()
            dt = (time.time()-start)/steps 
            left = dt*(training_steps-steps)/60
            print(f"iter {steps}/{training_steps} | loss {lossf:.4f} | lr {scheduler.get_lr()[0]} | est. time {left:2f}")
            
stop = time.time()
print(stop-start)

input_ids = torch.tensor(tok.encode("OCTAVIA")).unsqueeze(0)
generated = model.generate(input_ids, temperature = 1.2, top_k = 128)
