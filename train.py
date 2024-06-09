import functools
import time

import torch
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from toktokenizer import BPETokenizer
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_cosine_schedule_with_warmup)

from configs import Config, TrainConfig
from model import PicoGPT
from utils import cosine_loss, kl_div

teacher_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# teacher_model_id = "openai-community/gpt2"
tok = AutoTokenizer.from_pretrained(teacher_model_id)

config = Config(
    vocab_size = len(tok),
    block_size = 128,
    n_layer = 11,
    n_embd = 768,
    n_head = 16,
    n_query_groups = 4,
    tie_weights = True,
    rope_theta = 10000,
    neftune_noise_alpha=1.0,
)

train_config = TrainConfig(
    n_epochs = 3,
    batch_size = 1,
    lr = 1e-03, 
    gradient_accumulation_steps = 1,
    warmup_ratio = 0.0,
    grad_clip = 1.0,
    weight_decay = 0.0,
    log_ratio = 0.0,
    distill_temperature=1.0,
)

# dataset 
def collate_fn(inputs):
    return torch.tensor(inputs)

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ds = load_dataset("bigcode/starcoderdata", data_dir="rust", split="train[:1000]")
# ds = ds.filter(lambda x: [len(y)<256 and '//' in y for y in x['content']], batched = True)
# tokens = ds.map(lambda x: {'tokens': tok.encode(x['content']) + [tok.eos_token_id]}, batched=False)['tokens'] # added eos token

i_code = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train[:10]")
i_code = i_code.map(lambda x: {'tokens': tok.encode(f"### Instruction: {x['instruction']}\n### Output:\n{x['output']}")+[tok.eos_token_id]}, batched=False)
i_code_tokens = i_code.filter(lambda x: [len(y)<config.block_size for y in x['tokens']], batched = True)['tokens']

ds = CustomDataset(i_code_tokens)

dl = DataLoader(ds, batch_size = train_config.batch_size, shuffle = False, collate_fn = collate_fn)


# model
model = PicoGPT(config)
model.load_state_dict(torch.load("./pico_starcoder.pt", map_location=torch.device('cpu')))
model.train()
print(f"total model params: {model.get_num_params()/1e6} mil")

# distill model 
# TODO: explore quantized performance 
teacher = AutoModelForCausalLM.from_pretrained(teacher_model_id)
teacher.eval()


def train(model, train_config):
    c = train_config 

    # init wandb 
    if c.wandb_report:
        wandb.init(project = c.wandb_project, entity = c.wandb_entity)

    training_steps = len(dl)*c.n_epochs
    warmup_steps = int(c.warmup_ratio*training_steps)
    log_steps = max(int(training_steps*c.log_ratio),1)

    optimizer = model.configure_optimizers(train_config)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, training_steps)

    start = time.time()
    steps = 0 
    for epoch in range(c.n_epochs):
        for mini_batch in dl:
            out = model(mini_batch)
            logits = out['logits']

            ############# ce #####################
            ce_logits = logits[:,:-1].reshape(-1, logits.shape[-1])
            targets = mini_batch[:,1:].reshape(-1)

            ce_loss = F.cross_entropy(ce_logits, targets, reduction='mean')

            ############## kl ####################
            # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
            t = train_config.distill_temperature
            kl_inputs = logits.view(-1, logits.shape[-1])
            with torch.no_grad():
                teacher_out = teacher(mini_batch, output_hidden_states = True)

            kl_targets = teacher_out['logits']

            kl = kl_div(kl_inputs, kl_targets, t)

            # optional
            ############# cosine loss ###############
            # hidden_states = out['hidden_states']
            # teacher_hidden_states = teacher_out['hidden_states']
            #
            # cl = cosine_loss(hidden_states, teacher_hidden_states, 2) 

            loss = (ce_loss + kl)/2
            loss = loss/c.gradient_accumulation_steps
            loss.backward()

            if steps%c.gradient_accumulation_steps == 0 or steps == training_steps-1:
                if c.grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)

                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()

            steps+=1 
            if steps%log_steps == 0:
                lossf = c.gradient_accumulation_steps*loss.item()
                dt = (time.time()-start)/steps 
                left = dt*(training_steps-steps)/60
                print(f"iter {steps}/{training_steps} | loss {lossf:.4f} | lr {scheduler.get_last_lr()[0]:6f} | est. time {left:2f}")

            if c.wandb_report:
                wandb.log({'loss': c.gradient_accumulation_steps*loss.item(), 'ce': ce_loss, 'kl': kl, 'lr': scheduler.get_last_lr()[0]})
                
    stop = time.time()
    print(f"finished training in {stop-start}s")

# train(model, train_config)

model.eval()
input_ids = torch.tensor(tok.encode("fn")).unsqueeze(0)
generated = model.generate(input_ids, num_beams = 3, max_new_tokens=16, num_return_sequences=3)
# generated = model.generate(input_ids, do_sample=False, max_new_tokens = 32)

for i,g in enumerate(generated):
    print(f"GENERATION {i}")
    print(tok.decode(g, skip_spbatch_decodeecial_tokens = True))
    print("\n")
