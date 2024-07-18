import functools
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(1, "../")
from torch.utils.data import DataLoader

from configs import *
from dataset import *
from model import PicoGPT
from losses import *

tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
device = "cuda"

config = Config(
    vocab_size=len(tok),
    block_size=512,
    n_layer=20,
    n_embd=576,
    n_head=9,
    n_query_groups=3,
    tie_weights=True,
    rope_theta=10000,
    neftune_noise_alpha=1.0,
    dropout=0.1,
)


model = PicoGPT(config).to(device)
data = load_starcoder_test(tok)
# data = load_evol_py(tok)
#data = load_slimpajama(tok)

dl = DataLoader(
    data,
    batch_size=4,
    collate_fn=sft_collate_fn,
)
batch = next(iter(dl))

input_ids = batch["input_ids"].to(device)
attn_mask = batch["attn_mask"].to(device)
prompt_len = batch["prompt_len"].to(device)
seq_len = batch["seq_len"].to(device)


out = model(input_ids, attn_mask)
logits = out['logits']
print(logits)

loss = batched_cross_entropy(input_ids, logits, prompt_len, seq_len)
print(loss)

# attention hard check

bias = (
    torch.tril(torch.ones(config.block_size, config.block_size))
    .view(1, config.block_size, config.block_size)
    .to(dtype=torch.bool)
)

a = torch.tensor(
    [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0]]
)
T = a.shape[1]

attn_mask = a
attn_mask = attn_mask.unsqueeze(-1)
attn_mask = attn_mask @ attn_mask.transpose(-1, -2)
attn_mask = attn_mask.to(dtype=torch.bool) & bias[:, :T, :T]

# fill with -inf
# attn_mask = attn_mask.to(dtype=torch.float32)
# attn_mask.masked_fill_(attn_mask == 0, torch.finfo(torch.float32).min)  # nice


# weird batch 

ids = torch.tensor([[    1,  2529,   878,   270, 29918, 29896, 29881, 29936,    13,  5467,
           878,   270, 29918, 29906, 29881, 29936,     2,     2,     2,     2,
             2,     2],
        [    1,  2529,   878,  5844, 29936,    13,  5467,   878,  1410, 29875,
         29936,    13,  5467,   878,  7882, 29936,    13,  5467,   878,  4744,
         29936,     2],
        [    1,  2115,  8842,  2974, 29889,  6857,   293,   579, 29615,  4373,
            13,  1645,  8842,  2974, 29889,  2940,  2283,  5072,     2,     2,
             2,     2],
        [    1,  7787, 29889,  4668, 29889, 14196,   546,   276,  4011, 29889,
         14036, 29889,  1761, 29889,  1293,  5308,     2,     2,     2,     2,
             2,     2]]).to(device)

attn_mask = torch.tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.,
         0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
         1., 0., 0., 0.],
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.,
         0., 0., 0., 0.]])


train_config = TrainConfig(
    n_epochs=1,
    batch_size=4,
    lr=3e-05,
    min_lr=1e-05,
    gradient_accumulation_steps=8,
    warmup_ratio=0.03,
    grad_clip=1.0,
    weight_decay=0.1,
    save_steps=3000,
    log_steps=1,
    wandb_report=False,
    wandb_entity="nnethercott",
    wandb_project="picoGPT",
    distill_temperature=1.3,
    top_k=64,
    ckpt_path = None,
    save_path  = "./checkpoints/slim_test.pt",
    teacher_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ddp=False,
)

_ = model.configure_optimizers(train_config)
#model(ids, attn_mask)
