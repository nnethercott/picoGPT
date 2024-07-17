import functools
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(1, "../")
from torch.utils.data import DataLoader

from configs import Config
from dataset import *
from model import PicoGPT

tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

config = Config(
    vocab_size=len(tok),
    block_size=128,
    n_layer=1,
    n_embd=128,
    n_head=1,
    n_query_groups=1,
    tie_weights=True,
    rope_theta=10000,
    neftune_noise_alpha=1.0,
    dropout=0.1,
)


model = PicoGPT(config)
# starcoder = load_starcoder_test(tok)
data = load_evol_py(tok)

dl = DataLoader(
    data,
    batch_size=2,
    collate_fn=sft_collate_fn,
)
batch = next(iter(dl))

input_ids = batch["input_ids"]
attn_mask = batch["attn_mask"]
out = model(input_ids, attn_mask)

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
