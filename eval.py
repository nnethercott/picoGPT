import functools
import time

import torch
import torch.nn.functional as F
from configs import Config, TrainConfig
from datasets import load_dataset
from model import PicoGPT
from toktokenizer import BPETokenizer
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from utils import cosine_loss, kl_div

teacher_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tok = AutoTokenizer.from_pretrained(teacher_model_id)

config = Config(
    vocab_size=len(tok),
    block_size=128,
    n_layer=5,
    n_embd=2048,
    n_head=16,
    n_query_groups=4,
    tie_weights=True,
    rope_theta=10000,
    neftune_noise_alpha=0.0,
    dropout=0.1,
)

train_config = TrainConfig(
    n_epochs=1,
    batch_size=16,
    lr=2e-05,
    gradient_accumulation_steps=4,
    warmup_ratio=0.03,
    grad_clip=1.0,
    weight_decay=0.1,
    log_ratio=0.002,
    distill_temperature=1.1,
)


model = PicoGPT(config)
model.load_state_dict(torch.load("./checkpoints/pico_tinyllama_8.pt"))
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)
prompt = "Once upon a time there "
input_ids = torch.tensor(tok.encode(prompt)).unsqueeze(0).to(device)
generated = model.generate(
    input_ids, do_sample=True, max_new_tokens=128, temperature=0.7, top_k=32
)
print(tok.decode(generated, skip_special_tokens=True))
