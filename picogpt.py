import functools
import math
import os
import time

# torch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from toktokenizer import BPETokenizer
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# pico
from configs import Config, TrainConfig
from dataset import *
from losses import *
from model import PicoGPT
from utils import get_cosine_schedule_with_warmup

import lightning as L
from lit_model import LitPicoGPT
from lightning.pytorch.strategies import DeepSpeedStrategy

def setup():
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()


# https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/discussions/47
config = Config(
    vocab_size=len(tok),
    block_size=512,
    n_layer=20,
    n_embd=2048,
    n_head=32,
    n_query_groups=4,
    tie_weights=True,
    rope_theta=10000,
    neftune_noise_alpha=0.1,
    dropout=0.1,
)

train_config = TrainConfig(
    n_epochs=1,
    batch_size=4,
    lr=8.9e-05,
    min_lr=4e-05,
    warmup_ratio=0.0,
    grad_clip=1.0,
    betas = (0.999, 0.95),
    weight_decay=0.1,
    log_steps=5,
    wandb_report=False,
    ckpt_path=None,
    save_path=None,
    ddp=True,
)



from dataset import *
from torch.utils.data import DataLoader

data = load_starcoder_test(tok)
dl = DataLoader(
    data,
    batch_size=1,
    collate_fn=sft_collate_fn,
    pin_memory=True,
    num_workers = 2,
)

import json 
with open("ds_config.json", 'r') as f:
  ds_config = json.loads(f.read())

trainer = L.Trainer(
    devices=2, 
    accelerator="gpu", 
    max_epochs = 1,
    #strategy=DeepSpeedStrategy(config=ds_config),
    strategy = DeepSpeedStrategy(
      offload_optimizer=True,
      allgather_bucket_size=5e8,
      reduce_bucket_size=5e8
    ),
)
trainer.strategy.config["zero_force_ds_cpu_optimizer"] = False
#trainer.strategy.config["zero_allow_untested_optimizer"] = True

model = LitPicoGPT(model_config=config, train_config=train_config)
trainer.fit(model = model, train_dataloaders = dl)
