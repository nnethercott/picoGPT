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

import deepspeed

def setup():
    #dist.init_process_group("nccl")
    deepspeed.init_distributed()

def cleanup():
    dist.destroy_process_group()


# https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/discussions/47
config = Config(
    vocab_size=len(tok),
    block_size=512,
    n_layer=32,
    n_embd=512,
    n_head=32,
    n_query_groups=4,
    tie_weights=True,
    rope_theta=10000,
    neftune_noise_alpha=0.1,
    dropout=0.1,
)

train_config = TrainConfig(
    n_epochs=1,
    batch_size=2,
    lr=8.9e-05,
    min_lr=4e-05,
    warmup_ratio=0.0,
    grad_clip=1.0,
    weight_decay=0.1,
    log_steps=5,
    wandb_report=False,
    ckpt_path=None,
    save_path=None,
    ddp=True,
)


def train(model_config, train_config):
    c = train_config

    data = load_starcoder_test(tok)
    dl = DataLoader(
        data,
        batch_size=c.batch_size,
        collate_fn=sft_collate_fn,
        pin_memory=True,
    )
    training_steps = int(len(dl) * c.n_epochs)
    warmup_steps = int(c.warmup_ratio * training_steps)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # <|DEEPSPEED|>
    model = PicoGPT(model_config)
    model.print_total_trainable_parameters()

    model, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config="ds_config.json")
    

    start = time.time()
    steps = 0
    for epoch in range(c.n_epochs):
        for e, mini_batch in enumerate(dl):
            # might have to lump this in with model forward
            input_ids = mini_batch["input_ids"].to(device)
            attn_mask = mini_batch["attn_mask"].to(device)
            prompt_len = mini_batch["prompt_len"].to(device)
            seq_len = mini_batch["seq_len"].to(device)

            out = model(input_ids, attn_mask)
            logits = out["logits"]

            B, T, d = logits.shape

            ############# ce #####################
            loss = batched_cross_entropy(input_ids, logits, prompt_len, seq_len)

            model.backward(loss)
            model.step()

            steps+=1
            if steps % c.log_steps == 0:
                lossf = loss.item()
                dt = (time.time() - start) / (steps+1e-05)
                left = dt * (training_steps - steps) / 60

                if os.getenv("LOCAL_RANK") == "0":
                  print(
                      f"iter {steps}/{training_steps} | loss {lossf:.4f} | est. time {left:2f}"
                  )

    stop = time.time()
    print(f"finished training in {stop-start}s")


if __name__ == "__main__":
    setup()
    train(config, train_config)
    cleanup()

    #device = "cuda"
    #model = PicoGPT(config).to(device)
    #model.eval() #turn off neftune

    #model.load_state_dict(torch.load("/mnt/nate/checkpoints/slim_test.pt"))

    ## template = "{prompt}\n\n
    #prompt = "Potassium"
    ## prompt = template.format(prompt = prompt)
    #
    #input_ids = torch.tensor(tok.encode(prompt)).unsqueeze(0).to(device)
    #now = time.time()

    #generated = model.generate(
    #   #input_ids, do_sample=True, max_new_tokens=128, min_new_tokens = 10, temperature=1.2, top_k=32, eos_token_id = tok.eos_token_id,
    #   input_ids, do_sample = False, max_new_tokens = 64, num_beams=3, num_return_sequences=3, repetition_penalty=1.3, eos_token_id = tok.eos_token_id,
    #)
    #print(f'elapsed: {time.time()-now}')
    #print(prompt)

    #print(tok.decode(generated))
    #for g in generated:
    #   print(tok.decode(g, skip_special_tokens=True).strip())
    #   print("------------------------")
