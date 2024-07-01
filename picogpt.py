#TODO: create dataset.py
import functools
import time
import math
import os

# pico 
from configs import Config, TrainConfig
from utils import cosine_loss, kl_div, get_cosine_schedule_with_warmup
from model import PicoGPT
from dataset import *

from toktokenizer import BPETokenizer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import wandb

# torch 
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp 
import torch.distributed as dist 

def setup():
  dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()


# FIXME
tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

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
    batch_size=8,
    lr=1e-04,
    min_lr = 2e-05, 
    gradient_accumulation_steps=4,
    warmup_ratio=0.03,
    grad_clip=1.0,
    weight_decay=0.1,
    save_steps = 1000,
    log_steps = 50,
    wandb_report = True,
    wandb_entity = "nnethercott",
    wandb_project = "picoGPT",
    distill_temperature=1.2,
    top_k = 48,
    ckpt_path = "./checkpoints/pico_tinyllama_9.pt",
    save_path = "./checkpoints/pico_tinyllama_10.pt",
    teacher_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ddp = True,
)

def load_teacher(teacher_model_id):
  quantization_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype=torch.float32,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_use_double_quant=False,
  )
  teacher = AutoModelForCausalLM.from_pretrained(
      teacher_model_id, quantization_config=quantization_config
  )
  teacher.eval()
  return teacher 


def train(model_config, train_config):
  c = train_config

  dl = load_training_data(model_config, train_config, tok, rank = int(os.getenv("LOCAL_RANK")))
  training_steps = int(len(dl) * c.n_epochs)
  warmup_steps = int(c.warmup_ratio * training_steps)

  if c.ddp:
    # DDP setup 
    rank = int(os.environ["LOCAL_RANK"])
    master_rank = rank == 0
    world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    model = PicoGPT(model_config).to(device)

    if c.ckpt_path:
      print(f'loading checkpoint {c.ckpt_path} from file...')
      model.load_state_dict(torch.load(c.ckpt_path))

    # optimizers 
    optimizer = model.configure_optimizers(train_config)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, training_steps, min_lr = c.min_lr)

    model = DDP(model, device_ids = [rank])

  else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PicoGPT(model_config).to(device)
    if c.ckpt_path:
      print(f'loading checkpoint {c.ckpt_path} from file...')
      model.load_state_dict(torch.load(c.ckpt_path))

    optimizer = model.configure_optimizers(train_config)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, training_steps)
    master_rank = True
  
  teacher = load_teacher(c.teacher_model_id)


  if c.wandb_report:
    if master_rank:
      wandb.init(project=c.wandb_project, entity=c.wandb_entity)

  start = time.time()
  steps = 0
  for epoch in range(c.n_epochs):
      for e, mini_batch in enumerate(dl):
          mini_batch = mini_batch.to("cuda")
          out = model(mini_batch['input_ids'])
          logits = out["logits"]

          #TODO: masked_fill_ for prompt tokens in loss

          ############# ce #####################
          ce_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
          targets = mini_batch[:, 1:].reshape(-1)

          ce_loss = F.cross_entropy(ce_logits, targets, reduction="mean")

          ############## kl ####################
          # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
          t = c.distill_temperature
          with torch.no_grad():
              teacher_out = teacher(mini_batch, output_hidden_states=True)

          kl_targets = teacher_out["logits"]
          kl_targets = kl_targets.detach()

          kl = kl_div(logits, kl_targets, t, k=c.top_k)

          # optional
          ############# cosine loss ###############
          hidden_states = out["hidden_states"]
          teacher_hidden_states = teacher_out["hidden_states"]

          hidden_states = [s.to(device) for s in hidden_states]
          teacher_hidden_states = [s.to(device) for s in teacher_hidden_states]

          cl = cosine_loss(hidden_states, teacher_hidden_states, device, 4)

          loss = (ce_loss + kl) / 2  
          #loss = ce_loss
          loss = loss / c.gradient_accumulation_steps

          #if torch.isnan(loss):
          #    try:
          #        print(f'nan encountered for batch {tok.batch_decode(mini_batch)} at step: {steps}')
          #        del loss
          #        torch.cuda.empty_cache()
          #    finally:
          #        # load state dict from last checkpoint 
          #        model.load_state_dict(torch.load(save_path))
          #        continue 
          
          loss.backward()
          loss.detach().cpu()

          if steps % c.gradient_accumulation_steps == 0 or steps == training_steps - 1:
              if c.grad_clip:
                  nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)

              optimizer.step()
              optimizer.zero_grad()

          scheduler.step()

          steps += 1
          if steps % c.save_steps == 0:
              if master_rank:
                print(f"saving model at step: {steps} to file {c.save_path}...")
                torch.save(model.state_dict(), c.save_path)
                
              if c.ddp:
                # blocking
                dist.barrier()
                

          if steps % c.log_steps == 0:
              lossf = c.gradient_accumulation_steps * loss.item()
              dt = (time.time() - start) / (steps+1e-05)
              left = dt * (training_steps - steps) / 60

              if master_rank:
                print(
                    f"iter {steps}/{training_steps} | loss {lossf:.4f} | lr {scheduler.get_last_lr()[0]:6f} | est. time {left:2f}"
                )


          if c.wandb_report:
            if master_rank:
              wandb.log(
                  {
                      "loss": c.gradient_accumulation_steps * loss.item(),
                      "ce": ce_loss,
                      "ppl": 2**ce_loss,
                      "kl": kl,
                      "cosine": cl,  # /math.sqrt(config.n_embd),
                      "lr": scheduler.get_last_lr()[0],
                  }
              )


  stop = time.time()
  print(f"finished training in {stop-start}s")


if __name__ == "__main__":
  setup()
  train(config, train_config)
  cleanup()

# quick inference
#model.to(device)
#prompt = "A man"
#input_ids = torch.tensor(tok.encode(prompt)).unsqueeze(0).to(device)
#generated = model.generate(
#    input_ids, do_sample=True, max_new_tokens=128, temperature=0.8, top_k=16
#)
#print(tok.decode(generated, skip_special_tokens=True))
