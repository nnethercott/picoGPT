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
    block_size=512,
    n_layer=5,
    n_embd=2048,
    n_head=16,
    n_query_groups=4,
    tie_weights=True,
    rope_theta=10000,
    neftune_noise_alpha=1.0,
    dropout=0.1,
)

train_config = TrainConfig(
    n_epochs=1,
    batch_size=1,
    lr=3e-05,
    min_lr = 1e-05, 
    gradient_accumulation_steps=8,
    warmup_ratio=0.03,
    grad_clip=1.0,
    weight_decay=0.1,
    save_steps = 3000,
    log_steps = 50,
    wandb_report = True,
    wandb_entity = "nnethercott",
    wandb_project = "picoGPT",
    distill_temperature=1.3,
    top_k = 64,
    ckpt_path = "./checkpoints/pico_tinyllama_pretrain.pt",
    #ckpt_path  = "./checkpoints/pico_tinyllama_code_pretrain.pt",
    save_path = "/mnt/nate/pico_checkpoints/pico_tinyllama_pretrain2.pt",
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

  starcoder = load_starcoder("python", tok, rank = int(os.getenv("LOCAL_RANK")))
  slimpajama = load_slimpajama(tok, rank = int(os.getenv("LOCAL_RANK")))
  
  #alpaca = load_alpaca_instruct(tok, rank = int(os.getenv("LOCAL_RANK")), world_size = int(os.getenv("WORLD_SIZE")))
  #evol = load_evol_py(tok, rank = int(os.getenv("LOCAL_RANK")), world_size = int(os.getenv("WORLD_SIZE")))
  #chat = load_ultrachat(tok, rank = int(os.getenv("LOCAL_RANK")), world_size = int(os.getenv("WORLD_SIZE")))
  
  ds = InterpolatedDataset(
    {'data': starcoder, 'target_ratio': 1, 'is_main': True},
    {'data': slimpajama, 'target_ratio': 2, 'is_main': True},
    #{'data': alpaca, 'target_ratio': 1, 'is_main': True},
    #{'data': evol, 'target_ratio': 1, 'is_main': True},
    #{'data': chat, 'target_ratio': 5, 'is_main': True}
  ).generate(saturation_steps = 1)

  #ds = load_alpaca_instruct(tok, rank = int(os.getenv("LOCAL_RANK")))
  dl = DataLoader(ds, batch_size = 1, shuffle = False, collate_fn = sft_collate_fn)

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
    #scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, training_steps, min_lr = c.min_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1.0)

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
          input_ids = mini_batch['input_ids'].to(device)
          prompt_len = mini_batch['prompt_len'].to(device)
          out = model(input_ids)
          logits = out["logits"]

          ############# ce #####################
          ce_logits = logits[:, :-1].reshape(-1, logits.shape[-1])

          # masked fill instruct tokens
          targets = input_ids[:, 1:].clone().reshape(-1)
          y = torch.arange(targets.shape[-1], device = targets.device)
          mask = y<(prompt_len-1).repeat(y.shape)
          targets.masked_fill_(mask, -100)

          #print(tok.decode(targets[prompt_len:]))

          ce_loss = F.cross_entropy(ce_logits, targets, reduction="mean")

          ############## kl ####################
          # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
          t = c.distill_temperature
          with torch.no_grad():
              teacher_out = teacher(input_ids, output_hidden_states=True)

          kl_targets = teacher_out["logits"]
          kl_targets = kl_targets.detach()

          #TODO: potentially remove prompt tokens from kl loss
          kl = kl_div(logits, kl_targets, t, k=c.top_k)

          # optional
          ############# cosine loss ###############
          #hidden_states = out["hidden_states"]
          #teacher_hidden_states = teacher_out["hidden_states"]

          #hidden_states = [s.to(device) for s in hidden_states]
          #teacher_hidden_states = [s.to(device) for s in teacher_hidden_states]

          #cl = cosine_loss(hidden_states, teacher_hidden_states, device, 4)

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
          if steps % c.save_steps == 0 or steps == training_steps:
              if master_rank and c.save_path is not None:
                print(f"saving model at step: {steps} to file {c.save_path}...")
                torch.save(model.module.state_dict(), c.save_path)
                
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
                #print(
                #    f"iter {steps}/{training_steps} | ce {ce_loss.item():.4f} | kl: {kl.item():.4f} | lr {scheduler.get_last_lr()[0]:6f} | est. time {left:2f}"
                #)


          if c.wandb_report:
            if master_rank:
              wandb.log(
                  {
                      "loss": c.gradient_accumulation_steps * loss.item(),
                      "ce": ce_loss,
                      "ppl": 2**ce_loss,
                      "kl": kl,
                      #"cosine": cl,  # /math.sqrt(config.n_embd),
                      "lr": scheduler.get_last_lr()[0],
                  }
              )


  stop = time.time()
  print(f"finished training in {stop-start}s")


if __name__ == "__main__":
  setup()
  train(config, train_config)
  cleanup()

  #device = "cuda"
  #config.block_size = 512
  #device = "cuda"
  #model = PicoGPT(config).to(device)
  #model.eval() #turn off neftune

  #model.load_state_dict(torch.load("./checkpoints/pico_instruct.pt"))

  #template = "{prompt}\n\n"
  #prompt = "Write a function which generates a bar plot with title 'nate is cool`."
  #prompt = template.format(prompt = prompt)
  #
  #input_ids = torch.tensor(tok.encode(prompt)).unsqueeze(0).to(device)
  #now = time.time()
  #generated = model.generate(
  #    input_ids, do_sample=True, max_new_tokens=128, temperature=1.2, top_k=32
  #    #input_ids, do_sample = False, max_new_tokens = 64, num_beams=3, num_return_sequences=3, repetition_penalty=1.3, eos_token_id = tok.eos_token_id,
  #)
  #print(f'elapsed: {time.time()-now}')

  #print(prompt)

  #print(tok.decode(generated))
#  for g in generated:
#    print(tok.decode(g, skip_special_tokens=True).strip())
#    print("------------------------")
