#TODO: create dataset.py
import functools
import time
import math

# pico 
from configs import Config, TrainConfig
from utils import cosine_loss, kl_div
from model import PicoGPT

from datasets import load_dataset 
import datasets
from toktokenizer import BPETokenizer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    BitsAndBytesConfig,
)
import wandb

# torch 
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
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
    batch_size=16,
    lr=1e-04,
    gradient_accumulation_steps=4,
    warmup_ratio=0.03,
    grad_clip=1.0,
    weight_decay=0.1,
    save_steps = 1000,
    log_steps = 1,
    wandb_report = False,
    wandb_entity = "nnethercott",
    wandb_project = "picoGPT",
    distill_temperature=1.1,
    ckpt_path = "./checkpoints/pico_tinyllama_9.pt",
    save_path = "./checkpoints/pico_tinyllama_10.pt",
    teacher_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
)


# dataset prep 
# TODO: move collating fn to utils 
def collate_fn(inputs):
    return torch.tensor(inputs)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_training_data():
  #data = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)
  #texts = data.skip(800000).take(500)

  ## preprocessing
  ## https://stackoverflow.com/questions/76227219/can-i-convert-an-iterabledataset-to-dataset
  #def dataset_generator(dataset):
  #    yield from dataset

  #texts = datasets.Dataset.from_generator(functools.partial(dataset_generator, texts))
  #texts = texts.map(
  #    lambda x: {**x, "tokens": [tok.encode(y)[: config.block_size] for y in x["text"]]},
  #    batched=True,
  #)
  #texts = texts.filter(
  #    lambda x: [len(y) == config.block_size for y in x["tokens"]], batched=True
  #)
  #tokens = texts["tokens"]


  ## dataset object
  #ds = CustomDataset(tokens)
  #dl = DataLoader(
  #    ds, batch_size=train_config.batch_size, shuffle=False, collate_fn=collate_fn
  #)

  #tiny shakespeare 
  text = load_dataset("karpathy/tiny_shakespeare")['train'][0]['text']
  tokens = tok.encode(text)
  tokens = tokens[:-(len(tokens)%config.block_size)]
  tokens = [tokens[i*config.block_size:(i+1)*config.block_size] for i in range(len(tokens)//config.block_size)]
  ds = CustomDataset(tokens)
  dl = DataLoader(
      ds, batch_size=train_config.batch_size, shuffle=False, collate_fn=collate_fn
  )
  return dl 



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


# TODO: configure train config for scheduler params
def train(model_config, train_config):
  c = train_config

  if c.ddp:
    # DDP setup 
    rank = os.environ["LOCAL_RANK"]
    world_size = os.environ["WORLD_SIZE"]
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)
    model = PicoGPT(model_config).to(device)
    model = DDP(model, device_ids = [rank])

    # load data 
    dl = load_training_data()
  else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PicoGPT(model_config).to(device)
    dl = load_training_data()
  
  teacher = load_teacher(c.teacher_model_id)
  #model.load_state_dict(torch.load(c.ckpt_path))

  training_steps = int(len(dl) * c.n_epochs)
  warmup_steps = int(c.warmup_ratio * training_steps)
  optimizer = model.configure_optimizers(train_config)
  scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, training_steps)
  #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1.0)


  if c.wandb_report:
    wandb.init(project=c.wandb_project, entity=c.wandb_entity)

  start = time.time()
  steps = 0
  for epoch in range(c.n_epochs):
      for e, mini_batch in enumerate(dl):
          mini_batch = mini_batch.to("cuda")
          out = model(mini_batch)
          logits = out["logits"]

          # NOTE: if instruction tuning we need to omit loss for instruction input
          # TODO: change dataloader collator... 
          ############# ce #####################
          ce_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
          targets = mini_batch[:, 1:].reshape(-1)

          ce_loss = F.cross_entropy(ce_logits, targets, reduction="mean")

          ############## kl ####################
          # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
          t = c.distill_temperature
          kl_inputs = logits.view(-1, logits.shape[-1])
          with torch.no_grad():
              teacher_out = teacher(mini_batch, output_hidden_states=True)

          kl_targets = teacher_out["logits"]

          kl = kl_div(kl_inputs, kl_targets, t)

          # optional
          ############# cosine loss ###############
          hidden_states = out["hidden_states"]
          teacher_hidden_states = teacher_out["hidden_states"]

          hidden_states = [s.to(device) for s in hidden_states]
          teacher_hidden_states = [s.to(device) for s in teacher_hidden_states]

          cl = cosine_loss(hidden_states, teacher_hidden_states, device, 4)

          # loss = (ce_loss + kl + cl) / 3  # change this to just ce and kl
          loss = ce_loss
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
              print(f"saving model at step: {steps}")
              torch.save(model.state_dict(), save_path)

          if steps % c.log_steps == 0:
              lossf = c.gradient_accumulation_steps * loss.item()
              dt = (time.time() - start) / (steps+1e-05)
              left = dt * (training_steps - steps) / 60
              print(
                  f"iter {steps}/{training_steps} | loss {lossf:.4f} | lr {scheduler.get_last_lr()[0]:6f} | est. time {left:2f}"
              )

          if c.wandb_report:
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
