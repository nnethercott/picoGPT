# std
import time
import json 

# torch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

# transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# pico
from configs import Config, TrainConfig
from dataset import *
from model import PicoGPT, Block
from utils import get_cosine_schedule_with_warmup
from losses import *

# lightning
import lightning as L
from lightning.fabric.strategies import FSDPStrategy, DeepSpeedStrategy 

# deepspeed  (optim error https://github.com/microsoft/DeepSpeed/issues/1846#issuecomment-1080226911)
from deepspeed.ops.adam import DeepSpeedCPUAdam as Adam
Adam = torch.optim.AdamW

# https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/discussions/47
config = Config(
    vocab_size=len(tok),
    block_size=512,
    n_layer=23,
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
    batch_size=1,
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


# TODO: add type annotations
def train(fabric, state, dl):
    model = state['model']
    optimizer = state['optimizer']

    start = time.time()
    for e, batch in enumerate(dl):
        # make these tensors contiguous ? tensor.contiguous()
        input_ids = batch["input_ids"]
        attn_mask = batch["attn_mask"]
        prompt_len = batch["prompt_len"]
        seq_len = batch["seq_len"]

        is_accumulating = e%4 == 0

        with fabric.no_backward_sync(model, enabled=is_accumulating):
          out = model(input_ids, attn_mask)
          logits = out["logits"]
          #B, T, d = logits.shape

          loss = batched_cross_entropy(input_ids, logits, prompt_len, seq_len)
          fabric.backward(loss)

        if not is_accumulating:
          # grad clip throws errors using deepspeed, should be in ds_config.json
          #fabric.clip_gradients(model, optimizer, max_norm=1.0)
          optimizer.step()
          optimizer.zero_grad()


        dt = (time.time() - start) / ((e+1) + 1e-05)
        left = dt * (len(dl) - (e+1)) / 60
        throughput = input_ids.shape[0]/dt

        fabric.print(
                f"iter: {e}/{len(dl)}, loss: {loss.item():.4f}, est. time: {left}, throughput: {throughput}"
            )


def main():
  # <|DATALOADER|>
  #data = load_starcoder_test(tok)
  data = [{'input_ids':torch.randint(high=32000, size=(512,)).tolist(), 'prompt_len':0} for _ in range(5000)]
  data = CustomDataset(data)
  dl = DataLoader(
      data,
      batch_size=1,
      collate_fn=sft_collate_fn,
      pin_memory=True,
      num_workers = 2,
  )


  # <|FABRIC|>
  with open('ds_config.json') as f:
    ds_config = json.loads(f.read())

  #strategy="auto"
  strategy = FSDPStrategy(
                  auto_wrap_policy={Block},
                  activation_checkpointing_policy=None, #crashes for some reason
                  sharding_strategy=dist.fsdp.ShardingStrategy.FULL_SHARD,
                  state_dict_type="full",
                  limit_all_gathers=True,
                  cpu_offload=False,
              )
  #strategy = DeepSpeedStrategy(config = ds_config)

  fabric = L.Fabric(
      precision = "16-mixed",
      devices = 4,
      accelerator = "cuda",
      strategy = strategy,
  )
  fabric.launch()
  fabric.seed_everything(42)

  device = fabric.device

  # <|MODEL|>
  # Recommended for FSDP, TP and DeepSpeed
  with fabric.init_module(empty_init=False):
      model = PicoGPT(config)  # parameters are placed on the meta-device

  print("###################")
  model.print_total_trainable_parameters()
  print("###################")

  optimizer = torch.optim.AdamW(
          model.parameters(), lr=3e-05, weight_decay=0.1, betas=(0.99, 0.95),
      )
  model, optimizer = fabric.setup(model, optimizer)

  dl = fabric.setup_dataloaders(dl)

  state = {
    'model': model,
    'optimizer': optimizer,
  }

  train(fabric, state, dl)


if __name__ == "__main__":
    main()
