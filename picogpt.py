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


def setup():
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()


# TODO: change tokenizer and teacher to phi-3
tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")


config = Config(
    vocab_size=len(tok),
    block_size=512,
    n_layer=32,
    n_embd=784,
    n_head=18,
    n_query_groups=6,
    tie_weights=True,
    rope_theta=10000,
    neftune_noise_alpha=1.0,
    dropout=0.1,
)

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
    ckpt_path=None,
    save_path="./checkpoints/slim_test.pt",
    teacher_model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ddp=False,
)


def load_teacher(teacher_model_id):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_id,
        quantization_config=quantization_config,
    )
    teacher.eval()
    return teacher


#TODO: json dump config in checkpoints so we can reload later
def train(model_config, train_config):
    c = train_config

    # TODO: replace with data config & load
    data = load_starcoder_test(tok)
    dl = DataLoader(
        data,
        batch_size=4,
        collate_fn=sft_collate_fn,
    )
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
            print(f"loading checkpoint {c.ckpt_path} from file...")
            model.load_state_dict(torch.load(c.ckpt_path))

        # optimizers
        optimizer = model.configure_optimizers(train_config)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1.0)
        # scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, training_steps, min_lr = c.min_lr)

        model = DDP(model, device_ids=[rank])

    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = PicoGPT(model_config).to(device)
        if c.ckpt_path:
            print(f"loading checkpoint {c.ckpt_path} from file...")
            model.load_state_dict(torch.load(c.ckpt_path))

        optimizer = model.configure_optimizers(train_config)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, training_steps
        )
        master_rank = True

    if c.teacher_model_id is not None:
        teacher = load_teacher(c.teacher_model_id)

    if c.wandb_report:
        if master_rank:
            wandb.init(project=c.wandb_project, entity=c.wandb_entity)

    # get total params
    model.print_total_trainable_parameters()

    start = time.time()
    steps = 0
    for epoch in range(c.n_epochs):
        for e, mini_batch in enumerate(dl):
            input_ids = mini_batch["input_ids"].to(device)
            attn_mask = mini_batch["attn_mask"].to(device)
            prompt_len = mini_batch["prompt_len"].to(device)
            seq_len = mini_batch["seq_len"].to(device)

            # print(tok.batch_decode(input_ids))

            out = model(input_ids, attn_mask)
            logits = out["logits"]

            B, T, d = logits.shape

            ############# ce #####################
            ce_loss = batched_cross_entropy(input_ids, logits, prompt_len, seq_len)

            ############## kl ####################
            # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
            if c.teacher_model_id is not None:
                t = c.distill_temperature

                with torch.no_grad():
                    teacher_out = teacher(input_ids, output_hidden_states=True)

                kl_targets = teacher_out["logits"]
                kl_targets = kl_targets.detach()

                kl = topk_kl_div(logits, kl_targets, t, k=c.top_k)

            # optional
            ############# cosine loss ###############
            # hidden_states = out["hidden_states"]
            # teacher_hidden_states = teacher_out["hidden_states"]

            # hidden_states = [s.to(device) for s in hidden_states]
            # teacher_hidden_states = [s.to(device) for s in teacher_hidden_states]

            # cl = cosine_loss(hidden_states, teacher_hidden_states, device, 4)

            if c.teacher_model_id is not None:
              loss = ce_loss + kl
            else:
              loss = ce_loss

            loss = loss / c.gradient_accumulation_steps
            loss.backward()
            loss.detach().cpu()

            if (
                steps % c.gradient_accumulation_steps == 0
                or steps == training_steps - 1
            ):
                if c.grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)

                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()

            steps += 1
            if steps % c.save_steps == 0 or steps == training_steps:
                if master_rank and c.save_path is not None:
                    print(f"saving model at step: {steps} to file {c.save_path}...")
                    if c.ddp:
                      torch.save(model.module.state_dict(), c.save_path)
                    else:
                      torch.save(model.state_dict(), c.save_path)

                if c.ddp:
                    # blocking
                    dist.barrier()

            if steps % c.log_steps == 0:
                lossf = c.gradient_accumulation_steps * loss.item()
                dt = (time.time() - start) / (steps + 1e-05)
                left = dt * (training_steps - steps) / 60

                # TODO: turn this hardcode into a for k,v in losses.items()
                if master_rank:
                    print(
                        f"iter {steps}/{training_steps} | loss {lossf:.4f} | lr {scheduler.get_last_lr()[0]:6f} | est. time {left:2f}"
                    )
                    # print(
                    #    f"iter {steps}/{training_steps} | ce {ce_loss.item():.4f} | kl: {kl.item():.4f} | lr {scheduler.get_last_lr()[0]:6f} | est. time {left:2f}"
                    # )

            if c.wandb_report:
                if master_rank:
                    # TODO: turn this hardcode into a for k,v in losses.items()
                    wandb.log(
                        {
                            "loss": c.gradient_accumulation_steps * loss.item(),
                            "ce": ce_loss,
                            "ppl": 2**ce_loss,
                            "kl": kl,
                            # "cosine": cl,  # /math.sqrt(config.n_embd),
                            "lr": scheduler.get_last_lr()[0],
                        }
                    )

    stop = time.time()
    print(f"finished training in {stop-start}s")


if __name__ == "__main__":
    # setup()
    train(config, train_config)
    # cleanup()

    # device = "cuda"
    # config.block_size = 512
    # device = "cuda"
    # model = PicoGPT(config).to(device)
    # model.eval() #turn off neftune

    # model.load_state_dict(torch.load("./checkpoints/pico_instruct.pt"))

    # template = "{prompt}\n\n"
    # prompt = "Write a function which generates a bar plot with title 'nate is cool`."
    # prompt = template.format(prompt = prompt)
    #
    # input_ids = torch.tensor(tok.encode(prompt)).unsqueeze(0).to(device)
    # now = time.time()
    # generated = model.generate(
    #    input_ids, do_sample=True, max_new_tokens=128, temperature=1.2, top_k=32
    #    #input_ids, do_sample = False, max_new_tokens = 64, num_beams=3, num_return_sequences=3, repetition_penalty=1.3, eos_token_id = tok.eos_token_id,
    # )
    # print(f'elapsed: {time.time()-now}')

    # print(prompt)

    # print(tok.decode(generated))
#  for g in generated:
#    print(tok.decode(g, skip_special_tokens=True).strip())
#    print("------------------------")
