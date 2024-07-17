import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, optim


# llama: https://github.com/meta-llama/llama/blob/llama_v2/llama/model.py#L34
# tinyllama: https://github.com/jzhang38/TinyLlama/blob/main/lit_gpt/rmsnorm.py#L822
# maps embeddings to unit sphere radius sqrt(n_embd)
class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.pow(2).mean(
            dim=-1, keepdims=True
        )  # [B,C,N]->[B,C,1] instead of [B,C]
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed

    def reset_parameters(self):
        nn.init.ones_(self.weight)


def neftune_forward_hook(module, input, output, alpha):
    if module.training and alpha > 0.0:
        L, d = output.shape[-2:]
        alpha = alpha / torch.sqrt(torch.tensor(L * d))
        eps = torch.zeros_like(output).uniform_(-alpha, alpha)
        return output + eps

    return output


# https://github.com/huggingface/transformers/blob/e65502951593a76844e872fee9c56b805598538a/src/transformers/optimization.py#L135
def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
    min_ratio: float = 0.0,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return max(
        min_ratio, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    )


def get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_ratio=float(min_lr / (optimizer.param_groups[0]["lr"] + 1e-07)),
    )
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
