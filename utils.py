import torch
from torch import nn


# llama: https://github.com/meta-llama/llama/blob/llama_v2/llama/model.py#L34
# tinyllama: https://github.com/jzhang38/TinyLlama/blob/main/lit_gpt/rmsnorm.py#L822
# maps embeddings to unit sphere radius sqrt(n_embd) 
class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.pow(2).mean(dim=-1, keepdims=True) #[B,C,N]->[B,C,1] instead of [B,C]
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed

    def reset_parameters(self):
        nn.init.ones_(self.weight)

