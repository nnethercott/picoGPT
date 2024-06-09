import math

import torch
import torch.nn.functional as F
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


def neftune_forward_hook(module, input, output, alpha):
    if module.training and alpha>0.0:
        L,d = output.shape[-2:]
        alpha = alpha/torch.sqrt(torch.tensor(L*d))
        eps = torch.zeros_like(output).uniform_(-alpha, alpha)
        return output + eps 
    
    return output

def kl_div(inputs, targets, t):
    targets = targets.view(-1, targets.size(-1))
    inputs = inputs.view(-1, inputs.size(-1))

    targets = F.log_softmax(targets/t, dim = -1)
    inputs = F.log_softmax(inputs/t, dim = -1)

    kl = F.kl_div(inputs, targets, log_target=True, reduction = 'batchmean')

    return kl 


# NOTE: main intention is target model has 1/n the number of layers 
def cosine_loss(inputs, targets, n):
    device = inputs.device

    inputs = torch.cat(inputs, device=device)
    targets = torch.cat([h for e,h in enumerate(targets) if e%n == 0], device=device)

    assert inputs.shape == targets.shape, "make sure to properly configure student model"

    inputs = inputs.view(-1, inputs.shape[-1])
    targets = targets.view(-1, targets.shape[-1])

    B, d = inputs.shape
    y = torch.ones(B, device=device) 

    loss = F.cosine_embedding_loss(inputs, targets, y)
    return math.sqrt(d)*loss  #added by moi 


# class LoraLinear(nn.Module):
#     def __init__(self, in_features, out_features, bias, r, **kwargs):
#         super().__init__()
#
#         A = torch.randn((r, in_features))
#         B = torch.randn((out_features, r))
#         
#         self.in_features = in_features
#         self.out_features = out_features
#
#         if bias:
#             self.bias = nn.Parameter(torch.empty(out_features))
#         else:
#             self.register_parameter('bias', None)
#
#         self.weight = nn.Parameter(B@A, requires_grad=True)


