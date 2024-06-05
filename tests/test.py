import pytest
import torch

from config import Config
from model import CausalSelfAttention


@pytest.fixture 
def config():
    config = Config(
        n_embd = 128,
        n_head = 6,
        n_query_groups = 3,
    )
    return config


def test_self_attn(config):
    sa = CausalSelfAttention(config)

    #print total number of params 
    n_elements = sum(p.nelement() for p in sa.parameters())
    print(f"self attention head has: {n_elements} params")

    x = torch.randn(8,32,config.n_embd)
    y = sa(x)

    assert y.shape == x.shape 
