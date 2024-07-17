import functools
import sys

from transformers import AutoTokenizer

sys.path.insert(1, "../")
from torch.utils.data import DataLoader

from configs import Config
from dataset import load_starcoder_test, sft_collate_fn
from model import PicoGPT

tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

config = Config(
    vocab_size=len(tok),
    block_size=128,
    n_layer=1,
    n_embd=128,
    n_head=1,
    n_query_groups=1,
    tie_weights=True,
    rope_theta=10000,
    neftune_noise_alpha=1.0,
    dropout=0.1,
)


model = PicoGPT(config)
starcoder = load_starcoder_test(tok)

dl = DataLoader(
    starcoder, batch_size=2, collate_fn=functools.partial(sft_collate_fn, tok=tok)
)
batch = next(iter(dl))

# TODO: change the model forward call to inspect attention mask

out = model(batch)
