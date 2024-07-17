# picoGPT

![alt text](https://github.com/nnethercott/picoGPT/blob/main/media/picoGPT.png?raw=true)
smaller implementation of karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT/tree/9755682b981a45507f6eb9b11eadef8cb83cebd5) with a few changes.

# Usage

```python
from configs import Config
from model import PicoGPT

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

model = PicoGPT(model_config).to(device)

# load pretrained checkpoint with
model.load_state_dict(torch.load(your_ckpt_path))

#TODO: add tokenizer fwd call or model.generate()

```

## todos

- [x] [NEFTune](https://arxiv.org/abs/2310.05914)
- [x] [Grouped Query Attention](https://arxiv.org/pdf/2305.13245)
- [x] Beam search
- [x] [RMSNorm](https://arxiv.org/abs/1910.07467)
- [x] [Knowledge distillation (student-teacher learning)](https://arxiv.org/abs/1503.02531)
- [x] Cosine embedding loss
- [x] Attention mask & padding (dynamic)
- [x] DDP
- [ ] mixed-precision
- [ ] DPO
- [ ] quantization: [nf4](https://arxiv.org/abs/2305.14314), [1bit](https://github.com/kyegomez/BitNet)
- [ ] [slerp](https://en.wikipedia.org/wiki/Slerp)

## tiny-shakespeare

sample output after fine tuning a 9.3M llama-like model on 9.5M tokens of [karpathy/tiny_shakespeare](https://huggingface.co/datasets/karpathy/tiny_shakespeare) with knowledge distillation:

<!-- * `n_embd = 384`  -->
<!-- * `n_layer = 6`  -->
<!-- * `n_head = 4` -->
<!-- * `n_query_groups = 2` -->
<!-- * teacher model = [sadia72/gpt2-shakespeare](https://huggingface.co/sadia72/gpt2-shakespeare/tree/main) -->

<!-- ```python -->
<!-- config = Config( -->
<!--     vocab_size = len(tok), -->
<!--     block_size = 64, -->
<!--     n_layer = 6, -->
<!--     n_embd = 384, -->
<!--     n_head = 4, -->
<!--     n_query_groups = 2, -->
<!--     tie_weights = True, -->
<!--     rope_theta = 10000, -->
<!--     neftune_noise_alpha=0.0, -->
<!--     dropout = 0.1, -->
<!-- ) -->
<!---->
<!-- train_config = TrainConfig( -->
<!--     n_epochs = 30, -->
<!--     batch_size = 32, -->
<!--     lr = 1e-03, -->
<!--     gradient_accumulation_steps = 1, -->
<!--     warmup_ratio = 0.03, -->
<!--     grad_clip = 1.0, -->
<!--     weight_decay = 0.0, -->
<!--     distill_temperature=1.1, -->
<!-- ) -->
<!-- ``` -->

```
OPHELIA:
Good my lord, thou hateful dost companion ere Richmond
thoucester, who's here, and all the town with it,
That you should think me? then you shall be so holy
To the morning; as I see the duke:
If I be not consul, which is my distress
Of my tongue was made a guest,
And therefore, my noble brother.

YORK:
Madam.' God forbid your grace
Is very tongue of honour in this world;
How far best hope of it, weal is told.
```

## resources

- [nanoGPT](https://github.com/karpathy/nanoGPT/tree/9755682b981a45507f6eb9b11eadef8cb83cebd5)
- [TinyLlama](https://github.com/jzhang38/TinyLlama/tree/main)
- [llama paper](https://github.com/meta-llama/llama)
- [distilbert paper](https://arxiv.org/abs/1910.01108)
