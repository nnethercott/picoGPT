# picoGPT

![alt text](https://github.com/nnethercott/picoGPT/blob/main/media/picoGPT.png?raw=true)

smaller implementation of karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT/tree/9755682b981a45507f6eb9b11eadef8cb83cebd5) with a few changes.

## tiny-shakespeare

sample output after quick fine tuning on [karpathy/tiny_shakespeare](https://huggingface.co/datasets/karpathy/tiny_shakespeare).

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

## todos

- [x] [NEFTune](https://arxiv.org/abs/2310.05914)
- [x] [Grouped Query Attention](https://arxiv.org/pdf/2305.13245)
- [ ] Beam search
- [x] [RMSNorm](https://arxiv.org/abs/1910.07467)
- [x] [Knowledge distillation (student-teacher learning)](https://arxiv.org/abs/1503.02531)
- [x] Cosine embedding loss
- [ ] Attention mask & padding
- [x] DDP
- [ ] mixed-precision

## resources

- [nanoGPT](https://github.com/karpathy/nanoGPT/tree/9755682b981a45507f6eb9b11eadef8cb83cebd5)
- [TinyLlama](https://github.com/jzhang38/TinyLlama/tree/main)
- [llama paper](https://github.com/meta-llama/llama)
- [distilbert paper](https://arxiv.org/abs/1910.01108)
