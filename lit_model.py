from model import PicoGPT
from losses import *  
import lightning as L
import torch

class LitPicoGPT(L.LightningModule):
    def __init__(self, model_config, train_config):
        super().__init__()
        self.model = PicoGPT(model_config)
        self.train_config = train_config 
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attn_mask = batch["attn_mask"]
        prompt_len = batch["prompt_len"]
        seq_len = batch["seq_len"]
    
        out = self.model(input_ids, attn_mask)
        logits = out["logits"]
    
        B, T, d = logits.shape
        loss = batched_cross_entropy(input_ids, logits, prompt_len, seq_len)
        self.log("loss", loss, prog_bar=True)

        return loss
    
    def configure_optimizers(self):
        opt = self.model.configure_optimizers(self.train_config)
        return opt
    
# TODO: add loggers (console and wandb)

if __name__ == "__main__":
    from configs import *

    config = Config(
        vocab_size=32000,
        block_size=512,
        n_layer=1,
        n_embd=512,
        n_head=32,
        n_query_groups=4,
        tie_weights=True,
        rope_theta=10000,
        neftune_noise_alpha=0.1,
        dropout=0.1,
    )

    train_config = TrainConfig(
        n_epochs=1,
        batch_size=4,
        lr=8.9e-05,
        betas = (0.9, 0.95),
        min_lr=4e-05,
        warmup_ratio=0.0,
        grad_clip=1.0,
        weight_decay=0.1,
        log_steps=5,
        wandb_report=False,
        ckpt_path=None,
        save_path=None,
        ddp=True,
    )

    model = LitPicoGPT(model_config=config, train_config=train_config)
    
    from dataset import *
    from torch.utils.data import DataLoader

    data = load_starcoder_test(tok)
    dl = DataLoader(
        data,
        batch_size=1,
        collate_fn=sft_collate_fn,
        pin_memory=True,
    )

    batch = next(iter(dl))
    loss = model.training_step(batch, 0)

