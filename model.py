import functools
import math

import torch
import torch.nn.functional as F
import wandb
from rotary_embedding_torch import RotaryEmbedding
from torch import nn

from configs import Config
from utils import RMSNorm, neftune_forward_hook


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc_1 = config.linear_cls(config.n_embd, 4*config.n_embd, bias = config.bias)
        self.fc_2 = config.linear_cls(4*config.n_embd, config.n_embd, bias = config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.gelu(self.fc_1(x))
        x = self.dropout(self.fc_2(x))
        return x


# always dense linear layers for causal self attention?
class CausalSelfAttention(nn.Module):
    """
    rotary positional embeddings from: https://github.com/lucidrains/rotary-embedding-torch
    Supports: GQA, MHA, SA based on choices for `n_query_groups` in config
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        shape = (config.n_head + 2*config.n_query_groups)*config.head_size # n_head per query + (k+v)*n_query_groups
        
        self.c_attn = config.linear_cls(config.n_embd, shape, bias = config.bias)
        # self.c_attn = nn.Linear(config.n_embd, shape, bias = config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        # self.proj = nn.Linear(config.head_size*config.n_head, config.n_embd, bias = config.bias)
        self.proj = config.linear_cls(config.head_size*config.n_head, config.n_embd, bias = config.bias)

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            
            # (1,1,bsz,bsz) to match (B,nh,T,hs) dims
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

        self.rotary_emb = RotaryEmbedding(config.head_size, theta = config.rope_theta)

    def forward(self, x):
        B, T, C = x.shape 

        qkv = self.c_attn(x)

        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2 # queries per group + 1 key + 1 value 
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)

        # shape of q is (B,T, n_query_groups, q_per_kv, head_size), k and v have (B,T, n_query_groups, 1, head_size)
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2) #splits total_qkv into amount per q, per k, per v

        q = q.reshape(B,  T, -1, self.config.head_size)  # (B, T, nh_q, hs)
        k = k.reshape(B,  T, -1, self.config.head_size)  
        v = v.reshape(B,  T, -1, self.config.head_size)  
        
        y = self.scaled_dot_product_attention(q,k,v)
        y = y.reshape(B,T,self.config.n_head*self.config.head_size)

        y = self.proj(y)
        return y

    def scaled_dot_product_attention(self, q, k, v):
        T = q.shape[1]

        q = q.transpose(1, 2) # (B,T,nh_q, hs) -> (B,nhs,T,hs)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # rotary positional embeddings 
        # "dimensions should end with (seq_len, feature dimension), and any number of preceding dimensions (batch, heads, etc)"
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        if q.shape != k.shape:
            # repeat k,v enough times so we can shove into F.scaled_dot_product_attention
            k = k.repeat_interleave(q.shape[1]//k.shape[1], dim=1)
            v = v.repeat_interleave(q.shape[1]//v.shape[1], dim=1)


        if self.flash:
            y = F.scaled_dot_product_attention(q,k,v, attn_mask = None, dropout_p = self.config.dropout if self.training else 0.0, is_causal = True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att) if self.training else nn.Identity(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        return y.transpose(1,2).contiguous()


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_1 = config.norm_cls(config.n_embd, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = config.norm_cls(config.n_embd, config.norm_eps)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))
        return x

class PicoGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config 

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = config.norm_cls(config.n_embd, config.norm_eps),
        ))

        self.lm_head = config.linear_cls(config.n_embd, config.vocab_size, bias = config.bias)

        if config.tie_weights:
            self.transformer.wte.weight = self.lm_head.weight # weight tying 
        
        self.apply(self._init_weights)

        self.transformer.wte.register_forward_hook(
            functools.partial(neftune_forward_hook, alpha = config.neftune_noise_alpha)
        )


    # from karpathy (tiny llama has different init)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        if self.config.tie_weights:
            n_params -= sum(p.numel() for p in self.lm_head.parameters())

        return n_params


    def forward(self, x):
        assert x.shape[1]<=self.config.block_size, f"cannot forward seq of length {x.shape[1]}. max `block_size` configured to {self.config.block_size}"
        x = self.transformer.wte(x)
        # pos = torch.arange(0, x.shape[1], dtype=torch.long, device="cpu")
        # x = x + self.transformer.wpe(pos)

        # "one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer"
        hidden_states = [x] 

        for block in self.transformer.h:
            x = block(x)
            hidden_states.append(x)

        x = self.transformer.ln_f(x)

        return {'logits': self.lm_head(x), 'hidden_states': hidden_states}


    def configure_optimizers(self, train_config):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': train_config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr = train_config.lr)
        return optimizer


    def generate(self, input_ids, max_new_tokens = 64, temperature = 1.0, top_k = None, do_sample = False, eos_token_id = -1, num_beams = 1, num_return_sequences=1):
        new_tokens = [] 

        self.eval()
        with torch.no_grad():
            while len(new_tokens)<max_new_tokens:
                # truncate input_ids to context_length 
                input_ids = input_ids[:,-self.config.block_size:]

                logits = self.forward(input_ids)['logits'][:,-1]

                if do_sample:
                    if top_k is not None:
                        v, _ = torch.topk(logits, top_k)
                        logits[logits<v[:,[-1]]] = float('-inf')

                    probs = F.softmax(logits/temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    new_tokens.append(next_token.item())

                else: #greedy search subset of beam search 
                    new_tokens = self._beam_search(input_ids, max_new_tokens, num_beams, num_return_sequences)
                    return new_tokens

                # append 
                input_ids = torch.cat((input_ids, next_token), dim=1)

                if next_token.item() == eos_token_id:
                    break 

        return new_tokens

    

    # TODO: handle eos token generated case
    def _beam_search(self, input_ids, max_new_tokens=100, num_beams=1, num_return_sequences=1):
        size = input_ids.shape[1]

        beams = [{'cum_log_prob': 0., 'ids': input_ids}] 
        
        for _ in range(max_new_tokens):
            new_beams = [] 

            for beam in beams:
                # truncate 
                logits = self.forward(beam['ids'][:,-self.config.block_size:])['logits']
                probs = F.log_softmax(logits[:,-1,:], dim=-1) #avoid underflow 
                ps, ids = torch.topk(probs, k=num_beams)

                ps = ps.squeeze(0)
                ids = ids.squeeze(0)

                # log(p1*p2*p3) = log(p1)+log(p2)+log(p3)
                for p,i in zip(ps, ids):
                    new_beams.append({'cum_log_prob': beam['cum_log_prob'] + p, 'ids': torch.cat((beam['ids'], i.view(1,1)), dim=-1)})
                    
            beams = new_beams

            # keep `num_beams`
            beams = sorted(beams, key = lambda x: x['cum_log_prob'], reverse = True)[:num_beams]

        return [beam['ids'][:,size:].squeeze(0) for beam in beams[:num_return_sequences]]




