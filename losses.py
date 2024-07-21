import torch
import torch.nn.functional as F

TORCH_F32_MIN = torch.finfo(torch.float32).min


# TODO: potentially remove prompt tokens from kl loss
# TODO: pad tokens are still included here 
def topk_kl_div(input_ids, logits, targets, t=1.0, k=None, ignore_index=-100):
    assert logits.shape == targets.shape, f'logits: {logits.shape} != targets: {targets.shape}'

    if k is not None:
        B, T, d = targets.size()
        _, indices = torch.topk(targets, k, dim=-1)
        mask = (
            torch.ones_like(targets, requires_grad=False)
            .scatter_(-1, indices, 0)
            .to(logits.device, dtype=torch.bool)
        )
        logits[mask] = TORCH_F32_MIN
        targets[mask] = TORCH_F32_MIN

        del mask
        torch.cuda.empty_cache()

    input_ids = input_ids.view(-1)
    targets = targets.view(-1, targets.size(-1))
    logits = logits.view(-1, logits.size(-1))

    # hacky: not considering the kl loss over the eos token since it matches the padded one 
    targets = targets[input_ids != ignore_index, :]
    logits = logits[input_ids != ignore_index, :]

    targets = F.log_softmax(targets / t, dim=-1)
    logits = F.log_softmax(logits / t, dim=-1)

    kl = F.kl_div(logits, targets, log_target=True, reduction="batchmean")

    return kl


def batched_cross_entropy(input_ids, logits, prompt_len, seq_len):
    B, T, d = logits.shape

    logits = logits[:, :-1, :]
    targets = input_ids[:, 1:].clone()

    # mask prompt tokens
    y = torch.arange(T - 1, device=targets.device).unsqueeze(0).repeat((B, 1))
    prompt_len = prompt_len.unsqueeze(1).repeat(1, T - 1)
    prompt_len = prompt_len - 1  # lm loss shifted
    before_mask = y < prompt_len

    # mask pad tokens
    seq_len = seq_len.unsqueeze(1).repeat(1, T - 1)
    seq_len = seq_len - 1
    after_mask = y >= seq_len

    # mask for padded
    targets.masked_fill_(before_mask, -100)
    targets.masked_fill_(after_mask, -100)

    # print(targets)

    # debug
    # _targets = targets.clone()
    # _targets.masked_fill_(_targets == -100, tok.unk_token_id)
    # print(tok.batch_decode(_targets))

    # flatten
    # FIXME: potential mismatch in how logits get ordered with targets - revisit if training bad
    logits = logits.reshape((B * (T - 1), d))
    targets = targets.view(-1)

    # print(tok.decode(targets[prompt_len:]))

    return F.cross_entropy(logits, targets, reduction="mean", ignore_index = -100)


def cosine_loss(inputs, targets, device, n):
    inputs = torch.cat(inputs).to(device)
    targets = torch.cat([h for e, h in enumerate(targets) if e % n == 0]).to(device)

    assert (
        inputs.shape == targets.shape
    ), "make sure to properly configure student model"

    inputs = inputs.view(-1, inputs.shape[-1])
    targets = targets.view(-1, targets.shape[-1])

    B, d = inputs.shape
    y = torch.ones(B, device=device)

    loss = F.cosine_embedding_loss(inputs, targets, y)
    return loss
