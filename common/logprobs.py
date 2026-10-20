"""Memory-efficient response-token log-probabilities for an 8 GB GPU.

Why not common.generation.response_sequence_logprobs?  That helper materialises a float32
log-softmax over the full 151,936-token vocabulary for *every* position of the sequence
(prompt included): ~0.47 GB per 768-token sequence, plus its gradient. On an 8 GB laptop GPU
that is the difference between fitting and OOM. This module computes exactly the same quantity
(sum over response tokens of log pi(y_t | x, y_<t)) but

  1. asks the model for logits only on the tail of the sequence that contains response tokens
     (`logits_to_keep`, valid because the course collators LEFT-pad, so responses are right-aligned),
  2. does the float32 log-softmax one row at a time, and
  3. passes explicit `position_ids`, so left padding does not shift the rotary positions of the
     real tokens (HF otherwise numbers positions from the first *padded* column).
"""
from __future__ import annotations

import torch


def position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    pos = attention_mask.long().cumsum(-1) - 1
    return pos.clamp_min(0)


def response_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    return_entropy: bool = False,
):
    """Per-token log-probs of the response tokens under `model`.

    Args:
        input_ids, attention_mask: [B, L] left-padded prompt+response.
        response_mask: [B, L], 1 on response tokens (the tokens being scored), 0 elsewhere.

    Returns dict with
        token_logp [B, R]  log pi(token) for the last R columns (R = response window width),
        mask       [B, R]  response mask on that window,
        seq_logp   [B]     sum of token_logp over valid response tokens,
        entropy    [B, R]  full-distribution token entropy (only if return_entropy).
    """
    L = input_ids.shape[1]
    cols = torch.nonzero(response_mask.bool().any(0), as_tuple=False)
    if cols.numel() == 0:
        raise ValueError("response_mask selects no tokens")
    first = int(cols.min().item())
    if first == 0:
        raise ValueError("a response token in column 0 has no preceding context to predict it from")
    R = L - first  # width of the response window (right-aligned)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids_from_mask(attention_mask),
        use_cache=False,
        return_dict=True,
        logits_to_keep=R + 1,  # logits for positions L-R-1 .. L-1
    )
    logits = out.logits[:, :-1, :]  # position t predicts token t+1 -> predicts columns first..L-1
    labels = input_ids[:, first:]
    mask = response_mask[:, first:].to(torch.float32)

    token_logp, entropy = [], []
    for i in range(logits.shape[0]):  # one row at a time keeps the float32 copy small
        lp = torch.log_softmax(logits[i].float(), dim=-1)
        token_logp.append(lp.gather(-1, labels[i].unsqueeze(-1)).squeeze(-1))
        if return_entropy:
            with torch.no_grad():
                entropy.append(-(lp.exp() * lp).sum(-1))
        del lp
    token_logp = torch.stack(token_logp) * mask
    result = {"token_logp": token_logp, "mask": mask, "seq_logp": token_logp.sum(-1)}
    if return_entropy:
        result["entropy"] = torch.stack(entropy) * mask
    return result
