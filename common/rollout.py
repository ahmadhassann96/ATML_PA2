"""Shared on-policy rollout utilities for PPO (Task 2) and GRPO (Task 3).

Conventions (identical for every run and condition):
  * Prompts: the RL prompt pool, filtered to prompts whose chat-rendered length fits `max_prompt_length`
    (prompts are never truncated - cutting a chat template would remove the assistant header), then
    shuffled once with the config seed. Update u uses prompts [u*P, (u+1)*P) of that order, so every fork
    of a study sees exactly the same prompts in the same order.
  * Sampling: the config's generation settings (temperature/top-p) with the response cap given by the caller.
  * Reward: learned reward-model score of (prompt, response); a response that hits the cap without EOS gets
    `missing_eos_penalty` subtracted (the same rule the course used for the supplied cache).
  * Log-probs / values are computed with explicit position ids (left padding) over the response window only.
  * Dropout is disabled in trainable models so that the importance ratio is exactly 1 before the first update.
"""
from __future__ import annotations

import random

import torch

from common.generation import batch_generate, score_reward_pairs
from common.logprobs import position_ids_from_mask, response_logprobs
from common.models import reference_mode

SCORE_CHUNK = 4  # sequences per forward pass when scoring (memory bound on 8 GB)


def disable_dropout(model) -> int:
    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout) and m.p > 0:
            m.p = 0.0
            n += 1
    return n


def prompt_order(rows, tokenizer, max_prompt_length: int, seed: int):
    """Deterministic, length-filtered, shuffled prompt order shared by all runs of a study."""
    kept = []
    for r in rows:
        msgs = r["messages"] if isinstance(r.get("messages"), list) else [{"role": "user", "content": str(r.get("prompt", ""))}]
        n = len(tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
        if n <= max_prompt_length:
            kept.append({"prompt_id": r.get("prompt_id"), "source_index": r.get("source_index"), "messages": msgs})
    random.Random(seed).shuffle(kept)
    return kept


@torch.no_grad()
def generate(policy, tokenizer, messages_list, cfg, max_new_tokens: int, n_samples: int = 1):
    """Sample n_samples completions per prompt. Returns tensors aligned so that column j of the response
    window holds response token j. Rows are ordered prompt-major: [p0s0, p0s1, ..., p1s0, ...]."""
    gen = cfg["generation"]
    expanded = [m for m in messages_list for _ in range(n_samples)]
    # a trainable policy has config.use_cache=False (gradient checkpointing); generation needs the KV cache
    base = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
    prev = base.config.use_cache
    base.config.use_cache = True
    try:
        out = batch_generate(policy, tokenizer, expanded, max_prompt_length=4096, max_new_tokens=max_new_tokens,
                             temperature=float(gen["temperature"]), top_p=float(gen["top_p"]),
                             do_sample=bool(gen["do_sample"]))
    finally:
        base.config.use_cache = prev
    seq, width = out["sequences"], out["prompt_width"]
    rmask = out["response_mask"].float()
    attn = out["attention_mask"].clone()
    attn[:, width:] = rmask.long()  # never attend to post-EOS padding
    full_mask = torch.zeros_like(seq, dtype=torch.float32)
    full_mask[:, width:] = rmask
    return {
        "sequences": seq, "attention_mask": attn, "full_response_mask": full_mask, "prompt_width": width,
        "response_mask": rmask, "responses": out["responses"], "response_lengths": out["response_lengths"],
        "truncated": out["truncated"], "terminated_with_eos": out["terminated_with_eos"], "messages": expanded,
    }


@torch.no_grad()
def score_rewards(reward_model, reward_tokenizer, messages, responses, truncated, cfg):
    raw = []
    for i in range(0, len(responses), SCORE_CHUNK):
        raw += score_reward_pairs(reward_model, reward_tokenizer, messages[i:i + SCORE_CHUNK], responses[i:i + SCORE_CHUNK],
                                  max_length=int(cfg.get("reward_max_length", 1024))).tolist()
    pen = float(cfg.get("missing_eos_penalty", 0.0))
    eff = [r - pen if t else r for r, t in zip(raw, truncated)]
    return torch.tensor(raw, dtype=torch.float32), torch.tensor(eff, dtype=torch.float32)


def token_logprobs(model, roll, with_grad: bool = False, entropy: bool = False, rows=None):
    """Response-token log-probs under `model` for (a subset of) the rollout batch, chunked for memory.
    Returns dict(token_logp [B,R], mask [B,R], entropy [B,R] optional), R = response window width."""
    idx = list(range(roll["sequences"].shape[0])) if rows is None else list(rows)
    seq, attn, fm = roll["sequences"][idx], roll["attention_mask"][idx], roll["full_response_mask"][idx]
    first = roll["prompt_width"]
    outs = []
    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        for i in range(0, len(idx), SCORE_CHUNK):
            sl = slice(i, i + SCORE_CHUNK)
            # every sampled response has at least one token (EOS counts), so the window always starts at the
            # prompt width and all chunks share the same width R
            assert bool(fm[sl][:, first].bool().all()), "empty response column at prompt width"
            o = response_logprobs(model, seq[sl], attn[sl], fm[sl], return_entropy=entropy)
            outs.append(o)
    res = {"token_logp": torch.cat([o["token_logp"] for o in outs]), "mask": torch.cat([o["mask"] for o in outs])}
    if entropy:
        res["entropy"] = torch.cat([o["entropy"] for o in outs])
    return res


@torch.no_grad()
def reference_logprobs(model, roll, rows=None):
    with reference_mode(model):
        return token_logprobs(model, roll, with_grad=False, rows=rows)


def _value_parts(value_model):
    """(transformer body, scalar head) of the critic, whether or not it is wrapped by PEFT. The body is the
    inner Qwen2Model, so the head is applied only once, by us, to the final normalised hidden state."""
    base = value_model.get_base_model() if hasattr(value_model, "get_base_model") else value_model
    return getattr(base, base.base_model_prefix), base.score


def value_window(value_model, roll, with_grad: bool = False, rows=None):
    """Critic values V(s_t) for each response token t: the scalar head applied to the hidden state at the
    position just before token t (the state from which token t is generated). Shape [B, R], float32."""
    idx = list(range(roll["sequences"].shape[0])) if rows is None else list(rows)
    seq, attn = roll["sequences"][idx], roll["attention_mask"][idx]
    first, L = roll["prompt_width"], roll["sequences"].shape[1]
    body, head = _value_parts(value_model)
    trainable = [p for p in head.parameters() if p.requires_grad]
    head_dtype = (trainable or list(head.parameters()))[0].dtype
    outs = []
    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        for i in range(0, len(idx), SCORE_CHUNK):
            sl = slice(i, i + SCORE_CHUNK)
            o = body(input_ids=seq[sl], attention_mask=attn[sl], position_ids=position_ids_from_mask(attn[sl]),
                     use_cache=False, return_dict=True)
            h = o.last_hidden_state[:, first - 1: L - 1, :]
            outs.append(head(h.to(head_dtype)).squeeze(-1).float())
    return torch.cat(outs)


def float32_trainables(model) -> int:
    """Cast trainable parameters to float32 (Adam + grad scaling on fp16 parameters is unstable)."""
    n = 0
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
            n += 1
    return n
