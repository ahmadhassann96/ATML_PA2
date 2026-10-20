"""Common held-out evaluation protocol for the online-RL policies (PPO in Task 2, GRPO in Task 3).

For one policy (LoRA adapter, or None for the base SFT policy):
  * the first `n_prompts` prompts of the fixed held-out RL prompt pool (file order; prompts longer than
    `max_prompt_length` are skipped and counted), identical for every condition;
  * one sampled response per prompt with the config's generation settings, a fixed seed, and the
    evaluation response cap;
  * per response: learned reward (raw, and with the missing-EOS penalty), sampled-token KL to the frozen
    reference (sum over tokens and per-token mean, the course estimator), full-distribution entropy,
    response length, EOS/truncation flags.
Outputs: results/<task>/<name>/eval_heldout.json and generations_heldout.jsonl.
"""
from __future__ import annotations

import gc

import numpy as np
import torch

from common.data import read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json, set_seed, wall_timer
from common.models import load_policy, load_reward_model, load_tokenizer
from common.rollout import disable_dropout, generate, reference_logprobs, score_rewards, token_logprobs

GEN_BATCH = 32  # prompts per generate() call; a batch runs until its longest response ends, so fewer, larger batches are faster
N_BOOT = 1000


def length_stats(values):
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {}
    q25, q50, q75 = np.percentile(a, [25, 50, 75])
    return {"mean": float(a.mean()), "std": float(a.std()), "median": float(q50), "q25": float(q25),
            "q75": float(q75), "iqr": float(q75 - q25), "max": float(a.max()), "n": int(a.size)}


def bootstrap_ci(values, seed=0):
    a = np.asarray(values, dtype=float)
    if a.size < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    m = a[rng.integers(0, a.size, size=(N_BOOT, a.size))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def heldout_prompts(cfg, tokenizer, n_prompts):
    rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    keep, skipped = [], 0
    for r in rows:
        msgs = r["messages"]
        if len(tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)) > int(cfg["max_prompt_length"]):
            skipped += 1
            continue
        keep.append((r["prompt_id"], msgs))
        if n_prompts and len(keep) >= int(n_prompts):
            break
    return keep, skipped


def evaluate_policy(cfg, adapter, name, results_dir, n_prompts=96, max_new_tokens=768, label=None):
    label = label or name
    out_dir = repo_path(results_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = wall_timer()
    tok = load_tokenizer(cfg["base_model"])
    adapter = None if adapter in (None, "", "none", "None") else adapter
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    disable_dropout(policy)
    prompts, skipped = heldout_prompts(cfg, tok, n_prompts)
    prompts.sort(key=lambda p: len(tok.apply_chat_template(p[1], tokenize=False, add_generation_prompt=True)))
    print(f"[{label}] held-out evaluation: {len(prompts)} prompts ({skipped} skipped as longer than "
          f"{cfg['max_prompt_length']} tokens), cap {max_new_tokens} tokens, adapter={adapter}", flush=True)

    set_seed(int(cfg["seed"]))
    records = []
    n_batches = (len(prompts) + GEN_BATCH - 1) // GEN_BATCH
    for b in range(n_batches):
        chunk = prompts[b * GEN_BATCH:(b + 1) * GEN_BATCH]
        print(f"[{label}] generation batch {b + 1}/{n_batches} ({timer():.0f}s elapsed)", flush=True)
        roll = generate(policy, tok, [m for _, m in chunk], cfg, max_new_tokens)
        pol = token_logprobs(policy, roll, entropy=True)
        ref = reference_logprobs(policy, roll)
        mask = pol["mask"]
        diff = (pol["token_logp"] - ref["token_logp"]) * mask
        for i, (pid, msgs) in enumerate(chunk):
            n = int(mask[i].sum())
            records.append({
                "prompt_id": pid, "prompt": msgs[-1]["content"], "_messages": msgs,
                "response": roll["responses"][i], "response_tokens": roll["response_lengths"][i],
                "terminated_with_eos": roll["terminated_with_eos"][i], "truncated": roll["truncated"][i],
                "kl_seq": float(diff[i].sum()), "n_tokens": n,
                "entropy_mean": float(pol["entropy"][i].sum() / max(n, 1)),
                "logp_policy": float((pol["token_logp"][i] * mask[i]).sum()),
            })
        del roll, pol, ref
    del policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[{label}] scoring {len(records)} responses with the reward model", flush=True)
    rm, rtok = load_reward_model(cfg)
    raw, eff = score_rewards(rm, rtok, [r["_messages"] for r in records], [r["response"] for r in records],
                             [r["truncated"] for r in records], cfg)
    for r, a, e in zip(records, raw.tolist(), eff.tolist()):
        r["reward_raw"], r["reward"] = a, e
        r.pop("_messages")
    del rm

    n_tok = sum(r["n_tokens"] for r in records)
    rewards_raw = [r["reward_raw"] for r in records]
    rewards = [r["reward"] for r in records]
    metrics = {
        "name": name, "adapter": adapter, "n_prompts": len(records), "n_prompts_skipped_long": skipped,
        "max_new_tokens": int(max_new_tokens), "seed": int(cfg["seed"]), "generation": cfg["generation"],
        "reward_raw_mean": float(np.mean(rewards_raw)), "reward_raw_std": float(np.std(rewards_raw)),
        "reward_raw_ci95": bootstrap_ci(rewards_raw),
        "reward_mean": float(np.mean(rewards)), "reward_ci95": bootstrap_ci(rewards),
        "missing_eos_penalty": float(cfg.get("missing_eos_penalty", 0.0)),
        "kl_token_mean": float(sum(r["kl_seq"] for r in records) / max(n_tok, 1)),
        "kl_seq_mean": float(np.mean([r["kl_seq"] for r in records])),
        "kl_seq_ci95": bootstrap_ci([r["kl_seq"] for r in records]),
        "entropy_mean": float(np.mean([r["entropy_mean"] for r in records])),
        "response_tokens": length_stats([r["response_tokens"] for r in records]),
        "truncation_rate": float(np.mean([r["truncated"] for r in records])),
        "eos_rate": float(np.mean([r["terminated_with_eos"] for r in records])),
        "wall_clock_s": round(timer(), 1),
    }
    write_jsonl(out_dir / "generations_heldout.jsonl", records)
    save_json(out_dir / "eval_heldout.json", metrics)
    print(f"[{label}] reward={metrics['reward_raw_mean']:.3f} (with EOS penalty {metrics['reward_mean']:.3f}) "
          f"kl_tok={metrics['kl_token_mean']:.4f} entropy={metrics['entropy_mean']:.3f} "
          f"len={metrics['response_tokens']['mean']:.1f} trunc={metrics['truncation_rate']:.2f} "
          f"({metrics['wall_clock_s'] / 60:.1f} min)", flush=True)
    return metrics


# ----------------------------------------------------------------------------- qualitative candidates
def _short(t, n=500):
    t = " ".join(str(t).split())
    return t if len(t) <= n else t[:n] + " …"


def reward_change_candidates(results_dir, policy="standard", base="midpoint", k=6, ascending=False):
    """Held-out prompts ranked by reward change of `policy` over `base` on the same prompt. Read the texts and
    judge quality yourself: rising reward with a worse/longer/truncated answer is a reward-quality disagreement,
    rising reward with a genuinely better answer is agreement."""
    from pathlib import Path
    from common.live import read_jsonl_safe
    import pandas as pd
    rd = Path(results_dir)
    a = {r["prompt_id"]: r for r in read_jsonl_safe(rd / policy / "generations_heldout.jsonl")}
    b = {r["prompt_id"]: r for r in read_jsonl_safe(rd / base / "generations_heldout.jsonl")}
    rows = [{"prompt_id": p, "reward_change": a[p]["reward_raw"] - b[p]["reward_raw"],
             f"{policy}_reward": a[p]["reward_raw"], f"{base}_reward": b[p]["reward_raw"],
             f"{policy}_tokens": a[p]["response_tokens"], f"{base}_tokens": b[p]["response_tokens"],
             f"{policy}_truncated": a[p]["truncated"], "prompt": _short(a[p]["prompt"], 200),
             f"{policy}_response": _short(a[p]["response"]), f"{base}_response": _short(b[p]["response"])}
            for p in a.keys() & b.keys()]
    df = pd.DataFrame(rows)
    return df.sort_values("reward_change", ascending=ascending).head(k) if not df.empty else df
