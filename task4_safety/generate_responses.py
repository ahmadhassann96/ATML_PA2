"""Task 4 Step 1: deterministic XSTest responses from the four frozen policies (SFT, DPO, PPO, GRPO).

Every policy answers every XSTest prompt in the fixed CSV order with identical greedy decoding and the common cap
`safety_max_new_tokens`. Batches are the same consecutive slices of the CSV for every policy, so batch composition
(and therefore padding) is identical across policies.

    python -m task4_safety.generate_responses --config configs/feedback.yaml
Writes results/task4_safety/generated_<policy>.jsonl; existing files are kept (use --force to regenerate).
"""
from __future__ import annotations

import argparse
import gc

import pandas as pd
import torch

from common.data import load_yaml, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import wall_timer
from common.models import load_policy, load_tokenizer

BATCH_SIZE = 32  # same for every policy; greedy decoding, so only speed depends on it


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    df = load_xstest(cfg)
    records = []
    timer = wall_timer()
    n_batches = (len(df) + batch_size - 1) // batch_size
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        print(f"[{policy_name}] batch {start // batch_size + 1}/{n_batches} ({timer():.0f}s elapsed)", flush=True)
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for (_, row), response, n_tok, trunc in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"], gen["truncated"]):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": response,
                "response_tokens": int(n_tok),
                "truncated": bool(trunc),
            })
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", nargs="+", default=["sft", "dpo", "ppo", "grpo"])
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Policies:", list(policy_specs(cfg)))
    print("XSTest rows:", len(load_xstest(cfg)))
    out_dir = repo_path(cfg["results_dir"]) / "task4_safety"
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = policy_specs(cfg)
    for name in args.policies:
        adapter = specs[name]
        if adapter and not (repo_path(adapter) / "adapter_model.safetensors").exists():
            raise SystemExit(f"{name}: adapter {adapter} not found - finish Task {'1' if name == 'dpo' else '2' if name == 'ppo' else '3'} first")
        path = out_dir / f"generated_{name}.jsonl"
        if path.exists() and not args.force:
            print(f"{name}: {path} exists - skipping")
            continue
        rows = generate_for_policy(cfg, name, batch_size=args.batch_size)
        write_jsonl(path, rows)
        print(f"{name}: wrote {len(rows)} responses to {path} "
              f"(mean {sum(r['response_tokens'] for r in rows) / len(rows):.1f} tokens)", flush=True)


if __name__ == "__main__":
    main()
