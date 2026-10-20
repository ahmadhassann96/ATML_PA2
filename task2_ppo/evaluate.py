"""Task 2 held-out evaluation (common protocol in common/rl_eval.py).

    python -m task2_ppo.evaluate --adapter checkpoints/ppo_midpoint_policy --name midpoint
    python -m task2_ppo.evaluate --adapter outputs/task2_ppo/standard --name standard

Every PPO condition is evaluated on the same first `--n-prompts` held-out prompts with the same seed,
sampling settings and the evaluation cap `eval_max_response_length` from configs/ppo.yaml.
"""
from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl
from common.models import load_policy, load_reward_model, load_tokenizer
from common.rl_eval import evaluate_policy

DEFAULT_N_PROMPTS = 96


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def run_evaluation(config_path, adapter, name, n_prompts=DEFAULT_N_PROMPTS):
    cfg = load_yaml(config_path)
    return evaluate_policy(cfg, adapter, name, cfg["results_dir"], n_prompts=n_prompts,
                           max_new_tokens=int(cfg.get("eval_max_response_length", cfg["max_response_length"])),
                           label=f"eval:{name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter directory, or 'none' for the base SFT policy")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--n-prompts", type=int, default=DEFAULT_N_PROMPTS)
    args = ap.parse_args()
    run_evaluation(args.config, args.adapter, args.name, args.n_prompts)


if __name__ == "__main__":
    main()
