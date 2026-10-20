"""Task 1 Step 2: regularization-strength study.

Trains one short-run DPO fork per beta in configs/dpo.yaml `betas`, each from the ORIGINAL policy
initialization (fresh LoRA, same seed, same first `short_ablation_examples` training pairs, same
optimizer and LoRA config), evaluates every fork with the identical protocol, and writes a summary
table that also includes the one-epoch standard run and the SFT baseline (if already evaluated).

    python -m task1_dpo.ablate_beta --config configs/dpo.yaml
Re-running skips forks whose adapter and evaluation already exist.
"""
from __future__ import annotations

import argparse

from common.data import load_yaml, repo_path
from common.logging_utils import save_json
from common.runs import adapter_exists, last_train_record, load_json_if, run, write_csv


def fork_name(beta: float) -> str:
    return f"beta_{beta:g}"


def summary_row(cfg, name, budget, beta):
    res = cfg["results_dir"]
    ev = load_json_if(f"{res}/{name}/eval_dpo_standard_eval.json")
    if ev is None:
        return None
    tr = load_json_if(f"{res}/{name}/run_config.json") or {}
    last = last_train_record(res, name) or {}
    p, g = ev["pairs"]["overall"], ev.get("generation_metrics", {})
    rt = g.get("response_tokens", {})
    return {
        "condition": name,
        "budget": budget,
        "beta": beta,
        "train_pairs": tr.get("n_train_pairs"),
        "optimizer_updates": tr.get("optimizer_updates"),
        "final_train_loss": last.get("loss"),
        "final_train_acc": last.get("acc"),
        "heldout_dpo_loss": p["dpo_loss"],
        "heldout_pref_acc": p["preference_accuracy"],
        "heldout_pref_acc_ci_lo": p["preference_accuracy_ci95"][0],
        "heldout_pref_acc_ci_hi": p["preference_accuracy_ci95"][1],
        "heldout_mean_margin": p["mean_margin"],
        "kl_token_mean": g.get("kl_token_mean"),
        "kl_seq_mean": g.get("kl_seq_mean"),
        "reward_mean": g.get("reward_mean"),
        "reward_std": g.get("reward_std"),
        "entropy_mean": g.get("entropy_mean"),
        "len_mean": rt.get("mean"),
        "len_std": rt.get("std"),
        "len_iqr": rt.get("iqr"),
        "truncation_rate": g.get("truncation_rate"),
        "word_limit_compliance": (ev.get("word_limit") or {}).get("compliance_rate"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    n = int(cfg["short_ablation_examples"])
    res = cfg["results_dir"]

    if not args.summary_only:
        for beta in cfg["betas"]:
            name = fork_name(float(beta))
            out = f"outputs/task1_dpo/{name}"
            if not adapter_exists(out):
                run("task1_dpo.train", "--config", args.config, "--run-name", name, "--output", out,
                    "--beta", beta, "--max-examples", n)
            if not repo_path(f"{res}/{name}/eval_dpo_standard_eval.json").exists():
                run("task1_dpo.evaluate", "--config", args.config, "--adapter", out, "--name", name, "--beta", beta)

    rows = [r for r in (
        summary_row(cfg, "sft", "none (reference policy)", None),
        summary_row(cfg, "standard", f"1 epoch, full standard set", float(cfg["beta"])),
        *[summary_row(cfg, fork_name(float(b)), f"short run, first {n} pairs", float(b)) for b in cfg["betas"]],
    ) if r]
    write_csv(f"{res}/summary_beta.csv", rows)
    save_json(f"{res}/summary_beta.json", rows)


if __name__ == "__main__":
    main()
