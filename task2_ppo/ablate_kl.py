"""Task 2 Step 3: reward-overoptimisation / KL-pressure study.

Matched short continuations from the identical supplied midpoint (policy + critic) for every beta_KL in
configs/ppo.yaml `kl_values`, with epsilon fixed at `clip_epsilon`, the same prompt order, `fork_updates`
updates and the same evaluation protocol. The (epsilon=clip_epsilon, beta=kl_beta) fork is the same run as
in the clipping study and is trained only once.

    python -m task2_ppo.ablate_kl --config configs/ppo.yaml
"""
from __future__ import annotations

import argparse

from common.data import load_yaml, repo_path
from common.logging_utils import save_json
from common.runs import adapter_exists, run, write_csv
from task2_ppo.analyze_clipping import fork_name, fork_stability, heldout_row


def summarize(cfg):
    res_dir = repo_path(cfg["results_dir"])
    eps = float(cfg["clip_epsilon"])
    rows = [{"condition": "midpoint (start)", "kl_beta": None, **heldout_row(res_dir, "midpoint")},
            {"condition": "standard (20 updates)", "kl_beta": float(cfg["kl_beta"]),
             **fork_stability(res_dir, "standard"), **heldout_row(res_dir, "standard")}]
    for beta in [float(b) for b in cfg["kl_values"]]:
        name = fork_name(eps, beta)
        rows.append({"condition": name, "kl_beta": beta, **fork_stability(res_dir, name), **heldout_row(res_dir, name)})
    write_csv(res_dir / "summary_kl.csv", rows)
    save_json(res_dir / "summary_kl.json", rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("KL beta conditions:", cfg["kl_values"])
    print("Fork update budget:", cfg["fork_updates"])
    res_dir = repo_path(cfg["results_dir"])
    if not args.summary_only:
        eps, n = float(cfg["clip_epsilon"]), int(cfg["fork_updates"])
        for beta in [float(b) for b in cfg["kl_values"]]:
            name = fork_name(eps, beta)
            out = f"outputs/task2_ppo/{name}"
            if not adapter_exists(out):
                run("task2_ppo.continue_train", "--config", args.config, "--run-name", name, "--output", out,
                    "--updates", n, "--clip-epsilon", eps, "--kl-beta", beta)
            if not (res_dir / name / "eval_heldout.json").exists():
                run("task2_ppo.evaluate", "--config", args.config, "--adapter", out, "--name", name)
    summarize(cfg)


if __name__ == "__main__":
    main()
