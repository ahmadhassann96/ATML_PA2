"""Task 3 Step 3: canonical GRPO (1/T_k) vs Dr. GRPO-style (1/L) sequence normalisation.

Two matched `fork_updates`-update continuations from the identical supplied midpoint: same prompts and order, seed,
K, reward, beta_KL, epsilon, learning rate and generation settings; only `loss_type` differs. Both record each
completion's own gradient norm (--per-seq-grad), so the allocation of gradient magnitude across short and long
completions can be measured directly. Both forks are then evaluated with the common held-out protocol.

Length-conditioned statistics (from completions.jsonl, completions that entered the loss):
  * per-token loss weight |A_k| / norm_k / B and per-completion gradient norm, by length bin (split at the pooled
    median length of both forks) and by advantage sign;
  * log-log slope of per-completion gradient norm against completion length (0 = length-neutral).

    python -m task3_grpo.compare_normalization --config configs/grpo.yaml
"""
from __future__ import annotations

import argparse

import numpy as np

from common.data import load_yaml, repo_path
from common.live import read_jsonl_safe
from common.logging_utils import save_json
from common.runs import adapter_exists, load_json_if, run, write_csv

FORKS = {"norm_grpo": "grpo", "norm_dr_grpo": "dr_grpo"}


def length_conditioned(res_dir, names):
    comps = {n: [c for c in read_jsonl_safe(res_dir / n / "completions.jsonl") if c["in_loss"]] for n in names}
    pooled = [c["response_tokens"] for v in comps.values() for c in v]
    if not pooled:
        return {}, []
    cut = float(np.median(pooled))
    rows = []
    for n, cs in comps.items():
        for sign, sel in (("A>0", lambda c: c["advantage"] > 0), ("A<0", lambda c: c["advantage"] < 0)):
            for bin_name, inb in ((f"short (<= {cut:.0f} tok)", lambda c: c["response_tokens"] <= cut),
                                  (f"long (> {cut:.0f} tok)", lambda c: c["response_tokens"] > cut)):
                sub = [c for c in cs if sel(c) and inb(c)]
                g = [c["seq_grad_norm"] for c in sub if c.get("seq_grad_norm") is not None]
                rows.append({"fork": n, "advantage_sign": sign, "length_bin": bin_name, "n": len(sub),
                             "mean_length": float(np.mean([c["response_tokens"] for c in sub])) if sub else None,
                             "mean_abs_advantage": float(np.mean([abs(c["advantage"]) for c in sub])) if sub else None,
                             "mean_per_token_weight": float(np.mean([c["per_token_weight"] for c in sub])) if sub else None,
                             "mean_seq_grad_norm": float(np.mean(g)) if g else None})
    slopes = {}
    for n, cs in comps.items():
        pts = [(c["response_tokens"], c["seq_grad_norm"]) for c in cs if c.get("seq_grad_norm") and c["response_tokens"] > 0]
        if len(pts) >= 3 and len({p[0] for p in pts}) > 1:  # a slope needs more than one distinct length
            x, y = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
            slopes[n] = {"loglog_slope_gradnorm_vs_length": float(np.polyfit(x, y, 1)[0]),
                         "corr_length_gradnorm": float(np.corrcoef([p[0] for p in pts], [p[1] for p in pts])[0, 1]),
                         "n": len(pts)}
        L = [c["response_tokens"] for c in cs]
        A = [c["advantage"] for c in cs]
        if len(L) >= 3 and np.std(L) > 0:
            slopes.setdefault(n, {})["corr_length_advantage"] = float(np.corrcoef(L, A)[0, 1])
    return {"median_length_cut": cut, "slopes": slopes}, rows


def summarize(cfg):
    res_dir = repo_path(cfg["results_dir"])
    out = []
    for n, lt in FORKS.items():
        ev = load_json_if(res_dir / n / "eval_heldout.json") or {}
        log = read_jsonl_safe(res_dir / n / "train_log.jsonl")
        tr = load_json_if(res_dir / n / "train_summary.json") or {}
        out.append({"fork": n, "loss_type": lt, "train_generated_tokens": tr.get("generated_tokens"),
                    "train_reward_mean": float(np.mean([r["reward_raw"] for r in log])) if log else None,
                    "train_len_first": log[0]["response_tokens"] if log else None,
                    "train_len_last": log[-1]["response_tokens"] if log else None,
                    "train_len_mean": float(np.mean([r["response_tokens"] for r in log])) if log else None,
                    "train_grad_norm_mean": float(np.mean([r["grad_norm"] for r in log])) if log else None,
                    "heldout_reward": ev.get("reward_raw_mean"), "heldout_reward_ci95": ev.get("reward_raw_ci95"),
                    "heldout_kl_token": ev.get("kl_token_mean"), "heldout_kl_seq": ev.get("kl_seq_mean"),
                    "heldout_entropy": ev.get("entropy_mean"),
                    "heldout_len_mean": (ev.get("response_tokens") or {}).get("mean"),
                    "heldout_len_std": (ev.get("response_tokens") or {}).get("std"),
                    "heldout_truncation": ev.get("truncation_rate")})
    lc, lc_rows = length_conditioned(res_dir, list(FORKS))
    write_csv(res_dir / "summary_normalization.csv", out)
    write_csv(res_dir / "normalization_length_bins.csv", lc_rows)
    save_json(res_dir / "summary_normalization.json", {"forks": out, "length_conditioned": lc, "length_bins": lc_rows})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Fork updates:", cfg["fork_updates"])
    res_dir = repo_path(cfg["results_dir"])
    if not args.summary_only:
        for name, lt in FORKS.items():
            out = f"outputs/task3_grpo/{name}"
            if not adapter_exists(out):
                run("task3_grpo.continue_train", "--config", args.config, "--run-name", name, "--output", out,
                    "--updates", cfg["fork_updates"], "--loss-type", lt, "--per-seq-grad")
            if not (res_dir / name / "eval_heldout.json").exists():
                run("task3_grpo.evaluate", "--config", args.config, "--adapter", out, "--name", name)
    summarize(cfg)


if __name__ == "__main__":
    main()
