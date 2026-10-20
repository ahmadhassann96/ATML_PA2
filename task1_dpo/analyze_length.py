"""Task 1 Step 3: length-confounding study.

1. Dataset side: how strongly is "preferred" correlated with "longer" in each training file?
2. Trains the length-balanced DPO model (same config as the standard run, different data file).
3. Evaluates the standard and length-balanced models on the length-stratified held-out set
   (preferred-longer / length-matched / rejected-longer), plus generated length and word-limit
   compliance on the common prompt set.

    python -m task1_dpo.analyze_length --config configs/dpo.yaml
Requires the standard adapter (python -m task1_dpo.train --run-name standard) to exist.
"""
from __future__ import annotations

import argparse

import numpy as np

from common.data import load_yaml, preference_responses, read_jsonl, repo_path
from common.logging_utils import save_json
from common.models import load_tokenizer
from common.runs import adapter_exists, load_json_if, run, write_csv

STRAT_TAG = "dpo_length_stratified_eval"


def dataset_length_profile(path, tokenizer):
    rows = read_jsonl(path)
    diffs, chosen_longer = [], []
    for r in rows:
        if "chosen_tokens" in r and "rejected_tokens" in r:
            c, j = int(r["chosen_tokens"]), int(r["rejected_tokens"])
        else:
            yc, yr = preference_responses(r)
            c = len(tokenizer(yc, add_special_tokens=False)["input_ids"])
            j = len(tokenizer(yr, add_special_tokens=False)["input_ids"])
        diffs.append(c - j)
        chosen_longer.append(c > j)
    d = np.asarray(diffs, dtype=float)
    out = {
        "file": str(path), "n_pairs": len(rows),
        "frac_chosen_longer": float(np.mean(chosen_longer)),
        "mean_len_diff_chosen_minus_rejected": float(d.mean()),
        "median_len_diff": float(np.median(d)),
    }
    if rows and "length_stratum" in rows[0]:
        out["strata_counts"] = {s: sum(r["length_stratum"] == s for r in rows) for s in sorted({r["length_stratum"] for r in rows})}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = cfg["results_dir"]
    std_out, len_out = cfg["standard_output"], cfg["length_output"]

    if not args.summary_only:
        if not adapter_exists(std_out):
            raise SystemExit(f"standard adapter missing at {std_out}; run task1_dpo.train --run-name standard first")
        if not adapter_exists(len_out):
            run("task1_dpo.train", "--config", args.config, "--run-name", "length_balanced",
                "--dataset", cfg["paths"]["dpo_length_train"], "--output", len_out)
        # common prompt set: generated length + word-limit compliance (and standard held-out pairs)
        if not repo_path(f"{res}/length_balanced/eval_dpo_standard_eval.json").exists():
            run("task1_dpo.evaluate", "--config", args.config, "--adapter", len_out, "--name", "length_balanced")
        # length-stratified held-out pairs for both models
        for name, adapter in (("standard", std_out), ("length_balanced", len_out)):
            if not repo_path(f"{res}/{name}/eval_{STRAT_TAG}.json").exists():
                run("task1_dpo.evaluate", "--config", args.config, "--adapter", adapter, "--name", name,
                    "--eval-file", "dpo_length_eval", "--no-generate")

    tok = load_tokenizer(cfg["base_model"])
    data_profile = {
        "standard_train": dataset_length_profile(cfg["paths"]["dpo_standard_train"], tok),
        "length_balanced_train": dataset_length_profile(cfg["paths"]["dpo_length_train"], tok),
        "standard_eval": dataset_length_profile(cfg["paths"]["dpo_standard_eval"], tok),
        "length_stratified_eval": dataset_length_profile(cfg["paths"]["dpo_length_eval"], tok),
    }

    rows = []
    for name in ("standard", "length_balanced"):
        strat = load_json_if(f"{res}/{name}/eval_{STRAT_TAG}.json")
        common = load_json_if(f"{res}/{name}/eval_dpo_standard_eval.json")
        row = {"condition": name}
        if strat:
            row["strat_overall_acc"] = strat["pairs"]["overall"]["preference_accuracy"]
            for s, m in strat["pairs"].get("by_stratum", {}).items():
                row[f"acc_{s}"] = m["preference_accuracy"]
                row[f"acc_{s}_ci_lo"], row[f"acc_{s}_ci_hi"] = m["preference_accuracy_ci95"]
                row[f"margin_{s}"] = m["mean_margin"]
                row[f"n_{s}"] = m["n_pairs"]
        if common:
            g, wl = common.get("generation_metrics", {}), common.get("word_limit", {})
            row["gen_len_mean"] = g.get("response_tokens", {}).get("mean")
            row["gen_len_std"] = g.get("response_tokens", {}).get("std")
            row["gen_len_iqr"] = g.get("response_tokens", {}).get("iqr")
            row["reward_mean"] = g.get("reward_mean")
            row["kl_token_mean"] = g.get("kl_token_mean")
            row["word_limit_compliance"] = wl.get("compliance_rate")
            row["word_limit_words_mean"] = wl.get("response_words", {}).get("mean")
            row["word_limit_words_over_mean"] = wl.get("words_over_limit_mean")
        rows.append(row)
    sft = load_json_if(f"{res}/sft/eval_dpo_standard_eval.json")
    if sft:
        g, wl = sft.get("generation_metrics", {}), sft.get("word_limit", {})
        rows.append({"condition": "sft", "gen_len_mean": g.get("response_tokens", {}).get("mean"),
                     "gen_len_std": g.get("response_tokens", {}).get("std"), "reward_mean": g.get("reward_mean"),
                     "word_limit_compliance": wl.get("compliance_rate"),
                     "word_limit_words_mean": wl.get("response_words", {}).get("mean")})

    write_csv(f"{res}/summary_length.csv", rows)
    save_json(f"{res}/summary_length.json", {"dataset_length_profile": data_profile, "models": rows})
    for k, v in data_profile.items():
        print(f"{k:24s} chosen longer in {v['frac_chosen_longer']:.1%} of pairs (mean diff {v['mean_len_diff_chosen_minus_rejected']:+.1f} tokens)")


if __name__ == "__main__":
    main()
