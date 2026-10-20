"""Task 1 evaluation: held-out preference metrics, sampled KL, reward, length, word-limit compliance.

Examples (from the repo root):
    python -m task1_dpo.evaluate --adapter none --name sft
    python -m task1_dpo.evaluate --adapter outputs/task1_dpo/standard --name standard
    python -m task1_dpo.evaluate --adapter outputs/task1_dpo/standard --name standard \
        --eval-file dpo_length_eval --no-generate

Every condition uses the same prompts, prompt order, seed, decoding settings and generation cap.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common.data import load_yaml, prompt_messages, prompt_messages_from_preference, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import save_json, set_seed, wall_timer
from common.logprobs import response_logprobs
from common.metrics import parse_word_limit, word_count, word_limit_compliance
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from task1_dpo.train import filter_long_prompts, make_collate, pair_logps, to_device

PAIR_BATCH = 4        # preference pairs per forward pass
GEN_BATCH = 16        # prompts per generate() call
SCORE_BATCH = 4       # sequences per log-prob / reward forward pass
N_BOOT = 1000


# ----------------------------------------------------------------------------- small stats helpers
def length_stats(lengths):
    a = np.asarray(lengths, dtype=float)
    if a.size == 0:
        return {}
    q25, q50, q75 = np.percentile(a, [25, 50, 75])
    return {"mean": float(a.mean()), "std": float(a.std()), "median": float(q50),
            "q25": float(q25), "q75": float(q75), "iqr": float(q75 - q25), "max": float(a.max()), "n": int(a.size)}


def bootstrap_ci(values, seed=0):
    a = np.asarray(values, dtype=float)
    if a.size < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, size=(N_BOOT, a.size))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_path(cfg, key_or_path):
    return cfg["paths"].get(key_or_path, key_or_path)


# ----------------------------------------------------------------------------- 1. held-out pairs
@torch.no_grad()
def evaluate_pairs(model, tokenizer, rows, cfg, beta):
    rows, dropped = filter_long_prompts(rows, tokenizer, int(cfg["max_sequence_length"]))
    collate = make_collate(tokenizer, int(cfg["max_sequence_length"]))
    device = next(model.parameters()).device
    records = []
    n_batches = (len(rows) + PAIR_BATCH - 1) // PAIR_BATCH
    for i in range(0, len(rows), PAIR_BATCH):
        chunk = rows[i:i + PAIR_BATCH]
        if (i // PAIR_BATCH) % 10 == 0:
            print(f"[pairs] batch {i // PAIR_BATCH + 1}/{n_batches}", flush=True)
        chosen, rejected, _ = collate(chunk)
        chosen, rejected = to_device(chosen, device), to_device(rejected, device)
        pc, pr, lc, lr = pair_logps(model, chosen, rejected)
        with reference_mode(model):
            rc, rr, _, _ = pair_logps(model, chosen, rejected)
        m = (pc - rc) - (pr - rr)  # DPO implicit-reward margin (manual definition)
        for j, row in enumerate(chunk):
            records.append({
                "prompt_id": row.get("prompt_id"),
                "source_index": row.get("source_index"),
                "length_stratum": row.get("length_stratum"),
                "chosen_tokens": int(lc[j]),
                "rejected_tokens": int(lr[j]),
                "policy_logp_chosen": float(pc[j]), "policy_logp_rejected": float(pr[j]),
                "ref_logp_chosen": float(rc[j]), "ref_logp_rejected": float(rr[j]),
                "margin": float(m[j]),
                "dpo_loss": float(-F.logsigmoid(beta * m[j].float())),
                "correct": bool(m[j] > 0),
            })

    def summarize(recs):
        if not recs:
            return {}
        correct = [r["correct"] for r in recs]
        return {
            "n_pairs": len(recs),
            "dpo_loss": float(np.mean([r["dpo_loss"] for r in recs])),
            "preference_accuracy": float(np.mean(correct)),
            "preference_accuracy_ci95": bootstrap_ci(correct),
            "mean_margin": float(np.mean([r["margin"] for r in recs])),
            # likelihood-preference: does the model simply assign the chosen response higher total log-prob?
            "policy_logp_prefers_chosen": float(np.mean([r["policy_logp_chosen"] > r["policy_logp_rejected"] for r in recs])),
            "ref_logp_prefers_chosen": float(np.mean([r["ref_logp_chosen"] > r["ref_logp_rejected"] for r in recs])),
            "chosen_longer_frac": float(np.mean([r["chosen_tokens"] > r["rejected_tokens"] for r in recs])),
        }

    summary = {"beta_for_loss": beta, "n_dropped_long_prompts": len(dropped), "overall": summarize(records)}
    strata = sorted({r["length_stratum"] for r in records if r["length_stratum"]})
    if strata:
        summary["by_stratum"] = {s: summarize([r for r in records if r["length_stratum"] == s]) for s in strata}
    return summary, records


# ----------------------------------------------------------------------------- 2. generation-based metrics
@torch.no_grad()
def generate_and_score(model, tokenizer, prompts, cfg, max_new_tokens, seed, label="gen"):
    """prompts: list of (prompt_id, messages). Returns per-response records (no reward yet)."""
    device = next(model.parameters()).device
    set_seed(seed)
    gen = cfg["generation"]
    records = []
    n_batches = (len(prompts) + GEN_BATCH - 1) // GEN_BATCH
    timer = wall_timer()
    for i in range(0, len(prompts), GEN_BATCH):
        chunk = prompts[i:i + GEN_BATCH]
        b = i // GEN_BATCH + 1
        print(f"[{label}] generation batch {b}/{n_batches} ({timer():.0f}s elapsed)", flush=True)
        out = batch_generate(
            model, tokenizer, [m for _, m in chunk],
            max_prompt_length=4096,  # prompts were length-filtered already; never truncate the chat template
            max_new_tokens=max_new_tokens,
            temperature=float(gen["temperature"]), top_p=float(gen["top_p"]), do_sample=bool(gen["do_sample"]),
        )
        seq, attn, width = out["sequences"], out["attention_mask"], out["prompt_width"]
        rmask_full = torch.zeros_like(seq, dtype=torch.float32)
        rmask_full[:, width:] = out["response_mask"]
        attn = attn.clone()
        attn[:, width:] = out["response_mask"].long()  # do not attend to post-EOS padding
        for j in range(0, seq.shape[0], SCORE_BATCH):
            sl = slice(j, j + SCORE_BATCH)
            pol = response_logprobs(model, seq[sl], attn[sl], rmask_full[sl], return_entropy=True)
            with reference_mode(model):
                ref = response_logprobs(model, seq[sl], attn[sl], rmask_full[sl])
            diff = (pol["token_logp"] - ref["token_logp"]) * pol["mask"]
            for k in range(seq[sl].shape[0]):
                idx = j + k
                n = int(pol["mask"][k].sum())
                records.append({
                    "prompt_id": chunk[idx][0],
                    "prompt": chunk[idx][1][-1]["content"],
                    "response": out["responses"][idx],
                    "response_tokens": out["response_lengths"][idx],
                    "truncated": out["truncated"][idx],
                    "terminated_with_eos": out["terminated_with_eos"][idx],
                    "kl_seq": float(diff[k].sum()),           # sum_t [log pi - log pi_ref] on sampled tokens
                    "kl_token_sum_n": n,                       # tokens contributing (for the token-level mean)
                    "logp_policy": float(pol["seq_logp"][k]),
                    "entropy_mean": float(pol["entropy"][k].sum() / max(n, 1)),
                    "_messages": chunk[idx][1],
                })
    return records


@torch.no_grad()
def add_rewards(records, cfg):
    print(f"[reward] scoring {len(records)} responses with the reward model", flush=True)
    rm, rtok = load_reward_model(cfg)
    for i in range(0, len(records), SCORE_BATCH):
        chunk = records[i:i + SCORE_BATCH]
        scores = score_reward_pairs(rm, rtok, [r["_messages"] for r in chunk], [r["response"] for r in chunk],
                                    max_length=int(cfg.get("reward_max_length", 1024)))
        for r, s in zip(chunk, scores.tolist()):
            r["reward"] = float(s)
    del rm
    free_gpu()


def summarize_generations(records):
    n_tok = sum(r["kl_token_sum_n"] for r in records)
    kl_tok = sum(r["kl_seq"] for r in records) / max(n_tok, 1)  # == common.metrics.sampled_kl over all tokens
    rewards = [r["reward"] for r in records if "reward" in r]
    return {
        "n_responses": len(records),
        "kl_token_mean": kl_tok,
        "kl_seq_mean": float(np.mean([r["kl_seq"] for r in records])),
        "reward_mean": float(np.mean(rewards)) if rewards else None,
        "reward_std": float(np.std(rewards)) if rewards else None,
        "reward_ci95": bootstrap_ci(rewards) if rewards else None,
        "entropy_mean": float(np.mean([r["entropy_mean"] for r in records])),
        "response_tokens": length_stats([r["response_tokens"] for r in records]),
        "response_words": length_stats([word_count(r["response"]) for r in records]),
        "truncation_rate": float(np.mean([r["truncated"] for r in records])),
    }


# ----------------------------------------------------------------------------- driver
def run_evaluation(config_path, adapter, name, eval_file=None, beta=None, generate=True,
                   gen_limit=None, word_limit_samples=5, tag=None):
    cfg = load_yaml(config_path)
    beta = float(cfg["beta"] if beta is None else beta)
    eval_path = resolve_path(cfg, eval_file or "dpo_standard_eval")
    tag = tag or Path(eval_path).stem
    out_dir = repo_path(cfg["results_dir"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter = None if adapter in (None, "", "none", "None") else adapter
    timer = wall_timer()

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    rows = read_jsonl(eval_path)
    print(f"[eval:{name}] adapter={adapter} eval_file={eval_path} ({len(rows)} pairs)")

    metrics = {"name": name, "adapter": adapter, "eval_file": str(eval_path), "seed": int(cfg["seed"]),
               "generation": {**cfg["generation"], "max_new_tokens": int(cfg["max_generation_tokens"])}}
    pair_summary, pair_records = evaluate_pairs(model, tokenizer, rows, cfg, beta)
    metrics["pairs"] = pair_summary
    write_jsonl(out_dir / f"pairs_{tag}.jsonl", pair_records)
    o = pair_summary["overall"]
    print(f"[eval:{name}] pairs: acc={o['preference_accuracy']:.3f} loss={o['dpo_loss']:.4f} margin={o['mean_margin']:.3f}")

    if generate:
        max_new = int(cfg["max_generation_tokens"])
        prompt_budget = int(cfg["max_sequence_length"]) - max_new
        seen, prompts, skipped = set(), [], []
        for row in rows:
            pid = row.get("prompt_id")
            if pid in seen:
                continue
            seen.add(pid)
            msgs = prompt_messages_from_preference(row)
            n = len(tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
            (prompts if n <= prompt_budget else skipped).append((pid, msgs))
        if gen_limit:
            prompts = prompts[: int(gen_limit)]
        # sort by prompt length (stable, data-determined) to reduce padding; identical for all conditions
        prompts.sort(key=lambda p: len(tokenizer.apply_chat_template(p[1], tokenize=False, add_generation_prompt=True)))
        print(f"[eval:{name}] generating for {len(prompts)} prompts ({len(skipped)} skipped as > {prompt_budget} tokens)")
        gens = generate_and_score(model, tokenizer, prompts, cfg, max_new, seed=int(cfg["seed"]), label=f"eval:{name}")

        wl_rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
        wl_prompts = [(r["prompt_id"], prompt_messages(r)) for r in wl_rows for _ in range(int(word_limit_samples))]
        wl = generate_and_score(model, tokenizer, wl_prompts, cfg, max_new, seed=int(cfg["seed"]) + 1, label=f"eval:{name}:word-limit")

        del model
        free_gpu()  # make room for the 8-bit reward model
        add_rewards(gens, cfg)
        for r in wl:
            r["word_limit"] = parse_word_limit(r["prompt"])
            r["words"] = word_count(r["response"])
            r["compliant"] = word_limit_compliance(r["prompt"], r["response"])

        metrics["generation_metrics"] = summarize_generations(gens)
        metrics["generation_metrics"]["n_prompts_skipped_long"] = len(skipped)
        comp = [r["compliant"] for r in wl if r["compliant"] is not None]
        metrics["word_limit"] = {
            "n_prompts": len(wl_rows), "samples_per_prompt": int(word_limit_samples),
            "compliance_rate": float(np.mean(comp)) if comp else None,
            "compliance_ci95": bootstrap_ci(comp) if comp else None,
            "words_over_limit_mean": float(np.mean([max(0, r["words"] - r["word_limit"]) for r in wl if r["word_limit"]])),
            "response_words": length_stats([r["words"] for r in wl]),
            "response_tokens": length_stats([r["response_tokens"] for r in wl]),
        }
        for r in gens + wl:
            r.pop("_messages", None)
        write_jsonl(out_dir / "generations.jsonl", gens)
        write_jsonl(out_dir / "word_limit.jsonl", wl)
        g = metrics["generation_metrics"]
        print(f"[eval:{name}] gen: reward={g['reward_mean']:.3f} kl_tok={g['kl_token_mean']:.4f} "
              f"len={g['response_tokens']['mean']:.1f}±{g['response_tokens']['std']:.1f} "
              f"word-limit={metrics['word_limit']['compliance_rate']}")

    metrics["wall_clock_s"] = round(timer(), 1)
    save_json(out_dir / f"eval_{tag}.json", metrics)
    print(f"[eval:{name}] wrote {out_dir / f'eval_{tag}.json'} ({metrics['wall_clock_s'] / 60:.1f} min)")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter directory, or 'none' for the SFT baseline")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--eval-file", help="path or configs key (default: dpo_standard_eval)")
    ap.add_argument("--beta", type=float, help="beta used to report DPO loss (default: config beta)")
    ap.add_argument("--no-generate", action="store_true", help="only compute held-out pair metrics")
    ap.add_argument("--gen-limit", type=int, help="debug: cap the number of generation prompts")
    ap.add_argument("--word-limit-samples", type=int, default=5)
    args = ap.parse_args()
    run_evaluation(args.config, args.adapter, args.name, args.eval_file, args.beta,
                   generate=not args.no_generate, gen_limit=args.gen_limit,
                   word_limit_samples=args.word_limit_samples)


if __name__ == "__main__":
    main()
