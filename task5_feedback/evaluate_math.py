"""Task 5 Steps 1 and 3: SFT vs RLVR vs RLAIF on GSM8K (in-domain) and the fixed SVAMP subset (transfer).

For one dataset:
  1. Generation - each policy answers every problem in file order with identical greedy decoding, the common cap
     `math_max_new_tokens` and identical batches (results/task5_feedback/<dataset>/generations_<policy>.jsonl).
  2. Exact verification - released `exact_reward` (last designated `#### <number>` vs gold): exact accuracy, format
     compliance (a designated final was parsed), response length, truncation, failure type.
  3. AI pairwise evaluation - the released `PairwiseAIJudge.compare` (fixed rubric, deterministic A/B orientation
     balancing, persistent cache) compares each RL policy's response with SFT's on the same problem; win = 1,
     tie = 0.5, loss = 0. Byte-identical responses are scored as a tie without calling the judge (a preference between
     identical texts is undefined) and counted separately.
  4. Verifier-judge agreement on those pairs: where exactly one response is verified correct, does the judge prefer
     it? Where the verifier ties (both right or both wrong), how often does the judge also tie?

    python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
    python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
"""
from __future__ import annotations

import argparse
import ast
import json
import gc
import time

import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json, wall_timer
from common.models import load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final

GEN_BATCH = 32
RL_POLICIES = ["rlvr", "rlaif"]
N_BOOT = 1000


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def row_messages(row):
    """The course math files store `messages` as a Python-literal string; parse it (never re-wrap `prompt`, which is
    already chat-rendered)."""
    m = row.get("messages")
    if isinstance(m, str):
        m = ast.literal_eval(m)
    if not isinstance(m, list) or not m:
        raise ValueError(f"cannot parse messages for row {row.get('source_index')}")
    return m


def row_id(row):
    return str(row.get("prompt_id") or row.get("source_index"))


def boot_ci(x, seed=0):
    a = np.asarray(x, dtype=float)
    if a.size < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    m = a[rng.integers(0, a.size, size=(N_BOOT, a.size))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def failure_type(rec):
    if rec["correct"]:
        return "correct"
    if rec["pred"] is None:
        return "no_final_truncated" if rec["truncated"] else "no_designated_final"
    return "wrong_final_truncated" if rec["truncated"] else "wrong_final"


def generate_policy(cfg, rows, name, tok, label):
    model = load_frozen_policy(cfg, name)
    out, timer = [], wall_timer()
    n_b = (len(rows) + GEN_BATCH - 1) // GEN_BATCH
    for b in range(n_b):
        chunk = rows[b * GEN_BATCH:(b + 1) * GEN_BATCH]
        print(f"[{label}:{name}] generation batch {b + 1}/{n_b} ({timer():.0f}s elapsed)", flush=True)
        gen = batch_generate(model, tok, [row_messages(r) for r in chunk], max_prompt_length=4096,
                             max_new_tokens=int(cfg["math_max_new_tokens"]), temperature=0.0, top_p=1.0, do_sample=False)
        for r, resp, n, trunc in zip(chunk, gen["responses"], gen["response_lengths"], gen["truncated"]):
            pred = extract_designated_final(resp)
            rec = {"id": row_id(r), "policy": name, "question": r["question"], "gold_final": str(r["gold_final"]),
                   "response": resp, "response_tokens": int(n), "truncated": bool(trunc), "pred": pred,
                   "format_ok": pred is not None, "correct": bool(exact_reward(resp, str(r["gold_final"])))}
            for k in ("svamp_type", "svamp_id"):
                if k in r:
                    rec[k] = r[k]
            rec["failure_type"] = failure_type(rec)
            out.append(rec)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def policy_metrics(recs):
    c = [r["correct"] for r in recs]
    ft = {}
    for r in recs:
        ft[r["failure_type"]] = ft.get(r["failure_type"], 0) + 1
    L = np.array([r["response_tokens"] for r in recs], dtype=float)
    return {"n": len(recs), "exact_accuracy": float(np.mean(c)), "exact_accuracy_ci95": boot_ci(c),
            "format_compliance": float(np.mean([r["format_ok"] for r in recs])),
            "truncation_rate": float(np.mean([r["truncated"] for r in recs])),
            "response_tokens_mean": float(L.mean()), "response_tokens_std": float(L.std()),
            "response_tokens_median": float(np.median(L)), "failure_types": ft}


def pairwise_vs_sft(judge, gens, policy, label):
    sft = {r["id"]: r for r in gens["sft"]}
    out, t0, calls = [], time.perf_counter(), 0
    for i, r in enumerate(gens[policy]):
        s = sft[r["id"]]
        if r["response"] == s["response"]:
            pref, identical = "TIE", True
        else:
            if i % 25 == 0:
                print(f"[{label}:judge] {policy} vs sft {i + 1}/{len(gens[policy])}", flush=True)
            if judge._key(r["question"], r["response"], s["response"]) not in judge.cache:
                calls += 1   # only uncached calls cost judge inference
            pref, identical = judge.compare(r["question"], r["response"], s["response"]), False
        score = {"A": 1.0, "B": 0.0, "TIE": 0.5}[pref]
        v = int(r["correct"]) - int(s["correct"])   # +1 policy verified better, -1 worse, 0 verifier tie
        out.append({"id": r["id"], "policy": policy, "judge": pref, "win_score": score, "identical": identical,
                    "policy_correct": r["correct"], "sft_correct": s["correct"], "verifier_diff": v})
    elapsed = time.perf_counter() - t0
    return out, calls, elapsed


def pairwise_metrics(pairs):
    s = [p["win_score"] for p in pairs]
    dec = [p for p in pairs if p["verifier_diff"] != 0]
    vt = [p for p in pairs if p["verifier_diff"] == 0 and not p["identical"]]
    agree = [(p["verifier_diff"] > 0 and p["judge"] == "A") or (p["verifier_diff"] < 0 and p["judge"] == "B") for p in dec]
    return {"n_pairs": len(pairs), "win_rate_vs_sft": float(np.mean(s)), "win_rate_ci95": boot_ci(s),
            "wins": sum(p["judge"] == "A" for p in pairs), "losses": sum(p["judge"] == "B" for p in pairs),
            "ties_judge": sum(p["judge"] == "TIE" and not p["identical"] for p in pairs),
            "ties_identical_responses": sum(p["identical"] for p in pairs),
            "verifier_decisive_pairs": len(dec),
            "verifier_judge_agreement_on_decisive": float(np.mean(agree)) if agree else None,
            "judge_prefers_verified_wrong": int(sum((p["verifier_diff"] > 0 and p["judge"] == "B") or
                                                    (p["verifier_diff"] < 0 and p["judge"] == "A") for p in dec)),
            "judge_ties_on_decisive": int(sum(p["judge"] == "TIE" for p in dec)),
            "verifier_tie_pairs_nonidentical": len(vt),
            "judge_tie_rate_when_verifier_ties": float(np.mean([p["judge"] == "TIE" for p in vt])) if vt else None}


def run(config_path, dataset):
    cfg, rows, tok = load_math_evaluation(config_path, dataset)
    out = repo_path(cfg["results_dir"]) / "task5_feedback" / dataset
    out.mkdir(parents=True, exist_ok=True)
    label = f"task5:{dataset}"
    print(f"[{label}] {len(rows)} problems | policies {list(policy_specs(cfg))} | greedy, cap {cfg['math_max_new_tokens']}", flush=True)
    gens = {}
    for name in policy_specs(cfg):
        p = out / f"generations_{name}.jsonl"
        if p.exists():
            gens[name] = read_jsonl(p)
            print(f"[{label}] {name}: generations exist - reusing", flush=True)
        else:
            gens[name] = generate_policy(cfg, rows, name, tok, label)
            write_jsonl(p, gens[name])
        m = policy_metrics(gens[name])
        print(f"[{label}] {name}: accuracy {m['exact_accuracy']:.3f} format {m['format_compliance']:.3f} "
              f"len {m['response_tokens_mean']:.0f} trunc {m['truncation_rate']:.3f}", flush=True)

    judge = PairwiseAIJudge(cfg, repo_path(cfg["results_dir"]) / "task5_feedback" / "pairwise_judge_cache.json")
    old = out / "eval_math.json"
    old_cost = (json.loads(old.read_text(encoding="utf-8")).get("judge_cost", {}) if old.exists() else {})
    pairs, cost = {}, {}
    for pol in RL_POLICIES:
        pairs[pol], calls, secs = pairwise_vs_sft(judge, gens, pol, label)
        cost[pol] = {"judge_calls_uncached": calls, "judge_seconds": round(secs, 1),
                     "seconds_per_uncached_call": round(secs / calls, 3) if calls else None}
        if not calls and old_cost.get(pol, {}).get("seconds_per_uncached_call"):
            cost[pol] = old_cost[pol]   # everything was cached: keep the timing measured on the first run
        write_jsonl(out / f"pairwise_{pol}_vs_sft.jsonl", pairs[pol])

    metrics = {"dataset": dataset, "n_problems": len(rows), "decoding": "greedy",
               "max_new_tokens": int(cfg["math_max_new_tokens"]),
               "policies": {n: policy_metrics(g) for n, g in gens.items()},
               "pairwise_vs_sft": {p: pairwise_metrics(v) for p, v in pairs.items()},
               "judge_cost": cost, "judge_model": cfg["ai_judge_model"]}
    if dataset == "transfer" and "svamp_type" in gens["sft"][0]:
        metrics["accuracy_by_svamp_type"] = {
            n: {t: float(np.mean([r["correct"] for r in g if r.get("svamp_type") == t]))
                for t in sorted({r.get("svamp_type") for r in g})} for n, g in gens.items()}
    save_json(out / "eval_math.json", metrics)
    for p, m in metrics["pairwise_vs_sft"].items():
        print(f"[{label}] {p} vs sft: win rate {m['win_rate_vs_sft']:.3f} (W{m['wins']}/L{m['losses']}/T{m['ties_judge']}"
              f" +{m['ties_identical_responses']} identical) | verifier-judge agreement {m['verifier_judge_agreement_on_decisive']}",
              flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    args = ap.parse_args()
    run(args.config, args.dataset)


if __name__ == "__main__":
    main()
