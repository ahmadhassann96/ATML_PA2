"""Task 5 check: are the pairwise judge's TIEs real ties or parse fall-backs?

The released `PairwiseAIJudge.compare` decodes only 4 new tokens and maps anything without a standalone A / B / TIE
token to "TIE". This script re-runs the exact same prompt, physical A/B orientation (same hash rule) and greedy
decoding for a sample of comparisons already in the cache, and records the raw text with the released 4-token cap and
with a longer 48-token cap, so the report can separate
  * explicit ties   - the judge actually answered TIE,
  * parse fall-backs - the 4-token output contained no A / B / TIE token (e.g. it started explaining),
and check that re-running reproduces every cached decision (greedy decoding is deterministic).
Nothing in the released judge or its cache is modified.

    python -m task5_feedback.inspect_judge --config configs/feedback.yaml
Outputs: results/task5_feedback/judge_raw_outputs.jsonl, judge_parse_audit.json
"""
from __future__ import annotations

import argparse
import re
from collections import Counter

import torch

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from task5_feedback.rlaif import PAIRWISE_RUBRIC, PairwiseAIJudge

LABEL_RE = re.compile(r"\b(A|B|TIE)\b")


@torch.no_grad()
def raw_decode(judge, problem, a, b, max_new_tokens):
    key = judge._key(problem, a, b)
    swap = int(key[:8], 16) % 2 == 1
    aa, bb = (b, a) if swap else (a, b)
    ids = judge.tokenizer.apply_chat_template([{"role": "user", "content": PAIRWISE_RUBRIC.format(problem=problem, a=aa, b=bb)}],
                                              return_tensors="pt", add_generation_prompt=True)
    ids = ids.to(next(judge.model.parameters()).device)
    out = judge.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                               pad_token_id=judge.tokenizer.eos_token_id, eos_token_id=judge.tokenizer.eos_token_id)
    text = judge.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return text, swap, key


def mapped(text, swap):
    m = LABEL_RE.search(text.strip().upper())
    r = m.group(1) if m else "TIE"
    return ({"A": "B", "B": "A", "TIE": "TIE"}[r] if swap else r), bool(m)


def sample_items(res, n_policy):
    items = []
    for r in read_jsonl(res / "diagnostics" / "pair_records.jsonl"):
        items.append({"source": f"diagnostic:{r['category']}", "id": r["problem_id"], "problem": r["question"],
                      "a": r["better_response"], "b": r["worse_response"]})
    for ds in ("gsm", "transfer"):
        d = res / ds
        if not (d / "generations_sft.jsonl").exists():
            continue
        sft = {g["id"]: g for g in read_jsonl(d / "generations_sft.jsonl")}
        for pol in ("rlvr", "rlaif"):
            f = d / f"generations_{pol}.jsonl"
            if not f.exists():
                continue
            k = 0
            for g in read_jsonl(f):
                s = sft[g["id"]]
                if g["response"] != s["response"] and k < n_policy:
                    items.append({"source": f"{ds}:{pol}_vs_sft", "id": g["id"], "problem": g["question"],
                                  "a": g["response"], "b": s["response"]})
                    k += 1
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--n-policy", type=int, default=15, help="non-identical policy-vs-SFT pairs per dataset and policy")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = repo_path(cfg["results_dir"]) / "task5_feedback"
    judge = PairwiseAIJudge(cfg, res / "pairwise_judge_cache.json")
    items = sample_items(res, args.n_policy)
    print(f"[inspect_judge] {len(items)} comparisons", flush=True)
    out = []
    for i, it in enumerate(items):
        if i % 20 == 0:
            print(f"[inspect_judge] {i + 1}/{len(items)}", flush=True)
        raw4, swap, key = raw_decode(judge, it["problem"], it["a"], it["b"], 4)
        raw48, _, _ = raw_decode(judge, it["problem"], it["a"], it["b"], 48)
        d4, clean4 = mapped(raw4, swap)
        d48, clean48 = mapped(raw48, swap)
        out.append({**{k: it[k] for k in ("source", "id")}, "physical_swap": swap, "raw_4tok": raw4, "raw_48tok": raw48,
                    "decision_4tok": d4, "label_token_found_4tok": clean4, "decision_48tok": d48,
                    "label_token_found_48tok": clean48, "cached": judge.cache.get(key),
                    "reproduces_cache": judge.cache.get(key) in (None, d4)})
    write_jsonl(res / "judge_raw_outputs.jsonl", out)

    def summarise(rows):
        ties = [r for r in rows if r["decision_4tok"] == "TIE"]
        return {"n": len(rows), "decisions": dict(Counter(r["decision_4tok"] for r in rows)),
                "tie_rate": len(ties) / len(rows) if rows else None,
                "ties_explicit": sum(r["label_token_found_4tok"] for r in ties),
                "ties_parse_fallback": sum(not r["label_token_found_4tok"] for r in ties),
                "fallbacks_resolved_with_48_tokens": dict(Counter(r["decision_48tok"] for r in ties
                                                                  if not r["label_token_found_4tok"])),
                "raw_4tok_first_words": dict(Counter(r["raw_4tok"].strip().split(" ")[0][:20] for r in rows).most_common(8))}

    summary = {"overall": summarise(out),
               "by_source": {s: summarise([r for r in out if r["source"] == s]) for s in sorted({r["source"] for r in out})},
               "reproduces_cache_rate": sum(r["reproduces_cache"] for r in out) / len(out)}
    save_json(res / "judge_parse_audit.json", summary)
    o = summary["overall"]
    print(f"[inspect_judge] decisions {o['decisions']} | ties: {o['ties_explicit']} explicit, "
          f"{o['ties_parse_fallback']} parse fall-backs | 48-token re-decode of fall-backs: "
          f"{o['fallbacks_resolved_with_48_tokens']} | reproduces cache: {summary['reproduces_cache_rate']:.3f}", flush=True)
    print(f"[inspect_judge] most common raw 4-token outputs: {o['raw_4tok_first_words']}", flush=True)


if __name__ == "__main__":
    main()
