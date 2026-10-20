"""Task 5: combine the in-domain (GSM8K), controlled-diagnostic and transfer (SVAMP) results into the final
RLVR-vs-RLAIF comparison. No model is loaded; run after evaluate_math (both datasets) and score_perturbations.

Writes to results/task5_feedback/:
  summary_policies.csv          per dataset x policy: exact accuracy (+CI), format compliance, length, truncation,
                                AI pairwise win rate vs SFT (+CI, W/L/T), verifier-judge agreement
  summary_transfer.csv          per policy: in-domain vs transfer accuracy / win rate / length and the drop
  summary_diagnostics.csv       per perturbation category x mechanism: better / tie / wrong rates (+CI)
  summary_variant_rewards.csv   per variant: mean RLVR reward and mean RLAIF group reward (different scales)
  feedback_source_comparison.json   coverage, noise, exploitability and cost evidence, plus S_reason / S_outcome
  examples_diagnostic.jsonl     qualitative candidates: reasoning-only corruption, persuasive filler, wrong final,
                                gold distractor, chosen where the verifier and the judge disagree (or both fail)
  examples_policy.jsonl         policy-vs-SFT pairs where the verifier is decisive but the judge disagrees

    python -m task5_feedback.compare_feedback --config configs/feedback.yaml
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json

POLICIES = ["sft", "rlvr", "rlaif"]
RL = ["rlvr", "rlaif"]
DATASETS = {"gsm": "GSM8K (in-domain)", "transfer": "SVAMP (transfer)"}
MECHANISMS = ["rlvr", "rlaif_pairwise", "rlaif_group"]
GROUP_K = 4   # completions per prompt in the supplied group-based training (for the cost estimate only)


def _load(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def policy_table(evals):
    rows = []
    for ds, ev in evals.items():
        for pol in POLICIES:
            m = ev["policies"][pol]
            row = {"dataset": ds, "policy": pol, "n": m["n"], "exact_accuracy": m["exact_accuracy"],
                   "exact_accuracy_lo": m["exact_accuracy_ci95"][0], "exact_accuracy_hi": m["exact_accuracy_ci95"][1],
                   "format_compliance": m["format_compliance"], "response_tokens_mean": m["response_tokens_mean"],
                   "response_tokens_median": m["response_tokens_median"], "truncation_rate": m["truncation_rate"]}
            pw = ev["pairwise_vs_sft"].get(pol)
            if pw:
                row.update({"win_rate_vs_sft": pw["win_rate_vs_sft"], "win_rate_lo": pw["win_rate_ci95"][0],
                            "win_rate_hi": pw["win_rate_ci95"][1], "wins": pw["wins"], "losses": pw["losses"],
                            "judge_ties": pw["ties_judge"], "identical_to_sft": pw["ties_identical_responses"],
                            "verifier_decisive_pairs": pw["verifier_decisive_pairs"],
                            "verifier_judge_agreement": pw["verifier_judge_agreement_on_decisive"],
                            "judge_prefers_verified_wrong": pw["judge_prefers_verified_wrong"],
                            "judge_tie_rate_when_verifier_ties": pw["judge_tie_rate_when_verifier_ties"]})
            rows.append(row)
    return pd.DataFrame(rows)


def transfer_table(pt):
    if not {"gsm", "transfer"} <= set(pt.dataset):
        return pd.DataFrame()
    g, t = pt[pt.dataset == "gsm"].set_index("policy"), pt[pt.dataset == "transfer"].set_index("policy")
    rows = []
    for pol in POLICIES:
        r = {"policy": pol,
             "accuracy_in_domain": g.at[pol, "exact_accuracy"], "accuracy_transfer": t.at[pol, "exact_accuracy"],
             "accuracy_drop": g.at[pol, "exact_accuracy"] - t.at[pol, "exact_accuracy"],
             "format_in_domain": g.at[pol, "format_compliance"], "format_transfer": t.at[pol, "format_compliance"],
             "tokens_in_domain": g.at[pol, "response_tokens_mean"], "tokens_transfer": t.at[pol, "response_tokens_mean"],
             "tokens_change": t.at[pol, "response_tokens_mean"] - g.at[pol, "response_tokens_mean"]}
        if pol != "sft":
            r.update({"accuracy_gain_over_sft_in_domain": g.at[pol, "exact_accuracy"] - g.at["sft", "exact_accuracy"],
                      "accuracy_gain_over_sft_transfer": t.at[pol, "exact_accuracy"] - t.at["sft", "exact_accuracy"],
                      "win_rate_in_domain": g.at[pol, "win_rate_vs_sft"], "win_rate_transfer": t.at[pol, "win_rate_vs_sft"],
                      "win_rate_drop": g.at[pol, "win_rate_vs_sft"] - t.at[pol, "win_rate_vs_sft"]})
        rows.append(r)
    return pd.DataFrame(rows)


def diagnostics_table(diag):
    rows = []
    for cat, by_m in diag["by_category"].items():
        for m in MECHANISMS:
            r = by_m[m]
            rows.append({"category": cat, "better": diag["pairs"][cat]["better"], "worse": diag["pairs"][cat]["worse"],
                         "mechanism": m, "n": r["n"], "better_rate": r["better_rate"], "tie_rate": r["tie_rate"],
                         "wrong_rate": r["wrong_rate"], "better_lo": r["better_rate_ci95"][0],
                         "better_hi": r["better_rate_ci95"][1], "wrong_lo": r["wrong_rate_ci95"][0],
                         "wrong_hi": r["wrong_rate_ci95"][1]})
    return pd.DataFrame(rows)


def decisive_share(recs, policy_eval=False):
    """Coverage: share of comparisons on which a mechanism gives a non-tie signal."""
    if policy_eval:
        nonid = [p for p in recs if not p["identical"]]
        return {"n_nonidentical_pairs": len(nonid),
                "verifier_decisive": float(np.mean([p["verifier_diff"] != 0 for p in nonid])) if nonid else None,
                "judge_decisive": float(np.mean([p["judge"] != "TIE" for p in nonid])) if nonid else None}
    return {m: float(np.mean([r[m] != "tie" for r in recs])) for m in MECHANISMS}


def pick_diagnostic_examples(records, k=3):
    """A few per category, prioritising the cases that explain verifier-judge disagreement."""
    want = {
        "reasoning": [("rlaif_pairwise", "better"), ("rlaif_pairwise", "wrong"), ("rlaif_pairwise", "tie")],
        "filler": [("rlaif_pairwise", "wrong"), ("rlaif_pairwise", "tie"), ("rlaif_pairwise", "better")],
        "outcome": [("rlaif_pairwise", "wrong"), ("rlaif_pairwise", "tie"), ("rlaif_pairwise", "better")],
        "distractor": [("rlaif_pairwise", "wrong"), ("rlaif_pairwise", "tie"), ("rlaif_pairwise", "better")],
    }
    out = []
    for cat, prefs in want.items():
        sub = [r for r in records if r["category"] == cat]
        chosen = []
        for m, o in prefs:
            for r in sub:
                if r[m] == o and r not in chosen:
                    chosen.append(r)
                    break
            if len(chosen) >= k:
                break
        for r in chosen:
            out.append({"category": cat, "problem_id": r["problem_id"], "question": r["question"],
                        "gold_final": r["gold_final"], "verifier": r["rlvr"], "judge_pairwise": r["rlaif_pairwise"],
                        "judge_group": r["rlaif_group"], "better_variant": r["better_variant"],
                        "worse_variant": r["worse_variant"], "better_response": r["better_response"],
                        "worse_response": r["worse_response"]})
    return out


def pick_policy_examples(res, k=4):
    out = []
    for ds in DATASETS:
        d = res / ds
        gens = {p: {r["id"]: r for r in read_jsonl(d / f"generations_{p}.jsonl")} for p in POLICIES
                if (d / f"generations_{p}.jsonl").exists()}
        for pol in RL:
            f = d / f"pairwise_{pol}_vs_sft.jsonl"
            if not f.exists() or pol not in gens or "sft" not in gens:
                continue
            pairs = read_jsonl(f)
            bad = [p for p in pairs if (p["verifier_diff"] > 0 and p["judge"] != "A") or (p["verifier_diff"] < 0 and p["judge"] != "B")]
            for p in bad[:k]:
                g, s = gens[pol][p["id"]], gens["sft"][p["id"]]
                out.append({"dataset": ds, "policy": pol, "id": p["id"], "question": g["question"],
                            "gold_final": g["gold_final"], "policy_correct": g["correct"], "sft_correct": s["correct"],
                            "judge": p["judge"], "policy_response": g["response"], "sft_response": s["response"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = repo_path(cfg["results_dir"]) / "task5_feedback"

    evals = {ds: e for ds in DATASETS if (e := _load(res / ds / "eval_math.json"))}
    diag = _load(res / "diagnostics" / "diagnostic_scores.json")
    missing = [n for n, ok in [("evaluate_math --dataset gsm", "gsm" in evals),
                               ("evaluate_math --dataset transfer", "transfer" in evals),
                               ("score_perturbations", diag is not None)] if not ok]
    if missing:
        print("[task5:compare] missing results (run these first):", ", ".join(missing), flush=True)
    if not evals and diag is None:
        raise SystemExit(1)

    summary = {"missing_inputs": missing}
    if evals:
        pt = policy_table(evals)
        pt.to_csv(res / "summary_policies.csv", index=False)
        print(pt[["dataset", "policy", "exact_accuracy", "format_compliance", "response_tokens_mean"]
                 + [c for c in ("win_rate_vs_sft", "verifier_judge_agreement") if c in pt]].round(3).to_string(index=False))
        tt = transfer_table(pt)
        if not tt.empty:
            tt.to_csv(res / "summary_transfer.csv", index=False)
            summary["transfer"] = tt.round(4).to_dict(orient="records")
        summary["policies"] = pt.round(4).to_dict(orient="records")
        if "transfer" in evals and "accuracy_by_svamp_type" in evals["transfer"]:
            summary["transfer_accuracy_by_svamp_type"] = evals["transfer"]["accuracy_by_svamp_type"]

    coverage, noise, exploit, cost = {}, {}, {}, {}
    if diag is not None:
        dt = diagnostics_table(diag)
        dt.to_csv(res / "summary_diagnostics.csv", index=False)
        vr = pd.read_csv(res / "diagnostics" / "variant_rewards.csv", index_col=0)
        vr.to_csv(res / "summary_variant_rewards.csv")
        records = read_jsonl(res / "diagnostics" / "pair_records.jsonl")
        print(dt[["category", "mechanism", "better_rate", "tie_rate", "wrong_rate"]].round(3).to_string(index=False))
        summary["S_reason"], summary["S_outcome"] = diag["S_reason"], diag["S_outcome"]
        summary["S_outcome_all_outcome_changing_pairs"] = {m: v["better_rate"] for m, v in
                                                           diag["S_outcome_all_outcome_changing_pairs"].items()}
        coverage["diagnostic_pairs_decisive_share"] = decisive_share(records)
        coverage["diagnostic_decisive_share_by_category"] = {
            c: decisive_share([r for r in records if r["category"] == c]) for c in diag["by_category"]}
        noise["judge_order_check"] = diag.get("judge_order_check")
        noise["judge_position"] = diag.get("judge_position")
        noise["verifier_expected_mismatches"] = len(diag.get("verifier_expected_mismatches", []))
        exploit["filler_wrong_rate"] = {m: diag["by_category"]["filler"][m]["wrong_rate"] for m in MECHANISMS}
        exploit["filler_better_rate"] = {m: diag["by_category"]["filler"][m]["better_rate"] for m in MECHANISMS}
        exploit["distractor_wrong_or_tie_rate"] = {m: diag["by_category"]["distractor"][m]["wrong_rate"]
                                                   + diag["by_category"]["distractor"][m]["tie_rate"] for m in MECHANISMS}
        exploit["reasoning_corruption_undetected_rate"] = {m: 1 - diag["by_category"]["reasoning"][m]["better_rate"]
                                                           for m in MECHANISMS}
        cost["diagnostics"] = diag["cost"]
        write_jsonl(res / "examples_diagnostic.jsonl", pick_diagnostic_examples(records))

    for ds, ev in evals.items():
        pw = {p: read_jsonl(res / ds / f"pairwise_{p}_vs_sft.jsonl") for p in RL if (res / ds / f"pairwise_{p}_vs_sft.jsonl").exists()}
        coverage[f"{ds}_policy_vs_sft"] = {p: decisive_share(v, policy_eval=True) for p, v in pw.items()}
        noise[f"{ds}_verifier_judge_agreement_on_decisive"] = {p: m["verifier_judge_agreement_on_decisive"]
                                                              for p, m in ev["pairwise_vs_sft"].items()}
        noise[f"{ds}_judge_tie_rate_when_verifier_ties"] = {p: m["judge_tie_rate_when_verifier_ties"]
                                                           for p, m in ev["pairwise_vs_sft"].items()}
        exploit[f"{ds}_judge_prefers_verified_wrong"] = {p: m["judge_prefers_verified_wrong"]
                                                         for p, m in ev["pairwise_vs_sft"].items()}
        cost[f"{ds}_judge"] = ev["judge_cost"]
    # cost per training group: RLVR needs K verifier calls, direct RLAIF needs K(K-1)/2 judge calls
    sec = [c.get("seconds_per_uncached_call") for k, v in cost.items() if k.endswith("_judge") for c in v.values()]
    sec += [cost.get("diagnostics", {}).get("judge_seconds_per_uncached_call")]
    sec = [s for s in sec if s]
    if sec:
        jsec = float(np.median(sec))
        vsec = cost.get("diagnostics", {}).get("verifier_seconds_per_call")
        cost["per_training_group_estimate"] = {
            "K": GROUP_K, "verifier_calls": GROUP_K, "judge_calls": GROUP_K * (GROUP_K - 1) // 2,
            "judge_seconds_per_call_median": round(jsec, 3), "judge_seconds_per_group": round(jsec * GROUP_K * (GROUP_K - 1) / 2, 2),
            "verifier_seconds_per_group": vsec * GROUP_K if vsec else None,
            "note": "measured in this run with greedy 4-bit judge decoding (one forward pass per call); "
                    "the judge also needs its own weights in memory (Qwen2.5-3B in 4-bit)"}
    summary.update({"coverage": coverage, "noise": noise, "exploitability": exploit, "cost": cost})
    save_json(res / "feedback_source_comparison.json", summary)
    write_jsonl(res / "examples_policy.jsonl", pick_policy_examples(res))
    print("[task5:compare] wrote", res / "feedback_source_comparison.json", flush=True)


if __name__ == "__main__":
    main()
