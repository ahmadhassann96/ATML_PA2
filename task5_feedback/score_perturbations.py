"""Task 5 Step 2: reward-sensitivity and robustness on the fixed Controlled Reward Diagnostic Set.

The set has 20 GSM8K problems x 5 variants (100 responses). Two feedback mechanisms score it:
  * RLVR  - the released exact verifier `exact_reward` (last designated `#### n` vs gold), r in {0, 1};
  * RLAIF - the released `PairwiseAIJudge` (fixed rubric, deterministic A/B orientation balancing, cache).

Controlled pairs (better response first; every pair is within one problem):
  reasoning   clean_correct  vs corrupt_reasoning_correct_final   same correct final, degraded reasoning -> S_reason
  outcome     clean_correct  vs good_reasoning_wrong_final        reasoning ~fixed, final changed        -> S_outcome
  filler      clean_correct  vs persuasive_filler_correct          same solution + irrelevant persuasive filler
  distractor  clean_correct  vs gold_distractor_wrong_final        mentions the gold number, ends on a wrong final
For each pair and mechanism: better (prefers the diagnostically better response), tie, wrong (prefers the worse one).
  RLVR:            compare r(better) with r(worse).
  RLAIF pairwise:  judge.compare(question, better, worse): A = better, TIE = tie, B = wrong.
  RLAIF group:     the training-time RLAIF reward - each response's normalised pairwise win rate among the 5 variants
                   of its problem (`group_rewards`, (wins + 0.5 ties)/(K-1)) - compared between the two responses.
A tie for the filler pair is the defensible outcome (the filler adds nothing); the "better" label there means the
mechanism penalised the filler. All 10 within-problem judge comparisons are run (the 4 controlled pairs are among them,
so they are cached), which gives per-variant mean rewards for all five categories and a full variant-vs-variant
preference matrix. S_outcome is also reported over every within-problem pair whose verified finals differ
(3 correct-final x 2 wrong-final variants).

Order check: each controlled pair is also judged with the arguments swapped. Because the released judge chooses the
physical A/B presentation from a hash of its arguments, only the pairs whose two calls were shown in opposite physical
orders test position consistency; consistency is reported on those, and the raw first-position pick rate is reported
over every diagnostic call.

    python -m task5_feedback.score_perturbations --config configs/feedback.yaml
Outputs in results/task5_feedback/diagnostics/: diagnostic_scores.json, pair_records.jsonl, variant_rewards.csv,
judge_matrix.csv.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}
# fixed order: clean first, so group_rewards' (i < j) comparisons for the controlled pairs are exactly
# compare(question, clean, perturbed) - the same cache keys as the pairwise analysis
VARIANT_ORDER = ["clean_correct", "corrupt_reasoning_correct_final", "good_reasoning_wrong_final",
                 "persuasive_filler_correct", "gold_distractor_wrong_final"]
SHORT = {"clean_correct": "clean", "corrupt_reasoning_correct_final": "corrupt_reasoning",
         "good_reasoning_wrong_final": "wrong_final", "persuasive_filler_correct": "filler",
         "gold_distractor_wrong_final": "gold_distractor"}
PAIRS = {  # category -> (diagnostically better, worse)
    "reasoning": ("clean_correct", "corrupt_reasoning_correct_final"),
    "outcome": ("clean_correct", "good_reasoning_wrong_final"),
    "filler": ("clean_correct", "persuasive_filler_correct"),
    "distractor": ("clean_correct", "gold_distractor_wrong_final"),
}
MECHANISMS = ["rlvr", "rlaif_pairwise", "rlaif_group"]
N_BOOT = 1000


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def boot_ci(x, seed=0):
    a = np.asarray(x, dtype=float)
    if a.size < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    m = a[rng.integers(0, a.size, size=(N_BOOT, a.size))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def outcome_from_scores(r_better, r_worse, tol=1e-9):
    if r_better > r_worse + tol:
        return "better"
    if r_worse > r_better + tol:
        return "wrong"
    return "tie"


def outcome_from_judge(pref):  # judge.compare(question, better, worse)
    return {"A": "better", "B": "wrong", "TIE": "tie"}[pref]


def rates(outcomes):
    n = len(outcomes)
    out = {"n": n}
    for k in ("better", "tie", "wrong"):
        ind = [o == k for o in outcomes]
        out[f"{k}_rate"] = float(np.mean(ind)) if n else float("nan")
        out[f"{k}_rate_ci95"] = boot_ci(ind)
    return out


def swap_of(judge, problem, a, b):
    """The physical orientation the released judge uses for compare(problem, a, b) (True = shown as (b, a))."""
    return int(judge._key(problem, a, b)[:8], 16) % 2 == 1


def judge_variants(judge, groups, order_check=True):
    """All 10 within-problem comparisons (+ the 4 controlled pairs with arguments swapped). Returns
    prefs[(pid, vi, vj)] = 'A'|'B'|'TIE' for compare(question, resp[vi], resp[vj]) and the list of calls made."""
    prefs, calls = {}, []
    pids = list(groups)
    t0 = time.perf_counter()
    for n, pid in enumerate(pids):
        g = groups[pid]
        q = g["clean_correct"]["question"]
        print(f"[task5:diagnostics] judging problem {n + 1}/{len(pids)} ({time.perf_counter() - t0:.0f}s)", flush=True)
        todo = list(itertools.combinations(VARIANT_ORDER, 2))
        if order_check:
            todo += [(w, b) for b, w in PAIRS.values()]
        for vi, vj in todo:
            a, b = g[vi]["response"], g[vj]["response"]
            cached = judge._key(q, a, b) in judge.cache
            pref = judge.compare(q, a, b)
            swap = swap_of(judge, q, a, b)
            raw = {"A": "B", "B": "A", "TIE": "TIE"}[pref] if swap else pref   # the physical position picked
            prefs[(pid, vi, vj)] = pref
            calls.append({"problem_id": pid, "first": vi, "second": vj, "pref": pref, "physical_swap": swap,
                          "raw_position_pick": raw, "cached": cached})
    return prefs, calls, time.perf_counter() - t0


def group_rewards_from_prefs(prefs, pid):
    """Exactly `PairwiseAIJudge.group_rewards` for the 5 variants in VARIANT_ORDER, computed from the cached prefs."""
    k = len(VARIANT_ORDER)
    wins, ties = np.zeros(k), np.zeros(k)
    for i, j in itertools.combinations(range(k), 2):
        p = prefs[(pid, VARIANT_ORDER[i], VARIANT_ORDER[j])]
        if p == "A":
            wins[i] += 1
        elif p == "B":
            wins[j] += 1
        else:
            ties[i] += 1
            ties[j] += 1
    return dict(zip(VARIANT_ORDER, ((wins + 0.5 * ties) / (k - 1)).tolist()))


def run(config_path, order_check=True):
    cfg = load_yaml(config_path)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    out = repo_path(cfg["results_dir"]) / "task5_feedback" / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[task5:diagnostics] {len(groups)} problems x {len(VARIANT_ORDER)} variants", flush=True)

    # ---- RLVR: exact verifier on every response; check against the set's expected_exact_reward
    t0 = time.perf_counter()
    rlvr = {pid: {v: exact_reward(g[v]["response"], str(g[v]["gold_final"])) for v in VARIANT_ORDER}
            for pid, g in groups.items()}
    verifier_seconds = time.perf_counter() - t0
    mismatches = [{"problem_id": pid, "variant": v, "verifier": rlvr[pid][v],
                   "expected": float(groups[pid][v]["expected_exact_reward"]),
                   "extracted": extract_designated_final(groups[pid][v]["response"])}
                  for pid in groups for v in VARIANT_ORDER
                  if "expected_exact_reward" in groups[pid][v]
                  and rlvr[pid][v] != float(groups[pid][v]["expected_exact_reward"])]
    print(f"[task5:diagnostics] verifier vs expected_exact_reward mismatches: {len(mismatches)}", flush=True)

    # ---- RLAIF: fixed pairwise judge
    judge = PairwiseAIJudge(cfg, repo_path(cfg["results_dir"]) / "task5_feedback" / "pairwise_judge_cache.json")
    prefs, calls, judge_seconds = judge_variants(judge, groups, order_check)
    group_r = {pid: group_rewards_from_prefs(prefs, pid) for pid in groups}

    # ---- controlled pairs
    records = []
    for pid, g in groups.items():
        for cat, (vb, vw) in PAIRS.items():
            rec = {"problem_id": pid, "category": cat, "better_variant": vb, "worse_variant": vw,
                   "question": g[vb]["question"], "gold_final": str(g[vb]["gold_final"]),
                   "better_response": g[vb]["response"], "worse_response": g[vw]["response"],
                   "rlvr_better": rlvr[pid][vb], "rlvr_worse": rlvr[pid][vw],
                   "judge_pref": prefs[(pid, vb, vw)],
                   "rlaif_group_better": group_r[pid][vb], "rlaif_group_worse": group_r[pid][vw]}
            rec["rlvr"] = outcome_from_scores(rec["rlvr_better"], rec["rlvr_worse"])
            rec["rlaif_pairwise"] = outcome_from_judge(rec["judge_pref"])
            rec["rlaif_group"] = outcome_from_scores(rec["rlaif_group_better"], rec["rlaif_group_worse"])
            if order_check:
                rev = prefs[(pid, vw, vb)]   # compare(question, worse, better): B = prefers better
                rec["judge_pref_reversed_args"] = rev
                rec["rlaif_pairwise_reversed"] = {"B": "better", "A": "wrong", "TIE": "tie"}[rev]
                rec["opposite_physical_order"] = (swap_of(judge, rec["question"], g[vb]["response"], g[vw]["response"])
                                                  == swap_of(judge, rec["question"], g[vw]["response"], g[vb]["response"]))
            records.append(rec)
    write_jsonl(out / "pair_records.jsonl", records)
    df = pd.DataFrame(records)

    by_category = {cat: {m: rates(df.loc[df.category == cat, m].tolist()) for m in MECHANISMS} for cat in PAIRS}
    s_reason = {m: by_category["reasoning"][m]["better_rate"] for m in MECHANISMS}
    s_outcome = {m: by_category["outcome"][m]["better_rate"] for m in MECHANISMS}

    # S_outcome over every within-problem pair whose verified finals differ (correct-final vs wrong-final variants)
    correct_v = [v for v in VARIANT_ORDER if all(rlvr[p][v] == 1.0 for p in groups)]
    wrong_v = [v for v in VARIANT_ORDER if all(rlvr[p][v] == 0.0 for p in groups)]
    all_outcome = {"rlvr": [], "rlaif_pairwise": [], "rlaif_group": []}
    for pid in groups:
        for vc, vw in itertools.product(correct_v, wrong_v):
            all_outcome["rlvr"].append(outcome_from_scores(rlvr[pid][vc], rlvr[pid][vw]))
            p = prefs[(pid, vc, vw)] if (pid, vc, vw) in prefs else {"A": "B", "B": "A", "TIE": "TIE"}[prefs[(pid, vw, vc)]]
            all_outcome["rlaif_pairwise"].append(outcome_from_judge(p))
            all_outcome["rlaif_group"].append(outcome_from_scores(group_r[pid][vc], group_r[pid][vw]))
    outcome_all = {m: rates(v) for m, v in all_outcome.items()}

    # per-variant mean rewards (all five categories) - the two scales are NOT comparable utilities
    vr = pd.DataFrame([{"problem_id": pid, "variant": v, "rlvr_reward": rlvr[pid][v], "rlaif_group_reward": group_r[pid][v]}
                       for pid in groups for v in VARIANT_ORDER])
    variant_rewards = (vr.groupby("variant")[["rlvr_reward", "rlaif_group_reward"]].agg(["mean", "std"])
                       .reindex(VARIANT_ORDER))
    variant_rewards.columns = [f"{a}_{b}" for a, b in variant_rewards.columns]
    variant_rewards["rlaif_group_rank_mean"] = (vr.assign(rank=vr.groupby("problem_id").rlaif_group_reward
                                                          .rank(ascending=False, method="average"))
                                                .groupby("variant")["rank"].mean().reindex(VARIANT_ORDER))
    variant_rewards.to_csv(out / "variant_rewards.csv")

    # judge preference matrix: M[row, col] = fraction of problems where the judge prefers row over col
    mat = pd.DataFrame(np.nan, index=VARIANT_ORDER, columns=VARIANT_ORDER)
    tie_mat = mat.copy()
    for vi, vj in itertools.combinations(VARIANT_ORDER, 2):
        p = [prefs[(pid, vi, vj)] for pid in groups]
        mat.loc[vi, vj], mat.loc[vj, vi] = np.mean([x == "A" for x in p]), np.mean([x == "B" for x in p])
        tie_mat.loc[vi, vj] = tie_mat.loc[vj, vi] = np.mean([x == "TIE" for x in p])
    mat.to_csv(out / "judge_matrix.csv")
    tie_mat.to_csv(out / "judge_tie_matrix.csv")

    order = {}
    if order_check:
        opp = df[df.opposite_physical_order]
        order = {"pairs_with_opposite_physical_order": int(len(opp)),
                 "position_consistency": float((opp.rlaif_pairwise == opp.rlaif_pairwise_reversed).mean()) if len(opp) else None,
                 "consistency_by_category": {c: float((s.rlaif_pairwise == s.rlaif_pairwise_reversed).mean())
                                             for c, s in opp.groupby("category")}}
    cdf = pd.DataFrame(calls).drop_duplicates(["problem_id", "first", "second"])
    position = {"n_calls": int(len(cdf)),
                "raw_first_position_pick_rate": float((cdf.raw_position_pick == "A").mean()),
                "raw_second_position_pick_rate": float((cdf.raw_position_pick == "B").mean()),
                "tie_rate": float((cdf.raw_position_pick == "TIE").mean())}
    uncached = int((~cdf.cached).sum())

    summary = {
        "n_problems": len(groups), "n_responses": len(groups) * len(VARIANT_ORDER),
        "pairs": {c: {"better": SHORT[b], "worse": SHORT[w]} for c, (b, w) in PAIRS.items()},
        "by_category": by_category,
        "S_reason": s_reason, "S_outcome": s_outcome,
        "S_outcome_all_outcome_changing_pairs": outcome_all,
        "outcome_changing_variants": {"correct_final": [SHORT[v] for v in correct_v], "wrong_final": [SHORT[v] for v in wrong_v]},
        "verifier_expected_mismatches": mismatches,
        "judge_order_check": order, "judge_position": position,
        "cost": {"verifier_calls": len(groups) * len(VARIANT_ORDER), "verifier_seconds_total": verifier_seconds,
                 "verifier_seconds_per_call": verifier_seconds / (len(groups) * len(VARIANT_ORDER)),
                 "judge_calls_total": int(len(cdf)), "judge_calls_uncached_this_run": uncached,
                 "judge_seconds_this_run": round(judge_seconds, 1),
                 "judge_seconds_per_uncached_call": round(judge_seconds / uncached, 3) if uncached else None,
                 "judge_model": cfg["ai_judge_model"]},
    }
    old = out / "diagnostic_scores.json"
    if not uncached and old.exists():   # everything was cached: keep the judge timing measured on the first run
        oc = json.loads(old.read_text(encoding="utf-8")).get("cost", {})
        if oc.get("judge_seconds_per_uncached_call"):
            for k in ("judge_calls_uncached_this_run", "judge_seconds_this_run", "judge_seconds_per_uncached_call"):
                summary["cost"][k] = oc[k]
            summary["cost"]["judge_timing_from_first_run"] = True
    save_json(old, summary)

    print("[task5:diagnostics] category    mechanism        better  tie    wrong", flush=True)
    for c in PAIRS:
        for m in MECHANISMS:
            r = by_category[c][m]
            print(f"[task5:diagnostics] {c:11s} {m:15s}  {r['better_rate']:.2f}   {r['tie_rate']:.2f}   {r['wrong_rate']:.2f}",
                  flush=True)
    print(f"[task5:diagnostics] S_reason {s_reason} | S_outcome {s_outcome}", flush=True)
    if order:
        print(f"[task5:diagnostics] judge position consistency {order['position_consistency']} "
              f"on {order['pairs_with_opposite_physical_order']} opposite-order pairs", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--no-order-check", action="store_true", help="skip re-judging the controlled pairs with swapped arguments")
    args = ap.parse_args()
    run(args.config, order_check=not args.no_order_check)


if __name__ == "__main__":
    main()
