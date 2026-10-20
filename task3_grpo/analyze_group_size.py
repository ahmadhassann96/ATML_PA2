"""Task 3 Step 2: group size K in {2, 4, 8} at an equal generation budget (no training, no GPU).

Partition rule (documented once, used for every K): the supplied cache holds 8 completions for each of 24 held-out
prompts, i.e. 192 generations. For group size K, each prompt's 8 completions are split uniformly at random into
8/K disjoint groups of K (K=8: one group per prompt; K=4: two; K=2: four), so every K uses exactly the same 192
generations and the same prompts. The split is repeated over N_PARTITIONS random partitions (seeded) and statistics
are averaged, so results do not depend on the cache's generation order.

Per K (overall and per prompt-difficulty bin):
  * informative-group rate: share of groups whose reward std exceeds the helper tolerance (1e-6); sensitivity
    columns with practical tolerances (0.1, 0.25 reward units) are also reported;
  * mean within-group reward std (population std, as in the helper);
  * variance of the group-relative signal: mean squared centred reward E[(r - mean_group)^2], and the
    partition-to-partition variance of each completion's normalised advantage (how much a completion's learning
    signal depends on which other samples it was grouped with), averaged over completions.
Difficulty bins (defined once): prompts ranked by their mean reward over all 8 cached completions and split into
terciles: "hard" (lowest third of mean reward), "medium", "easy" (highest third).

    python -m task3_grpo.analyze_group_size --config configs/grpo.yaml
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from common.runs import write_csv

ZERO_STD_TOL = 1e-6
PRACTICAL_TOLS = (0.1, 0.25)
N_PARTITIONS = 500


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int, rng=None):
    """Return K-sized groups while keeping total cached completions fixed.

    Each prompt's first 8 completions are permuted (rng) and cut into 8 // k disjoint groups of size k, so all
    group sizes use the same 8 * n_prompts generations. Returns a list of (prompt_key, [rows]) groups.
    """
    if 8 % k:
        raise ValueError("k must divide 8")
    rng = rng or np.random.default_rng(0)
    groups = []
    for pid, rows in by_prompt.items():
        rows8 = rows[:8]
        perm = rng.permutation(8)
        for j in range(8 // k):
            groups.append((pid, [rows8[i] for i in perm[j * k:(j + 1) * k]]))
    return groups


def difficulty_bins(by_prompt):
    means = {pid: float(np.mean([float(r["reward"]) for r in rows[:8]])) for pid, rows in by_prompt.items()}
    ranked = sorted(means, key=means.get)
    n = len(ranked)
    names = ("hard (low mean reward)", "medium", "easy (high mean reward)")
    bins = {}
    for i, pid in enumerate(ranked):
        bins[pid] = names[min(2, i * 3 // n)]
    return bins, means


def study(by_prompt, ks, n_partitions=N_PARTITIONS, seed=6304):
    bins, means = difficulty_bins(by_prompt)
    rng = np.random.default_rng(seed)
    results = []
    for k in ks:
        acc = defaultdict(lambda: defaultdict(list))
        adv_samples = defaultdict(list)  # (pid, generation_index) -> advantages over partitions
        for _ in range(n_partitions):
            for pid, grp in regroup_equal_generation_budget(by_prompt, k, rng):
                r = np.array([float(x["reward"]) for x in grp])
                mu, sd = r.mean(), r.std()
                a = (r - mu) / (sd + ZERO_STD_TOL)
                for x, ai in zip(grp, a):
                    adv_samples[(pid, x.get("generation_index"))].append(ai)
                for key in ("all", bins[pid]):
                    d = acc[key]
                    d["informative"].append(sd > ZERO_STD_TOL)
                    for t in PRACTICAL_TOLS:
                        d[f"informative_tol{t:g}"].append(sd > t)
                    d["std"].append(sd)
                    d["centred_sq"].extend(((r - mu) ** 2).tolist())
        adv_var = defaultdict(list)
        for (pid, _), vals in adv_samples.items():
            v = float(np.var(vals))
            adv_var["all"].append(v)
            adv_var[bins[pid]].append(v)
        for key, d in acc.items():
            row = {"K": k, "bin": key, "groups_per_partition": len(d["std"]) // n_partitions,
                   "informative_rate": float(np.mean(d["informative"])),
                   "mean_group_reward_std": float(np.mean(d["std"])),
                   "mean_centred_reward_sq": float(np.mean(d["centred_sq"])),
                   "advantage_partition_variance": float(np.mean(adv_var[key]))}
            for t in PRACTICAL_TOLS:
                row[f"informative_rate_tol{t:g}"] = float(np.mean(d[f"informative_tol{t:g}"]))
            results.append(row)
    order = {"all": 0, "hard (low mean reward)": 1, "medium": 2, "easy (high mean reward)": 3}
    results.sort(key=lambda r: (order.get(r["bin"], 9), r["K"]))
    return results, bins, means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--partitions", type=int, default=N_PARTITIONS)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    print("Cached prompts:", len(by_prompt), "| total generations:", 8 * len(by_prompt))
    print("Group sizes to analyze:", cfg["group_sizes"])
    results, bins, means = study(by_prompt, [int(k) for k in cfg["group_sizes"]], args.partitions, int(cfg["seed"]))
    res_dir = repo_path(cfg["results_dir"])
    res_dir.mkdir(parents=True, exist_ok=True)
    write_csv(res_dir / "group_size_study.csv", results)
    save_json(res_dir / "group_size_study.json", {
        "partition_rule": "each prompt's 8 cached completions randomly split into 8/K disjoint groups of K; "
                          f"{args.partitions} seeded partitions averaged; total generations fixed at {8 * len(by_prompt)}",
        "difficulty_rule": "terciles of per-prompt mean reward over all 8 cached completions",
        "zero_std_tolerance": ZERO_STD_TOL, "practical_tolerances": list(PRACTICAL_TOLS),
        "prompt_bins": bins, "prompt_mean_reward": means, "results": results})
    for r in results:
        if r["bin"] == "all":
            print(f"K={r['K']}: informative {r['informative_rate']:.3f} (tol0.1 {r['informative_rate_tol0.1']:.3f}) "
                  f"mean std {r['mean_group_reward_std']:.3f} centred^2 {r['mean_centred_reward_sq']:.3f} "
                  f"adv partition var {r['advantage_partition_variance']:.3f}")


if __name__ == "__main__":
    main()
