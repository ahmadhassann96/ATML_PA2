"""Task 3 figures: live GRPO training curves and report figures, built only from files in results/task3_grpo/.

Fixed colors per condition: standard GRPO blue, canonical-normalisation fork aqua, Dr. GRPO fork orange, supplied
midpoint neutral gray. Difficulty bins (an ordered category) use violet / yellow / green, "all prompts" is ink.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data import repo_path
from common.live import read_jsonl_safe
from common.plotting import INK, MUTED, save, style  # noqa: F401
from common.rl_eval import _short, reward_change_candidates as _rcc

style()

COLORS = {"standard": "#2a78d6", "norm_grpo": "#1baf7a", "norm_dr_grpo": "#eb6834", "midpoint": "#8a8984"}
LABELS = {"standard": "standard GRPO (20 updates)", "norm_grpo": "canonical GRPO (1/T_k)",
          "norm_dr_grpo": "Dr. GRPO (1/L)", "midpoint": "supplied midpoint"}
BIN_COLORS = {"all": INK, "hard (low mean reward)": "#4a3aa7", "medium": "#eda100", "easy (high mean reward)": "#008300"}
NORM_FORKS = ["norm_grpo", "norm_dr_grpo"]


def res_dir(cfg) -> Path:
    return repo_path(cfg["results_dir"])


def _json(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _smooth(y, k):
    return pd.Series(np.asarray(y, dtype=float)).rolling(k, min_periods=1).mean().to_numpy()


PANELS = [("reward_raw", "learned reward (mean of K)"), ("group_reward_std_mean", "within-group reward std"),
          ("frac_zero_std_groups", "uninformative groups (std≈0)"), ("kl_token_mean", "KL to reference (per token)"),
          ("policy_term", "policy loss term"), ("grad_norm", "grad norm (pre-clip)"),
          ("entropy_mean", "entropy (per token)"), ("response_tokens", "completion length (tokens)"),
          ("truncated_frac", "truncated (hit cap)"), ("masked_completion_frac", "completions masked from loss")]


def train_curves(cfg, runs, smooth=3):
    rd = res_dir(cfg)
    fig, axes = plt.subplots(2, 5, figsize=(15, 5.4))
    axes = axes.ravel()
    status = []
    for run in runs:
        log = read_jsonl_safe(rd / run / "train_log.jsonl")
        if not log:
            continue
        rc = _json(rd / run / "run_config.json") or {}
        total = rc.get("updates")
        x = [r["update"] for r in log]
        c = COLORS.get(run, "#4a3aa7")
        for ax, (key, title) in zip(axes, PANELS):
            y = [r.get(key, np.nan) for r in log]
            ax.plot(x, y, color=c, alpha=0.25, linewidth=1)
            ax.plot(x, _smooth(y, smooth), color=c, label=LABELS.get(run, run))
            ax.set_title(title, loc="left")
            ax.set_xlabel("update")
        last = log[-1]
        eta = f", ETA {last['elapsed_s'] / last['update'] * (total - last['update']) / 60:.1f} min" if total and last["update"] < total else ""
        vram = f", peak VRAM {last['peak_vram_gib']:.2f} GiB" if last.get("peak_vram_gib") else ""
        status.append(f"{LABELS.get(run, run)}: update {last['update']}/{total or '?'}{eta}{vram}")
    if not status:
        plt.close(fig)
        return None
    axes[3].axhline(0, color=MUTED, linewidth=0.8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 4), fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("\n".join("   |   ".join(status[i:i + 2]) for i in range(0, len(status), 2)), x=0.01, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def group_size_study(cfg):
    s = _json(res_dir(cfg) / "group_size_study.json")
    if not s:
        return None
    df = pd.DataFrame(s["results"])
    ks = sorted(df["K"].unique())
    panels = [("informative_rate", "informative-group rate (std > 1e-6)"),
              ("informative_rate_tol0.1", "…with practical tolerance std > 0.1"),
              ("mean_group_reward_std", "mean within-group reward std"),
              ("advantage_partition_variance", "advantage variance across regroupings")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
    for ax, (col, title) in zip(axes, panels):
        for b, c in BIN_COLORS.items():
            sub = df[df["bin"] == b].sort_values("K")
            if sub.empty:
                continue
            ax.plot(sub["K"], sub[col], marker="o", markersize=6, color=c, linewidth=2.4 if b == "all" else 1.6,
                    label=b if b != "all" else "all prompts")
        ax.set_xticks(ks)
        ax.set_xlabel("group size K (same 192 generations)")
        ax.set_title(title, loc="left", fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Group-size study on the supplied K=8 cache: each prompt's 8 completions split into 8/K random groups "
                 f"({s['partition_rule'].split(';')[1].strip()})", x=0.01, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def normalization_study(cfg):
    rd = res_dir(cfg)
    s = _json(rd / "summary_normalization.json")
    comps = {n: [c for c in read_jsonl_safe(rd / n / "completions.jsonl") if c["in_loss"]] for n in NORM_FORKS}
    if not any(comps.values()):
        return None
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.3))
    # (a) per-completion gradient norm vs length
    ax = axes[0]
    for n, cs in comps.items():
        pos = [c for c in cs if c["advantage"] > 0 and c.get("seq_grad_norm")]
        neg = [c for c in cs if c["advantage"] < 0 and c.get("seq_grad_norm")]
        ax.scatter([c["response_tokens"] for c in pos], [c["seq_grad_norm"] for c in pos], s=22, color=COLORS[n],
                   label=f"{LABELS[n]}, A>0")
        ax.scatter([c["response_tokens"] for c in neg], [c["seq_grad_norm"] for c in neg], s=22, facecolor="none",
                   edgecolor=COLORS[n], linewidth=1.2, label=f"{LABELS[n]}, A<0")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("completion length (tokens)")
    ax.set_title("Per-completion gradient norm", loc="left")
    ax.legend(fontsize=6.5, loc="lower left")
    # (b) per-token weight by length bin and advantage sign
    ax = axes[1]
    if s and s.get("length_bins"):
        lb = pd.DataFrame(s["length_bins"])
        cats = list(dict.fromkeys(zip(lb["advantage_sign"], lb["length_bin"])))
        w = 0.38
        for i, n in enumerate(NORM_FORKS):
            vals = [lb[(lb.fork == n) & (lb.advantage_sign == a) & (lb.length_bin == b)]["mean_per_token_weight"].mean()
                    for a, b in cats]
            ax.bar(np.arange(len(cats)) + (i - 0.5) * (w + 0.02), vals, width=w, color=COLORS[n], label=LABELS[n],
                   edgecolor="white", linewidth=1)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([f"{a}\n{b.split(' (')[0]}" for a, b in cats], fontsize=8)
        ax.set_yscale("log")
        ax.set_title("Per-token loss weight |A|/norm/B", loc="left")
        ax.legend(fontsize=7)
    # (c) completion length over updates
    ax = axes[2]
    for n in NORM_FORKS:
        log = read_jsonl_safe(rd / n / "train_log.jsonl")
        if log:
            ax.plot([r["update"] for r in log], [r["response_tokens"] for r in log], marker="o", markersize=4,
                    color=COLORS[n], label=LABELS[n])
    ax.set_xlabel("update (same prompts)")
    ax.set_title("Training completion length", loc="left")
    # (d) held-out outcomes
    ax = axes[3]
    evs = {n: _json(rd / n / "eval_heldout.json") for n in ["midpoint", *NORM_FORKS]}
    mets = [("reward_raw_mean", "reward"), ("len", "length / 100"), ("kl_seq_mean", "KL per seq x 100")]
    w = 0.26
    for i, (n, ev) in enumerate(evs.items()):
        if not ev:
            continue
        vals = [ev["response_tokens"]["mean"] / 100 if k == "len" else ev[k] * (100 if k == "kl_seq_mean" else 1)
                for k, _ in mets]
        ax.bar(np.arange(len(mets)) + (i - 1) * (w + 0.01), vals, width=w, color=COLORS[n], label=LABELS[n],
               edgecolor="white", linewidth=1)
    ax.set_xticks(range(len(mets)))
    ax.set_xticklabels([m[1] for m in mets])
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_title("Held-out outcomes", loc="left")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.35)
    ax.legend(fontsize=7, loc="upper left", ncol=1)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- qualitative candidates
def reward_change_candidates(cfg, policy="standard", base="midpoint", k=6, ascending=False):
    return _rcc(res_dir(cfg), policy, base, k, ascending)


def extreme_completions(cfg, run="norm_dr_grpo", k=4):
    """Longest and shortest training completions that entered the loss, with their advantage and weight."""
    cs = [c for c in read_jsonl_safe(res_dir(cfg) / run / "completions.jsonl") if c["in_loss"]]
    cs.sort(key=lambda c: c["response_tokens"])
    pick = cs[:k] + cs[-k:]
    return pd.DataFrame([{"update": c["update"], "tokens": c["response_tokens"], "reward": c["reward"],
                          "advantage": c["advantage"], "per_token_weight": c["per_token_weight"],
                          "seq_grad_norm": c.get("seq_grad_norm"), "response": _short(c["response"], 300)} for c in pick])


def uninformative_groups(cfg, run="standard"):
    """Training prompt groups whose K rewards were (near-)identical: no relative signal."""
    cs = read_jsonl_safe(res_dir(cfg) / run / "completions.jsonl")
    by = {}
    for c in cs:
        by.setdefault((c["update"], c["prompt_id"]), []).append(c)
    rows = []
    for (u, pid), g in by.items():
        r = np.array([c["reward"] for c in g])
        rows.append({"update": u, "prompt_id": pid, "reward_std": float(r.std()), "reward_mean": float(r.mean()),
                     "lengths": [c["response_tokens"] for c in g], "first_response": _short(g[0]["response"], 200)})
    df = pd.DataFrame(rows)
    return df.sort_values("reward_std").head(6) if not df.empty else df
