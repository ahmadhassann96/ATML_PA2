"""Task 1 figures: live training curves and report figures, all built from files in results/task1_dpo/.

Colors follow the condition, never its rank, and are fixed across every figure so a reader can
match a run between panels. SFT (the reference policy) is drawn in neutral gray.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data import repo_path
from common.live import read_jsonl_safe
from common.plotting import GRID, INK, MUTED, save, style  # noqa: F401  (save re-exported for the notebook)

COLORS = {
    "standard": "#2a78d6",         # blue
    "length_balanced": "#eb6834",  # orange
    "beta_0.03": "#1baf7a",        # aqua
    "beta_0.1": "#eda100",         # yellow
    "beta_0.3": "#e87ba4",         # magenta
    "sft": "#8a8984",              # neutral gray = reference policy
}
LABELS = {
    "standard": "standard DPO (1 epoch, β=0.1)",
    "length_balanced": "length-balanced DPO",
    "beta_0.03": "short run β=0.03",
    "beta_0.1": "short run β=0.10",
    "beta_0.3": "short run β=0.30",
    "sft": "SFT (reference)",
}
def results_dir(cfg) -> Path:
    return repo_path(cfg["results_dir"])


def _json(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _smooth(y, k=5):
    """Trailing moving average that is defined from the first point (window grows up to k)."""
    return pd.Series(np.asarray(y, dtype=float)).rolling(k, min_periods=1).mean().to_numpy()


# ----------------------------------------------------------------------------- live / training curves
def train_curves(cfg, runs, smooth=5):
    """Loss, train preference accuracy, implicit-reward margin and grad norm vs optimizer update."""
    res = results_dir(cfg)
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.9))
    panels = [("loss", "DPO loss (train)"), ("acc", "preference accuracy (train batch)"),
              ("margin", "implicit-reward margin m"), ("grad_norm", "grad norm (pre-clip)")]
    status = []
    for run in runs:
        log = read_jsonl_safe(res / run / "train_log.jsonl")
        if not log:
            continue
        rc = _json(res / run / "run_config.json") or {}
        total = rc.get("optimizer_updates")
        x = [r["update"] for r in log]
        color = COLORS.get(run, "#4a3aa7")
        for ax, (key, title) in zip(axes, panels):
            y = [r[key] for r in log]
            ax.plot(x, y, color=color, alpha=0.25, linewidth=1)
            ax.plot(x, _smooth(y, smooth), color=color, label=LABELS.get(run, run))
            ax.set_title(title, loc="left")
            ax.set_xlabel("optimizer update")
        last = log[-1]
        eta = ""
        if total and last["update"] < total:
            eta = f", ETA {last['elapsed_s'] / last['update'] * (total - last['update']) / 60:.1f} min"
        vram = f", peak VRAM {last['peak_vram_gib']:.2f} GiB" if last.get("peak_vram_gib") else ""
        status.append(f"{LABELS.get(run, run)}: update {last['update']}/{total or '?'}{eta}{vram}")
    axes[0].axhline(np.log(2), color=MUTED, linestyle=":", linewidth=1)
    axes[0].annotate("log 2 (init)", xy=(0, np.log(2)), xycoords=("axes fraction", "data"),
                     xytext=(2, 3), textcoords="offset points", color=MUTED, fontsize=8)
    axes[2].axhline(0, color=MUTED, linewidth=0.8)
    if not status:
        plt.close(fig)
        return None
    handles, labels = axes[0].get_legend_handles_labels()
    keep = [i for i, l in enumerate(labels) if l]
    fig.legend([handles[i] for i in keep], [labels[i] for i in keep], loc="lower center",
               ncol=min(len(keep), 5), fontsize=8, bbox_to_anchor=(0.5, -0.02))
    lines = ["   |   ".join(status[i:i + 2]) for i in range(0, len(status), 2)]
    fig.suptitle("\n".join(lines), x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


# ----------------------------------------------------------------------------- report figures
def beta_study(cfg):
    """Small multiples: held-out preference accuracy, KL, reward, generated length vs beta."""
    p = results_dir(cfg) / "summary_beta.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    forks = df[df["condition"].str.startswith("beta_")].sort_values("beta")
    ref = {r["condition"]: r for _, r in df.iterrows() if r["condition"] in ("standard", "sft")}
    panels = [("heldout_pref_acc", "held-out preference accuracy", ("heldout_pref_acc_ci_lo", "heldout_pref_acc_ci_hi")),
              ("kl_token_mean", "KL from reference (per token)", None),
              ("reward_mean", "reward-model score (mean)", None),
              ("len_mean", "generated length (tokens, mean)", None)]
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.9))
    for ax, (col, title, ci) in zip(axes, panels):
        x, y = forks["beta"].to_numpy(), forks[col].to_numpy(dtype=float)
        if ci:
            err = np.vstack([y - forks[ci[0]].to_numpy(float), forks[ci[1]].to_numpy(float) - y])
            ax.errorbar(x, y, yerr=err, color=INK, linewidth=1.2, capsize=3, zorder=1)
        ax.plot(x, y, color=INK, linewidth=1.5, zorder=2)
        for xi, yi, name in zip(x, y, forks["condition"]):
            ax.scatter([xi], [yi], s=60, color=COLORS.get(name), edgecolor="white", linewidth=2, zorder=3)
        for name, r in ref.items():
            if pd.notna(r.get(col)) and not (name == "sft" and col in ("heldout_pref_acc",)):
                ax.axhline(r[col], color=COLORS[name], linestyle="--", linewidth=1.2)
                # standard labelled at the right end, SFT at the left, so close reference lines stay legible
                right = name == "standard"
                ax.annotate("standard (1 epoch)" if right else "SFT", xy=(1 if right else 0, r[col]),
                            xycoords=("axes fraction", "data"), xytext=(-2 if right else 2, 3),
                            textcoords="offset points", ha="right" if right else "left", fontsize=7.5,
                            color=COLORS[name])
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b:g}" for b in x])
        ax.minorticks_off()
        ax.set_xlabel("β (short-run forks, same data/steps)")
        ax.set_title(title, loc="left")
    fig.suptitle("Regularization strength: short-run forks (points) vs one-epoch standard run and SFT (dashed lines)",
                 x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    return fig


def length_strata(cfg):
    """Held-out preference accuracy per length stratum, standard vs length-balanced (95% bootstrap CI)."""
    s = _json(results_dir(cfg) / "summary_length.json")
    if not s:
        return None
    strata = ["preferred_longer", "length_matched", "rejected_longer"]
    names = {"preferred_longer": "preferred longer", "length_matched": "length matched", "rejected_longer": "rejected longer"}
    models = [m for m in s["models"] if m["condition"] in ("standard", "length_balanced") and f"acc_{strata[0]}" in m]
    if not models:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    w = 0.38
    for i, m in enumerate(models):
        xs = np.arange(len(strata)) + (i - (len(models) - 1) / 2) * (w + 0.02)
        ys = np.array([m[f"acc_{t}"] for t in strata], dtype=float)
        lo = np.array([m.get(f"acc_{t}_ci_lo", np.nan) for t in strata], dtype=float)
        hi = np.array([m.get(f"acc_{t}_ci_hi", np.nan) for t in strata], dtype=float)
        ax.bar(xs, ys, width=w, color=COLORS[m["condition"]], label=LABELS[m["condition"]], edgecolor="white", linewidth=2)
        ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt="none", ecolor=INK, elinewidth=1, capsize=3)
        for xx, yy in zip(xs, ys):
            ax.annotate(f"{yy:.2f}", (xx, 0.02), ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")
    ax.axhline(0.5, color=MUTED, linestyle=":", linewidth=1)
    ax.set_xticks(range(len(strata)))
    ax.set_xticklabels([f"{names[t]}\n(n={models[0].get(f'n_{t}', '?')})" for t in strata])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("held-out preference accuracy")
    ax.set_title("Length-stratified held-out pairs (95% bootstrap CI; dotted = chance)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def generated_lengths(cfg, conditions=("sft", "standard", "length_balanced", "beta_0.03", "beta_0.1", "beta_0.3")):
    """Distribution of generated response length on the common held-out prompts."""
    res = results_dir(cfg)
    data, names = [], []
    for c in conditions:
        g = read_jsonl_safe(res / c / "generations.jsonl")
        if g:
            data.append([r["response_tokens"] for r in g])
            names.append(c)
    if not data:
        return None
    cap = int(cfg.get("max_generation_tokens", 256))
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(names) + 1.1))
    bp = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops={"color": "white", "linewidth": 2})
    for patch, n in zip(bp["boxes"], names):
        patch.set_facecolor(COLORS.get(n, "#4a3aa7"))
        patch.set_edgecolor(COLORS.get(n, "#4a3aa7"))
    for i, (d, n) in enumerate(zip(data, names), start=1):
        ax.scatter([np.mean(d)], [i], marker="D", s=28, color=INK, zorder=3)
        ax.annotate(f"mean {np.mean(d):.0f}, trunc {np.mean(np.asarray(d) >= cap):.0%}", (cap, i), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color=MUTED, annotation_clip=False)
    ax.axvline(cap, color=MUTED, linestyle=":", linewidth=1)
    ax.set_yticks(range(1, len(names) + 1))
    ax.set_yticklabels([LABELS.get(n, n) for n in names])
    ax.set_xlabel(f"generated response tokens (cap {cap}; box = IQR, ◆ = mean)")
    ax.set_title("Generated length on the common held-out prompts", loc="left")
    fig.tight_layout()
    return fig


def word_limit(cfg, conditions=("sft", "standard", "length_balanced", "beta_0.03", "beta_0.1", "beta_0.3")):
    res = results_dir(cfg)
    rows = []
    for c in conditions:
        ev = _json(res / c / "eval_dpo_standard_eval.json")
        if ev and ev.get("word_limit"):
            w = ev["word_limit"]
            rows.append((c, w["compliance_rate"], *(w.get("compliance_ci95") or [np.nan, np.nan]), w["response_words"]["mean"]))
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(rows) + 1.1))
    for i, (c, rate, lo, hi, words) in enumerate(rows):
        ax.barh(i, rate, color=COLORS.get(c, "#4a3aa7"), height=0.6, edgecolor="white", linewidth=2)
        ax.errorbar(rate, i, xerr=[[rate - lo], [hi - rate]], fmt="none", ecolor=INK, elinewidth=1, capsize=3)
        ax.annotate(f"{rate:.0%}  (mean {words:.0f} words)", (min(hi, 1) if not np.isnan(hi) else rate, i),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABELS.get(r[0], r[0]) for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.35)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("explicit word-limit compliance (10 prompts × samples; 95% bootstrap CI)")
    ax.set_title("Word-limit compliance on the common prompt set", loc="left")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- qualitative candidates
def _short(t, n=500):
    t = " ".join(str(t).split())
    return t if len(t) <= n else t[:n] + " …"


def reward_vs_quality_candidates(cfg, policy="standard", base="sft", k=5):
    """Prompts where the policy's reward rose most over SFT, with length/truncation alongside.
    These are CANDIDATES for 'higher reward but worse response': read them and judge correctness,
    concision and instruction-following yourself before quoting any in the report."""
    res = results_dir(cfg)
    a = {r["prompt_id"]: r for r in read_jsonl_safe(res / policy / "generations.jsonl")}
    b = {r["prompt_id"]: r for r in read_jsonl_safe(res / base / "generations.jsonl")}
    rows = []
    for pid in a.keys() & b.keys():
        ra, rb = a[pid], b[pid]
        rows.append({"prompt_id": pid, "reward_gain": ra["reward"] - rb["reward"],
                     "len_policy": ra["response_tokens"], "len_base": rb["response_tokens"],
                     "truncated_policy": ra["truncated"], "prompt": _short(ra["prompt"], 200),
                     f"{policy}_response": _short(ra["response"]), f"{base}_response": _short(rb["response"])})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["len_ratio"] = df["len_policy"] / df["len_base"].clip(lower=1)
    return df.sort_values(["reward_gain", "len_ratio"], ascending=False).head(k)


def word_limit_failures(cfg, policy="standard", k=5):
    res = results_dir(cfg)
    rows = [r for r in read_jsonl_safe(res / policy / "word_limit.jsonl") if r.get("compliant") == 0.0]
    rows.sort(key=lambda r: r["words"] - (r["word_limit"] or 0), reverse=True)
    return pd.DataFrame([{"prompt_id": r["prompt_id"], "limit": r["word_limit"], "words": r["words"],
                          "prompt": r["prompt"], "response": _short(r["response"])} for r in rows[:k]])


def stratum_flips(cfg, stratum="rejected_longer", k=5):
    """Held-out pairs in one length stratum where standard and length-balanced DPO disagree."""
    res = results_dir(cfg)
    tag = "pairs_dpo_length_stratified_eval.jsonl"
    s = {r["prompt_id"]: r for r in read_jsonl_safe(res / "standard" / tag) if r["length_stratum"] == stratum}
    l = {r["prompt_id"]: r for r in read_jsonl_safe(res / "length_balanced" / tag) if r["length_stratum"] == stratum}
    rows = [{"prompt_id": p, "standard_margin": s[p]["margin"], "length_balanced_margin": l[p]["margin"],
             "chosen_tokens": s[p]["chosen_tokens"], "rejected_tokens": s[p]["rejected_tokens"]}
            for p in s.keys() & l.keys() if s[p]["correct"] != l[p]["correct"]]
    df = pd.DataFrame(rows)
    return df.sort_values("length_balanced_margin", ascending=False).head(k) if not df.empty else df
