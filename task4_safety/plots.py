"""Task 4 figures, built from results/task4_safety/.

Policy colors (fixed; SFT is the neutral-gray reference, every policy also has its own marker shape and a direct
label so identity never relies on color): SFT gray, DPO blue, PPO orange, GRPO violet.
Judge-label colors (stack order = palette order): SAFE_ANSWER blue, OVER_REFUSAL orange, JUSTIFIED_REFUSAL aqua,
UNSAFE_COMPLIANCE yellow, AMBIGUOUS gray.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data import read_jsonl, repo_path
from common.plotting import INK, MUTED, save, style  # noqa: F401

style()
POLICIES = ["sft", "dpo", "ppo", "grpo"]
P_COLORS = {"sft": "#8a8984", "dpo": "#2a78d6", "ppo": "#eb6834", "grpo": "#4a3aa7"}
P_MARKERS = {"sft": "o", "dpo": "s", "ppo": "^", "grpo": "D"}
P_LABELS = {"sft": "SFT", "dpo": "DPO", "ppo": "PPO", "grpo": "GRPO"}
LABELS = ["SAFE_ANSWER", "OVER_REFUSAL", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "AMBIGUOUS"]
L_COLORS = {"SAFE_ANSWER": "#2a78d6", "OVER_REFUSAL": "#eb6834", "JUSTIFIED_REFUSAL": "#1baf7a",
            "UNSAFE_COMPLIANCE": "#eda100", "AMBIGUOUS": "#b9b8b3"}


def out_dir(cfg) -> Path:
    return repo_path(cfg["results_dir"]) / "task4_safety"


def _summary(cfg):
    p = out_dir(cfg) / "safety_summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def judged(cfg):
    frames = [pd.DataFrame(read_jsonl(out_dir(cfg) / f"judged_{p}.jsonl")) for p in POLICIES
              if (out_dir(cfg) / f"judged_{p}.jsonl").exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def calibration(cfg):
    s = _summary(cfg)
    if not s:
        return None
    pol = [p for p in POLICIES if p in s["policies"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    safe_metrics = [("safe_answer_rate", "safe answer"), ("over_refusal_rate", "OVER_REFUSAL label")]
    if "refusal_on_safe_rate" in s["policies"][pol[0]]:
        safe_metrics.append(("refusal_on_safe_rate", "any refusal label\n(class-mapped over-refusal)"))
    for ax, cls, metrics in [(axes[0], "SAFE prompts", safe_metrics),
                             (axes[1], "UNSAFE prompts", [("justified_refusal_rate", "justified refusal"),
                                                          ("unsafe_compliance_rate", "unsafe compliance")])]:
        w = 0.19
        for j, (key, lab) in enumerate(metrics):
            for i, p in enumerate(pol):
                m = s["policies"][p]
                x = j + (i - (len(pol) - 1) / 2) * (w + 0.01)
                y, (lo, hi) = m[key], m[key + "_ci95"]
                ax.bar(x, y, width=w, color=P_COLORS[p], edgecolor="white", linewidth=2,
                       hatch="//" if key == "refusal_on_safe_rate" else None, label=P_LABELS[p] if j == 0 else None)
                ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="none", ecolor=INK, elinewidth=1, capsize=2)
                ax.annotate(f"{y:.2f}", (x, hi), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7, color=MUTED)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([m[1] for m in metrics], fontsize=8)
        ax.set_ylim(0, 1.12)
        ax.set_title(f"{cls} (AI judge, 95% CI)", loc="left")
    axes[0].legend(fontsize=8, ncol=4, loc="upper right")
    # (c) how much each trained policy actually changed relative to SFT on the same prompts
    ax = axes[2]
    df = judged(cfg)
    others = [p for p in pol if p != "sft"]
    if not df.empty and "sft" in set(df.policy) and others:
        piv_r = df.pivot_table(index="xstest_id", columns="policy", values="response", aggfunc="first")
        piv_l = df.pivot_table(index="xstest_id", columns="policy", values="label", aggfunc="first")
        w = 0.36
        for i, (piv, lab, hatch) in enumerate([(piv_r, "response text differs", None),
                                                (piv_l, "judge label differs", "//")]):
            vals = [float((piv[p] != piv["sft"]).mean()) for p in others]
            x = np.arange(len(others)) + (i - 0.5) * w
            ax.bar(x, vals, width=w - 0.02, color=[P_COLORS[p] for p in others], edgecolor="white", hatch=hatch,
                   label=lab)
            for xi, v in zip(x, vals):
                ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7,
                            color=MUTED)
        ax.set_xticks(range(len(others)))
        ax.set_xticklabels([P_LABELS[p] for p in others])
        ax.set_ylim(0, 1.12)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.set_title("Change vs SFT on the same 450 prompts", loc="left")
    fig.tight_layout()
    return fig


def categories(cfg):
    df = judged(cfg)
    if df.empty:
        return None
    pol = [p for p in POLICIES if p in set(df.policy)]
    types = (df[["benchmark_class", "type"]].drop_duplicates().sort_values(["benchmark_class", "type"])["type"].tolist())
    fig, axes = plt.subplots(1, len(pol), figsize=(3.4 * len(pol) + 1.6, 6.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, pol):
        sub = df[df.policy == p]
        frac = (sub.groupby("type").label.value_counts(normalize=True).unstack(fill_value=0)
                .reindex(index=types, columns=LABELS, fill_value=0))
        left = np.zeros(len(types))
        for lab in LABELS:
            ax.barh(range(len(types)), frac[lab].to_numpy(), left=left, color=L_COLORS[lab], edgecolor="white",
                    linewidth=1, height=0.8, label=lab.replace("_", " ").lower())
            left += frac[lab].to_numpy()
        ax.set_title(P_LABELS[p], loc="left")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.grid(axis="y", visible=False)
        n_safe = int((df[["benchmark_class", "type"]].drop_duplicates().benchmark_class == "SAFE").sum())
        ax.axhline(n_safe - 0.5, color=INK, linewidth=1)
    axes[0].set_yticks(range(len(types)))
    axes[0].set_yticklabels(types, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].annotate("SAFE types ↑ | UNSAFE contrasts ↓", xy=(0, 0), xytext=(0, -26), textcoords="offset points",
                     xycoords="axes fraction", fontsize=8, color=MUTED)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.55, -0.01))
    fig.suptitle("AI-judge label distribution per XSTest category", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def confusion(cfg):
    p = out_dir(cfg) / "manual_vs_ai_confusion.csv"
    if not p.exists():
        return None
    c = pd.read_csv(p, index_col=0).reindex(index=LABELS, columns=LABELS, fill_value=0)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    im = ax.imshow(c.to_numpy(), cmap="Blues")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            v = int(c.iloc[i, j])
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                        color="white" if v > c.to_numpy().max() * 0.5 else INK)
    short = [l.replace("_", "\n").lower() for l in LABELS]
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(short, fontsize=7.5)
    ax.set_yticks(range(len(LABELS)))
    ax.set_yticklabels(short, fontsize=7.5)
    ax.set_xlabel("AI judge label")
    ax.set_ylabel("manual label")
    ax.grid(False)
    s = _summary(cfg) or {}
    ma = s.get("manual_audit", {})
    ax.set_title(f"Manual audit vs AI judge (distinct responses: n={ma.get('n_items', '?')}, "
                 f"agreement {ma.get('agreement_items', float('nan')):.2f}, κ={ma.get('kappa_items', float('nan')):.2f})",
                 loc="left", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    return fig


def lengths(cfg):
    df = judged(cfg)
    if df.empty:
        return None
    pol = [p for p in POLICIES if p in set(df.policy)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3), sharex=True)
    for ax, cls in zip(axes, ("SAFE", "UNSAFE")):
        data = [df[(df.policy == p) & (df.benchmark_class == cls)].response_tokens.to_numpy() for p in pol]
        bp = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops={"color": "white", "linewidth": 2})
        for patch, p in zip(bp["boxes"], pol):
            patch.set_facecolor(P_COLORS[p])
            patch.set_edgecolor(P_COLORS[p])
        for i, d in enumerate(data, start=1):
            ax.scatter([d.mean()], [i], marker="D", s=24, color=INK, zorder=3)
        ax.set_yticks(range(1, len(pol) + 1))
        ax.set_yticklabels([P_LABELS[p] for p in pol])
        ax.set_title(f"Response length, {cls} prompts (◆ mean)", loc="left")
        ax.set_xlabel("response tokens (cap 256)")
    fig.tight_layout()
    return fig


def examples(cfg, label, cls=None, policy=None, k=4):
    df = judged(cfg)
    if df.empty:
        return df
    d = df[df.label == label]
    if cls:
        d = d[d.benchmark_class == cls]
    if policy:
        d = d[d.policy == policy]
    d = d.sort_values("confidence", ascending=False)
    return d[["policy", "xstest_id", "type", "benchmark_class", "label", "confidence", "prompt", "response"]].head(k)


def policy_disagreements(cfg, k=8):
    """Prompts on which the four policies receive different judge labels (where policy differences show)."""
    df = judged(cfg)
    if df.empty:
        return df
    piv = df.pivot_table(index=["xstest_id", "benchmark_class", "type", "prompt"], columns="policy", values="label",
                         aggfunc="first")
    piv = piv[piv.nunique(axis=1) > 1]
    return piv.reset_index().head(k)
