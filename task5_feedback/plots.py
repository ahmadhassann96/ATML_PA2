"""Task 5 figures, built from results/task5_feedback/ (no model is loaded).

Policy colors (fixed; SFT is the neutral-gray reference, each policy also has its own marker and a direct label):
SFT gray, RLVR aqua, RLAIF violet. Outcome colors for the diagnostic study: prefers the better response blue,
tie gray, prefers the worse response orange.
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
POLICIES = ["sft", "rlvr", "rlaif"]
P_COLORS = {"sft": "#8a8984", "rlvr": "#1baf7a", "rlaif": "#4a3aa7"}
P_LABELS = {"sft": "SFT", "rlvr": "RLVR", "rlaif": "RLAIF"}
DATASETS = {"gsm": "GSM8K", "transfer": "SVAMP"}
MECH = ["rlvr", "rlaif_pairwise", "rlaif_group"]
M_LABELS = {"rlvr": "RLVR verifier", "rlaif_pairwise": "RLAIF judge (pair)", "rlaif_group": "RLAIF group reward"}
O_COLORS = {"better": "#2a78d6", "tie": "#b9b8b3", "wrong": "#eb6834"}
CATEGORIES = {"reasoning": "reasoning-only corruption\n(clean vs corrupted steps, same final)",
              "outcome": "outcome change\n(clean vs same steps, wrong final)",
              "filler": "persuasive filler\n(clean vs clean + filler)",
              "distractor": "gold distractor\n(clean vs mentions gold, wrong final)"}
VARIANTS = ["clean_correct", "corrupt_reasoning_correct_final", "good_reasoning_wrong_final",
            "persuasive_filler_correct", "gold_distractor_wrong_final"]
V_SHORT = {"clean_correct": "clean", "corrupt_reasoning_correct_final": "corrupt reasoning\n(correct final)",
           "good_reasoning_wrong_final": "good reasoning\n(wrong final)", "persuasive_filler_correct": "persuasive filler\n(correct final)",
           "gold_distractor_wrong_final": "gold distractor\n(wrong final)"}
FAIL = ["correct", "wrong_final", "no_designated_final", "wrong_final_truncated", "no_final_truncated"]
F_COLORS = {"correct": "#2a78d6", "wrong_final": "#eb6834", "no_designated_final": "#eda100",
            "wrong_final_truncated": "#f2a37f", "no_final_truncated": "#f5d27a"}


def out_dir(cfg) -> Path:
    return repo_path(cfg["results_dir"]) / "task5_feedback"


def _json(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def evals(cfg):
    return {ds: e for ds in DATASETS if (e := _json(out_dir(cfg) / ds / "eval_math.json"))}


def policies(cfg):
    ev = evals(cfg)
    if not ev:
        return None
    ds = list(ev)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), gridspec_kw={"width_ratios": [1.2, 1, 1.2, 1.2]})
    w = 0.26

    def bars(ax, key, ci_key=None, fmt="{:.2f}", pols=POLICIES):
        for i, p in enumerate(pols):
            for j, d in enumerate(ds):
                src = ev[d]["policies"][p] if key in ev[d]["policies"][p] else ev[d]["pairwise_vs_sft"][p]
                x = j + (i - (len(pols) - 1) / 2) * w
                y = src[key]
                ax.bar(x, y, width=w - 0.02, color=P_COLORS[p], edgecolor="white", label=P_LABELS[p] if j == 0 else None)
                top = y
                if ci_key:
                    lo, hi = src[ci_key]
                    ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="none", ecolor=INK, elinewidth=1, capsize=2)
                    top = hi
                ax.annotate(fmt.format(y), (x, top), xytext=(0, 2), textcoords="offset points", ha="center",
                            fontsize=7, color=MUTED)
        ax.set_xticks(range(len(ds)))
        ax.set_xticklabels([DATASETS[d] for d in ds])

    bars(axes[0], "exact_accuracy", "exact_accuracy_ci95")
    axes[0].set_ylim(0, 1.2)
    axes[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[0].set_title("Exact final-answer accuracy (95% CI)", loc="left")
    axes[0].legend(fontsize=8, ncol=3, loc="upper left")
    bars(axes[1], "win_rate_vs_sft", "win_rate_ci95", pols=["rlvr", "rlaif"])
    axes[1].axhline(0.5, color=MUTED, linewidth=1, linestyle="--")
    axes[1].annotate("= SFT", (len(ds) - 0.5, 0.5), xytext=(0, 3), textcoords="offset points", fontsize=7,
                     color=MUTED, ha="right")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("AI-judge win rate vs SFT (tie = 0.5)", loc="left")
    bars(axes[2], "format_compliance")
    axes[2].set_ylim(0, 1.08)
    axes[2].set_title("Format compliance (#### n parsed)", loc="left")
    bars(axes[3], "response_tokens_mean", fmt="{:.0f}")
    axes[3].set_title("Mean response length (tokens)", loc="left")
    fig.tight_layout()
    return fig


def diagnostics(cfg):
    d = _json(out_dir(cfg) / "diagnostics" / "diagnostic_scores.json")
    if not d:
        return None
    fig, axes = plt.subplots(1, 4, figsize=(16, 2.9), sharey=True)
    for ax, (cat, title) in zip(axes, CATEGORIES.items()):
        for i, m in enumerate(MECH):
            r = d["by_category"][cat][m]
            left = 0.0
            for o in ("better", "tie", "wrong"):
                v = r[f"{o}_rate"]
                ax.barh(i, v, left=left, color=O_COLORS[o], edgecolor="white", height=0.7,
                        label=o if (cat == "reasoning" and i == 0) else None)
                if v >= 0.08:
                    ax.text(left + v / 2, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                            color="white" if o != "tie" else INK)
                left += v
        ax.set_xlim(0, 1)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(MECH)))
    axes[0].set_yticklabels([M_LABELS[m] for m in MECH])
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, ["prefers better", "tie", "prefers worse"], loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Controlled diagnostic pairs ({d['n_problems']} problems per category): "
                 f"S_reason = better-rate on reasoning, S_outcome = better-rate on outcome", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def variant_rewards(cfg):
    p = out_dir(cfg) / "diagnostics" / "variant_rewards.csv"
    if not p.exists():
        return None
    vr = pd.read_csv(p, index_col=0).reindex(VARIANTS)
    n = (_json(out_dir(cfg) / "diagnostics" / "diagnostic_scores.json") or {}).get("n_problems", 20)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.3), sharey=True)
    for ax, col, color, title in [(axes[0], "rlvr_reward", P_COLORS["rlvr"], "RLVR: exact verifier reward (0/1)"),
                                  (axes[1], "rlaif_group_reward", P_COLORS["rlaif"],
                                   "RLAIF: group win rate among the 5 variants")]:
        y = vr[f"{col}_mean"].to_numpy()
        se = vr[f"{col}_std"].fillna(0).to_numpy() / np.sqrt(n)
        ax.barh(range(len(VARIANTS)), y, xerr=se, color=color, edgecolor="white", height=0.65,
                error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 2})
        for i, (v, e) in enumerate(zip(y, se)):
            ax.text(v + e + 0.02, i, f"{v:.2f}", va="center", fontsize=7.5, color=MUTED)
        ax.set_xlim(0, 1.12)
        ax.set_title(title + " (mean ± s.e.)", loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(VARIANTS)))
    axes[0].set_yticklabels([V_SHORT[v] for v in VARIANTS], fontsize=8)
    axes[0].invert_yaxis()
    axes[1].annotate("different scales: not calibrated utilities", xy=(1, 0), xycoords="axes fraction",
                     xytext=(0, -24), textcoords="offset points", ha="right", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    return fig


def judge_matrix(cfg):
    p = out_dir(cfg) / "diagnostics" / "judge_matrix.csv"
    if not p.exists():
        return None
    m = pd.read_csv(p, index_col=0).reindex(index=VARIANTS, columns=VARIANTS)
    t = pd.read_csv(out_dir(cfg) / "diagnostics" / "judge_tie_matrix.csv", index_col=0).reindex(index=VARIANTS, columns=VARIANTS)
    fig, ax = plt.subplots(figsize=(6.6, 5))
    im = ax.imshow(m.to_numpy(dtype=float), cmap="Blues", vmin=0, vmax=1)
    for i in range(len(VARIANTS)):
        for j in range(len(VARIANTS)):
            if i == j:
                continue
            v, tv = m.iloc[i, j], t.iloc[i, j]
            ax.text(j, i, f"{v:.2f}\n(tie {tv:.2f})", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.6 else INK)
    lab = [V_SHORT[v] for v in VARIANTS]
    ax.set_xticks(range(len(VARIANTS)))
    ax.set_xticklabels(lab, fontsize=7, rotation=30, ha="right")
    ax.set_yticks(range(len(VARIANTS)))
    ax.set_yticklabels(lab, fontsize=7)
    ax.set_xlabel("column variant")
    ax.set_ylabel("row variant")
    ax.grid(False)
    ax.set_title("AI judge: fraction of problems where row is preferred to column", loc="left", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    return fig


def failure_types(cfg):
    ev = evals(cfg)
    if not ev:
        return None
    fig, axes = plt.subplots(1, len(ev), figsize=(5.5 * len(ev), 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (d, e) in zip(axes, ev.items()):
        for i, p in enumerate(POLICIES):
            ft, n = e["policies"][p]["failure_types"], e["policies"][p]["n"]
            left = 0.0
            for f in FAIL:
                v = ft.get(f, 0) / n
                ax.barh(i, v, left=left, color=F_COLORS[f], edgecolor="white", height=0.7,
                        label=f.replace("_", " ") if (i == 0 and d == list(ev)[0]) else None)
                if v >= 0.07:
                    ax.text(left + v / 2, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                            color="white" if f in ("correct", "wrong_final") else INK)
                left += v
        ax.set_xlim(0, 1)
        ax.set_title(f"{DATASETS[d]}: outcome per response", loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(POLICIES)))
    axes[0].set_yticklabels([P_LABELS[p] for p in POLICIES])
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def svamp_types(cfg):
    e = evals(cfg).get("transfer")
    if not e or "accuracy_by_svamp_type" not in e:
        return None
    acc = pd.DataFrame(e["accuracy_by_svamp_type"])[POLICIES]
    gens = read_jsonl(out_dir(cfg) / "transfer" / "generations_sft.jsonl")
    counts = pd.Series([g.get("svamp_type") for g in gens]).value_counts()
    fig, ax = plt.subplots(figsize=(7, 2.8))
    w = 0.26
    for i, p in enumerate(POLICIES):
        x = np.arange(len(acc)) + (i - 1) * w
        ax.bar(x, acc[p].to_numpy(), width=w - 0.02, color=P_COLORS[p], edgecolor="white", label=P_LABELS[p])
    ax.set_xticks(range(len(acc)))
    ax.set_xticklabels([f"{t}\n(n={counts.get(t, 0)})" for t in acc.index], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("SVAMP transfer accuracy by problem type", loc="left")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    return fig


def disagreement_counts(cfg):
    """Verifier-vs-judge 3x3 table per RL policy and dataset (rows: verifier says policy better/tie/worse than SFT)."""
    rows = []
    for d in DATASETS:
        for p in ("rlvr", "rlaif"):
            f = out_dir(cfg) / d / f"pairwise_{p}_vs_sft.jsonl"
            if not f.exists():
                continue
            df = pd.DataFrame(read_jsonl(f))
            df = df[~df.identical]
            v = df.verifier_diff.map({1: "verifier: policy right, SFT wrong", 0: "verifier: tie",
                                      -1: "verifier: policy wrong, SFT right"})
            j = df.judge.map({"A": "judge: policy", "B": "judge: SFT", "TIE": "judge: tie"})
            t = pd.crosstab(v, j)
            t.insert(0, "policy", p)
            t.insert(0, "dataset", d)
            rows.append(t)
    return pd.concat(rows).fillna(0) if rows else pd.DataFrame()
