"""Task 2 figures: live PPO training curves and report figures, built only from files in results/task2_ppo/.

Colors follow the condition and are fixed across every figure. The (eps=0.2, beta=0.1) short fork is the
shared reference fork of both studies and is yellow in both. The supplied midpoint is neutral gray.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data import repo_path
from common.live import read_jsonl_safe
from common.rl_eval import _short, reward_change_candidates as _rcc
from common.plotting import GRID, INK, MUTED, save, style  # noqa: F401

style()

COLORS = {
    "standard": "#2a78d6",          # blue
    "eps0.05_kl0.1": "#1baf7a",     # aqua
    "eps0.2_kl0.1": "#eda100",      # yellow (shared fork)
    "eps0.5_kl0.1": "#e87ba4",      # magenta
    "eps0.2_kl0": "#4a3aa7",        # violet
    "eps0.2_kl0.2": "#008300",      # green
    "midpoint": "#8a8984",          # gray
}
LABELS = {
    "standard": "standard PPO (20 updates)",
    "eps0.05_kl0.1": "ε=0.05, β_KL=0.1", "eps0.2_kl0.1": "ε=0.2, β_KL=0.1", "eps0.5_kl0.1": "ε=0.5, β_KL=0.1",
    "eps0.2_kl0": "ε=0.2, β_KL=0", "eps0.2_kl0.2": "ε=0.2, β_KL=0.2", "midpoint": "supplied midpoint",
}
CLIP_FORKS = ["eps0.05_kl0.1", "eps0.2_kl0.1", "eps0.5_kl0.1"]
KL_FORKS = ["eps0.2_kl0", "eps0.2_kl0.1", "eps0.2_kl0.2"]


def res_dir(cfg) -> Path:
    return repo_path(cfg["results_dir"])


def _json(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _smooth(y, k):
    return pd.Series(np.asarray(y, dtype=float)).rolling(k, min_periods=1).mean().to_numpy()


PANELS = [("reward_raw", "learned reward (rollout)"), ("kl_token_mean", "KL to reference (per token)"),
          ("policy_loss", "policy loss"), ("value_loss", "value loss"), ("value_explained_variance", "critic explained variance"),
          ("entropy_mean", "entropy (per token)"), ("grad_norm_policy", "policy grad norm (pre-clip)"),
          ("clip_fraction", "clip fraction"), ("policy_change_k3", "policy change per update (k3)"),
          ("response_tokens", "response length (tokens)")]


def train_curves(cfg, runs, smooth=3, panels=PANELS):
    rd = res_dir(cfg)
    ncol = 5
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 2.7 * nrow), squeeze=False)
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
        for ax, (key, title) in zip(axes, panels):
            y = [r.get(key, np.nan) for r in log]
            ax.plot(x, y, color=c, alpha=0.25, linewidth=1)
            ax.plot(x, _smooth(y, smooth), color=c, label=LABELS.get(run, run))
            ax.set_title(title, loc="left")
            ax.set_xlabel("update")
        last = log[-1]
        eta = ""
        if total and last["update"] < total:
            eta = f", ETA {last['elapsed_s'] / last['update'] * (total - last['update']) / 60:.1f} min"
        vram = f", peak VRAM {last['peak_vram_gib']:.2f} GiB" if last.get("peak_vram_gib") else ""
        status.append(f"{LABELS.get(run, run)}: update {last['update']}/{total or '?'}{eta}{vram}")
    for ax in axes[len(panels):]:
        ax.axis("off")
    if not status:
        plt.close(fig)
        return None
    for ax, (key, _) in zip(axes, panels):
        if key in ("kl_token_mean", "policy_loss", "value_explained_variance"):
            ax.axhline(0, color=MUTED, linewidth=0.8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 6), fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("\n".join("   |   ".join(status[i:i + 2]) for i in range(0, len(status), 2)), x=0.01, ha="left", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def clipping_study(cfg):
    rd = res_dir(cfg)
    cached = _json(rd / "clipping_cached.json")
    if not cached:
        return None
    npz = rd / "clipping_cached_log_ratios.npz"
    lr = np.load(npz) if npz.exists() else None
    eps_list = [float(e) for e in cfg["clip_values"]]
    names = [f"eps{e:g}_kl{float(cfg['kl_beta']):g}" for e in eps_list]
    steps = max(r["step"] for r in cached["results"])
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.1))

    # (a) log-ratio distribution: midpoint vs after the optimisation steps, per epsilon
    ax = axes[0]
    if lr is not None:
        bins = np.linspace(-0.6, 0.6, 61)
        ax.hist(lr["midpoint"], bins=bins, color=COLORS["midpoint"], alpha=0.6, label="midpoint vs cached old")
        for e, n in zip(eps_list, names):
            k = f"eps{e:g}_step{steps}"
            if k in lr.files:
                ax.hist(lr[k], bins=bins, histtype="step", linewidth=1.6, color=COLORS[n], label=f"after {steps} steps, ε={e:g}")
        for e, n in zip(eps_list, names):
            for b in (np.log(1 - e), np.log(1 + e)):
                ax.axvline(b, color=COLORS[n], linestyle=":", linewidth=1)
        ax.set_yscale("log")
        ax.set_xlabel("log ratio  log π_θ − log π_old  (dotted: ±ε band)")
        ax.set_title("Cached batch: token log-ratios", loc="left")
        ax.legend(fontsize=7, loc="upper left")

    # (b) clip / affected fraction per epsilon, midpoint and after steps
    ax = axes[1]
    w = 0.2
    for j, (key, lab) in enumerate([("clip_fraction", "outside band"), ("affected_fraction", "clip binds")]):
        for s_i, step in enumerate((0, steps)):
            vals = [next(r[key] for r in cached["results"] if r["epsilon"] == e and r["step"] == step) for e in eps_list]
            xs = np.arange(len(eps_list)) + (j * 2 + s_i - 1.5) * (w + 0.01)
            ax.bar(xs, vals, width=w, color=[COLORS[n] for n in names], alpha=1.0 if step else 0.45,
                   hatch="" if key == "clip_fraction" else "///", edgecolor="white", linewidth=1)
    for i, e in enumerate(eps_list):
        nf = next((r.get("noise_floor_clip_fraction") for r in cached["results"] if r["epsilon"] == e and r["step"] == 0), None)
        if nf is not None:
            ax.hlines(nf, i - 0.45, i + 0.45, colors=INK, linestyles="--", linewidth=1.2,
                      label="precision noise floor" if i == 0 else None)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xticks(range(len(eps_list)))
    ax.set_xticklabels([f"ε={e:g}" for e in eps_list])
    ax.set_ylabel("fraction of response tokens")
    ax.set_title("Clip fraction (solid) / clip binds (hatched)\nlight = midpoint, dark = after steps", loc="left", fontsize=9)

    # (c) per-update policy movement in the matched forks
    ax = axes[2]
    for n, e in zip(names, eps_list):
        log = read_jsonl_safe(rd / n / "train_log.jsonl")
        if log:
            ax.plot([r["update"] for r in log], [r["policy_change_k3"] for r in log], color=COLORS[n], marker="o",
                    markersize=4, label=f"ε={e:g}")
    ax.set_xlabel("update")
    ax.set_title("Fork stability: policy change per update (k3)", loc="left")
    ax.legend(fontsize=8)

    # (d) held-out reward of the forks
    ax = axes[3]
    for i, (n, e) in enumerate(zip(names, eps_list)):
        ev = _json(rd / n / "eval_heldout.json")
        if ev:
            y, (lo, hi) = ev["reward_raw_mean"], ev["reward_raw_ci95"]
            ax.errorbar([i], [y], yerr=[[y - lo], [hi - y]], fmt="o", color=COLORS[n], ecolor=INK, capsize=3, markersize=8)
    mid = _json(rd / "midpoint" / "eval_heldout.json")
    if mid:
        ax.axhline(mid["reward_raw_mean"], color=COLORS["midpoint"], linestyle="--", linewidth=1.2)
        ax.annotate("midpoint", xy=(0, mid["reward_raw_mean"]), xycoords=("axes fraction", "data"), xytext=(2, 3),
                    textcoords="offset points", fontsize=7.5, color=MUTED)
    ax.set_xticks(range(len(eps_list)))
    ax.set_xticklabels([f"ε={e:g}" for e in eps_list])
    ax.set_title("Held-out reward after the fork (95% CI)", loc="left")
    fig.tight_layout()
    return fig


def kl_study(cfg):
    rd = res_dir(cfg)
    betas = [float(b) for b in cfg["kl_values"]]
    eps = float(cfg["clip_epsilon"])
    names = [f"eps{eps:g}_kl{b:g}" for b in betas]
    evs = [_json(rd / n / "eval_heldout.json") for n in names]
    if not any(evs):
        return None
    mid = _json(rd / "midpoint" / "eval_heldout.json")
    panels = [("reward_raw_mean", "held-out reward", "reward_raw_ci95"), ("kl_token_mean", "held-out KL (per token)", None),
              ("entropy_mean", "held-out entropy (per token)", None), ("len", "held-out response length", None)]
    fig, axes = plt.subplots(2, 4, figsize=(15, 5.6))
    for ax, (key, title, ci) in zip(axes[0], panels):
        for i, (n, ev) in enumerate(zip(names, evs)):
            if not ev:
                continue
            y = ev["response_tokens"]["mean"] if key == "len" else ev[key]
            if ci:
                lo, hi = ev[ci]
                ax.errorbar([i], [y], yerr=[[y - lo], [hi - y]], fmt="none", ecolor=INK, capsize=3)
            ax.scatter([i], [y], s=70, color=COLORS[n], edgecolor="white", linewidth=2, zorder=3)
        if mid:
            yref = mid["response_tokens"]["mean"] if key == "len" else mid[key]
            ax.axhline(yref, color=COLORS["midpoint"], linestyle="--", linewidth=1.2)
            ax.annotate("midpoint", xy=(0, yref), xycoords=("axes fraction", "data"), xytext=(2, 3),
                        textcoords="offset points", fontsize=7.5, color=MUTED)
        ax.set_xticks(range(len(betas)))
        ax.set_xticklabels([f"β={b:g}" for b in betas])
        ax.set_title(title, loc="left")
    traj = [("reward_raw", "rollout reward"), ("kl_token_mean", "rollout KL (per token)"),
            ("entropy_mean", "rollout entropy"), ("response_tokens", "rollout length")]
    for ax, (key, title) in zip(axes[1], traj):
        for n, b in zip(names, betas):
            log = read_jsonl_safe(rd / n / "train_log.jsonl")
            if log:
                ax.plot([r["update"] for r in log], [r[key] for r in log], color=COLORS[n], marker="o", markersize=3,
                        label=f"β_KL={b:g}")
        ax.set_xlabel("update (same prompts across forks)")
        ax.set_title(title, loc="left")
    axes[1][0].legend(fontsize=8)
    fig.suptitle(f"KL-pressure study: matched {cfg['fork_updates']}-update forks from the midpoint (ε={eps:g})",
                 x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- qualitative candidates
def reward_change_candidates(cfg, policy="standard", base="midpoint", k=6, ascending=False):
    """Held-out prompts ranked by reward change of `policy` over `base` (see common.rl_eval)."""
    return _rcc(res_dir(cfg), policy, base, k, ascending)


def high_reward_truncated(cfg, run="eps0.2_kl0", k=6):
    """Held-out responses that hit the length cap without EOS yet score highly (raw reward)."""
    rd = res_dir(cfg)
    rows = [r for r in read_jsonl_safe(rd / run / "generations_heldout.jsonl") if r["truncated"]]
    rows.sort(key=lambda r: r["reward_raw"], reverse=True)
    return pd.DataFrame([{"prompt_id": r["prompt_id"], "reward_raw": r["reward_raw"], "tokens": r["response_tokens"],
                          "prompt": _short(r["prompt"], 200), "response": _short(r["response"])} for r in rows[:k]])
