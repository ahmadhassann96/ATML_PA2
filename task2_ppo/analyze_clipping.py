"""Task 2 Step 2: clipping study.

Part A - supplied cached rollout batch (no new generation):
  * rebuilds the 32 cached prompt+response token sequences and checks them against the cache
    (recomputed reference log-probs and critic values must match the cached ones);
  * advantages from the cached quantities: KL-shaped rewards (cached old/ref log-probs, effective reward,
    beta_KL from the config), cached critic values, GAE(gamma, lambda), whitened over the batch;
  * for every epsilon: the ratio rho_t = pi_theta / pi_old of the supplied midpoint policy against the cached
    old log-probs, and then again after each of `ppo_epochs` clipped-surrogate optimisation steps on this batch
    (each epsilon starts from the identical midpoint weights). Reported per point:
      clip_fraction     - share of valid tokens with rho outside [1-eps, 1+eps]          (manual definition)
      affected_fraction - share of valid tokens where the clip binds, i.e. min() selects the clipped term
                          and the token's gradient is zeroed: (A>0 and rho>1+eps) or (A<0 and rho<1-eps)
      surrogate_unclipped / surrogate_clipped - mean rho*A and mean min(rho*A, clip(rho)*A)
Part B - matched short continuation forks from the same midpoint for every epsilon (beta_KL fixed), each
  evaluated with the common held-out protocol; stability statistics come from the fork training logs.

    python -m task2_ppo.analyze_clipping --config configs/ppo.yaml
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json, set_seed
from common.metrics import masked_mean
from common.models import load_policy, load_tokenizer, load_value_model, reference_mode, trainable_parameters
from common.rollout import disable_dropout, float32_trainables, token_logprobs, value_window
from common.runs import adapter_exists, load_json_if, run, write_csv
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards

CHUNK = 4
# A rollout counts as misaligned (different tokenisation of the same text) only if its recomputed reference log-probs
# disagree with the cached ones far beyond numerical-precision noise: precision noise between this float16 run and the
# course's cache is ~0.02 nats per token on average with occasional ~0.1-0.3 outliers; a token-shifted sequence is
# off by several nats on most tokens.
ALIGN_MEAN_TOL = 0.1
ALIGN_MAX_TOL = 2.0


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def fork_name(eps: float, beta: float) -> str:
    return f"eps{eps:g}_kl{beta:g}"


# ----------------------------------------------------------------------------- Part A helpers
def build_cached_batch(cfg, tok, rows):
    """Left-padded prompt + right-padded response token batch reproducing the cached rollouts."""
    prompts = {}
    for key in ("rl_prompt_eval", "rl_prompt_train"):
        for r in read_jsonl(cfg["paths"][key]):
            prompts.setdefault(r["prompt_id"], r["messages"])
    kept, skipped = [], []
    for r in rows:
        msgs = prompts.get(r.get("prompt_id"))
        T = int(r["old_logprobs"].numel())
        base_ids = tok(r["response"], add_special_tokens=False)["input_ids"]
        eos = [tok.eos_token_id] if tok.eos_token_id is not None else []
        variants = [base_ids + eos, base_ids] if r.get("terminated_with_eos") else [base_ids, base_ids + eos]
        ids = next((v for v in variants if len(v) == T), variants[0])
        if msgs is None or len(ids) != T:
            skipped.append({"source_index": r["source_index"], "cached_tokens": T, "retokenized": len(ids),
                            "prompt_found": msgs is not None})
            continue
        kept.append((r, tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True), ids))
    W = max(len(p) for _, p, _ in kept)
    R = max(len(x) for _, _, x in kept)
    pad = tok.pad_token_id
    seq, attn, fmask = [], [], []
    for _, p, x in kept:
        seq.append([pad] * (W - len(p)) + p + x + [pad] * (R - len(x)))
        attn.append([0] * (W - len(p)) + [1] * (len(p) + len(x)) + [0] * (R - len(x)))
        fmask.append([0] * W + [1] * len(x) + [0] * (R - len(x)))

    def stack(key, fill=0.0):
        return torch.stack([torch.cat([r[key].float(), torch.full((R - r[key].numel(),), fill)]) for r, _, _ in kept])

    roll = {"sequences": torch.tensor(seq), "attention_mask": torch.tensor(attn),
            "full_response_mask": torch.tensor(fmask, dtype=torch.float32), "prompt_width": W}
    cache = {"old": stack("old_logprobs"), "ref": stack("ref_logprobs"),
             "values": stack("values") if "values" in kept[0][0] else None,
             "mask": torch.tensor([[1.0] * len(x) + [0.0] * (R - len(x)) for _, _, x in kept]),
             "reward": torch.tensor([float(r.get("effective_terminal_reward", r.get("raw_terminal_reward"))) for r, _, _ in kept]),
             "source_index": [r["source_index"] for r, _, _ in kept]}
    return roll, cache, skipped


def to(roll, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in roll.items()}


def clip_metrics(new_logp, old_logp, adv, mask, eps):
    ratio = torch.exp(new_logp - old_logp)
    m = mask.bool()
    outside = (ratio < 1 - eps) | (ratio > 1 + eps)
    binds = ((adv > 0) & (ratio > 1 + eps)) | ((adv < 0) & (ratio < 1 - eps))
    s1 = ratio * adv
    s2 = ratio.clamp(1 - eps, 1 + eps) * adv
    lr = (new_logp - old_logp)[m]
    return {
        "clip_fraction": float(outside[m].float().mean()),
        "affected_fraction": float(binds[m].float().mean()),
        "surrogate_unclipped": float(masked_mean(s1, mask)),
        "surrogate_clipped": float(masked_mean(torch.minimum(s1, s2), mask)),
        "ratio_mean": float(ratio[m].mean()), "ratio_max": float(ratio[m].max()), "ratio_min": float(ratio[m].min()),
        "log_ratio_abs_mean": float(lr.abs().mean()),
        "kl_k3_old_to_new": float((torch.exp(-lr) - 1 + lr).mean()),
    }


def all_token_logp(policy, roll, mask):
    out = token_logprobs(policy, roll)["token_logp"]
    return out * mask


def cached_batch_study(cfg, out_dir):
    set_seed(int(cfg["seed"]))
    tok = load_tokenizer(cfg["base_model"])
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    roll, cache, skipped = build_cached_batch(cfg, tok, rows)
    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
    disable_dropout(policy)
    float32_trainables(policy)
    policy.eval()
    device = next(policy.parameters()).device
    roll = to(roll, device)
    mask = cache["mask"].to(device)
    old, ref_c = cache["old"].to(device) * mask, cache["ref"].to(device) * mask

    # --- validation of the reconstructed batch against the cache
    new0 = all_token_logp(policy, roll, mask)
    with reference_mode(policy):
        ref_now = all_token_logp(policy, roll, mask)
    # a same-length but differently tokenised response would silently misalign the cached per-token values:
    # drop any rollout whose recomputed reference log-probs disagree with the cache
    absdiff = (ref_now - ref_c).abs() * mask
    row_max = absdiff.amax(-1)
    row_mean = absdiff.sum(-1) / mask.sum(-1).clamp_min(1.0)
    bad = [i for i in range(mask.shape[0]) if float(row_mean[i]) > ALIGN_MEAN_TOL or float(row_max[i]) > ALIGN_MAX_TOL]
    misaligned = [cache["source_index"][i] for i in bad]
    for i in bad:
        mask[i] = 0.0
    old, ref_c, new0, ref_now = old * mask, ref_c * mask, new0 * mask, ref_now * mask
    m = mask.bool()
    validation = {
        "n_rollouts": len(cache["source_index"]) - len(misaligned), "n_tokens": int(mask.sum()), "skipped": skipped,
        "dropped_misaligned": misaligned,
        "alignment_rule": f"drop a rollout if mean |ref diff| > {ALIGN_MEAN_TOL} or max |ref diff| > {ALIGN_MAX_TOL}",
        "ref_logp_abs_diff_mean": float((ref_now - ref_c)[m].abs().mean()),
        "ref_logp_abs_diff_max": float((ref_now - ref_c)[m].abs().max()),
        "midpoint_vs_cached_old_abs_diff_mean": float((new0 - old)[m].abs().mean()),
        "midpoint_vs_cached_old_abs_diff_max": float((new0 - old)[m].abs().max()),
    }
    values = cache["values"].to(device) * mask if cache["values"] is not None else None
    try:
        critic = load_value_model(cfg, cfg["paths"]["ppo_midpoint_value"], train_mode="frozen")
        v_now = value_window(critic, roll) * mask
        if values is not None:
            validation["critic_value_abs_diff_mean"] = float((v_now - values)[m].abs().mean())
            validation["critic_value_abs_diff_max"] = float((v_now - values)[m].abs().max())
        else:
            values = v_now
        del critic
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as exc:  # validation only; the cached values are what the study uses
        validation["critic_check_error"] = repr(exc)
    print(f"[clip] cached batch: {validation['n_rollouts']} rollouts / {validation['n_tokens']} response tokens used "
          f"({len(skipped)} skipped: re-tokenisation length mismatch or missing prompt; {len(misaligned)} dropped as misaligned)",
          flush=True)
    print("[clip] validation:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in validation.items() if k != "skipped"}, flush=True)

    # --- advantages from cached quantities
    beta, gamma, lam = float(cfg["kl_beta"]), float(cfg.get("gamma", 1.0)), float(cfg.get("gae_lambda", 0.95))
    rewards = shaped_rewards(cache["reward"].to(device), old, ref_c, mask, beta)
    adv, _ = compute_gae(rewards, values, mask, gamma=gamma, lam=lam)
    adv = normalize_advantages(adv, mask)

    # --- per-epsilon geometry at the midpoint and after optimisation steps on this batch
    start = {n: p.detach().clone() for n, p in policy.named_parameters() if p.requires_grad}
    results, log_ratios = [], {"midpoint": (new0 - old)[m].float().cpu().numpy()}
    for eps in [float(e) for e in cfg["clip_values"]]:
        with torch.no_grad():
            for n, p in policy.named_parameters():
                if n in start:
                    p.copy_(start[n])
        params = trainable_parameters(policy)
        opt = AdamW(params, lr=float(cfg["policy_learning_rate"]))
        scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 12, enabled=device.type == "cuda")
        noise_ratio = torch.exp(ref_now - ref_c)
        noise_out = ((noise_ratio < 1 - eps) | (noise_ratio > 1 + eps))[m].float().mean()
        point = {"epsilon": eps, "step": 0, **clip_metrics(new0, old, adv, mask, eps),
                 # share of tokens pushed outside the band by numerical precision alone (identical reference model)
                 "noise_floor_clip_fraction": float(noise_out)}
        results.append(point)
        total = mask.sum()
        for step in range(1, int(cfg["ppo_epochs"]) + 1):
            retries = 0
            while True:
                policy.train()
                for i in range(0, mask.shape[0], CHUNK):
                    rows_i = list(range(i, min(i + CHUNK, mask.shape[0])))
                    mi = mask[rows_i]
                    if float(mi.sum()) == 0:
                        continue
                    new = token_logprobs(policy, roll, with_grad=True, rows=rows_i)["token_logp"]
                    loss, _, _ = ppo_policy_loss(new, old[rows_i], adv[rows_i], mi, eps=eps)
                    scaler.scale(loss * (mi.sum() / total)).backward()
                scaler.unscale_(opt)
                gn = float(torch.nn.utils.clip_grad_norm_(params, float(cfg.get("max_grad_norm", 1.0))))
                scaler.step(opt)      # skipped internally if the scaled gradients overflowed
                scaler.update()       # ...in which case the scale is halved
                opt.zero_grad(set_to_none=True)
                if np.isfinite(gn) or retries >= 8:
                    break
                retries += 1          # redo the pass so every epsilon gets the same number of real steps
            policy.eval()
            new_s = all_token_logp(policy, roll, mask)
            results.append({"epsilon": eps, "step": step, "grad_norm": gn, "overflow_retries": retries,
                            **clip_metrics(new_s, old, adv, mask, eps)})
            log_ratios[f"eps{eps:g}_step{step}"] = (new_s - old)[m].float().cpu().numpy()
        r = results[-1]
        print(f"[clip] eps={eps:g}: midpoint clip={point['clip_fraction']:.4f} affected={point['affected_fraction']:.4f} | "
              f"after {cfg['ppo_epochs']} steps clip={r['clip_fraction']:.4f} affected={r['affected_fraction']:.4f} "
              f"surrogate {r['surrogate_unclipped']:+.4f} -> clipped {r['surrogate_clipped']:+.4f}", flush=True)

    save_json(out_dir / "clipping_cached.json", {
        "definition": {"clip_fraction": "share of valid tokens with ratio outside [1-eps, 1+eps]",
                       "affected_fraction": "share of valid tokens where min() selects the clipped term (zero gradient)",
                       "advantages": "GAE on KL-shaped cached rewards with cached critic values, whitened over batch tokens",
                       "steps": "clipped-surrogate AdamW steps on the cached batch from the identical midpoint weights"},
        "kl_beta": beta, "policy_learning_rate": float(cfg["policy_learning_rate"]),
        "validation": validation, "results": results})
    np.savez_compressed(out_dir / "clipping_cached_log_ratios.npz", **log_ratios)
    return results


# ----------------------------------------------------------------------------- summary
def fork_stability(res_dir, name):
    from common.live import read_jsonl_safe
    log = read_jsonl_safe(res_dir / name / "train_log.jsonl")
    if not log:
        return {}
    pc = [r["policy_change_k3"] for r in log]
    gn = [r["grad_norm_policy"] for r in log]
    rw = [r["reward_raw"] for r in log]
    return {"train_updates": len(log), "train_generated_tokens": log[-1]["generated_tokens_cum"],
            "train_clip_fraction_mean": float(np.mean([r["clip_fraction"] for r in log])),
            "policy_change_k3_mean": float(np.mean(pc)), "policy_change_k3_max": float(np.max(pc)),
            "grad_norm_mean": float(np.mean(gn)), "grad_norm_std": float(np.std(gn)),
            "train_reward_mean": float(np.mean(rw)), "train_kl_final": log[-1]["kl_token_mean"],
            "value_loss_mean": float(np.mean([r["value_loss"] for r in log]))}


def heldout_row(res_dir, name):
    ev = load_json_if(res_dir / name / "eval_heldout.json")
    if not ev:
        return {}
    return {"heldout_reward_raw": ev["reward_raw_mean"], "heldout_reward_raw_ci_lo": ev["reward_raw_ci95"][0],
            "heldout_reward_raw_ci_hi": ev["reward_raw_ci95"][1], "heldout_reward_eos_penalised": ev["reward_mean"],
            "heldout_kl_token": ev["kl_token_mean"], "heldout_kl_seq": ev["kl_seq_mean"],
            "heldout_entropy": ev["entropy_mean"], "heldout_len_mean": ev["response_tokens"]["mean"],
            "heldout_len_std": ev["response_tokens"]["std"], "heldout_truncation": ev["truncation_rate"]}


def summarize(cfg):
    res_dir = repo_path(cfg["results_dir"])
    cached = load_json_if(res_dir / "clipping_cached.json") or {"results": []}
    beta = float(cfg["kl_beta"])
    rows = []
    for eps in [float(e) for e in cfg["clip_values"]]:
        pts = [r for r in cached["results"] if r["epsilon"] == eps]
        p0 = next((r for r in pts if r["step"] == 0), {})
        pN = max(pts, key=lambda r: r["step"]) if pts else {}
        name = fork_name(eps, beta)
        rows.append({"epsilon": eps, "fork": name,
                     "cached_clip_fraction_midpoint": p0.get("clip_fraction"),
                     "cached_noise_floor_clip_fraction": p0.get("noise_floor_clip_fraction"),
                     "cached_affected_fraction_midpoint": p0.get("affected_fraction"),
                     "cached_clip_fraction_after_steps": pN.get("clip_fraction"),
                     "cached_affected_fraction_after_steps": pN.get("affected_fraction"),
                     "cached_surrogate_clipped_after_steps": pN.get("surrogate_clipped"),
                     "cached_kl_k3_after_steps": pN.get("kl_k3_old_to_new"),
                     **fork_stability(res_dir, name), **heldout_row(res_dir, name)})
    write_csv(res_dir / "summary_clipping.csv", rows)
    save_json(res_dir / "summary_clipping.json", rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--cached-only", action="store_true", help="only Part A (no forks)")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res_dir = repo_path(cfg["results_dir"])
    res_dir.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        if not (res_dir / "clipping_cached.json").exists():
            cached_batch_study(cfg, res_dir)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if not args.cached_only:
            beta, n = float(cfg["kl_beta"]), int(cfg["fork_updates"])
            for eps in [float(e) for e in cfg["clip_values"]]:
                name = fork_name(eps, beta)
                out = f"outputs/task2_ppo/{name}"
                if not adapter_exists(out):
                    run("task2_ppo.continue_train", "--config", args.config, "--run-name", name, "--output", out,
                        "--updates", n, "--clip-epsilon", eps, "--kl-beta", beta)
                if not (res_dir / name / "eval_heldout.json").exists():
                    run("task2_ppo.evaluate", "--config", args.config, "--adapter", out, "--name", name)
    summarize(cfg)


if __name__ == "__main__":
    main()
