"""Task 3: GRPO continuation from the supplied midpoint adapter (critic-free).

One update u:
  1. take prompts [u*P, (u+1)*P) of the shared prompt order (common.rollout.prompt_order);
  2. sample K = num_generations completions per prompt (cap max_completion_length);
  3. learned reward per completion; group-relative advantages A_k = (r_k - mean_group) / (std_group + eps);
  4. completions that hit the cap without EOS are masked out of the loss (mask_truncated_completions) but still
     count in their group's mean/std;
  5. one clipped-surrogate step (policy_epochs=1, so pi_old = pi_theta and the ratio is exactly 1) with the
     k3 KL penalty to the frozen reference, normalised per sequence by 1/T_k ("grpo") or by the constant
     max_completion_length ("dr_grpo", Dr. GRPO-style).
Per-update diagnostics go to results/task3_grpo/<run>/train_log.jsonl, per-completion records (length, reward,
advantage, per-token loss weight, optional per-completion gradient norm) to completions.jsonl.

    python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
    python -m task3_grpo.continue_train --run-name norm_dr_grpo --updates 8 --loss-type dr_grpo --per-seq-grad
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.models import load_policy, load_reward_model, load_tokenizer, trainable_parameters
from common.rollout import (
    disable_dropout,
    float32_trainables,
    generate,
    prompt_order,
    reference_logprobs,
    score_rewards,
    token_logprobs,
)
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences

ZERO_STD_TOL = 1e-6  # same eps as group_relative_advantages: a group with std below this has no relative signal


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def norm_denominator(loss_type, lengths, max_completion_length):
    """Per-sequence normaliser of the summed token objective: realised length T_k (GRPO) or constant L (Dr. GRPO)."""
    if loss_type == "grpo":
        return lengths.clamp_min(1.0)
    return torch.full_like(lengths, float(max_completion_length))


def per_sequence_grad_norms(policy, roll, adv, loss_mask, loss_type, L, params):
    """Gradient norm of each completion's own contribution to the policy term of the batch loss,
    -(1/B) * sum_t min(rho A, clip(rho) A) / norm_k, at rho = 1 (on-policy). Costs one extra forward/backward per
    completion; used by the normalisation study to measure how gradient magnitude is allocated across lengths."""
    B = loss_mask.shape[0]
    out = []
    for k in range(B):
        if float(loss_mask[k].sum()) == 0:
            out.append(0.0)
            continue
        lp = token_logprobs(policy, roll, with_grad=True, rows=[k])["token_logp"]
        m = loss_mask[k:k + 1]
        ratio = torch.exp(lp - lp.detach())                                   # = 1, carries the gradient of log pi
        obj = (ratio * adv[k]) * m
        denom = norm_denominator(loss_type, m.sum(-1), L)
        contrib = -(obj.sum(-1) / denom).sum() / B
        (contrib * 1024.0).backward()                                         # scaled against fp16 underflow
        g = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None)) / 1024.0
        out.append(float(g))
        for p in params:
            p.grad = None
    return out


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo",
             run_name: str = "standard", per_seq_grad: bool = False):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    policy, tok, rm, rtok, opt = (bundle[k] for k in ("policy", "tokenizer", "reward_model", "reward_tokenizer", "optimizer"))
    n_dropout = disable_dropout(policy)
    float32_trainables(policy)
    params = trainable_parameters(policy)
    device = next(policy.parameters()).device
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    U, P, K = int(cfg["updates"]), int(cfg.get("prompts_per_update", 1)), int(cfg["num_generations"])
    eps, beta, L = float(cfg["clip_epsilon"]), float(cfg["kl_beta"]), int(cfg["max_completion_length"])
    if int(cfg.get("policy_epochs", 1)) != 1:
        raise NotImplementedError("this continuation implements the released policy_epochs=1 setting")
    order = prompt_order(bundle["prompt_rows"], tok, int(cfg["max_prompt_length"]), int(cfg["seed"]))
    res_dir = repo_path(cfg["results_dir"]) / run_name
    res_dir.mkdir(parents=True, exist_ok=True)
    for f in ("train_log.jsonl", "completions.jsonl"):
        (res_dir / f).unlink(missing_ok=True)
    save_json(res_dir / "run_config.json", {
        "run_name": run_name, "start_policy": cfg["paths"]["grpo_midpoint_policy"], "loss_type": loss_type,
        "updates": U, "prompts_per_update": P, "num_generations": K, "policy_epochs": 1, "clip_epsilon": eps,
        "kl_beta": beta, "kl_estimator": "k3: exp(ref-new) - (ref-new) - 1, token mean over loss mask",
        "learning_rate": float(cfg["learning_rate"]), "max_prompt_length": int(cfg["max_prompt_length"]),
        "max_completion_length": L, "mask_truncated_completions": bool(cfg.get("mask_truncated_completions", True)),
        "advantage": "(r - group mean) / (group population std + 1e-6); truncated completions kept in group statistics",
        "zero_std_tolerance": ZERO_STD_TOL, "dropout_modules_disabled": n_dropout, "per_seq_grad": per_seq_grad,
        "generation": cfg["generation"], "seed": int(cfg["seed"]), "prompt_ids": [p["prompt_id"] for p in order[: U * P]],
        "output": str(out)})
    print(f"[{run_name}] GRPO continuation ({loss_type}): {U} updates x {P} prompt(s) x K={K}, eps={eps}, "
          f"beta_KL={beta}, cap {L} tokens{', per-completion grad norms' if per_seq_grad else ''}", flush=True)

    timer = wall_timer()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    gen_tokens = 0
    for u in range(U):
        batch = order[u * P:(u + 1) * P]
        policy.eval()
        roll = generate(policy, tok, [b["messages"] for b in batch], cfg, L, n_samples=K)
        raw, _ = score_rewards(rm, rtok, roll["messages"], roll["responses"], roll["truncated"], {**cfg, "missing_eos_penalty": 0.0})
        groups = torch.arange(P).repeat_interleave(K)
        adv = group_relative_advantages(raw, groups).to(device)
        ref = reference_logprobs(policy, roll)["token_logp"]
        gen_mask = roll["response_mask"].to(device)                       # all generated tokens
        loss_mask = mask_truncated_sequences(gen_mask, roll["truncated"]) if cfg.get("mask_truncated_completions", True) else gen_mask
        lengths = gen_mask.sum(-1)

        seq_g = per_sequence_grad_norms(policy, roll, adv, loss_mask, loss_type, L, params) if per_seq_grad else None

        policy.train()
        new = token_logprobs(policy, roll, with_grad=True, entropy=True)
        lp = new["token_logp"]
        loss, stats = grpo_policy_loss(lp, lp.detach(), adv, loss_mask, ref, eps, beta, loss_type=loss_type,
                                       max_completion_length=L)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = float(torch.nn.utils.clip_grad_norm_(params, float(cfg.get("max_grad_norm", 1.0))))
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        policy.eval()

        r = raw.view(P, K)
        g_std = r.std(-1, unbiased=False)
        gen_tokens += int(lengths.sum())
        kl_all = ((lp.detach() - ref) * gen_mask)
        denom = norm_denominator(loss_type, loss_mask.sum(-1), L)
        rec = {
            "update": u + 1, "loss_type": loss_type, "prompt_ids": [b["prompt_id"] for b in batch],
            "reward_raw": float(raw.mean()), "reward_max": float(raw.max()), "reward_min": float(raw.min()),
            "group_reward_std_mean": float(g_std.mean()),
            "frac_zero_std_groups": float((g_std < ZERO_STD_TOL).float().mean()),
            "kl_token_mean": float(kl_all.sum() / gen_mask.sum()), "kl_seq_mean": float(kl_all.sum(-1).mean()),
            "kl_k3_loss": float(stats["sampled_kl"]), "loss": float(loss), "policy_term": float(stats["policy_term"]),
            "grad_norm": gn, "clip_fraction": float(stats["clip_fraction"]),
            "entropy_mean": float((new["entropy"] * gen_mask).sum() / gen_mask.sum()),
            "response_tokens": float(lengths.mean()), "truncated_frac": float(np.mean(roll["truncated"])),
            "masked_completion_frac": float((loss_mask.sum(-1) == 0).float().mean()),
            "generated_tokens_cum": gen_tokens, "elapsed_s": round(timer(), 1),
        }
        if seq_g is not None:
            rec["per_seq_grad_norm_mean"] = float(np.mean(seq_g))
        if device.type == "cuda":
            rec["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        append_jsonl(res_dir / "train_log.jsonl", rec)
        for i in range(P * K):
            append_jsonl(res_dir / "completions.jsonl", {
                "update": u + 1, "prompt_id": batch[i // K]["prompt_id"], "sample": i % K,
                "response_tokens": int(lengths[i]), "truncated": bool(roll["truncated"][i]),
                "in_loss": bool(loss_mask[i].sum() > 0), "reward": float(raw[i]), "advantage": float(adv[i]),
                # weight each of this completion's tokens receives in the policy term (|A| / normaliser / B)
                "per_token_weight": float(abs(adv[i]) / denom[i] / (P * K)) if loss_mask[i].sum() > 0 else 0.0,
                "seq_grad_norm": seq_g[i] if seq_g is not None else None,
                "response": roll["responses"][i]})
        eta = rec["elapsed_s"] / (u + 1) * (U - u - 1)
        print(f"[{run_name}] update {u + 1}/{U} reward={rec['reward_raw']:+.3f} grp_std={rec['group_reward_std_mean']:.3f} "
              f"zero_std={rec['frac_zero_std_groups']:.2f} kl_tok={rec['kl_token_mean']:+.4f} loss={rec['loss']:+.4f} "
              f"|g|={gn:.3f} len={rec['response_tokens']:.0f} trunc={rec['truncated_frac']:.2f} t={rec['elapsed_s']:.0f}s "
              f"eta={eta / 60:.1f}min", flush=True)
        del roll, new, lp, ref, loss

    policy.save_pretrained(str(out))
    summary = {"run_name": run_name, "loss_type": loss_type, "updates": U, "generated_tokens": gen_tokens,
               "wall_clock_s": round(timer(), 1),
               "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3) if device.type == "cuda" else None,
               "adapter": str(out)}
    save_json(res_dir / "train_summary.json", summary)
    print(f"[{run_name}] saved adapter to {out} ({summary['wall_clock_s'] / 60:.1f} min, peak VRAM {summary['peak_vram_gib']} GiB)",
          flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--per-seq-grad", action="store_true", help="also measure each completion's gradient norm")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name, args.per_seq_grad)


if __name__ == "__main__":
    main()
