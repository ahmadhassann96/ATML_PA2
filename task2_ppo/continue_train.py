"""Task 2: PPO continuation from the supplied midpoint (policy LoRA + critic).

One update u:
  1. take prompts [u*P, (u+1)*P) of the shared prompt order (common.rollout.prompt_order);
  2. sample one response per prompt from the current policy (rollout cap = max_response_length);
  3. learned reward with the missing-EOS penalty; old-policy, reference and critic values on the sampled tokens;
  4. KL-shaped token rewards r_t = R*1[t=T] - beta_KL (log pi_old - log pi_ref), GAE(gamma, lambda) advantages
     and returns, advantages whitened over the batch's response tokens;
  5. `ppo_epochs` passes over the batch: clipped-surrogate policy step and value-MSE critic step.
Everything per update is appended to results/task2_ppo/<run>/train_log.jsonl; sampled rollouts to rollouts.jsonl.

    python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
    python -m task2_ppo.continue_train --run-name eps0.05_kl0.1 --updates 8 --clip-epsilon 0.05 --output outputs/task2_ppo/eps0.05_kl0.1
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import masked_mean
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    trainable_parameters,
    value_parameter_groups,
)
from common.rollout import (
    disable_dropout,
    float32_trainables,
    generate,
    prompt_order,
    reference_logprobs,
    score_rewards,
    token_logprobs,
    value_window,
)
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss

CHUNK = 4  # rollouts per forward/backward pass inside an optimisation epoch


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def explained_variance(values, returns, mask):
    v, r = values[mask.bool()], returns[mask.bool()]
    var_r = r.var(unbiased=False)
    return float(1.0 - (r - v).var(unbiased=False) / var_r) if var_r > 0 else float("nan")


def optimise_epoch(policy, value_model, roll, old, adv, returns, mask, cfg, eps, opt_p, opt_v, scaler_p, params_p, params_v):
    """One pass over the rollout batch: clipped policy step + critic step. Exact global token means are
    obtained by weighting each chunk's masked mean by its share of valid tokens."""
    B, total = mask.shape[0], mask.sum().clamp_min(1.0)
    max_norm = float(cfg.get("max_grad_norm", 1.0))
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "clip_fraction": 0.0, "ratio_max": 0.0, "ratio_min": float("inf"),
             "policy_change_k3": 0.0}
    policy.train()
    for i in range(0, B, CHUNK):
        rows = list(range(i, min(i + CHUNK, B)))
        m = mask[rows]
        w = float(m.sum() / total)
        new = token_logprobs(policy, roll, with_grad=True, rows=rows)["token_logp"]
        loss, ratio, clip_frac = ppo_policy_loss(new, old[rows], adv[rows], m, eps=eps)
        scaler_p.scale(loss * w).backward()
        log_ratio = torch.log(ratio.clamp_min(1e-12))
        stats["policy_loss"] += float(loss) * w
        stats["clip_fraction"] += float(clip_frac) * w
        stats["policy_change_k3"] += float(masked_mean((ratio - 1.0) - log_ratio, m)) * w
        stats["ratio_max"] = max(stats["ratio_max"], float(ratio[m.bool()].max()))
        stats["ratio_min"] = min(stats["ratio_min"], float(ratio[m.bool()].min()))
    scaler_p.unscale_(opt_p)
    stats["grad_norm_policy"] = float(torch.nn.utils.clip_grad_norm_(params_p, max_norm))
    scaler_p.step(opt_p)
    scaler_p.update()
    opt_p.zero_grad(set_to_none=True)
    policy.eval()

    value_model.train()
    vc = float(cfg.get("value_coef", 0.5))
    for i in range(0, B, CHUNK):
        rows = list(range(i, min(i + CHUNK, B)))
        m = mask[rows]
        w = float(m.sum() / total)
        v = value_window(value_model, roll, with_grad=True, rows=rows)
        vloss = value_mse_loss(v, returns[rows], m)
        (vc * vloss * w).backward()
        stats["value_loss"] += float(vloss) * w
    stats["grad_norm_value"] = float(torch.nn.utils.clip_grad_norm_(params_v, max_norm))
    opt_v.step()
    opt_v.zero_grad(set_to_none=True)
    value_model.eval()
    return stats


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    policy, value_model = bundle["policy"], bundle["value_model"]
    tok, rm, rtok = bundle["tokenizer"], bundle["reward_model"], bundle["reward_tokenizer"]
    opt_p, opt_v = bundle["policy_optimizer"], bundle["value_optimizer"]
    n_dropout = disable_dropout(policy) + disable_dropout(value_model)
    n_cast = float32_trainables(policy) + float32_trainables(value_model)
    params_p, params_v = trainable_parameters(policy), trainable_parameters(value_model)
    device = next(policy.parameters()).device
    scaler_p = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    U, P = int(cfg["updates"]), int(cfg.get("prompts_per_update", 1))
    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    gamma, lam = float(cfg.get("gamma", 1.0)), float(cfg.get("gae_lambda", 0.95))
    order = prompt_order(bundle["prompt_rows"], tok, int(cfg["max_prompt_length"]), int(cfg["seed"]))
    if U * P > len(order):
        raise ValueError(f"need {U * P} prompts, only {len(order)} fit max_prompt_length")

    res_dir = repo_path(cfg["results_dir"]) / run_name
    res_dir.mkdir(parents=True, exist_ok=True)
    for f in ("train_log.jsonl", "rollouts.jsonl"):
        (res_dir / f).unlink(missing_ok=True)
    save_json(res_dir / "run_config.json", {
        "run_name": run_name, "start_policy": cfg["paths"]["ppo_midpoint_policy"], "start_value": cfg["paths"]["ppo_midpoint_value"],
        "updates": U, "prompts_per_update": P, "ppo_epochs": int(cfg["ppo_epochs"]), "clip_epsilon": eps, "kl_beta": beta,
        "gamma": gamma, "gae_lambda": lam, "value_coef": float(cfg.get("value_coef", 0.5)),
        "policy_learning_rate": float(cfg["policy_learning_rate"]), "value_lora_learning_rate": float(cfg["value_lora_learning_rate"]),
        "value_head_learning_rate": float(cfg["value_head_learning_rate"]), "value_train_mode": cfg.get("value_train_mode"),
        "max_prompt_length": int(cfg["max_prompt_length"]), "max_response_length": int(cfg["max_response_length"]),
        "missing_eos_penalty": float(cfg.get("missing_eos_penalty", 0.0)), "reward_max_length": int(cfg.get("reward_max_length", 1024)),
        "advantage_whitening": "batch response tokens", "dropout_modules_disabled": n_dropout,
        "params_cast_to_fp32": n_cast, "generation": cfg["generation"], "seed": int(cfg["seed"]),
        "prompt_ids": [p["prompt_id"] for p in order[: U * P]], "output": str(out),
    })
    print(f"[{run_name}] PPO continuation: {U} updates x {P} prompt(s), {cfg['ppo_epochs']} epochs/update, "
          f"eps={eps}, beta_KL={beta}, cap {cfg['max_response_length']} tokens", flush=True)

    timer = wall_timer()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    gen_tokens = 0
    for u in range(U):
        batch = order[u * P:(u + 1) * P]
        policy.eval()
        value_model.eval()
        roll = generate(policy, tok, [b["messages"] for b in batch], cfg, int(cfg["max_response_length"]))
        raw, eff = score_rewards(rm, rtok, roll["messages"], roll["responses"], roll["truncated"], cfg)
        oldd = token_logprobs(policy, roll, entropy=True)
        old, mask = oldd["token_logp"], oldd["mask"]
        ref = reference_logprobs(policy, roll)["token_logp"]
        values = value_window(value_model, roll) * mask
        eff_d = eff.to(old.device)
        rewards = shaped_rewards(eff_d, old, ref, mask, beta)
        adv, returns = compute_gae(rewards, values, mask, gamma=gamma, lam=lam)
        adv_n = normalize_advantages(adv, mask)

        epochs = []
        for e in range(int(cfg["ppo_epochs"])):
            epochs.append(optimise_epoch(policy, value_model, roll, old, adv_n, returns, mask, cfg, eps,
                                         opt_p, opt_v, scaler_p, params_p, params_v))
        gen_tokens += int(mask.sum())
        n_tok = mask.sum(-1)
        kl_tok = (old - ref) * mask
        rec = {
            "update": u + 1,
            "prompt_ids": [b["prompt_id"] for b in batch],
            "reward_raw": float(raw.mean()), "reward": float(eff.mean()),
            "kl_token_mean": float(kl_tok.sum() / mask.sum()), "kl_seq_mean": float(kl_tok.sum(-1).mean()),
            "entropy_mean": float((oldd["entropy"] * mask).sum() / mask.sum()),
            "response_tokens": float(n_tok.mean()), "truncated_frac": float(np.mean(roll["truncated"])),
            "policy_loss": float(np.mean([s["policy_loss"] for s in epochs])),
            "value_loss": float(np.mean([s["value_loss"] for s in epochs])),
            "grad_norm_policy": float(np.mean([s["grad_norm_policy"] for s in epochs])),
            "grad_norm_value": float(np.mean([s["grad_norm_value"] for s in epochs])),
            "clip_fraction": float(np.mean([s["clip_fraction"] for s in epochs])),
            "clip_fraction_per_epoch": [s["clip_fraction"] for s in epochs],
            "ratio_max": max(s["ratio_max"] for s in epochs), "ratio_min": min(s["ratio_min"] for s in epochs),
            # policy movement between consecutive epochs (k3 estimator of KL(old||new)); 0 in epoch 1 by construction
            "policy_change_k3": float(epochs[-1]["policy_change_k3"]),
            "value_mean": float(values.sum() / mask.sum()), "return_mean": float((returns * mask).sum() / mask.sum()),
            "value_explained_variance": explained_variance(values, returns, mask),
            "value_return_corr": float(np.corrcoef(values[mask.bool()].cpu(), returns[mask.bool()].cpu())[0, 1])
            if int(mask.sum()) > 2 else float("nan"),
            "generated_tokens_cum": gen_tokens,
            "elapsed_s": round(timer(), 1),
        }
        if device.type == "cuda":
            rec["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        append_jsonl(res_dir / "train_log.jsonl", rec)
        for i, b in enumerate(batch):
            append_jsonl(res_dir / "rollouts.jsonl", {
                "update": u + 1, "prompt_id": b["prompt_id"], "prompt": b["messages"][-1]["content"],
                "response": roll["responses"][i], "response_tokens": roll["response_lengths"][i],
                "truncated": roll["truncated"][i], "reward_raw": float(raw[i]), "reward": float(eff[i])})
        eta = rec["elapsed_s"] / (u + 1) * (U - u - 1)
        print(f"[{run_name}] update {u + 1}/{U} reward={rec['reward_raw']:+.3f} kl_tok={rec['kl_token_mean']:+.4f} "
              f"pol_loss={rec['policy_loss']:+.4f} v_loss={rec['value_loss']:.3f} clip={rec['clip_fraction']:.3f} "
              f"len={rec['response_tokens']:.0f} |g|={rec['grad_norm_policy']:.3f} t={rec['elapsed_s']:.0f}s "
              f"eta={eta / 60:.1f}min", flush=True)
        del roll, oldd, old, ref, values, rewards, adv, adv_n, returns

    policy.save_pretrained(str(out))
    value_model.save_pretrained(str(out / "critic"))
    summary = {"run_name": run_name, "updates": U, "generated_tokens": gen_tokens, "wall_clock_s": round(timer(), 1),
               "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3) if device.type == "cuda" else None,
               "adapter": str(out)}
    save_json(res_dir / "train_summary.json", summary)
    print(f"[{run_name}] saved policy adapter to {out} ({summary['wall_clock_s'] / 60:.1f} min, "
          f"peak VRAM {summary['peak_vram_gib']} GiB)", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
