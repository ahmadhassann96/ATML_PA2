from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.logprobs import response_logprobs
from common.models import load_policy, load_tokenizer, reference_mode, trainable_parameters
from task1_dpo.dpo import dpo_loss

# Leave at least this many response tokens after the prompt; longer prompts are dropped (and logged).
MIN_RESPONSE_TOKENS = 32


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected), [r.get("prompt_id") for r in rows]
    return collate


def filter_long_prompts(rows, tokenizer, max_length):
    """encode_prompt_response raises if the prompt alone does not fit. Drop such rows up front
    (identically for every condition, because the filter depends only on the data and max_length)."""
    kept, dropped = [], []
    for row in rows:
        n = len(tokenizer.apply_chat_template(prompt_messages_from_preference(row), tokenize=True, add_generation_prompt=True))
        (kept if n <= max_length - MIN_RESPONSE_TOKENS else dropped).append(row)
    return kept, dropped


def pair_logps(model, chosen, rejected):
    """Summed response log-probs for chosen and rejected batches under the current model state."""
    c = response_logprobs(model, chosen["input_ids"], chosen["attention_mask"], chosen["response_mask"])
    r = response_logprobs(model, rejected["input_ids"], rejected["attention_mask"], rejected["response_mask"])
    return c["seq_logp"], r["seq_logp"], c["mask"].sum(-1), r["mask"].sum(-1)


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    rows, dropped = filter_long_prompts(rows, tokenizer, int(cfg["max_sequence_length"]))
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    generator = torch.Generator().manual_seed(int(cfg["seed"]))  # fixed shuffle order across conditions
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=generator,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "dataset_path": str(path),
        "rows": rows,
        "dropped": dropped,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg, model, loader, optimizer = bundle["cfg"], bundle["model"], bundle["loader"], bundle["optimizer"]
    beta = bundle["beta"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg["results_dir"]) / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    device = next(model.parameters()).device
    accum = int(cfg["grad_accum_steps"])
    epochs = int(cfg.get("epochs", 1))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    micro_per_epoch = len(loader)
    total_updates = epochs * math.ceil(micro_per_epoch / accum)
    params = trainable_parameters(model)
    # The frozen base runs in float16 while LoRA weights are float32; scale the loss so small
    # gradients flowing back through fp16 activations do not underflow (no-op on CPU / bf16).
    use_scaler = (device.type == "cuda" and next(model.parameters()).dtype == torch.float16
                  and all(p.dtype == torch.float32 for p in params))
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    save_json(results_dir / "run_config.json", {
        "run_name": run_name,
        "dataset": bundle["dataset_path"],
        "n_train_pairs": len(bundle["rows"]),
        "dropped_long_prompt_ids": [r.get("prompt_id") for r in bundle["dropped"]],
        "beta": beta,
        "epochs": epochs,
        "batch_size": int(cfg["batch_size"]),
        "grad_accum_steps": accum,
        "effective_batch_pairs": int(cfg["batch_size"]) * accum,
        "optimizer_updates": total_updates,
        "learning_rate": float(cfg["learning_rate"]),
        "lr_schedule": "constant",
        "max_sequence_length": int(cfg["max_sequence_length"]),
        "max_grad_norm": max_grad_norm,
        "lora": cfg["lora"],
        "seed": int(cfg["seed"]),
        "dtype": cfg.get("dtype"),
        "grad_scaler": use_scaler,
        "output_adapter": str(output),
    })
    print(f"[{run_name}] {len(bundle['rows'])} pairs ({len(bundle['dropped'])} dropped for long prompts), "
          f"beta={beta}, {micro_per_epoch} micro-batches/epoch, {total_updates} optimizer updates")

    elapsed = wall_timer()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    update, window = 0, []
    order = []

    for epoch in range(epochs):
        for step, (chosen, rejected, prompt_ids) in enumerate(loader):
            order.extend(prompt_ids)
            chosen, rejected = to_device(chosen, device), to_device(rejected, device)

            # Reference log-probs: same network with the LoRA adapter disabled == frozen initial policy.
            with torch.no_grad(), reference_mode(model):
                ref_c, ref_r, _, _ = pair_logps(model, chosen, rejected)
            pol_c, pol_r, len_c, len_r = pair_logps(model, chosen, rejected)

            loss, stats = dpo_loss(pol_c, pol_r, ref_c.float(), ref_r.float(), beta)
            scaler.scale(loss / accum).backward()

            window.append({
                "loss": loss.item(),
                "acc": stats["preference_accuracy"].item(),
                "margin": (stats["logit_mean"].item() / beta),
                "reward_chosen": (beta * (pol_c - ref_c)).mean().item(),
                "reward_rejected": (beta * (pol_r - ref_r)).mean().item(),
                "len_chosen": len_c.float().mean().item(),
                "len_rejected": len_r.float().mean().item(),
            })

            last = step + 1 == micro_per_epoch
            if (step + 1) % accum == 0 or last:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm).item()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                rec = {k: sum(w[k] for w in window) / len(window) for k in window[0]}
                rec.update({
                    "update": update,
                    "epoch": epoch,
                    "micro_batches": len(window),
                    "grad_norm": grad_norm,
                    "lr": optimizer.param_groups[0]["lr"],
                    "elapsed_s": round(elapsed(), 1),
                })
                if device.type == "cuda":
                    rec["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
                append_jsonl(log_path, rec)
                window = []
                if update == 1 or update % 5 == 0 or update == total_updates:
                    eta = rec["elapsed_s"] / update * (total_updates - update)
                    print(f"[{run_name}] update {update}/{total_updates} loss={rec['loss']:.4f} acc={rec['acc']:.3f} "
                          f"margin={rec['margin']:.3f} |g|={grad_norm:.3f} t={rec['elapsed_s']:.0f}s eta={eta / 60:.1f}min", flush=True)

    model.save_pretrained(str(output))
    save_json(results_dir / "train_order_prompt_ids.json", order)
    summary = {
        "run_name": run_name,
        "updates": update,
        "wall_clock_s": round(elapsed(), 1),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3) if device.type == "cuda" else None,
        "adapter": str(output),
    }
    save_json(results_dir / "train_summary.json", summary)
    print(f"[{run_name}] saved adapter to {output} ({summary['wall_clock_s'] / 60:.1f} min)")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
