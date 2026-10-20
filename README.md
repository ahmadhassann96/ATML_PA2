# ATML PA2 - LLM Post-Training

<!-- FINAL_STUDENT_SETUP -->

## Quick start

```bash
git clone https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining.git
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
```

The fixed datasets, cached diagnostics, and supplied
continuation checkpoints are downloaded from:

https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets

Pinned release revision:

`0b350481fb03f5525a35bcdec4131bd4fe487f98`

---
# ATML PA2 - LLM Post-Training

This is the **student starter repository** for ATML PA2. The released code is intentionally incomplete: Tasks 1-3 provide model/data loading, objective helpers, checkpoint restoration, and experiment entry points, but **you must implement the training loops and ablation orchestration yourself**. Each of Tasks 1-3 also contains one deliberate algorithmic defect in its core objective code; identifying and correcting these defects is part of validating your implementation.

Task 4 supplies the fixed AI safety judge and response-generation utilities, but you must write the evaluation/aggregation code. Task 5 supplies the exact RLVR verifier, the fixed pairwise AI judge used for RLAIF evaluation, and data/model loaders; you must implement the requested evaluation and analysis.

## 1. Clone and install

```bash
git clone https://github.com/COURSE_ORG/ATML-PA2-LLM-PostTraining.git
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
```

## 2. Download the course assets

The large course-created checkpoints and fixed data are distributed as a GitHub Release asset rather than normal Git files. After cloning, run:

```bash
python -m scripts.download_assets
python -m scripts.validate_assets
```

If your instructor provides a direct asset URL separately, use:

```bash
python -m scripts.download_assets --url '<ASSET_URL>'
```

Public base/reward/judge models are downloaded from Hugging Face at runtime and are **not** included in the course asset archive.

The installer also materializes the fixed 100-example Task 5 transfer set from the official SVAMP challenge-set source if it is not already present. The tiny Task 1 word-limit prompt set is tracked directly in this repository.

## 3. Environment check

```bash
python -m scripts.check_environment
```

Run commands from the repository root. The reference environment used to prepare the release pins Transformers 4.57.1, TRL 0.27.2, PEFT 0.17.1, and Tokenizers 0.22.1.

## 4. Supplied course checkpoints

After `download_assets`, these directories should exist:

```text
checkpoints/ppo_midpoint_policy/
checkpoints/ppo_midpoint_value/
checkpoints/grpo_midpoint_policy/
checkpoints/rlvr_policy/
checkpoints/rlaif_policy/
```

PPO and GRPO begin from the supplied continuation checkpoints. RLVR and RLAIF are supplied frozen evaluation policies; students do not retrain them.

The PPO value checkpoint is intentionally released as the exact staff midpoint state, including its imperfect held-out value calibration. Treat critic behavior as an analysis variable rather than assuming a perfect baseline, and start every PPO fork from the identical supplied policy/value state. The default continuation generation cap is 512 tokens for feasibility; frozen evaluation uses the larger cap specified in `configs/ppo.yaml`.

## 5. Task entry points

### Objective validation (run first)

```bash
python -m scripts.test_objectives
```

Hand-checkable unit tests for the corrected DPO loss (`task1_dpo/dpo.py`: reference margin must be
subtracted), PPO clipped surrogate (`task2_ppo/ppo.py`: pessimistic `min`, not `max`) and GRPO
advantages (`task3_grpo/grpo.py`: normalise within each prompt group, not over the whole batch).

### Task 1 - DPO

```bash
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter none --name sft
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.ablate_beta --config configs/dpo.yaml
python -m task1_dpo.analyze_length --config configs/dpo.yaml
```

Or all of the above in order, either interactively with live training curves in `task1_dpo/task1_dpo.ipynb`
(a thin driver: it calls exactly these commands and plots their saved logs), or unattended on Windows with
`powershell -ExecutionPolicy Bypass -File .\scripts\run_task1.ps1`. Report figures are written to `report/figures/`.

Outputs: `results/task1_dpo/<condition>/` holds `run_config.json`, `train_log.jsonl` (per optimizer update),
`train_order_prompt_ids.json`, `eval_*.json`, per-pair `pairs_*.jsonl`, `generations.jsonl` and
`word_limit.jsonl`; `results/task1_dpo/summary_beta.csv` and `summary_length.csv` collect the tables.
Log-probabilities are computed by `common/logprobs.py` (response-token window only, explicit position ids
for left padding, per-row float32 log-softmax) to fit an 8 GB GPU.

### Task 2 - PPO

```bash
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter checkpoints/ppo_midpoint_policy --name midpoint
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml
python -m task2_ppo.ablate_kl --config configs/ppo.yaml
```

Or interactively with live training curves in `task2_ppo/task2_ppo.ipynb` (thin driver over the same commands).
Shared RL machinery lives in `common/rollout.py` (fixed seeded prompt order, sampling, learned reward with the
missing-EOS penalty, response-window log-probs and critic values with explicit position ids, dropout disabled so the
first-epoch ratio is exactly 1) and `common/rl_eval.py` (held-out protocol: first 96 held-out prompts within the prompt
cap, one seeded sample each, evaluation cap 768). Short forks are named `eps<ε>_kl<β>`; the `eps0.2_kl0.1` fork is
shared by the clipping and KL studies. Outputs per condition in `results/task2_ppo/<name>/`: `run_config.json`,
`train_log.jsonl` (per update: reward, KL, policy/value loss, entropy, grad norms, clip fraction, policy change,
critic explained variance, length, VRAM, wall-clock), `rollouts.jsonl`, `train_summary.json`, `eval_heldout.json`,
`generations_heldout.jsonl`; study tables in `summary_clipping.csv`, `summary_kl.csv`, `clipping_cached.json`.

### Task 3 - GRPO

```bash
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter checkpoints/grpo_midpoint_policy --name midpoint
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml
python -m task3_grpo.compare_normalization --config configs/grpo.yaml
```

Or interactively in `task3_grpo/task3_grpo.ipynb`. GRPO uses the same rollout and held-out evaluation machinery as PPO
(`common/rollout.py`, `common/rl_eval.py`). Each update: K = 4 completions for one prompt, per-group advantages,
completions that hit the 512-token cap without EOS masked from the loss (kept in the group statistics), one clipped
step with the k3 KL penalty (policy_epochs = 1). The group-size study needs no GPU: each prompt's 8 cached completions
are split at random into 8/K disjoint groups (same 192 generations for every K, 500 seeded partitions averaged);
difficulty bins are terciles of each prompt's mean cached reward. The normalisation study runs matched 8-update forks
`norm_grpo` (1/T_k) and `norm_dr_grpo` (1/L) with per-completion gradient norms (`--per-seq-grad`). Outputs in
`results/task3_grpo/`: per-run `train_log.jsonl`, `completions.jsonl`, `run_config.json`, `train_summary.json`,
`eval_heldout.json`, `generations_heldout.jsonl`; `group_size_study.{csv,json}`, `summary_normalization.{csv,json}`,
`normalization_length_bins.csv`.

### Task 4 - Safety calibration

```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml
python -m task4_safety.judge_responses --config configs/feedback.yaml
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml
# label results/task4_safety/manual_audit_sheet.csv (blinded), then:
python -m task4_safety.evaluate_safety --config configs/feedback.yaml
```

Or interactively in `task4_safety/task4_safety.ipynb` (includes a keyboard helper for the blinded audit). Policies are
fixed: SFT (base), standard DPO, standard PPO, standard GRPO. Responses use greedy decoding, cap 256, identical batches
for every policy. The released judge prompt/parser/`judge_one` are unchanged; judging is batched (16) and checked
against `judge_one` on 24 items (`judge_batch_validation.json`); identical (prompt, response) pairs are judged once.
The manual audit uses the released fixed 60 prompts; each distinct response is labelled once, blind to policy, XSTest
class and AI label, and the label is attached to every policy that produced it (`manual_audit_key_map.csv`).
Besides the four label rates, `refusal_on_safe_rate` = P(judge label is JUSTIFIED_ or OVER_REFUSAL | SAFE prompt) is
reported as a class-mapped over-refusal view, because the fixed judge decides benignness itself and can file a refusal of a
benign XSTest prompt under JUSTIFIED_REFUSAL; the manual audit checks this.
Outputs in `results/task4_safety/`: `generated_*.jsonl`, `judged_*.jsonl`, `safety_summary.{csv,json}`,
`category_label_{counts,fractions}.csv`, `manual_vs_ai_confusion.csv`, `manual_vs_ai_disagreements.csv`.

### Task 5 - RLVR vs RLAIF

The exact verifier and pairwise AI judge are supplied; you implement the evaluation/analysis.

```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

Or interactively in `task5_feedback/task5_feedback.ipynb`. Policies are fixed: SFT (base), the supplied
`checkpoints/rlvr_policy` and `checkpoints/rlaif_policy` (no training, independent of Tasks 1-4). The released
verifier (`rlvr.exact_reward`) and pairwise judge (`rlaif.PairwiseAIJudge`, cache in
`results/task5_feedback/pairwise_judge_cache.json`) are used unchanged.
- `evaluate_math` (`--dataset gsm` = the fixed 300-problem GSM8K subset, `--dataset transfer` = the fixed SVAMP file,
  used unchanged): greedy decoding, cap `math_max_new_tokens`, identical batches for all three policies; exact accuracy
  (bootstrap CI), format compliance, length, truncation, failure types; each RL response judged against SFT's on the same
  problem (win 1 / tie 0.5 / loss 0; byte-identical responses are a tie without a judge call); verifier-judge agreement
  on pairs the verifier separates. SVAMP accuracy is also broken down by problem type.
- `score_perturbations`: the 20 x 5 diagnostic set. Controlled pairs, each against `clean_correct`: reasoning-only
  corruption (S_reason), wrong final (S_outcome), persuasive filler, gold distractor. Mechanisms: exact verifier, the
  judge's direct pairwise preference, and the RLAIF training reward (group win rate among the 5 variants, identical to
  `group_rewards`). Better / tie / wrong-preference rates per category, per-variant mean rewards for all five variants,
  the full variant-vs-variant judge matrix, S_outcome over every outcome-changing pair, a swapped-argument position check,
  and verifier checks against `expected_exact_reward`.
- `inspect_judge` (optional check): re-runs the judge's exact prompt/orientation on the diagnostic pairs and a sample of
  policy-vs-SFT pairs and records the raw 4-token (released) and 48-token outputs, separating explicit TIEs from parse
  fall-backs (`judge_raw_outputs.jsonl`, `judge_parse_audit.json`).
- `compare_feedback`: no model; combines everything.

Outputs in `results/task5_feedback/`: `{gsm,transfer}/generations_*.jsonl`, `pairwise_*_vs_sft.jsonl`, `eval_math.json`;
`diagnostics/diagnostic_scores.json`, `pair_records.jsonl`, `variant_rewards.csv`, `judge_matrix.csv`;
`summary_policies.csv`, `summary_transfer.csv`, `summary_diagnostics.csv`, `summary_variant_rewards.csv`,
`feedback_source_comparison.json` (coverage, noise, exploitability, cost), `examples_diagnostic.jsonl`,
`examples_policy.jsonl`.

### Report

```bash
python report/make_report_assets.py          # report/figures/report_*.pdf and report/tables/*.tex from results/
cd report && latexmk -pdf report.tex          # report/report.pdf (NeurIPS 2025 style, preprint option)
```

Every table and derived number in the PDF is generated from the saved files in `results/` by
`report/make_report_assets.py`; nothing is typed by hand.

## 6. Reproducibility rules

- Do not alter course-provided data, cached rollouts, or supplied checkpoints.
- Start every short fork from the **same supplied midpoint checkpoint**.
- Keep prompt IDs, generated-token/update budgets, seed, and evaluation procedure matched across ablations.
- Commit your code, configs, small JSON/CSV logs, and figures. Do not commit downloaded checkpoints, raw course assets, or model caches.
- Record peak VRAM and wall-clock time for the standard PPO and GRPO continuations.

See the assignment manual for the required experiments, metrics, and report questions.
