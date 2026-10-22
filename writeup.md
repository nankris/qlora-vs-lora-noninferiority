# QLoRA vs LoRA: fresh-model non-inferiority replication

**Verdict: INCONCLUSIVE.** H1 accuracy: INCONCLUSIVE; H2 memory: PROVEN; H3 speed prediction: PROVEN.

**ONE headline number: -0.31 percentage points**, QLoRA minus LoRA test accuracy (95% CI [-1.12, 0.75]).

**Plain-English takeaway:** On this short-text task, the data could not rule out QLoRA losing one accuracy point, even with the larger test set. It used 46.2% of LoRA's peak allocated training memory (63.5% of reserved memory) and took 1.42 times as long to train.

## Main results

Five paired seeds on the same 10,991 fresh DBpedia-14 test items. Accuracy CIs resample both paired seeds and common test items, with 10,000 draws.

| Method | Accuracy and 95% CI (%) | Seed SD (pp) | Peak allocated / reserved (GiB) | Train min/run | Eval min/run |
|---|---|---|---|---|---|
| LoRA | 96.84% [96.24, 97.38] | 0.659 | 3.619 / 4.177 | 3.300 | 2.578 |
| QLoRA | 96.53% [95.94, 97.13] | 0.702 | 1.673 / 2.651 | 4.682 | 2.854 |

The table is rendered in `figures/table.png`. Resource means are over five fits per arm.

## Fixed hypothesis decisions

| Hypothesis | Estimate | 95% CI | Pre-registered rule | Verdict |
|---|---|---|---|---|
| H1: accuracy QLoRA minus LoRA | -0.311 pp | [-1.12, 0.75] pp | lower bound > -1 pp | INCONCLUSIVE |
| H2: allocated VRAM QLoRA / LoRA | 0.4623 | [0.4622, 0.4623] | upper bound < 0.50 | PROVEN |
| H3: train time QLoRA / LoRA | 1.4185 | [1.3971, 1.4415] | CI above 1 and inside [1.1137, 1.6705] | PROVEN |

Overall requires both H1 and H2. A disproven component disproves the conjunction. H3 cannot rescue or overturn it. PROVEN is a local result under this recipe, not a universal claim.

## What was run

Qwen/Qwen3-1.7B, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; BF16 LoRA r16 vs NF4 double-quantized QLoRA r16, BF16 computation and adapters. Five seeds 1701–1705 per arm, paired initial adapters and training order; 120 steps, batch 16, LR 2e-4, AdamW with FP32 master weights/states, gradient checkpointing. There are 17,432,576 trainable parameters per fit. A fixed 4,096-item pool supplies 1,920 unique training exposures per seed. Both methods score the same A–N class codes at the last prompt token. These are model forward passes, not generated reasoning answers. Only final-position vocabulary logits are materialized; full-token language-model training has a different activation budget.

Prompts are limited to 128 tokens. The DBpedia legend and suffix use 78, leaving only 50 tokens for article content. This matters for interpreting accuracy. Token counts from the actual Qwen tokenizer and run records: `{"lora": {"training_prompt_tokens": 1174895, "training_target_tokens": 9600, "eval_prompt_tokens": 6713660, "eval_scored_tokens": 54955}, "qlora": {"training_prompt_tokens": 1174895, "training_target_tokens": 9600, "eval_prompt_tokens": 6713660, "eval_scored_tokens": 54955}}`. Target/scored tokens are classification positions; no free-form output tokens were generated.

The sizing pilot used a different dataset, AG News: one paired seed, 480 training exposures, 256 test items and 30 steps per arm. Disagreement was 9.77%, Wilson upper bound 14.02%. Targeting a 0.7 pp item half-width selected 10,991 confirmation items. Pilot accuracy was 85.16% LoRA and 87.50% QLoRA. Pilot outcomes are excluded from all confirmatory CIs. The cross-task sizing is approximate and does not guarantee seed-level power.

Registration: initial protocol commit `4ce14e4`; pilot-derived final registration `a3a5d23`, before the VM confirmation start timestamp in `results/remote_logs/confirmation_start.json`. The earlier Qwen3-4B run in [my PEFT head-to-head](https://github.com/nankris/peft-head-to-head) had already supported its QLoRA comparison; DoRA made its overall result inconclusive. This is a replication under a stricter memory threshold, not a correction of an unproven old accuracy claim.

## Per-seed accuracy

| Seed | LoRA (%) | QLoRA (%) | QLoRA minus LoRA (pp) |
|---|---|---|---|
| 1701 | 95.81 | 97.50 | +1.692 |
| 1702 | 96.87 | 96.33 | -0.537 |
| 1703 | 97.31 | 95.75 | -1.556 |
| 1704 | 96.72 | 96.09 | -0.628 |
| 1705 | 97.50 | 96.97 | -0.528 |

## Measurement and caveats

- This is an explicit BF16 preparation policy, not the default `prepare_model_for_kbit_training` recipe: [PEFT 0.17.1 source](https://github.com/huggingface/peft/blob/v0.17.1/src/peft/utils/other.py#L102-L144) upcasts non-quantized BF16 parameters to FP32. Our registered worker freezes the base, enables checkpointing and retains BF16 explicitly; copying a default helper can change memory and numerical behavior.
- Peak allocated VRAM is the preregistered PyTorch metric. Reserved VRAM and driver snapshots are also saved. The ratio of mean peak reserved memory is 0.635, so the allocated-memory finding does not imply halving reservations or peak driver-level process memory; driver snapshots are not continuous peak measurements. Allocated, reserved, and process memory answer different capacity questions.
- Timing excludes loading, tokenization and evaluation, includes synchronized training and recovery checkpoint writes. Raw `total_seconds` starts after Python imports, so it is not fully end-to-end per fit; the complete named-session lifetime is captured separately. Every fit uses a fresh subprocess; the confirmation spans two L4 hosts after the first was pruned. Arm order alternates across seeds; the odd number of seeds leaves three QLoRA-first pairs.
- Only five seeds: seed tails are poorly estimated. Test items are reused across seeds, so this is not 54,955 independent test examples.
- Passing non-inferiority would rule out a practically unacceptable average loss under the chosen one-point margin. It does not establish identical predictions, equality, or per-class non-inferiority. The margin is an operational tolerance, not a measured human perception threshold.
- One model, one short-document dataset, one GPU and one fixed recipe. There is no hyperparameter sweep, long-context test, generation benchmark, zero-shot baseline, or convergence claim. Public benchmark contamination in pretraining is possible and was not measured.
- The runtime confidence interval reflects these five seed timings, not uncertainty across GPU models, library versions or Colab hosts. Nearly identical memory peaks do not imply similarly precise portability.

## Compute and reproducibility

Timing: estimated 150 min; actual 341.0 min elapsed including allocation waits. GPU: NVIDIA L4. Two-VM compute: 3.091 CU (elapsed × measured 1.54 CU/h; shared balance is not an isolated bill).

The first `qlora-vs-lora` VM was pruned at 20:46 UTC after five saved confirmation fits. A replacement L4 resumed only the five missing fits with the registered protocol unchanged, then was stopped after saved-result verification. Seed 1703 spans the two hosts; seeds 1701–1702 ran on the first and 1704–1705 on the second. Total GPU training time including pilot: 41.88 min; evaluation: 27.27 min; combined billed VM lifetimes: 120.44 min. The 6 CU cap was respected. Other sessions were neither used nor stopped. Raw predictions, gold labels, per-step losses, token counts and dataset split indices are under `results/`; logs and nvidia-smi evidence are in `figures/colab_run.txt` and `results/remote_logs/`.

Run `scripts/audit_results.py`, `scripts/analyze.py`, `scripts/figures.py`, then `scripts/report.py` with the local ML Python to regenerate the tables and charts. The remote worker and driver skip completed JSON results and resume exact optimizer checkpoints. Model packages are pinned in `scripts/bootstrap_setup.py`; the full VM package snapshot is saved in `results/remote_logs/environment.txt`.

Source: [Dettmers et al., QLoRA](https://arxiv.org/abs/2305.14314), which motivates the accuracy-preservation claim, not this exact model's speed or memory ratio.
