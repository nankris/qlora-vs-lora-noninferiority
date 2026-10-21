"""Render the factual short report from audited, saved results only."""
import json
from pathlib import Path
r=Path(__file__).resolve().parents[1];a=json.loads((r/'results/analysis.json').read_text());c=json.loads((r/'results/compute.json').read_text());t=json.loads((r/'results/timing.json').read_text());p=json.loads((r/'results/pilot_registration.json').read_text())
def ci(x,digits=2):return f'[{x[0]:.{digits}f}, {x[1]:.{digits}f}]'
l,q=a['rows'];delta=a['difference_pp'];mem=a['memory_ratio'];speed=a['speed_ratio']
timing=f"Timing: estimated {t['estimated_minutes']} min; actual {t['actual_minutes']:.1f} min elapsed including allocation waits. GPU: NVIDIA L4. Two-VM compute: {c['estimated_cu']:.3f} CU (elapsed × measured 1.54 CU/h; shared balance is not an isolated bill)."
if a['H1']=='PROVEN':
 takeaway_accuracy='On this short-text task, the confidence interval ruled out an average accuracy loss of one percentage point for QLoRA.'
elif a['H1']=='DISPROVEN':
 takeaway_accuracy='On this short-text task, QLoRA lost more than one accuracy point under the registered confidence-interval rule.'
else:
 takeaway_accuracy='On this short-text task, the data could not rule out QLoRA losing one accuracy point, even with the larger test set.'
seed_rows=[]
for seed in range(1701,1706):
    sl=json.loads((r/f'results/confirm_lora_{seed}.json').read_text());sq=json.loads((r/f'results/confirm_qlora_{seed}.json').read_text());seed_rows.append(f"| {seed} | {100*sl['accuracy']:.2f} | {100*sq['accuracy']:.2f} | {100*(sq['accuracy']-sl['accuracy']):+.3f} |")
seed_table='\n'.join(seed_rows)
rows='\n'.join(f"| {x['method']} | {x['accuracy_percent']:.2f}% {ci(x['accuracy_ci'])} | {x['seed_sd_pp']:.3f} | {x['peak_GiB']:.3f} / {x['reserved_GiB']:.3f} | {x['train_minutes']:.3f} | {x['eval_minutes']:.3f} |" for x in a['rows'])
text=f'''# QLoRA vs LoRA: fresh-model non-inferiority replication

**Verdict: {a['verdict']}.** H1 accuracy: {a['H1']}; H2 memory: {a['H2']}; H3 speed prediction: {a['H3']}.

**ONE headline number: {delta:+.2f} percentage points**, QLoRA minus LoRA test accuracy (95% CI {ci(a['difference_ci'])}).

**Plain-English takeaway:** {takeaway_accuracy} It used {mem*100:.1f}% of LoRA's peak allocated training memory ({q['reserved_GiB']/l['reserved_GiB']*100:.1f}% of reserved memory) and took {speed:.2f} times as long to train.

## Main results

Five paired seeds on the same {a['n_test']:,} fresh DBpedia-14 test items. Accuracy CIs resample both paired seeds and common test items, with 10,000 draws.

| Method | Accuracy and 95% CI (%) | Seed SD (pp) | Peak allocated / reserved (GiB) | Train min/run | Eval min/run |
|---|---|---|---|---|---|
{rows}

The table is rendered in `figures/table.png`. Resource means are over five fits per arm.

## Fixed hypothesis decisions

| Hypothesis | Estimate | 95% CI | Pre-registered rule | Verdict |
|---|---|---|---|---|
| H1: accuracy QLoRA minus LoRA | {delta:+.3f} pp | {ci(a['difference_ci'])} pp | lower bound > -1 pp | {a['H1']} |
| H2: allocated VRAM QLoRA / LoRA | {mem:.4f} | {ci(a['memory_ratio_ci'],4)} | upper bound < 0.50 | {a['H2']} |
| H3: train time QLoRA / LoRA | {speed:.4f} | {ci(a['speed_ratio_ci'],4)} | CI above 1 and inside {ci(p['prediction_interval'],4)} | {a['H3']} |

Overall requires both H1 and H2. A disproven component disproves the conjunction. H3 cannot rescue or overturn it. PROVEN is a local result under this recipe, not a universal claim.

## What was run

Qwen/Qwen3-1.7B, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; BF16 LoRA r16 vs NF4 double-quantized QLoRA r16, BF16 computation and adapters. Five seeds 1701–1705 per arm, paired initial adapters and training order; 120 steps, batch 16, LR 2e-4, AdamW with FP32 master weights/states, gradient checkpointing. There are 17,432,576 trainable parameters per fit. A fixed 4,096-item pool supplies 1,920 unique training exposures per seed. Both methods score the same A–N class codes at the last prompt token. These are model forward passes, not generated reasoning answers. Only final-position vocabulary logits are materialized; full-token language-model training has a different activation budget.

Prompts are limited to 128 tokens. The DBpedia legend and suffix use 78, leaving only 50 tokens for article content. This matters for interpreting accuracy. Token counts from the actual Qwen tokenizer and run records: `{json.dumps(a['token_counts'])}`. Target/scored tokens are classification positions; no free-form output tokens were generated.

The sizing pilot used a different dataset, AG News: one paired seed, 480 training exposures, 256 test items and 30 steps per arm. Disagreement was {p['pilot_disagreement']*100:.2f}%, Wilson upper bound {p['pilot_disagreement_wilson_upper']*100:.2f}%. Targeting a 0.7 pp item half-width selected {a['n_test']:,} confirmation items. Pilot accuracy was {p['pilot_accuracy_lora']*100:.2f}% LoRA and {p['pilot_accuracy_qlora']*100:.2f}% QLoRA. Pilot outcomes are excluded from all confirmatory CIs. The cross-task sizing is approximate and does not guarantee seed-level power.

Registration: initial protocol commit `4ce14e4`; pilot-derived final registration `a3a5d23`, before the VM confirmation authorization timestamp in `results/remote_logs/confirmation_start.json`. The old #101 writeup actually used Qwen3-4B and already supported its QLoRA comparison; DoRA made its overall result inconclusive. This is a replication under a stricter memory threshold, not a correction of an unproven old accuracy claim.

## Per-seed accuracy

| Seed | LoRA (%) | QLoRA (%) | QLoRA minus LoRA (pp) |
|---|---|---|---|
{seed_table}

## Measurement and caveats

- This is an explicit BF16 preparation policy, not the default `prepare_model_for_kbit_training` recipe: [PEFT 0.17.1 source](https://github.com/huggingface/peft/blob/v0.17.1/src/peft/utils/other.py#L102-L144) upcasts non-quantized BF16 parameters to FP32. Our registered worker freezes the base, enables checkpointing and retains BF16 explicitly; copying a default helper can change memory and numerical behavior.
- Peak allocated VRAM is the preregistered PyTorch metric. Reserved VRAM and driver snapshots are also saved. The ratio of mean peak reserved memory is {q['reserved_GiB']/l['reserved_GiB']:.3f}, so the allocated-memory finding does not imply halving reservations or peak driver-level process memory; driver snapshots are not continuous peak measurements. Allocated, reserved, and process memory answer different capacity questions.
- Timing excludes loading, tokenization and evaluation, includes synchronized training and recovery checkpoint writes. Raw `total_seconds` starts after Python imports, so it is not fully end-to-end per fit; the complete named-session lifetime is captured separately. Every fit uses a fresh subprocess; the confirmation spans two L4 hosts after the first was pruned. Arm order alternates across seeds; the odd number of seeds leaves three QLoRA-first pairs.
- Only five seeds: seed tails are poorly estimated. Test items are reused across seeds, so this is not {a['n_test']*5:,} independent test examples.
- Passing non-inferiority would rule out a practically unacceptable average loss under the chosen one-point margin. It does not establish identical predictions, equality, or per-class non-inferiority. The margin is an operational tolerance, not a measured human perception threshold.
- One model, one short-document dataset, one GPU and one fixed recipe. There is no hyperparameter sweep, long-context test, generation benchmark, zero-shot baseline, or convergence claim. Public benchmark contamination in pretraining is possible and was not measured.
- The runtime confidence interval reflects these five seed timings, not uncertainty across GPU models, library versions or Colab hosts. Nearly identical memory peaks do not imply similarly precise portability.

## Compute and reproducibility

{timing}

The first `b9-101c` VM was pruned at 20:46 UTC after five saved confirmation fits. A replacement L4 resumed only the five missing fits with the registered protocol unchanged, then was stopped after saved-result verification. Seed 1703 spans the two hosts; seeds 1701–1702 ran on the first and 1704–1705 on the second. Total GPU training time including pilot: {c['training_minutes']:.2f} min; evaluation: {c['evaluation_minutes']:.2f} min; combined billed VM lifetimes: {c['session_minutes']:.2f} min. The 6 CU cap was respected. Other sessions were neither used nor stopped. Raw predictions, gold labels, per-step losses, token counts and dataset split indices are under `results/`; logs and nvidia-smi evidence are in `figures/colab_run.txt` and `results/remote_logs/`.

Run `scripts/audit_results.py`, `scripts/analyze.py`, `scripts/figures.py`, then `scripts/report.py` with the local ML Python to regenerate the tables and charts. The remote worker and driver skip completed JSON results and resume exact optimizer checkpoints. Model packages are pinned in `scripts/bootstrap_setup.py`; the full VM package snapshot is saved in `results/remote_logs/environment.txt`.

Source: [Dettmers et al., QLoRA](https://arxiv.org/abs/2305.14314), which motivates the accuracy-preservation claim, not this exact model's speed or memory ratio.
'''
(r/'writeup.md').write_text(text)
# Keep preregistration intact; replace only the clearly marked execution addendum.
readme=(r/'README.md').read_text().split('\n## Final execution record')[0]
(r/'README.md').write_text(readme+f"\n## Final execution record\n\n{timing}\n\nGPU: NVIDIA L4. Verdict: **{a['verdict']}** (H1 {a['H1']}, H2 {a['H2']}, H3 {a['H3']}). Full results and limitations: `writeup.md`; publication post: `blog.md`. Two L4 VM lifetimes accounted; replacement stopped; raw evidence saved.\n")
print(timing)
