# 101c: QLoRA within one point of LoRA at under half the VRAM?

Question: Does NF4 QLoRA preserve classification accuracy relative to BF16 LoRA while halving training memory on Qwen/Qwen3-1.7B?

Source: Dettmers et al., [QLoRA](https://arxiv.org/abs/2305.14314), which reports preserving 16-bit finetuning performance. Prior #101's saved writeup actually names Qwen3-4B, despite the request describing 0.6B. This experiment uses a fresh 1.7B model and DBpedia task regardless.

## Protocol registered before pilot
Pilot: one paired seed (700), AG News, 512 training / 256 test items, 30 steps. Only sizing, speed and implementation diagnostics; excluded from confirmation. This obeys the hypothesis-rules requirement to pilot on a different dataset. Confirmation: public HF `fancyzhx/dbpedia_14`, 4,096 training items and initially 6,000 test items (may increase BEFORE confirmation using pilot disagreement). Five paired seeds 1701–1705, same training items and order within each pair, 120 optimizer steps, batch 16 (microbatch 4 × accumulation 4), fixed length 128 tokens. Single-letter class codes A–N, next-token cross-entropy and constrained classification. No generated explanations. Record tokenizer token counts, not character estimates.

Both arms: all linear attention/MLP projections, rank 16, alpha 32, dropout 0, LR 2e-4, AdamW torch FP32 states, constant LR, weight decay 0, gradient clipping 1, gradient checkpointing, BF16 computation, SDPA. QLoRA: bitsandbytes NF4 double quantization, frozen base; LoRA: frozen BF16 base. No per-arm tuning. Adapter parameters explicitly BF16 in both arms. Frozen embeddings/head remain BF16 in both; do not cast all QLoRA weights to FP32. One fresh subprocess per run, same GPU. Quantization preparation is explicit rather than PEFT's automatic FP32 casting.

H1 primary: QLoRA minus LoRA mean accuracy, percentage points. PROVEN if paired two-level (seeds and common test items) percentile bootstrap 95% lower bound > -1.0 pp; DISPROVEN if upper bound < -1.0 pp; otherwise INCONCLUSIVE. Margin rationale supplied by the experiment brief: 1 pp is smaller than users notice on most classification tasks, treated here as a practical tolerance rather than a universal usability fact. No claim of exact equality.

H2: QLoRA peak PyTorch allocated training VRAM / LoRA <0.50. Pair-bootstrap seeds 95% CI: PROVEN if upper bound <0.50, DISPROVEN if lower bound >=0.50, else INCONCLUSIVE. Peak reserved and nvidia-smi also recorded; primary memory metric is allocated, reset immediately before training including resident model/optimizer.

H3: training wall-clock ratio predicted from pilot, numeric prediction and tolerance registered after pilot and BEFORE confirmation. Ratio uses synchronized training seconds, excluding loading and evaluation. Report end-to-end seconds separately. Seed bootstrap CI. Confirmation outcome rule fixed in addendum.

Overall question PROVEN only if H1 and H2 PROVEN; DISPROVEN if either DISPROVEN; otherwise INCONCLUSIVE. H3 is diagnostic and cannot change primary verdict. 10,000 bootstrap draws, RNG 101.

Sizing: >=5 paired seeds and >=6,000 fresh test items. Approximate 95% item half-width is 1.96 sqrt(discordance / n), with a separately reported seed component. Pilot-derived sizing and H3 will be appended and committed before confirmation. Five seeds give coarse information about seed tails; report this limitation.

## Execution and budget
NVIDIA L4 only, named session `b9-101c`; detached VM processes, per-run atomic JSON and checkpointing, frequent local copies. Hard experiment cap 6 CU, conservatively cap VM lifetime at 3 hours if L4 rate <=2 CU/h; record actual assigned rate and adjust cap down if higher. Never create below daily floor. Stop session on completion/failure.

Timing: estimated 150 minutes (setup 20, GPU run 100, analysis 10, writeup 20); actual and CU pending. See results/timing.json.

Implementation clarification before pilot: 120 × 16 = 1,920 training exposures per seed, sampled without replacement from the fixed 4,096-item training pool using that seed's order. Both arms see the identical 1,920 examples within each seed. Thus the effective training size per fit is 1,920, not all 4,096. Pilot uses 480 exposures from its 512-item pool. No epoch-based reshuffling occurs in these short runs.

Measurement clarification before pilot: training wall time includes the periodic checkpoint writes needed for reboot recovery, including their CPU serialization time; loading, tokenization and evaluation are excluded. Checkpoint policy is identical across arms. The effective recipe is supervised next-token classification, not chat reasoning or free-form generation. Accuracy cannot be generalized to those workloads.

Prior-result provenance: #101's saved writeup already reports QLoRA minus LoRA +0.10 pp, 95% CI [-0.35,+0.55], and calls its QLoRA comparison supported with three seeds. Its overall INCONCLUSIVE verdict came from DoRA. Thus this follow-up is a fresh-model/fresh-task replication with more seeds and a stricter memory threshold, not the first evidence that QLoRA can clear a 1 pp margin. This discrepancy with the brief is disclosed rather than silently repeated.

Pre-run tokenizer check (no task outcomes observed): the DBpedia legend and suffix occupy 78 tokens, leaving **50 content tokens** at the registered 128-token limit. Every A-N class code is exactly one token. This is a short-text classification comparison and cannot establish performance on full documents.

Recovery implementation clarification before pilot execution: checkpoints save FP32 optimizer master weights as well as adapters and optimizer state, so recovery preserves the unrounded master values. This serialization overhead is included equally in both arms.

## Pilot-derived addendum, committed before confirmation
Pilot uses AG News only. LoRA accuracy 85.16%, QLoRA 87.50%; n=256. Prediction disagreement 9.77% (Wilson 95% upper bound 14.02%). Sizing with that upper bound and a 0.7 pp item half-width requires 10,991 items; confirmation fixed to **10,991 DBpedia-14 test items and five paired seeds**. This is approximate sizing, not a guarantee of power: task transfer and seed variance remain uncertain.

H3 numeric prediction: QLoRA training takes **1.3921×** LoRA, from the paired pilot training times 68.87 / 49.47 seconds. Predict the confirmatory 95% seed-bootstrap ratio CI lies inside [1.1137, 1.6705] AND above 1.0. A 20% relative prediction tolerance represents the cost-estimation error we consider operationally material. PROVEN if both conditions hold; DISPROVEN if the CI is disjoint from that interval or upper bound <=1.0; otherwise INCONCLUSIVE. If the pilot ratio is below one, the required slowdown prediction is visibly unsupported by the pilot and the outcome is still evaluated without changing this rule.

## Final execution record

Timing: estimated 150 min; actual 341.0 min elapsed including allocation waits. GPU: NVIDIA L4. Two-VM compute: 3.091 CU (elapsed × measured 1.54 CU/h; shared balance is not an isolated bill).

GPU: NVIDIA L4. Verdict: **INCONCLUSIVE** (H1 INCONCLUSIVE, H2 PROVEN, H3 PROVEN). Full results and limitations: `writeup.md`; publication post: `blog.md`. Two L4 VM lifetimes accounted; replacement stopped; raw evidence saved.
