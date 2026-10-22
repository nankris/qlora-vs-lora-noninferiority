"""Build a publication draft from completed, audited measurements."""
import json,re
from pathlib import Path
r=Path(__file__).resolve().parents[1];a=json.loads((r/'results/analysis.json').read_text());c=json.loads((r/'results/compute.json').read_text());t=json.loads((r/'results/timing.json').read_text());l,q=a['rows'];d=a['difference_pp'];mem=a['memory_ratio'];speed=a['speed_ratio'];lo,hi=a['difference_ci'];reg=json.loads((r/'results/pilot_registration.json').read_text())
if a['H1']=='PROVEN':accuracy_finding='The accuracy claim passed: the lower confidence bound cleared the registered minus-one-point margin.'
elif a['H1']=='DISPROVEN':accuracy_finding='The accuracy claim failed: even the upper confidence bound was below the registered minus-one-point margin.'
else:accuracy_finding='The accuracy claim remains inconclusive: the confidence interval crosses the registered minus-one-point margin.'
if a['H2']=='PROVEN':memory_finding='The strict memory claim passed: the entire seed-bootstrap interval was below half.'
elif a['H2']=='DISPROVEN':memory_finding='The strict memory claim failed: the seed-bootstrap interval was at or above half. A useful saving does not make the registered threshold pass.'
else:memory_finding='The memory claim remains inconclusive because its interval crosses the half-memory threshold.'
if a['H3']=='PROVEN':speed_finding='That speed prediction passed its registered tolerance.'
elif a['H3']=='DISPROVEN':speed_finding='That speed prediction failed its registered rule. I am keeping the miss rather than replacing the prediction with the observed ratio.'
else:speed_finding='That speed prediction was inconclusive under its registered rule.'
if a['H1']=='PROVEN' and a['H2']=='PROVEN':title='QLoRA cleared the accuracy margin at less than half the allocated VRAM'
elif a['H1']=='PROVEN':title='QLoRA cleared the accuracy margin but missed the half-memory target'
elif a['H1']=='DISPROVEN':title='QLoRA missed the one-point accuracy margin on DBpedia'
else:title='QLoRA’s one-point accuracy margin remains unresolved on DBpedia'
if a['H1']=='INCONCLUSIVE':
    postmortem='My main prediction remains unresolved. I expected the accuracy interval to clear minus one point, and it did not. Having 10,991 articles does not guarantee a decisive answer when training seeds also contribute uncertainty. The pilot-based sizing addressed test-item uncertainty; one pilot seed could not settle the seed component.'
elif a['H1']=='DISPROVEN':
    postmortem='My main accuracy prediction failed. I expected the interval to clear minus one point; instead even its upper bound was below the margin. The memory saving cannot rescue that failed accuracy claim. The result belongs in the record with the same hypothesis I started with.'
else:
    postmortem='The primary accuracy prediction passed this local test. That still does not turn a one-point tolerance into a universal deployment rule, or make the pilot a reliable forecast of every seed.'
text=f'''# {title}

*A five-seed Qwen3-1.7B comparison, with 10,991 shared test examples and the decision rules committed first.*

The [QLoRA paper](https://arxiv.org/abs/2305.14314) makes a concrete promise: lower memory use while “preserving full 16-bit finetuning task performance.” If you maintain a fine-tuning pipeline, that promise affects a concrete decision: can you fit the job on the GPU you already have without giving away useful accuracy?

The phrase sounds simple. Testing it requires defining both “preserving” and “memory.” Two rounded accuracy numbers can hide an unacceptable loss. A smaller model footprint does not automatically mean the entire training process uses less than half the memory.

My headline result is **{d:+.2f} percentage points**: QLoRA accuracy minus LoRA accuracy, with a 95% confidence interval of **[{lo:+.2f}, {hi:+.2f}] points**. The combined verdict under the rules I committed before confirmation is **{a['verdict']}**.

{accuracy_finding} {memory_finding}

## What I tested

I used [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) on NVIDIA L4 GPUs in Colab. The task was classifying short article excerpts into the fourteen categories in the public [DBpedia-14 dataset](https://huggingface.co/datasets/fancyzhx/dbpedia_14). The model returned a score for each permitted class code, A through N. It did not generate explanations or reasoning traces.

Both methods trained rank-16 adapters on the same attention and MLP projections. LoRA kept the frozen base weights in BF16. QLoRA stored eligible frozen linear weights in NF4 with double quantization. Both computed in BF16 and used BF16 adapters with FP32 optimizer master weights and states.

There were five paired seeds per method. Within each pair, adapter initialization and the training-example order matched. Each fit ran 120 optimizer steps at effective batch size 16, with learning rate 0.0002 and no method-specific tuning. That means 1,920 training examples per fit, drawn without replacement from a fixed 4,096-item pool.

Every fit saw the same 10,991 held-out test items. Repeating those items across seeds measures training variation; it does not turn them into 54,955 independent articles. The confidence intervals account for both seeds and shared test items.

![Experimental pipeline](figures/diagram.png)

The diagram shows the separation between the pilot and confirmation. The two training branches differ in base-weight precision, while the adapter recipe and test set stay paired.

## Why I used a pilot, then changed datasets

Before confirmation, I ran one paired pilot on AG News: thirty steps, 480 training exposures and 256 test items. The pilot served three purposes: checking the implementation, estimating disagreement for sizing, and predicting training time.

The models disagreed on {reg['pilot_disagreement']*100:.2f}% of pilot predictions. I used the upper Wilson confidence bound on that rate, {reg['pilot_disagreement_wilson_upper']*100:.2f}%, to size an approximate 0.7-point item-level confidence half-width. That calculation selected 10,991 confirmation items. It was an explicit sizing calculation, not a round sample chosen because it looked sufficient.

There is a catch: pilot disagreement on AG News need not transfer to DBpedia. Neither does one pilot seed estimate the tails of seed variation. The sizing is a reasoned starting point, not a guarantee that the test will pass.

The final sample size and speed prediction were committed before I started the confirmation job. Pilot outcomes are excluded from the reported confirmation intervals.

## The accuracy result

LoRA averaged **{l['accuracy_percent']:.2f}%** accuracy. QLoRA averaged **{q['accuracy_percent']:.2f}%**. The paired seed differences ranged from {min(a['paired_seed_difference_pp']):+.2f} to {max(a['paired_seed_difference_pp']):+.2f} points.

![Main results table](figures/table.png)

The table puts accuracy, peak allocated training memory and training time side by side. Its accuracy intervals include both test-item and training-seed uncertainty.

I resampled paired seeds and common test items together for 10,000 bootstrap draws, keeping the comparison paired throughout.

![Accuracy difference and registered margin](figures/noninferiority.png)

The orange point is the mean difference; its whisker is the 95% interval. Blue dots show each paired seed. The dashed line is the registered minus-one-percentage-point margin, so the location of the whole interval determines the non-inferiority decision.

I chose that margin because a one-point average loss was the practical tolerance specified for this experiment. It is not a measured claim that every user ignores a one-point error increase. An engineer handling rare, expensive mistakes may reasonably demand a tighter margin.

Non-inferiority also does not establish identical models. Two classifiers can have similar average accuracy while making different mistakes. This experiment tests average accuracy under a defined recipe, not interchangeable behavior on every category or example.

## The memory saving had a time price

LoRA peaked at **{l['peak_GiB']:.2f} GiB** of allocated training memory. QLoRA peaked at **{q['peak_GiB']:.2f} GiB**, a ratio of **{mem:.3f}**. These include resident model weights and training allocations, measured after resetting PyTorch's peak counter immediately before training. The frozen embeddings and output head stayed in BF16 in both arms. This memory result belongs to that precision policy and the shared FP32 optimizer recipe. The [PEFT 0.17.1 preparation helper](https://github.com/huggingface/peft/blob/v0.17.1/src/peft/utils/other.py#L102-L144) casts non-quantized BF16 parameters to FP32. Our registered worker freezes the base and enables checkpointing explicitly, retaining BF16 for those parameters. A copy of this experiment that uses the helper unchanged is a different memory recipe.

![Memory and training time](figures/resources.png)

The left panel compares peak allocated VRAM; the right compares synchronized training time. QLoRA took **{q['train_minutes']:.2f} minutes** per fit against **{l['train_minutes']:.2f} minutes** for LoRA, or **{speed:.2f} times as long**.

Allocated memory is the registered primary metric, but it is not the whole capacity story. Peak reserved memory averaged {l['reserved_GiB']:.2f} GiB for LoRA and {q['reserved_GiB']:.2f} GiB for QLoRA. That is {q['reserved_GiB']/l['reserved_GiB']*100:.1f}% of LoRA's reservation, so reserved memory was not halved. The under-half finding applies to allocated memory. The allocator can reserve more than live tensors need. Driver-level process memory adds another view, and the saved logs include nvidia-smi snapshots.

Training time excludes model loading, tokenization and evaluation. It includes the checkpoint writes used for recovery. Both methods followed the same checkpoint policy. This measures the training loop we actually ran, not an idealized kernel benchmark.

A four-bit checkpoint does not promise a four-times-faster training loop. Storage precision and computation precision differ here: both arms use BF16 computation, and NF4 relies on the quantization implementation. I did not profile kernels, so this experiment measures the time penalty without assigning it to a single cause.

The pilot predicted a **{reg['pilot_speed_ratio']:.3f}×** slowdown. Before confirmation I required the timing interval to stay above one and within 20% of that prediction. {speed_finding}

## What surprised me, and what I got wrong

{postmortem}

Seed 1701 favored QLoRA by 1.69 points; seed 1703 favored LoRA by 1.56. The mean hides that swing. More test articles alone would not remove training variation.

The earlier Qwen3-4B run tied on training speed. The new pilot did not: QLoRA took 39% longer. That changed the operational expectation before confirmation, and the pilot-derived speed prediction was then committed.

The pilot gave QLoRA higher accuracy than LoRA, 87.50% versus 85.16%. It would have been tempting to predict an accuracy improvement. I did not: 256 examples and one seed are too little evidence for that stronger claim, particularly when confirmation changes tasks.

One correction to the premise of this follow-up: the earlier Qwen3-4B run in [my PEFT head-to-head](https://github.com/nankris/peft-head-to-head) had already supported its QLoRA accuracy comparison. Its overall inconclusive verdict came from DoRA. This is a fresh-model replication with a stricter memory target.

The first Colab VM was pruned after five saved confirmation fits. A replacement L4 ran only the missing fits under the unchanged protocol. Seed 1703 crosses hosts, which limits the paired timing comparison.

The registered outcomes remain H1 **{a['H1']}**, H2 **{a['H2']}**, and H3 **{a['H3']}**. Accuracy and memory determine the combined verdict; speed is diagnostic.

## What I would do in production

1. **Set an error budget before switching.** Choose a margin tied to the cost of mistakes, then compare paired predictions on representative held-out examples. Average accuracy may need a companion metric for critical classes.
2. **Measure the whole workload's memory.** Keep allocated, reserved and process memory separate. Repeat at production sequence lengths and batch sizes before selecting a smaller GPU.
3. **Price the time tradeoff.** Use measured training and evaluation times, including loading if jobs are short. A configuration that fits can be valuable even when each step takes longer.
4. **Preserve a reproducible baseline.** Pin model revisions and package versions, save split indices and predictions, and keep the BF16 LoRA recipe available. Change one consequential setting at a time.

## Where this result stops

The largest task limitation is the prompt budget. At 128 tokens, the category legend and answer suffix consume 78 tokens. Only fifty tokens remain for article content. This is short-excerpt classification. The loop trains only the final class-answer position; a full-token language-modeling loss would have a different memory and compute budget.

There is also only one model size, one GPU type and five seeds. Five is better than one, but it still gives a coarse view of training variability. The fixed short training schedule does not establish convergence or the best achievable accuracy for either method. Public benchmark exposure during pretraining is possible and was not measured.

The comparison does not cover long-context generation, tool use, instruction following or serving throughput. The memory ratio depends on what remains unquantized and on the optimizer and activation budget. The runtime ratio depends on this L4 and these library implementations. This also uses a shared AdamW recipe, rather than reproducing every detail of the original QLoRA paper.

Timing: estimated {t['estimated_minutes']} minutes; actual {t['actual_minutes']:.1f} minutes including slot waits. The two L4 VM lifetimes used an estimated {c['estimated_cu']:.3f} CU from their combined elapsed lifetimes at the observed 1.54 CU/hour rate, below the six-unit cap. Shared account balance changes cannot isolate this experiment's bill.

Measure the memory saving, price the extra training time, and require your own error budget to pass.

Code: https://github.com/nankris/qlora-vs-lora-noninferiority
'''
(r/'blog.md').write_text(text)
words=len(text.split());print('Blog words:',words)
assert 1200<=words<=1700,words
assert '\u2014' not in text
