# 08: Speed finding

- **Experiment:** QLoRA vs LoRA non-inferiority (this repo)
- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST. Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
Speed finding (1.42× slower) contradicts the earlier Qwen3-4B run in [my PEFT head-to-head](https://github.com/nankris/peft-head-to-head) (QLoRA = LoRA speed on 4B); training time includes checkpoint writes (FP32 masters serialized) — overhead equal in principle but could inflate/deflate ratio; different model size. Worth noting in any cross-experiment narrative.

## Status
Open, documented. Nothing in the results was changed because of it. The README, writeup or blog post states where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data (registered before the data are seen), not an edit to these numbers.
