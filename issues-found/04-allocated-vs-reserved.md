# 04: Allocated vs reserved

- **Experiment:** QLoRA vs LoRA non-inferiority (this repo)
- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST. Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
Allocated vs reserved: headline "under half the VRAM" holds only for torch allocated (0.462); reserved 0.635, so practical capacity savings smaller. Disclosed but the title-level framing ("under half the VRAM") leans on the friendlier metric.

## Status
Open, documented. Nothing in the results was changed because of it. The README, writeup or blog post states where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data (registered before the data are seen), not an edit to these numbers.
