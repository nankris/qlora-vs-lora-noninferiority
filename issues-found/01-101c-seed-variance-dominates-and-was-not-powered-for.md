# 01 (101c): Seed variance dominates and was not powered for

- **Experiment:** 101c (this repo)
- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST (fact sheet `101c-qlora-vs-lora-noninferiority.md`). Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
Seed variance dominates and was not powered for: seed SD ~0.7 pp per arm, paired diff SD 1.2 pp vs a 1 pp margin; 5 seeds cannot resolve that. 10,991 items solved the wrong variance component (writeup admits).

## Status
Open, documented. Nothing in the results was changed because of it. The README/writeup or the Substack post states where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data (hypothesis rules), not an edit to these numbers.
