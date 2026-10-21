# Resume state
Initial README committed at 4ce14e4. No GPU run started as of latest allocation log.
Both Colab slots were occupied by other experiments. Never touch them.
Check `colab sessions` for b9-101c first and `results/allocation_attempts.jsonl`.
Wait at least five minutes between slot allocation retries, up to two hours total; read daily floor before every `colab new`. First refusal approximately 17:45 UTC on 2026-10-18. Source setup.sh before CLI.

Once allocated:
1. `colab exec -s b9-101c -f scripts/bootstrap_setup.py` starts detached install and writes VM deadline (3 hours max).
2. Upload scripts/train.py and scripts/driver.py to /content/exp101c/.
3. Check setup.log and setup.done; `colab exec -s b9-101c -f scripts/launch.py` starts detached pilot, unless confirm.authorized exists.
4. `bash scripts/sync.sh` frequently. Pilot results must both finish before sizing.
5. Inspect pilot and correct implementation failures, recording changes before confirmation.
6. `python scripts/pilot_register.py` computes sizing and H3 then commits README and raw pilot results. No confirm results may exist.
7. Upload updated train.py. Write `/content/exp101c/confirm.authorized` on named VM, then launch.py starts confirmation. Never authorize before the registration commit succeeds.
8. Poll/sync frequently; check CU via rate × named-session elapsed, not shared balance alone. Hard cap6CU, VMdeadline3h at L4 1.54CU/h is conservative. Stop b9-101c immediately on completion/failure, after syncing.
9. Local analysis/figures, H3 evaluation, writeup and 1200–1700 word blog. Diagram has been rendered and visually checked; regenerate from scripts/diagram.py if sample size changes.
10. Update timing actual, README compute/timing, include evidence, final commit. Finish 5-line summary.

Known: protocol uses 120 steps/batch16 = 1920 actual examples per seed from 4096-item pool. Report accurately. LoRA and QLoRA adapter parameters are BF16, optimizer uses explicit FP32 master copies and FP32 AdamW states. Scoring is supervised next-token class codes, not free generation. Checkpoint overhead included in training wall clock.

Update 2026-10-18 19:25 UTC: b9-101c allocated on L4 at 19:23:49 UTC. Setup complete; pilot launch requested. Preserve this named VM across sandbox reboot. VM deadline 22:23:57 UTC (3h, estimated max 4.62 CU). Do not allocate a replacement without checking named session. FP32 optimizer master weights added to checkpoints before pilot launch.

Confirmation authorized after preregistration commit a3a5d23. Pilot sizing fixed 10,991 DBpedia test items; H3 ratio prediction 1.3921 with interval [1.1137,1.6705]. Detached confirm PID 11412 on b9-101c. Do not launch again without checking PID and logs. Sync results with scripts/sync.sh, then finish analyze/audit/figures and manuscripts.

Reporting scripts now ready (do not generate until all ten results exist):
- `scripts/monitor.py` polls and snapshots with no changes to training. If sandbox reboot killed monitor, check named VM with poll.py before restarting monitor; remote driver remains detached.
- On confirm.done and ten `confirm_{arm}_{seed}.json`: run audit_results.py, sync once more, then stop_completed.py IMMEDIATELY to stop b9-101c and record results/stop.json. Never touch another session.
- Run analyze.py and figures.py. Inspect 3 charts + table. Run finish_accounting.py, report.py, blog.py. Blog generator asserts 1200–1700 words/no em dash; review actual prose and verdicts. Re-run accounting/report/blog at final save to update end timing.
- report.py preserves the registered README and appends a clearly marked final execution record. Preregistration hypotheses must not change.
- Finish by git add/commit locally; no push. Five-line final summary verdict, headline delta, key accuracy/memory/time, GPU estimated CU, hash.
- Completed fits so far: consult results/wait_status.json and monitor.log. First confirm QLoRA finished; first LoRA evaluating at 19:42 UTC. Both pilot JSON files and pilot-derived registration committed. No confirmatory analysis yet.

Recovery 22:47 UTC: original VM was pruned at 20:46:18 UTC (CLI history saved). First session upper compute ~2.12 CU. Replacement b9-101c L4 allocated at 22:47, detached recovery driver restores five completed confirmation JSONs and skips them. Missing LoRA 1703, LoRA/QLoRA 1704/1705 only. No training changes. Replacement deadline 2h, total upper 5.21 CU. Run sync.sh often. First VM logs preserved in results/first_remote_logs. Final accounting scripts MUST be updated for TWO sessions. Do not use original single-session accounting as-is.

Final recovery review: nine confirmation fits saved locally; only confirm_lora_1705 remains. Named detached VM driver is active. Both-VM accounting script is updated; finalization is ready. Do not rerun completed fits.

COMPLETE: all ten confirmation fits audited. b9-101c stopped successfully. Analysis, three 150 dpi charts, full table PNG, rendered Excalidraw diagram, README execution record, writeup and 1655-word blog saved. Verdict INCONCLUSIVE; H1 CI [-1.12,+0.75] crosses -1 pp, H2/H3 PROVEN. Two-VM estimated compute 3.091 CU. No further GPU work is needed.
