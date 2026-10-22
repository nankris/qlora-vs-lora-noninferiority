#!/usr/bin/env bash
set -euo pipefail
[ -z "${ENV_SETUP:-}" ] || source "$ENV_SETUP"
cd "$(dirname "$0")/.."
timeout 30s colab exec -s qlora-vs-lora -f scripts/poll.py | tee -a figures/colab_run.txt
timeout 30s colab download -s qlora-vs-lora /content/qlora_vs_lora/snapshot.tar.gz results/snapshot.tar.gz
python - <<'PY'
import tarfile
with tarfile.open('results/snapshot.tar.gz') as t:t.extractall('.',filter='data')
PY
timeout 15s colab usage | tee -a results/usage.log
