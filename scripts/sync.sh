#!/usr/bin/env bash
set -euo pipefail
source /workspace/tools/setup.sh
cd /workspace/work/tig/101c-qlora-vs-lora-noninferiority
timeout 30s colab exec -s b9-101c -f scripts/poll.py | tee -a figures/colab_run.txt
timeout 30s colab download -s b9-101c /content/exp101c/snapshot.tar.gz results/snapshot.tar.gz
python - <<'PY'
import tarfile
with tarfile.open('results/snapshot.tar.gz') as t:t.extractall('.',filter='data')
PY
timeout 15s colab usage | tee -a results/usage.log
