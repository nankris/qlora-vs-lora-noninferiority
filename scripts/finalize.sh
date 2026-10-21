#!/usr/bin/env bash
set -euo pipefail
source /workspace/tools/setup.sh
cd /workspace/work/tig/101c-qlora-vs-lora-noninferiority
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
PYTHON=/workspace/tools/mlenv/bin/python
"$PYTHON" scripts/audit_results.py
"$PYTHON" - <<'PY'
import json
from pathlib import Path
assert json.loads(Path('results/audit.json').read_text())['complete']
assert Path('results/remote_logs/confirm.done').exists()
PY
if [ ! -f results/stop.json ]; then
  "$PYTHON" scripts/stop_completed.py
fi
"$PYTHON" scripts/finish_accounting.py
"$PYTHON" scripts/analyze.py
"$PYTHON" scripts/figures.py
"$PYTHON" scripts/report.py
"$PYTHON" scripts/blog.py
"$PYTHON" scripts/proof.py
"$PYTHON" scripts/validate_artifacts.py
