"""Stop only this completed experiment's named session, and keep evidence."""
import json,subprocess,time
from pathlib import Path
r=Path(__file__).resolve().parents[1]
assert len(list((r/'results').glob('confirm_*_*.json')))==10
assert (r/'results/remote_logs/confirm.done').exists()
start=time.time();p=subprocess.run(['colab','stop','-s','qlora-vs-lora'],capture_output=True,text=True)
x={'session':'qlora-vs-lora','start_unix':start,'end_unix':time.time(),'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
(r/'results/stop.json').write_text(json.dumps(x,indent=2));print(p.stdout,p.stderr)
with (r/'figures/colab_run.txt').open('a') as f:f.write('\nSTOP '+json.dumps(x)+'\n')
raise SystemExit(p.returncode)
