"""Short-sleep acquisition loop. Original retry deadline is in acquire_once.py."""
from pathlib import Path
import subprocess,time,json,datetime,sys
root=Path(__file__).resolve().parents[1]
while True:
 rows=[json.loads(x) for x in (root/'results/allocation_attempts.jsonl').read_text().splitlines()]
 last=datetime.datetime.fromisoformat(rows[-1]['time_utc']).timestamp() if rows else 0
 delay=max(0,last+302-time.time())
 if delay>0:
  time.sleep(min(30,delay));continue
 p=subprocess.run([sys.executable,str(root/'scripts/acquire_once.py')],cwd=root)
 if p.returncode in [0,2,4]:raise SystemExit(p.returncode)
 if p.returncode==5:time.sleep(30)
