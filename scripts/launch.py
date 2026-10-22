"""colab exec this only after installing, uploading train.py and driver.py."""
import subprocess,os
from pathlib import Path
r=Path('/content/qlora_vs_lora');phase='confirm' if (r/'confirm.start').exists() else 'pilot'
assert (r/'setup.done').exists()
pidfile=r/f'{phase}.pid'
if pidfile.exists():
    try:os.kill(int(pidfile.read_text()),0);raise RuntimeError('job already running')
    except ProcessLookupError:pass
p=subprocess.Popen(['nohup','python','-u',str(r/'driver.py'),phase],stdout=open(r/f'{phase}.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
pidfile.write_text(str(p.pid));print('detached',phase,'PID',p.pid)
