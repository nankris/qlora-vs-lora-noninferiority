"""Restore only completed results, preserve old logs locally, then detach confirmation."""
from pathlib import Path
import tarfile,time,subprocess
r=Path('/content/qlora_vs_lora')
with tarfile.open(r/'restore.tar.gz') as t:
 for m in t.getmembers():
  if m.name.startswith('results/') and '/' not in m.name[len('results/'):]:
   t.extract(m,path=r,filter='data')
# Prior accounting allows 6-2.12 CU; limit replacement lifetime to two hours.
(r/'deadline.txt').write_text(str(time.time()+2*3600))
(r/'confirm.start').write_text('Recovery of the committed confirmation start a3a5d23; unchanged protocol')
code="""from pathlib import Path
import time,subprocess
r=Path('/content/qlora_vs_lora')
while not (r/'setup.done').exists(): time.sleep(3)
subprocess.run(['python','/content/qlora_vs_lora/driver.py','confirm'],check=True)
"""
(r/'recovery_driver.py').write_text(code)
p=subprocess.Popen(['nohup','python','-u',str(r/'recovery_driver.py')],stdout=open(r/'confirm.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
(r/'confirm.pid').write_text(str(p.pid));print('Detached recovery',p.pid,'saved',list((r/'results').glob('confirm*json')))
