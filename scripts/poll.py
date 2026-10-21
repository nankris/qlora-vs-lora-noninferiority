"""Execute on the named VM with colab exec, then download snapshot.tar.gz."""
import subprocess,json,os,tarfile,time
from pathlib import Path
r=Path('/content/exp101c')
print(subprocess.check_output(['nvidia-smi'],text=True))
for name in ['setup.log','pilot.log','confirm.log']:
    p=r/name
    if p.exists():print(name,p.read_text()[-5000:])
print('DONE',list(r.glob('*.done')))
for pidfile in r.glob('*.pid'):
    pid=pidfile.read_text().strip()
    print('PROCESS',pidfile.name,subprocess.run(['ps','-p',pid,'-o','pid,stat,etime,args'],capture_output=True,text=True).stdout)
print('CHECKPOINTS',[(p.name,p.stat().st_size) for p in r.glob('*.pt')])
if not (r/'environment.txt').exists():
    (r/'environment.txt').write_text(subprocess.check_output(['python','-m','pip','freeze'],text=True))
with tarfile.open(r/'snapshot.tar.gz.tmp','w:gz') as t:
    for p in (r/'results').glob('*'):t.add(p,arcname='results/'+p.name)
    for p in r.glob('*.log'):t.add(p,arcname='results/remote_logs/'+p.name)
    for p in r.glob('*.json'):t.add(p,arcname='results/remote_logs/'+p.name)
    for p in r.glob('*.txt'):t.add(p,arcname='results/remote_logs/'+p.name)
    for p in r.glob('*.done'):t.add(p,arcname='results/remote_logs/'+p.name)
    for p in r.glob('*.py'):t.add(p,arcname='results/remote_code/'+p.name)
os.replace(r/'snapshot.tar.gz.tmp',r/'snapshot.tar.gz')
print('snapshot written',time.time())
