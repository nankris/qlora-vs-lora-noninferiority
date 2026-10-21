import subprocess,sys,time,json,os
from pathlib import Path
root=Path('/content/exp101c');phase=sys.argv[1];deadline=float((root/'deadline.txt').read_text())
for seed in ([700] if phase=='pilot' else range(1701,1706)):
    for arm in (['lora','qlora'] if seed%2==0 else ['qlora','lora']):
        if time.time()>deadline:raise RuntimeError('CU safety deadline reached')
        subprocess.run([sys.executable,str(root/'train.py'),'--phase',phase,'--arm',arm,'--seed',str(seed)],check=True,timeout=max(1,deadline-time.time()))
(root/f'{phase}.done').write_text(str(time.time()))
