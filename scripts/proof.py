"""Create readable run evidence while preserving the complete polling transcript."""
import json,re,shutil
from pathlib import Path
r=Path(__file__).resolve().parents[1];f=r/'figures/colab_run.txt';archive=r/'results/poll_history.txt'
if not archive.exists():shutil.copyfile(f,archive)
history=archive.read_text();sections=['Experiment 101c: recorded terminal evidence.\nFull polling transcript: results/poll_history.txt\nFull VM logs: results/remote_logs/\n']
for name in ['confirmation_start.json','started.json']:
    p=r/'results/remote_logs'/name
    if p.exists():sections.append(name+'\n'+p.read_text())
p=r/'results/remote_logs/nvidia-smi.txt'
if p.exists():sections.append('GPU at setup\n'+p.read_text())
# Include one busy-GPU sample directly from the recorded polling transcript.
for block in re.split(r'(?=Sun Oct)',history):
    if '100%      Default' in block and 'NVIDIA-SMI' in block:
        sections.append('One recorded active-GPU snapshot\n'+block.split('setup.log')[0]);break
for folder,phase in [('first_remote_logs','pilot'),('first_remote_logs','confirm'),('remote_logs','confirm')]:
    p=r/f'results/{folder}/{phase}.log';lines=p.read_text().splitlines()
    kept=[s for s in lines if ' step ' in s or s.startswith('RESULT ')]
    sections.append(phase.upper()+' training checkpoints and measured results\n'+'\n'.join(kept))
for name in ['stop.json','compute.json','audit.json']:
    sections.append(name+'\n'+(r/'results'/name).read_text())
f.write_text('\n\n'.join(sections)+'\n')
print('Evidence excerpts:',f)
