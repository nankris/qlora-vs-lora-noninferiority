import subprocess,os,time,json
from pathlib import Path
r=Path('/content/qlora_vs_lora');r.mkdir(exist_ok=True)
# Detached installation survives client loss.
p=subprocess.Popen(['bash','-c','python -m pip install -q "transformers==4.57.1" "peft==0.17.1" "bitsandbytes==0.48.1" "datasets==4.3.0" accelerate && touch /content/qlora_vs_lora/setup.done'],stdout=open(r/'setup.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
print('setup PID',p.pid)
(r/'nvidia-smi.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
print(subprocess.check_output(['nvidia-smi'],text=True))
(r/'deadline.txt').write_text(str(time.time()+3*3600))
(r/'started.json').write_text(json.dumps({'unix':time.time(),'deadline_unix':time.time()+3*3600}))
