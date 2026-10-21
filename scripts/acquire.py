"""Respect 5-minute slot retries, 15-minute floor retries and 2-hour limit.
Does not inspect, stop or change any other session.
"""
import subprocess,time,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
start=time.time()
while time.time()-start<7200:
    time.sleep(300)
    now=time.time();usage=subprocess.run(['colab','usage'],capture_output=True,text=True)
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),usage.stdout,flush=True)
    import re
    m=re.search(r'Current balance:\s*([\d.]+)',usage.stdout)
    floor=float(Path('/workspace/work/tig/_batch9/cu_floor.txt').read_text().strip())
    if not m:continue
    if float(m.group(1))<=floor:
        print('Below floor, waiting 15 minutes',flush=True);time.sleep(900);continue
    result=subprocess.run(['colab','new','--gpu','L4','-s','b9-101c'],capture_output=True,text=True)
    print(result.stdout,result.stderr,flush=True)
    if result.returncode==0:
        (root/'results/allocation.json').write_text(json.dumps({'unix':time.time(),'wait_minutes':(time.time()-start)/60,'output':result.stdout}))
        subprocess.run(['colab','exec','-s','b9-101c','-f',str(root/'scripts/bootstrap_setup.py')],check=True)
        for f in ['train.py','driver.py']:
            subprocess.run(['colab','upload','-s','b9-101c',str(root/'scripts'/f),'/content/exp101c/'+f],check=True)
        for _ in range(60):
            p=subprocess.run(['colab','exec','-s','b9-101c'],input="from pathlib import Path; print('READY', Path('/content/exp101c/setup.done').exists())",capture_output=True,text=True)
            if 'READY True' in p.stdout:break
            time.sleep(10)
        else:
            subprocess.run(['colab','stop','-s','b9-101c']);raise RuntimeError('Setup timeout')
        subprocess.run(['colab','exec','-s','b9-101c','-f',str(root/'scripts/launch.py')],check=True)
        break
else: print('Two-hour allocation wait exhausted',flush=True)
