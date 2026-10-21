from pathlib import Path
import time,subprocess
r=Path('/content/exp101c')
while not (r/'setup.done').exists(): time.sleep(3)
subprocess.run(['python','/content/exp101c/driver.py','confirm'],check=True)
