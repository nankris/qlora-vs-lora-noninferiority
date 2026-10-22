from pathlib import Path
import time,subprocess
r=Path('/content/qlora_vs_lora')
while not (r/'setup.done').exists(): time.sleep(3)
subprocess.run(['python','/content/qlora_vs_lora/driver.py','confirm'],check=True)
