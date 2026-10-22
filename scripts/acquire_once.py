"""One allocation attempt, safe to rerun after a sandbox reboot."""
import datetime,json,os,re,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
r=root/'results'
now=datetime.datetime.now(datetime.timezone.utc)
# The first refusal is documented in the original resume record.
deadline=datetime.datetime.fromisoformat('2026-10-18T19:45:00+00:00')
if now>=deadline:
 print('Two-hour allocation window exhausted. No new session requested.');raise SystemExit(2)
rows=[json.loads(x) for x in (r/'allocation_attempts.jsonl').read_text().splitlines()]
if rows and (now-datetime.datetime.fromisoformat(rows[-1]['time_utc'])).total_seconds()<300:
 print('Five-minute retry interval has not elapsed.');raise SystemExit(3)
s=subprocess.run(['colab','sessions'],capture_output=True,text=True,check=True)
if '[qlora-vs-lora]' in s.stdout:
 print('Named session exists. Reconnect instead of allocating.');raise SystemExit(4)
u=subprocess.run(['colab','usage'],capture_output=True,text=True,check=True)
print(u.stdout,flush=True)
floor=float(Path(os.environ.get('CU_FLOOR_FILE','cu_floor.txt')).read_text().strip())
balance=float(re.search(r'Current balance:\s*([\d.]+)',u.stdout)[1])
if balance<=floor:
 (r/'floor_wait.json').write_text(json.dumps({'retry_after_unix':time.time()+900,'balance':balance,'floor':floor}))
 print('Daily floor reached. Wait 15 minutes.');raise SystemExit(5)
p=r/'floor_wait.json'
if p.exists() and time.time()<json.loads(p.read_text())['retry_after_unix']:
 print('Fifteen-minute floor interval has not elapsed.');raise SystemExit(5)
result=subprocess.run(['colab','new','--gpu','L4','-s','qlora-vs-lora'],capture_output=True,text=True)
print(result.stdout,result.stderr,flush=True)
with (r/'allocation_attempts.jsonl').open('a') as f:
 f.write(json.dumps({'time_utc':now.isoformat(),'result':result.stdout+result.stderr,'balance':balance,'floor':floor,'exit_code':result.returncode})+'\n')
if result.returncode==0:
 (r/'allocation.json').write_text(json.dumps({'unix':time.time(),'output':result.stdout,'balance_before':balance}))
raise SystemExit(result.returncode)
