"""Short polls and frequent snapshots; remote GPU driver is detached separately."""
import subprocess,time,json,datetime
from pathlib import Path
r=Path(__file__).resolve().parents[1];allocation=json.loads((r/'results/replacement_allocation.json').read_text())
def stop(reason):
 start=time.time();p=subprocess.run(['colab','stop','-s','b9-101c'],capture_output=True,text=True)
 record={'session':'b9-101c','start_unix':start,'end_unix':time.time(),'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'reason':reason}
 (r/'results/stop.json').write_text(json.dumps(record,indent=2));print('STOP',record,flush=True)
 with (r/'figures/colab_run.txt').open('a') as f:f.write('\nSTOP '+json.dumps(record)+'\n')
while True:
 try:
  p=subprocess.run(['bash','scripts/sync.sh'],cwd=r,capture_output=True,text=True,timeout=100)
  (r/'results/latest_sync.txt').write_text(p.stdout+p.stderr)
  if p.returncode:print('SYNC retry needed',p.returncode,flush=True)
  else:
   log=r/'results/remote_logs/confirm.log';lines=log.read_text().splitlines() if log.exists() else []
   steps=[x for x in lines if ' step ' in x or x.startswith('RESULT')]
   files=list((r/'results').glob('confirm_*_*.json'))
   print(datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed',len(files),'latest',steps[-1][:220] if steps else 'loading',flush=True)
   status={'state':'confirmatory_job_running','completed':len(files),'session':'b9-101c','replacement_elapsed_minutes':(time.time()-allocation['ready_unix'])/60,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
   (r/'results/wait_status.json').write_text(json.dumps(status,indent=2))
   if (r/'results/remote_logs/confirm.done').exists():
    audit=subprocess.run(['/workspace/tools/mlenv/bin/python','scripts/audit_results.py'],cwd=r)
    stop('All ten confirmation fits saved; audit exit '+str(audit.returncode));break
   if any('Traceback (most recent call last)' in x for x in lines):stop('Remote failure; inspect logs');break
  if time.time()>allocation['ready_unix']+2*3600:stop('Replacement two-hour safety deadline');break
 except subprocess.TimeoutExpired:print('SYNC timeout; retrying',flush=True)
 time.sleep(40)
