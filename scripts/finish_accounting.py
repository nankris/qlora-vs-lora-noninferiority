"""Account for both named L4 lifetimes, including the pruned first VM."""
import json,time,datetime
from pathlib import Path
r=Path(__file__).resolve().parents[1]/'results'
stop=json.loads((r/'stop.json').read_text());assert stop['returncode']==0
first=json.loads((r/'first_session.json').read_text())
attempts=[json.loads(x) for x in (r/'allocation_attempts.jsonl').read_text().splitlines()]
start1=datetime.datetime.fromisoformat([x for x in attempts if x.get('exit_code')==0][-1]['time_utc']).timestamp()
end1=datetime.datetime.fromisoformat(first['terminated']['timestamp']).timestamp()
second=json.loads((r/'replacement_allocation.json').read_text())
# Replacement request began before ready timestamp; CLI session creation recorded
# immediately on success. Add 10 seconds conservatively for allocation overhead.
start2=second['ready_unix']-10;end2=stop['end_unix']
duration=end1-start1+end2-start2
runs=[json.loads(p.read_text()) for p in r.glob('*_lora_*.json')]+[json.loads(p.read_text()) for p in r.glob('*_qlora_*.json')]
assert len(runs)==12,len(runs)
train=sum(x['train_seconds'] for x in runs)/60;ev=sum(x['eval_seconds'] for x in runs)/60
compute={'gpu':'NVIDIA L4','session':'b9-101c','n_vms':2,'rate_cu_per_hour':1.54,'rate_evidence':'Recorded account rates 1.54/hr for one L4, 3.08/hr for two. Shared balance cannot isolate spend.','sessions':[{'start_unix':start1,'end_unix':end1,'end_reason':'pruned (CLI history)','minutes':(end1-start1)/60},{'start_unix':start2,'end_unix':end2,'end_reason':'explicit stop after completion','minutes':(end2-start2)/60}], 'session_minutes':duration/60,'estimated_cu':duration/3600*1.54,'accounting_note':'Conservative elapsed estimate includes entire first VM until recorded pruning, all idle time, and 10 seconds before replacement ready. Not a separate account invoice.','training_minutes':train,'evaluation_minutes':ev,'complete_pilot_runs':2,'complete_confirmatory_runs':10,'cap_cu':6,'stopped':True}
assert compute['estimated_cu']<6
(r/'compute.json').write_text(json.dumps(compute,indent=2))
t=json.loads((r/'timing.json').read_text());now=time.time();initial=datetime.datetime.fromisoformat(t['start_ist']).timestamp();actual=(now-initial)/60;writeup=(now-end2)/60
t.update(end_ist=datetime.datetime.fromtimestamp(now,datetime.timezone(datetime.timedelta(hours=5,minutes=30))).isoformat(),actual_minutes=actual,GPU='NVIDIA L4',per_stage_minutes={'setup':actual-train-ev-writeup,'GPU run':train,'eval':ev,'writeup':writeup},stage_definition='GPU run/eval sum completed worker measurements including pilot. Writeup is elapsed after final VM stop including analysis/figures. Setup is remaining elapsed, including allocation wait, downloads, loading, interruptions, lost partial work, registration and monitoring. Early drafting overlapped GPU work and is not added twice.',estimated_cu=compute['estimated_cu'])
(r/'timing.json').write_text(json.dumps(t,indent=2))
print(json.dumps(compute,indent=2));print(json.dumps(t,indent=2))
