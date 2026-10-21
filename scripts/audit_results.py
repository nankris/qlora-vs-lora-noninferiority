"""Audit actual saved predictions and provenance; never fill missing runs."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
r=root/'results'
report={'complete':False,'checks':[],'missing':[]}
for phase,seeds in [('pilot',[700]),('confirm',range(1701,1706))]:
 for seed in seeds:
  pair=[]
  for arm in ['lora','qlora']:
   p=r/f'{phase}_{arm}_{seed}.json'
   if not p.exists():report['missing'].append(p.name);continue
   x=json.loads(p.read_text());pair.append(x)
   assert len(x['pred'])==len(x['gold'])
   expected=256 if phase=='pilot' else json.loads((r/'pilot_registration.json').read_text())['confirmation_n']
   assert len(x['gold'])==expected
   assert x['model_revision']=='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
   assert x['gpu']=='NVIDIA L4'
   assert all(0<=v<(4 if phase=='pilot' else 14) for v in x['pred']+x['gold'])
   assert abs(x['accuracy']-sum(a==b for a,b in zip(x['pred'],x['gold']))/len(x['gold']))<1e-12
   assert x['training_target_tokens']==(480 if phase=='pilot' else 1920)
   assert x['train_seconds']>0 and x['peak_allocated_bytes']>0
  if len(pair)==2:
   assert pair[0]['gold']==pair[1]['gold']
   assert pair[0]['initial_adapter_sha256']==pair[1]['initial_adapter_sha256']
   assert pair[0]['training_prompt_tokens']==pair[1]['training_prompt_tokens']
   report['checks'].append(f'{phase} seed {seed}: paired gold, initialization, token counts and computed accuracy match')
report['complete']=not report['missing']
(r/'audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
