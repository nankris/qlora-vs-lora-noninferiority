"""Paired crossed bootstrap: resample common seeds and common test items."""
import json,csv
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];r=root/'results';rng=np.random.default_rng(101)
lo=[json.loads((r/f'confirm_lora_{s}.json').read_text()) for s in range(1701,1706)]
ql=[json.loads((r/f'confirm_qlora_{s}.json').read_text()) for s in range(1701,1706)]
assert all(x['gold']==lo[0]['gold'] for x in lo+ql)
gold=np.array(lo[0]['gold']);l=np.array([x['pred'] for x in lo])==gold;q=np.array([x['pred'] for x in ql])==gold;d=q.astype(float)-l
N=l.shape[1];boots=[];abs_l=[];abs_q=[];mem=[];speed=[]
for _ in range(10000):
    si=rng.integers(0,5,5);ii=rng.integers(0,N,N)
    boots.append(d[si][:,ii].mean()*100);abs_l.append(l[si][:,ii].mean()*100);abs_q.append(q[si][:,ii].mean()*100)
    mem.append(np.mean([ql[s]['peak_allocated_bytes'] for s in si])/np.mean([lo[s]['peak_allocated_bytes'] for s in si]))
    speed.append(np.mean([ql[s]['train_seconds'] for s in si])/np.mean([lo[s]['train_seconds'] for s in si]))
def ci(x):return np.percentile(x,[2.5,97.5]).tolist()
h1=ci(boots);h2=ci(mem)
ver1='PROVEN' if h1[0]>-1 else ('DISPROVEN' if h1[1]<-1 else 'INCONCLUSIVE')
ver2='PROVEN' if h2[1]<.5 else ('DISPROVEN' if h2[0]>=.5 else 'INCONCLUSIVE')
verdict='PROVEN' if ver1==ver2=='PROVEN' else ('DISPROVEN' if 'DISPROVEN' in [ver1,ver2] else 'INCONCLUSIVE')
summary={'verdict':verdict,'H1':ver1,'H2':ver2,'difference_pp':d.mean()*100,'difference_ci':h1,'memory_ratio':np.mean([x['peak_allocated_bytes'] for x in ql])/np.mean([x['peak_allocated_bytes'] for x in lo]),'memory_ratio_ci':h2,'speed_ratio':np.mean([x['train_seconds'] for x in ql])/np.mean([x['train_seconds'] for x in lo]),'speed_ratio_ci':ci(speed),'n_test':N,'n_seeds':5,'rows':[]}
for name,runs,acc,b in [('LoRA',lo,l,abs_l),('QLoRA',ql,q,abs_q)]:
    summary['rows'].append({'method':name,'accuracy_percent':acc.mean()*100,'accuracy_ci':ci(b),'seed_sd_pp':acc.mean(1).std(ddof=1)*100,'peak_GiB':np.mean([x['peak_allocated_bytes'] for x in runs])/2**30,'reserved_GiB':np.mean([x['peak_reserved_bytes'] for x in runs])/2**30,'train_minutes':np.mean([x['train_seconds'] for x in runs])/60,'eval_minutes':np.mean([x['eval_seconds'] for x in runs])/60})
reg=json.loads((r/'pilot_registration.json').read_text())
sl,sh=summary['speed_ratio_ci'];pl,ph=reg['prediction_interval']
summary['H3']='PROVEN' if sl>1 and sl>=pl and sh<=ph else ('DISPROVEN' if sh<=1 or sh<pl or sl>ph else 'INCONCLUSIVE')
summary['paired_seed_difference_pp']=(100*d.mean(axis=1)).tolist()
summary['prediction_disagreement_rate']=float(np.mean(np.array([x['pred'] for x in lo])!=np.array([x['pred'] for x in ql])))
summary['bootstrap_draws']=10000
summary['token_counts']={arm:{k:sum(x[k] for x in runs) for k in ['training_prompt_tokens','training_target_tokens','eval_prompt_tokens','eval_scored_tokens']} for arm,runs in [('lora',lo),('qlora',ql)]}
(r/'analysis.json').write_text(json.dumps(summary,indent=2))
with (r/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=summary['rows'][0]);w.writeheader();w.writerows(summary['rows'])
print(json.dumps(summary,indent=2))
# A compact, inspectable export; predictions and losses remain in the raw JSONs.
records=[]
for path in sorted(r.glob('*_lora_*.json'))+sorted(r.glob('*_qlora_*.json')):
    x=json.loads(path.read_text());records.append({k:x[k] for k in ['tag','phase','arm','seed','accuracy','train_seconds','eval_seconds','total_seconds','peak_allocated_bytes','peak_reserved_bytes','training_prompt_tokens','training_target_tokens','eval_prompt_tokens','eval_scored_tokens','initial_adapter_sha256','model_revision','gpu','torch']})
with (r/'runs.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
