"""Run locally only after copying complete pilot results, before confirmation."""
import json,math,subprocess
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];r=root/'results'
assert not list(r.glob('confirm_*.json')),'Confirmation already started; do not re-register'
l=json.loads((r/'pilot_lora_700.json').read_text());q=json.loads((r/'pilot_qlora_700.json').read_text());assert l['gold']==q['gold']
n=len(l['gold']);dis=float(np.mean(np.array(l['pred'])!=q['pred']));z=1.96
upper=(dis+z*z/(2*n)+z*math.sqrt(dis*(1-dis)/n+z*z/(4*n*n)))/(1+z*z/n)
needed=math.ceil(z*z*upper/.007**2);nconfirm=max(6000,needed)
if nconfirm>14000:raise RuntimeError('Pilot sizing exceeds pre-confirmation feasible test cap; review before proceeding')
ratio=q['train_seconds']/l['train_seconds'];low=.8*ratio;high=1.2*ratio
reg={'pilot_n':n,'pilot_disagreement':dis,'pilot_disagreement_wilson_upper':upper,'sizing_n_for_0.7pp_halfwidth':needed,'confirmation_n':nconfirm,'pilot_speed_ratio':ratio,'prediction_interval':[low,high],'pilot_accuracy_lora':l['accuracy'],'pilot_accuracy_qlora':q['accuracy']}
(r/'pilot_registration.json').write_text(json.dumps(reg,indent=2))
text=f'''
## Pilot-derived addendum, committed before confirmation
Pilot uses AG News only. LoRA accuracy {l['accuracy']*100:.2f}%, QLoRA {q['accuracy']*100:.2f}%; n={n}. Prediction disagreement {dis*100:.2f}% (Wilson 95% upper bound {upper*100:.2f}%). Sizing with that upper bound and a 0.7 pp item half-width requires {needed:,} items; confirmation fixed to **{nconfirm:,} DBpedia-14 test items and five paired seeds**. This is approximate sizing, not a guarantee of power: task transfer and seed variance remain uncertain.

H3 numeric prediction: QLoRA training takes **{ratio:.4f}×** LoRA, from the paired pilot training times {q['train_seconds']:.2f} / {l['train_seconds']:.2f} seconds. Predict the confirmatory 95% seed-bootstrap ratio CI lies inside [{low:.4f}, {high:.4f}] AND above 1.0. A 20% relative prediction tolerance represents the cost-estimation error we consider operationally material. PROVEN if both conditions hold; DISPROVEN if the CI is disjoint from that interval or upper bound <=1.0; otherwise INCONCLUSIVE. If the pilot ratio is below one, the required slowdown prediction is visibly unsupported by the pilot and the outcome is still evaluated without changing this rule.
'''
with (root/'README.md').open('a') as f:f.write(text)
p=root/'scripts/train.py';s=p.read_text();s=s.replace('ntest=256 if pilot else 6000',f'ntest=256 if pilot else {nconfirm}');p.write_text(s)
print(json.dumps(reg,indent=2))
subprocess.run(['git','add','README.md','scripts/train.py','results/pilot_registration.json','results/pilot_lora_700.json','results/pilot_qlora_700.json'],cwd=root,check=True)
subprocess.run(['git','commit','-m','Freeze pilot-derived sample size and wall-time prediction before confirmation'],cwd=root,check=True)
