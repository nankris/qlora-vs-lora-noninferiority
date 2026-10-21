"""Check final publication artifacts against raw outcomes and saved registration."""
import json,re
from pathlib import Path
from PIL import Image
root=Path(__file__).resolve().parents[1];r=root/'results';a=json.loads((r/'analysis.json').read_text());audit=json.loads((r/'audit.json').read_text());c=json.loads((r/'compute.json').read_text());t=json.loads((r/'timing.json').read_text())
assert audit['complete'] and a['n_test']==10991 and a['n_seeds']==5
assert a['difference_ci'][0]<=a['difference_pp']<=a['difference_ci'][1]
assert a['H1']==('PROVEN' if a['difference_ci'][0]>-1 else ('DISPROVEN' if a['difference_ci'][1]<-1 else 'INCONCLUSIVE'))
assert c['stopped'] and c['estimated_cu']<6 and c['n_vms']==2
assert abs(sum(t['per_stage_minutes'].values())-t['actual_minutes'])<1e-6
blog=(root/'blog.md').read_text();assert 1200<=len(blog.split())<=1700
assert '\u2014' not in blog
for name in ['README.md','writeup.md','blog.md']:
 s=(root/name).read_text();assert a['verdict'] in s and 'Timing:' in s
 assert not re.search(r'(?i)actual and CU pending\s*$',s)
for path in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',blog):assert (root/path).is_file(),path
pngs=['accuracy.png','resources.png','noninferiority.png','table.png','diagram.png']
for name in pngs:
 with Image.open(root/'figures'/name) as im:
  assert min(im.size)>=300
  if name!='diagram.png':assert all(abs(x-150)<1 for x in im.info['dpi'])
assert json.loads((root/'figures/diagram.excalidraw').read_text())['type']=='excalidraw'
assert 'NVIDIA L4' in (root/'figures/colab_run.txt').read_text()
summary={'validated':True,'blog_words':len(blog.split()),'charts':pngs,'complete_confirmatory_fits':10,'compute_cu':c['estimated_cu'],'verdict':a['verdict']}
(r/'artifact_validation.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
