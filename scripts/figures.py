import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];a=json.loads((root/'results/analysis.json').read_text());out=root/'figures'
plt.rcParams.update({'font.size':12,'axes.titlesize':16,'axes.labelsize':12,'figure.dpi':150,'savefig.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
colors=['#277da8','#d17928'];rows=a['rows']
fig,ax=plt.subplots(figsize=(8,4.8))
for i,r in enumerate(rows):
    lo,hi=r['accuracy_ci'];ax.errorbar(i,r['accuracy_percent'],yerr=[[r['accuracy_percent']-lo],[hi-r['accuracy_percent']]],fmt='o',capsize=7,color=colors[i],markersize=10,label=r['method'])
    seeds=[json.loads((root/f"results/confirm_{r['method'].lower()}_{s}.json").read_text())['accuracy']*100 for s in range(1701,1706)]
    ax.scatter(np.linspace(i-.10,i+.10,5),seeds,color=colors[i],s=18,alpha=.65)
ax.set_xticks([0,1],[r['method'] for r in rows]);ax.set_ylabel('Test accuracy (%)');ax.set_title('DBpedia-14 accuracy across five paired seeds');ax.grid(axis='y',alpha=.2)
fig.text(.12,.01,'Whiskers: paired seed × item bootstrap 95% CIs. Small dots: individual seeds.',fontsize=9);fig.tight_layout(rect=(0,.05,1,1));fig.savefig(out/'accuracy.png');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,4.6))
for ax,key,title,unit in zip(axs,['peak_GiB','train_minutes'],['Peak allocated training VRAM','Synchronized training wall time'],['GiB','Minutes per run']):
    vals=[r[key] for r in rows];ax.bar([r['method'] for r in rows],vals,color=colors,width=.6);ax.set_title(title);ax.set_ylabel(unit)
    for i,v in enumerate(vals):ax.text(i,v*1.02,f'{v:.2f}',ha='center')
    ax.set_ylim(0,max(vals)*1.18);ax.grid(axis='y',alpha=.2)
fig.suptitle('The memory saving and its time cost',fontsize=18);fig.text(.06,.01,'VRAM: PyTorch allocated. Training excludes loading/evaluation; includes recovery checkpoints.',fontsize=9);fig.tight_layout(rect=(0,.05,1,1));fig.savefig(out/'resources.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(8.5,5.2));lo,hi=a['difference_ci'];d=a['difference_pp'];seed_d=a['paired_seed_difference_pp']
left=min(-1.3,lo-.2,min(seed_d)-.2);right=max(.5,hi+.2,max(seed_d)+.2)
ax.axvspan(-1,right,alpha=.08,color=colors[0]);ax.axvline(-1,ls='--',color='#ba3434',label='Registered margin: -1 pp');ax.axvline(0,lw=1,color='#999999')
ax.errorbar(d,0,xerr=[[d-lo],[hi-d]],fmt='o',color=colors[1],capsize=8,markersize=10)
ax.scatter(seed_d,range(1,6),s=38,color=colors[0],alpha=.8)
ax.set_yticks(range(6),['Mean + 95% CI']+[f'Seed {seed}' for seed in range(1701,1706)])
ax.invert_yaxis();ax.set_xlim(left,right);ax.set_ylim(5.6,-1.0)
ax.text(.98,.95,f'{d:+.2f} pp; 95% CI [{lo:+.2f}, {hi:+.2f}]',transform=ax.transAxes,ha='right',va='top',fontsize=11)
ax.set_xlabel('QLoRA minus LoRA accuracy (percentage points)');ax.set_title(f"Accuracy non-inferiority: {a['H1']}");ax.legend(loc='lower left',fontsize=10);ax.grid(axis='y',alpha=.12)
fig.text(.05,.01,'Seed dots show observed paired differences; whisker resamples paired seeds and shared test items.',fontsize=9)
fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/'noninferiority.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(13,3.0));ax.axis('off');data=[]
for r in rows:data.append([r['method'],f"{r['accuracy_percent']:.2f}% [{r['accuracy_ci'][0]:.2f}, {r['accuracy_ci'][1]:.2f}]",f"{r['seed_sd_pp']:.3f}",f"{r['peak_GiB']:.3f} / {r['reserved_GiB']:.3f}",f"{r['train_minutes']:.3f}",f"{r['eval_minutes']:.3f}"])
t=ax.table(cellText=data,colLabels=['Method','Accuracy and 95% CI (%)','Seed SD (pp)','Peak allocated /\nreserved (GiB)','Train min/run','Eval min/run'],colWidths=[.09,.29,.11,.23,.14,.14],loc='center',cellLoc='center');t.auto_set_font_size(False);t.set_fontsize(11);t.scale(1,2.5)
for (i,j),cell in t.get_celld().items():
    cell.set_edgecolor('#dddddd')
    if i==0:cell.set_facecolor('#243b53');cell.set_text_props(color='white',weight='bold')
    else:cell.set_facecolor('#f0f5f9' if i%2 else '#ffffff')
ax.set_title('Qwen3-1.7B on DBpedia-14 | five seeds per method',pad=10);fig.tight_layout();fig.savefig(out/'table.png');plt.close(fig)
