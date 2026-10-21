import json
from pathlib import Path
root=Path(__file__).resolve().parents[1];els=[]
def element(kind,x,y,w,h,**kw):
    e={'id':str(len(els)+1),'type':kind,'x':x,'y':y,'width':w,'height':h,'angle':0,'strokeColor':'#243b53','backgroundColor':'transparent','fillStyle':'solid','strokeWidth':2,'strokeStyle':'solid','roughness':1,'opacity':100,'groupIds':[],'frameId':None,'index':None,'roundness':{'type':3} if kind=='rectangle' else None,'seed':len(els)+21,'version':1,'versionNonce':1,'isDeleted':False,'boundElements':None,'updated':1,'link':None,'locked':False};e.update(kw);els.append(e)
def box(x,y,w,h,title,body,color):
    element('rectangle',x,y,w,h,backgroundColor=color)
    text(x+15,y+18,w-30,25,title,20)
    text(x+15,y+49,w-30,h-52,body,18)
def text(x,y,w,h,s,size=18):
    
    for i,line in enumerate(s.split('\n')):
        element('text',x,y+i*size*1.5,w,size*1.5,text=line,fontSize=size,fontFamily=1,textAlign='left',verticalAlign='top',containerId=None,originalText=line,autoResize=True,lineHeight=1.25,baseline=size)
def arrow(x,y,dx,dy):
    element('arrow',x,y,abs(dx),abs(dy),points=[[0,0],[dx,dy]],lastCommittedPoint=None,startBinding=None,endBinding=None,startArrowhead=None,endArrowhead='arrow',elbowed=False)
text(30,45,1090,35,'Same model. Same training. Different base-weight precision.',20)
box(30,85,280,130,'Pilot: AG News','Qwen3-1.7B, one seed\n30 steps per arm\nSets speed prediction','#fff4d6')
box(370,85,360,130,'Freeze the protocol','Commit before confirmation\n1 pp accuracy margin\n5 seeds, 10,991 fresh test items','#e5f3ff')
arrow(315,150,45,0)
box(30,285,280,155,'DBpedia-14','4,096-item training pool\n1,920 seen per paired seed\n120 steps, batch 16\nRank 16 adapters','#e7f7ed')
box(380,255,330,105,'LoRA: BF16 base','Frozen base + BF16 adapters\nFive independent runs','#e5f3ff')
box(380,395,330,105,'QLoRA: NF4 base','Frozen base + BF16 adapters\nFive paired-seed runs','#f1e7ff')
arrow(315,330,55,-30);arrow(315,380,55,55);arrow(550,220,0,25)
box(790,285,330,170,'Compare paired results','Accuracy: 95% bootstrap CI\nPeak allocated training VRAM\nSynchronized training time\nL4 GPU, maximum 6 CU','#fff4d6')
arrow(720,305,60,35);arrow(720,445,60,-40)
text(30,550,1090,55,'Primary claim passes only if the accuracy CI clears -1 pp\nand QLoRA uses less than half the allocated training VRAM.',20)
(root/'figures/diagram.excalidraw').write_text(json.dumps({'type':'excalidraw','version':2,'source':'https://excalidraw.com','elements':els,'appState':{'viewBackgroundColor':'#ffffff','gridSize':None},'files':{}},indent=2))
