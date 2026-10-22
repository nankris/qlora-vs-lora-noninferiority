"""Remote resumable worker. Run each arm in a clean subprocess via driver.py."""
import os,time,json,argparse,random,hashlib
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,set_seed
from peft import LoraConfig,get_peft_model
from datasets import load_dataset
p=argparse.ArgumentParser();p.add_argument('--arm',choices=['lora','qlora']);p.add_argument('--seed',type=int);p.add_argument('--phase',choices=['pilot','confirm']);a=p.parse_args()
root=Path('/content/qlora_vs_lora');out=root/'results';out.mkdir(exist_ok=True,parents=True)
tag=f'{a.phase}_{a.arm}_{a.seed}';dest=out/f'{tag}.json'
if dest.exists(): print('SKIP',tag,flush=True);raise SystemExit
start=time.time();set_seed(a.seed);torch.set_num_threads(2)
model_id='Qwen/Qwen3-1.7B';tok=AutoTokenizer.from_pretrained(model_id);tok.pad_token=tok.eos_token
pilot=a.phase=='pilot';ntrain=512 if pilot else 4096;ntest=256 if pilot else 10991;steps=30 if pilot else 120
name='fancyzhx/ag_news' if pilot else 'fancyzhx/dbpedia_14';ds=load_dataset(name)
trids=np.random.default_rng(101).permutation(len(ds['train']))[:ntrain];teids=np.random.default_rng(202).permutation(len(ds['test']))[:ntest]
classes=ds['train'].features['label'].names;codes=list('ABCDEFGHIJKLMN')[:len(classes)]
codeids=[tok.encode(c,add_special_tokens=False)[0] for c in codes]
assert all(len(tok.encode(c,add_special_tokens=False))==1 for c in codes)
legend='; '.join(f'{c}={s}' for c,s in zip(codes,classes))
def encode(row):
    text=row.get('text',row.get('content',''))
    # Reserve space for label legend and answer suffix; truncate content, not suffix.
    prefix=tok.encode('Classify the text. '+legend+'\nText: ',add_special_tokens=False)
    suffix=tok.encode('\nAnswer:',add_special_tokens=False)
    content=tok.encode(text,add_special_tokens=False)[:128-len(prefix)-len(suffix)]
    ids=prefix+content+suffix
    assert len(ids)<=128 and len(content)>0
    return ids,int(row['label'])
train=[encode(ds['train'][int(i)]) for i in trids];test=[encode(ds['test'][int(i)]) for i in teids]
manifest=out/f'{a.phase}_data.json'
if not manifest.exists(): manifest.write_text(json.dumps({'dataset':name,'train_indices':trids.tolist(),'test_indices':teids.tolist(),'classes':classes,'model':model_id,'dataset_fingerprint':{k:v._fingerprint for k,v in ds.items()},'train_prompt_tokens':sum(len(x) for x,y in train),'test_prompt_tokens':sum(len(x) for x,y in test),'label_token_ids':codeids,'legend':legend,'max_prompt_length':128,'prefix_tokens':len(tok.encode('Classify the text. '+legend+'\nText: ',add_special_tokens=False)),'suffix_tokens':len(tok.encode('\nAnswer:',add_special_tokens=False))},indent=2))
kw={'torch_dtype':torch.bfloat16,'device_map':{'':'cuda:0'},'attn_implementation':'sdpa'}
if a.arm=='qlora': kw['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
m=AutoModelForCausalLM.from_pretrained(model_id,**kw);m.config.use_cache=False
for param in m.parameters():param.requires_grad=False
m.enable_input_require_grads();m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
set_seed(a.seed)  # Pair adapter initialization independent of quantized loading path.
m=get_peft_model(m,LoraConfig(r=16,lora_alpha=32,lora_dropout=0,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
for param in m.parameters():
    if param.requires_grad:param.data=param.data.to(torch.bfloat16)
# FP32 Adam states via FP32 master optimizer copies, identical for both methods.
params=[p for p in m.parameters() if p.requires_grad];initial_hash=hashlib.sha256(b''.join(p.detach().cpu().view(torch.uint8).numpy().tobytes() for p in params)).hexdigest();masters=[p.detach().float().clone().requires_grad_() for p in params]
opt=torch.optim.AdamW(masters,lr=2e-4,weight_decay=0)
def batch(items):
    lengths=[len(x) for x,y in items];mx=max(lengths);ids=torch.full((len(items),mx),tok.pad_token_id,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
    for j,(x,y) in enumerate(items):ids[j,:len(x)]=torch.tensor(x,device='cuda');mask[j,:len(x)]=1
    return ids,mask,torch.tensor(lengths,device='cuda')-1,torch.tensor([codeids[y] for x,y in items],device='cuda')
checkpoint=root/f'{tag}.pt';rng=np.random.default_rng(a.seed);order=rng.permutation(ntrain);losses=[];elapsed=0;begin=0;oldpeak=0;oldreserved=0
if checkpoint.exists():
    ck=torch.load(checkpoint,weights_only=False);oldpeak=ck.get('peak_allocated_bytes',0);oldreserved=ck.get('peak_reserved_bytes',0);m.load_state_dict(ck['adapter'],strict=False)
    for master,saved in zip(masters,ck['masters']):master.data.copy_(saved)
    opt.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state(ck['cuda_rng']);begin=ck['step'];losses=ck['losses'];elapsed=ck['elapsed']
m.train();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.time()
for step in range(begin,steps):
    total=0
    for micro in range(4):
        ix=[int(order[(step*16+micro*4+j)%ntrain]) for j in range(4)]
        ids,mask,pos,target=batch([train[i] for i in ix])
        # Only materialize final-position vocabulary logits (logits_to_keep tensor indexes varies by length).
        hidden=m.base_model.model.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
        logits=m.base_model.model.lm_head(hidden[torch.arange(len(ix),device='cuda'),pos])
        loss=torch.nn.functional.cross_entropy(logits.float(),target);(loss/4).backward();total+=loss.item()/4
    for master,param in zip(masters,params):master.grad=param.grad.float()
    torch.nn.utils.clip_grad_norm_(masters,1.0);opt.step();opt.zero_grad(set_to_none=True)
    for master,param in zip(masters,params):param.data.copy_(master.data)
    m.zero_grad(set_to_none=True);losses.append(total)
    if (step+1)%10==0:
        torch.cuda.synchronize();spent=elapsed+time.time()-t
        ck={'adapter':{k:v.cpu() for k,v in m.state_dict().items() if 'lora_' in k},'optimizer':opt.state_dict(),'masters':[p.detach().cpu() for p in masters],'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'step':step+1,'losses':losses,'elapsed':spent,'peak_allocated_bytes':max(oldpeak,torch.cuda.max_memory_allocated()),'peak_reserved_bytes':max(oldreserved,torch.cuda.max_memory_reserved())}
        torch.save(ck,str(checkpoint)+'.tmp');os.replace(str(checkpoint)+'.tmp',checkpoint)
        print(tag,'step',step+1,'loss',round(total,4),'seconds',round(spent,2),flush=True)
torch.cuda.synchronize();train_s=elapsed+time.time()-t;peak=max(torch.cuda.max_memory_allocated(),ck.get('peak_allocated_bytes',0) if checkpoint.exists() else 0);reserved=max(torch.cuda.max_memory_reserved(),ck.get('peak_reserved_bytes',0) if checkpoint.exists() else 0)
m.eval();pred=[];evalstart=time.time()
with torch.no_grad():
    for st in range(0,len(test),32):
        items=test[st:st+32];ids,mask,pos,target=batch(items)
        h=m.base_model.model.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
        logits=m.base_model.model.lm_head(h[torch.arange(len(items),device='cuda'),pos]);pred.extend(logits[:,codeids].argmax(-1).cpu().tolist())
torch.cuda.synchronize();eval_s=time.time()-evalstart
res={'tag':tag,'arm':a.arm,'seed':a.seed,'phase':a.phase,'accuracy':float(np.mean(np.array(pred)==[y for x,y in test])),'pred':pred,'gold':[y for x,y in test],'train_seconds':train_s,'eval_seconds':eval_s,'total_seconds':time.time()-start,'peak_allocated_bytes':peak,'peak_reserved_bytes':reserved,'losses':losses,'initial_adapter_sha256':initial_hash,'trainable_parameters':sum(p.numel() for p in params),'training_prompt_tokens':sum(len(train[int(order[(i%ntrain)])][0]) for i in range(steps*16)),'training_target_tokens':steps*16,'eval_prompt_tokens':sum(len(x) for x,y in test),'eval_scored_tokens':len(test),'model_revision':getattr(m.config,'_commit_hash',None),'gpu':torch.cuda.get_device_name(),'torch':torch.__version__}
Path(str(dest)+'.tmp').write_text(json.dumps(res));os.replace(str(dest)+'.tmp',dest);checkpoint.unlink(missing_ok=True);print('RESULT',json.dumps({k:v for k,v in res.items() if k not in ['pred','gold','losses']}),flush=True)
