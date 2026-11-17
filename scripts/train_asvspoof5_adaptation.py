"""New-domain CE/KD, identical historical source controls, epoch-resumable."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import sys,json,random,time,contextlib,io
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from aasist_replication_common import setup_gpu,make_model,preprocess,sha,read,save
from preservation_utils import source_kl
R=Path(__file__).resolve().parents[1];O=R/'results/asvspoof5_adaptation_v1';BASE=R/'external/w2v2lcnn/models_0.63/try/anti-spoofing_feat_model.pt';SEEDS=[20260912,20260913,20260914]
sys.path.insert(0,str(R/'external/w2v2lcnn'));import model

def forward(net,x):
 with contextlib.redirect_stdout(io.StringIO()):return net(x)[1]
def main():
 setup_gpu();preflight=json.loads((O/'preflight.json').read_text());assert preflight['passed'];
 for path,h in preflight['input_sha256'].items():assert sha(R/path)==h,path
 assert json.loads((O/'audio_audit.json').read_text())['passed'];rows=read(O/'training.csv');manifest=read(O/'manifest.csv');oldrows=read(R/'results/replication_v1/adaptation_training.csv');n=len(rows);assert n==10320;target=[i for i,r in enumerate(rows) if r['is_source']=='False'];indices={r['record_id']:i for i,r in enumerate(manifest)};allwaves=np.load(O/'windows.npy',mmap_mode='r');prepared=O/'training_prepared.json'
 inputs=[Path(__file__),R/'scripts/preservation_utils.py',R/'scripts/aasist_replication_common.py',O/'preflight.json',R/'docs/asvspoof5-adaptation-v1.md',O/'training.csv',O/'audio_audit.json',R/'results/preservation_v1/frontend_provenance.json',R/'results/replication_v1/training_complete.json',R/'results/aasist_replication_v1/training_complete.json']
 rec=dict(input_sha256={str(p.relative_to(R)):sha(p) for p in inputs},seeds=SEEDS,epochs=8,batch_size=32,lr=1e-4,weight_decay=1e-4)
 if prepared.exists():assert json.loads(prepared.read_text())==rec
 else:save(prepared,rec)
 if not (O/'training_cache.json').exists():
  oldinfo=json.loads((R/'results/preservation_v1/feature_cache.json').read_text());oldpath=R/'results/preservation_v1/training_features.npy';assert sha(oldpath)==oldinfo['sha256'];oldfeat=np.load(oldpath,mmap_mode='r');feat=np.lib.format.open_memmap(O/'training_features.npy',mode='w+',dtype=np.float16,shape=oldfeat.shape);waves=np.lib.format.open_memmap(O/'training_waveforms.npy',mode='w+',dtype=np.float32,shape=(n,64600))
  for i,r in enumerate(rows):
   if r['is_source']=='True':assert r['path']==oldrows[i]['path'];feat[i]=oldfeat[i];waves[i]=preprocess(r['path'])
   else:waves[i]=allwaves[indices[r['record_id']]]
  from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Model
  from transformers.utils.hub import cached_file
  front=json.loads((R/'results/preservation_v1/frontend_provenance.json').read_text());repo=front['repo'];rev=front['cached_files'][0]['cache_snapshot']
  for f in front['cached_files']:assert sha(cached_file(repo,f['filename'],revision=rev,local_files_only=True))==f['sha256']
  proc=Wav2Vec2FeatureExtractor.from_pretrained(repo,revision=rev,local_files_only=True);enc=Wav2Vec2Model.from_pretrained(repo,revision=rev,local_files_only=True,use_safetensors=False).eval().cuda()
  with torch.inference_mode():
   for start in range(0,len(target),16):
    ix=target[start:start+16];x=proc(list(waves[ix]),sampling_rate=16000,return_tensors='pt').input_values.cuda();feat[ix]=enc(x).last_hidden_state.unsqueeze(1).transpose(2,3).half().cpu().numpy()
    if start%320==0:print('TARGET FEATURES',start,len(target),flush=True)
  feat.flush();waves.flush();del feat,waves,enc;torch.cuda.empty_cache();save(O/'training_cache.json',dict(feature_sha256=sha(O/'training_features.npy'),waveform_sha256=sha(O/'training_waveforms.npy')))
 ci=json.loads((O/'training_cache.json').read_text());assert sha(O/'training_features.npy')==ci['feature_sha256'] and sha(O/'training_waveforms.npy')==ci['waveform_sha256'];features=np.load(O/'training_features.npy',mmap_mode='r');waves=np.load(O/'training_waveforms.npy',mmap_mode='r');source=torch.tensor([r['is_source']=='True' for r in rows],device='cuda');(O/'checkpoints').mkdir(exist_ok=True);results=[]
 for family,folder in [('xlsr_lcnn','replication_v1'),('aasist','aasist_replication_v1')]:
  old=json.loads((R/'results'/folder/'training_complete.json').read_text());base=BASE if family=='xlsr_lcnn' else R/'external/aasist/models/weights/AASIST.pth';factory=lambda:torch.load(base,map_location='cpu',weights_only=False) if family=='xlsr_lcnn' else make_model(base);cache=features if family=='xlsr_lcnn' else waves;labels=torch.tensor([int(r['label_0bona_1spoof']) if family=='xlsr_lcnn' else 1-int(r['label_0bona_1spoof']) for r in rows],device='cuda');teacher=factory().eval().cuda()
  for p in teacher.parameters():p.requires_grad_(False)
  tl=torch.empty((n,2),device='cuda')
  with torch.inference_mode():
   for i in range(0,n,32):tl[i:i+32]=forward(teacher,torch.from_numpy(cache[i:i+32].copy()).float().cuda())
  for seed in SEEDS:
   for method in ['full_ce','kd10']:
    name=f'{family}_{method}_s{seed}';ck=O/'checkpoints'/f'{name}.pt';done=ck.with_suffix('.json');resume=ck.with_suffix('.resume.pt')
    if done.exists():r=json.loads(done.read_text());assert sha(ck)==r['sha256'];results.append(r);continue
    assert not ck.exists();torch.manual_seed(seed);random.seed(seed);np.random.seed(seed);student=factory().cuda();torch.manual_seed(seed);rng=np.random.RandomState(seed);opt=torch.optim.Adam(student.parameters(),lr=1e-4,weight_decay=1e-4);idx=np.arange(n);start=0;history=[]
    if resume.exists():
     d=torch.load(resume,map_location='cpu',weights_only=False);assert d['prepared_sha256']==sha(prepared);student.load_state_dict(d['model']);opt.load_state_dict(d['optimizer']);idx=d['idx'];rng.set_state(d['rng']);torch.set_rng_state(d['torch_rng']);torch.cuda.set_rng_state(d['cuda_rng']);random.setstate(d['python_rng']);np.random.set_state(d['numpy_rng']);start=d['epoch'];history=d['history']
    for ep in range(start,8):
     student.train();rng.shuffle(idx);loss_sum=0.
     for pos in range(0,n,32):
      b=idx[pos:pos+32];opt.zero_grad(set_to_none=True);logits=forward(student,torch.from_numpy(cache[b].copy()).float().cuda());ce=F.cross_entropy(logits,labels[b]);kd=source_kl(logits,tl[b],source[b]) if method=='kd10' else logits.sum()*0;loss=ce+10*kd;assert torch.isfinite(loss);loss.backward();opt.step();loss_sum+=float(loss.detach())*len(b)
     history.append(dict(epoch=ep+1,loss=loss_sum/n));tmp=resume.with_suffix('.tmp');torch.save(dict(model=student.state_dict(),optimizer=opt.state_dict(),idx=idx,rng=rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),epoch=ep+1,history=history,prepared_sha256=sha(prepared)),tmp);tmp.replace(resume);print(name,'EPOCH',ep+1,'loss',loss_sum/n,flush=True)
    student.eval().cpu();torch.save(student if family=='xlsr_lcnn' else student.state_dict(),ck);r=dict(family=family,model=f'{method}_s{seed}',method=method,seed=seed,path=str(ck.relative_to(R)),sha256=sha(ck),steps=2584,history=history,reused=False);save(done,r);results.append(r);del student,opt;torch.cuda.empty_cache()
  for seed in SEEDS:
   a=old['models'][f'source_only_s{seed}'];assert sha(R/a['path'])==a['sha256'];results.append(dict(family=family,model=f'source_only_s{seed}',method='source_only',seed=seed,path=a['path'],sha256=a['sha256'],steps=2584,reused=True))
  results.append(dict(family=family,model='original',method='original',seed='',path=str(base.relative_to(R)),sha256=sha(base),steps=0,reused=True));del teacher,tl;torch.cuda.empty_cache()
 assert len(results)==20
 if not (O/'training_complete.json').exists():save(O/'training_complete.json',dict(success=True,models=results,prepared_sha256=sha(prepared)))
 print('TRAINING COMPLETE',flush=True)
if __name__=='__main__':main()
