"""Sealed calibration before eval; resumable all-model inference, source retention."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import sys,json,csv,contextlib,io,time
from pathlib import Path
import numpy as np
from aasist_replication_common import read,sha,preprocess
from asvspoof5_metrics import calibrate,freeze
R=Path(__file__).resolve().parents[1];O=R/'results/asvspoof5_adaptation_v1'
def save(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2)
def write(p,rows):
 with p.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def now():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def main():
 import torch
 from aasist_replication_common import setup_gpu,make_model
 from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Model
 from transformers.utils.hub import cached_file
 setup_gpu();torch.manual_seed(20260919)
 sys.path.insert(0,str(R/'external/w2v2lcnn'));import model
 models=json.loads((O/'training_complete.json').read_text())['models'];assert len(models)==20
 manifest=read(O/'manifest.csv');audit=read(O/'audio_audit_rows.csv');assert len(audit)==len(manifest)==49310
 assert json.loads((O/'audio_audit.json').read_text())['passed'];assert sha(O/'windows.npy')==json.loads((O/'audio_audit.json').read_text())['windows_sha256']
 for m in models:assert sha(R/m['path'])==m['sha256']
 source_ids=set(json.loads((R/'results/interpolation_v1/prepared.json').read_text())['selection_speakers'])
 source=[r for r in read(R/'results/asvspoof2019_la_audit/validation.csv') if r['speaker_id'] in source_ids];assert len(source_ids)==10
 inputs=[Path(__file__),R/'scripts/asvspoof5_metrics.py',R/'scripts/aasist_replication_common.py',R/'docs/asvspoof5-scoring-v1.md',R/'docs/asvspoof5-adaptation-v1.md',O/'manifest.csv',O/'audio_audit.json',O/'audio_audit_rows.csv',O/'training_complete.json',R/'results/preservation_v1/frontend_provenance.json',R/'results/interpolation_v1/prepared.json',R/'results/asvspoof2019_la_audit/validation.csv',R/'external/w2v2lcnn/model.py',R/'external/aasist/models/AASIST.py',R/'external/aasist/config/AASIST.conf']
 for folder in ['replication_v1','aasist_replication_v1']:inputs.append(R/'results'/folder/'thresholds.json')
 meta=dict(input_sha256={str(p.relative_to(R)):sha(p) for p in inputs},models=models,source_records=len(source),batch=8,canonical_columns=['bonafide_logit','spoof_logit'])
 prep=O/'scoring_prepared.json'
 if prep.exists():assert json.loads(prep.read_text())==meta
 else:
  save(prep,meta);write(O/'source_selection.csv',source)
  snap=O/'scoring_code';snap.mkdir()
  for p in inputs:
   if p.suffix in ['.py','.md']:(snap/p.name).write_bytes(p.read_bytes())
 front=json.loads((R/'results/preservation_v1/frontend_provenance.json').read_text());repo=front['repo'];rev=front['cached_files'][0]['cache_snapshot']
 for f in front['cached_files']:assert sha(cached_file(repo,f['filename'],revision=rev,local_files_only=True))==f['sha256']
 proc=Wav2Vec2FeatureExtractor.from_pretrained(repo,revision=rev,local_files_only=True);enc=Wav2Vec2Model.from_pretrained(repo,revision=rev,local_files_only=True,use_safetensors=False).eval().cuda()
 nets=[(torch.load(R/m['path'],map_location='cpu',weights_only=False) if m['family']=='xlsr_lcnn' else make_model(R/m['path'])).eval().cuda() for m in models]
 for net in [enc,*nets]:
  for param in net.parameters():param.requires_grad_(False)
 def predict(x):
  with torch.inference_mode(),contextlib.redirect_stdout(io.StringIO()):
   h=enc(proc(list(x),sampling_rate=16000,return_tensors='pt').input_values.cuda()).last_hidden_state.unsqueeze(1).transpose(2,3).half().float();tx=torch.from_numpy(x.copy()).cuda();zz=[]
   for m,net in zip(models,nets):
    z=net(h if m['family']=='xlsr_lcnn' else tx)[1].cpu().numpy();zz.append(z if m['family']=='xlsr_lcnn' else z[:,::-1])
  z=np.stack(zz,axis=1).astype(float);assert np.isfinite(z).all();return z
 probes=read(R/'results/learning_vs_calibration_v1/historical_probe_manifest.csv');x=np.stack([preprocess(r['path']) for r in probes]);bat=predict(x);single=np.concatenate([predict(xx[None]) for xx in x]);checks=[]
 for j,m in enumerate(models):
  error=float(abs(bat[:,j]-single[:,j]).max());assert error<=.005
  d=dict(family=m['family'],model=m['model'],batch_single_error=error)
  if m['method'] in ['original','source_only']:
   folder='replication_v1' if m['family']=='xlsr_lcnn' else 'aasist_replication_v1';old={r['utterance_id']:r for r in read(R/'results'/folder/'scores'/f"external_{m['model']}.csv")};expected=np.array([[float(old[r['utterance_id']][k]) for k in ['bonafide_logit','spoof_logit']] for r in probes]);d['historical_error']=float(abs(bat[:,j]-expected).max());assert d['historical_error']<=.005
  checks.append(d)
 if not (O/'scoring_probes.json').exists():save(O/'scoring_probes.json',checks)
 waves=np.load(O/'windows.npy',mmap_mode='r')
 def run_stage(stage,indices=None):
  count=len(source) if stage=='source' else len(indices);folder=O/('score_chunks_'+stage);folder.mkdir(exist_ok=True);parts=[];started=time.monotonic()
  seal=sha(O/'thresholds_frozen.json') if stage!='calibration' else None
  for begin in range(0,count,160):
   end=min(begin+160,count);p=folder/f'{begin:06d}_{end:06d}.npz';done=p.with_suffix('.json')
   if done.exists():
    d=json.loads(done.read_text());assert d['prepared_sha256']==sha(prep) and d['threshold_seal']==seal and d['sha256']==sha(p);parts.append(np.load(p)['logits']);continue
   z=[]
   for i in range(begin,end,8):
    if stage=='source':x=np.stack([preprocess(r['path']) for r in source[i:min(i+8,end)]])
    else:
     ix=indices[i:min(i+8,end)];x=np.array(waves[ix]);
     import hashlib
     for j,xx in zip(ix,x):assert hashlib.sha256(xx.tobytes()).hexdigest()==audit[j]['window_sha256']
    z.append(predict(x))
   z=np.concatenate(z);tmp=p.with_suffix('.partial.npz');np.savez(tmp,logits=z);tmp.replace(p)
   save(done,dict(begin=begin,end=end,sha256=sha(p),prepared_sha256=sha(prep),threshold_seal=seal,completed_utc=now()));parts.append(z)
   print(stage,end,'/',count,'elapsed',round(time.monotonic()-started,1),flush=True)
  z=np.concatenate(parts);assert z.shape==(count,20,2);p=O/f'{stage}_logits.npz';done=O/f'{stage}_scoring_complete.json'
  if not done.exists():
   np.savez(p,logits=z,indices=np.array(indices if indices is not None else range(count)));save(done,dict(records=count,model_records=count*20,sha256=sha(p),completed_utc=now(),threshold_seal=seal))
  else:assert sha(p)==json.loads(done.read_text())['sha256']
  return z
 ci=[i for i,r in enumerate(manifest) if r['role']=='calibration'];assert len(ci)==1200
 cal=run_stage('calibration',ci);cr=[manifest[i] for i in ci];thresholds=[];(O/'thresholds').mkdir(exist_ok=True)
 for j,m in enumerate(models):
  v=cal[:,j,0]-cal[:,j,1];t=calibrate(v,[r['speaker_id'] for r in cr]);p=O/'thresholds'/f'{j:02d}.json'
  if not p.exists():freeze(p,t,cr,v,m['sha256'])
  d=json.loads(p.read_text());assert d['threshold']==t and d['model_sha256']==m['sha256'];thresholds.append(dict(index=j,family=m['family'],model=m['model'],threshold=t,record_sha256=sha(p)))
 seal=O/'thresholds_frozen.json'
 if not seal.exists():save(seal,dict(thresholds=thresholds,calibration_logits_sha256=sha(O/'calibration_logits.npz'),prepared_sha256=sha(prep),frozen_utc=now(),evaluation_started=False))
 else:
  d=json.loads(seal.read_text());assert d['thresholds']==thresholds and d['calibration_logits_sha256']==sha(O/'calibration_logits.npz') and d['prepared_sha256']==sha(prep)
 print('ALL 20 THRESHOLDS FROZEN',flush=True)
 gate=O/'evaluation_started.json'
 if not gate.exists():save(gate,dict(started_utc=now(),thresholds_sha256=sha(seal)))
 else:assert json.loads(gate.read_text())['thresholds_sha256']==sha(seal)
 ei=[i for i,r in enumerate(manifest) if r['role']=='evaluation'];assert len(ei)==45530;run_stage('evaluation',ei);run_stage('source')
 if not (O/'scoring_complete.json').exists():save(O/'scoring_complete.json',dict(success=True,completed_utc=now(),thresholds_sha256=sha(seal),artifact_sha256={str((O/f).relative_to(R)):sha(O/f) for f in ['calibration_logits.npz','evaluation_logits.npz','source_logits.npz','source_selection.csv','scoring_prepared.json','scoring_probes.json','thresholds_frozen.json','evaluation_started.json']}))
 print('SCORING COMPLETE',flush=True)
if __name__=='__main__':main()
