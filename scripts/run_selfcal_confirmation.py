"""Audit local archive subset; seal design; score enrollment then held-out test."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import sys,tarfile,json,hashlib,contextlib,io,time
import numpy as np
import soundfile as sf
from selfcal_common import *
def audit():
 rows=read(O/'manifest.csv');selection=json.loads((O/'selection.json').read_text());assert sha(O/'manifest.csv')==selection['manifest_sha256'];wanted={'flac_D/'+r['record_id']+'.flac':r for r in rows};found={};archives={}
 if (O/'audio_audit.json').exists():
  d=json.loads((O/'audio_audit.json').read_text());assert d['passed'] and sha(O/'windows.npy')==d['windows_sha256'];return
 for p in sorted((R/'data/downloads/asvspoof5').glob('flac_D_*.tar')):
  rec=json.loads(p.with_name(p.name+'.verified.json').read_text());assert sha(p)==rec['sha256'];archives[str(p.relative_to(R))]=rec['sha256']
  with tarfile.open(p,'r:') as t:
   for member in t:
    name=member.name.removeprefix('./')
    if name not in wanted:continue
    assert member.isfile() and name not in found;blob=t.extractfile(member).read();dest=R/wanted[name]['path'];dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():assert hashlib.sha256(dest.read_bytes()).digest()==hashlib.sha256(blob).digest()
    else:dest.write_bytes(blob)
    found[name]=hashlib.sha256(blob).hexdigest()
  print('EXTRACTED',p.name,len(found),flush=True)
 assert len(found)==7800
 old=read(B/'audio_audit_rows.csv');oldpcm={r['pcm_sha256'] for r in old};oldwin={r['window_sha256'] for r in old};oldwin.update(json.loads((R/'results/aasist_replication_v1/waveform_cache.json').read_text())['model_input_sha256']);groups={};issues=[];records=[]
 waves=np.lib.format.open_memmap(O/'windows.partial.npy',mode='w+',dtype=np.float32,shape=(7800,64600))
 for i,r in enumerate(rows):
  x,sr=sf.read(R/r['path'],dtype='float32',always_2d=True);assert sr==16000 and x.shape[1]==1 and len(x)>0 and np.isfinite(x).all();x=x[:,0];w=x[:64600] if len(x)>=64600 else np.resize(x,64600);waves[i]=w;pcm=hashlib.sha256(x.tobytes()).hexdigest();wh=hashlib.sha256(w.tobytes()).hexdigest()
  if pcm in oldpcm or wh in oldwin:issues.append(dict(record_id=r['record_id'],reason='overlap_with_exposed_waveform'))
  for kind,h in [('pcm',pcm),('window',wh)]:
   if (kind,h) in groups:
    prior=groups[kind,h]
    if prior['role']!=r['role'] or prior['speaker_id']!=r['speaker_id']:issues.append(dict(record_id=r['record_id'],reason='cross_role_or_identity_duplicate',other=prior['record_id']))
   else:groups[kind,h]=r
  records.append(dict(index=i,record_id=r['record_id'],file_sha256=sha(R/r['path']),pcm_sha256=pcm,window_sha256=wh,frames=len(x)))
 waves.flush();del waves;save(O/'duplicate_audit.json',issues);assert not issues,'Potential leakage: do not replace silently'
 (O/'windows.partial.npy').rename(O/'windows.npy');write(O/'audio_audit_rows.csv',records);save(O/'audio_audit.json',dict(passed=True,records=7800,archives=archives,windows_sha256=sha(O/'windows.npy'),audit_sha256=sha(O/'audio_audit_rows.csv'),cross_role_or_exposed_duplicates=0));print('AUDIO AUDIT PASSED',flush=True)
def seal():
 paths=[R/'docs/selfcal-confirmation-v1.md',O/'manifest.csv',O/'identity_pool.csv',O/'selection.json',O/'audio_audit.json',O/'audio_audit_rows.csv',B/'training_complete.json',B/'thresholds_frozen.json',R/'results/independent_data_audit_v1/ASVspoof5.dev.track_1.tsv',R/'results/preservation_v1/frontend_provenance.json']
 paths += [R/'scripts'/n for n in ['prepare_selfcal_confirmation.py','selfcal_common.py','run_selfcal_confirmation.py','analyze_selfcal_confirmation.py','validate_selfcal_confirmation.py','execute_selfcal_confirmation.py','aasist_replication_common.py']]
 paths += [R/'external/aasist/config/AASIST.conf',R/'external/aasist/models/AASIST.py',R/'external/w2v2lcnn/model.py',R/'results/learning_vs_calibration_v1/historical_probe_manifest.csv']
 models=json.loads((B/'training_complete.json').read_text())['models'];assert len(models)==20
 for m in models:assert sha(R/m['path'])==m['sha256']
 rec=dict(frozen_utc=utc(),models=models,input_sha256={str(p.relative_to(R)):sha(p) for p in paths},archive_sha256=json.loads((O/'audio_audit.json').read_text())['archives'],identities=json.loads((O/'selection.json').read_text())['identities'],arms=['old_global_001','pooled_005','self_005'],alpha=.05,enrollment_per_identity=20,test_genuine_per_identity=30,test_spoof_per_attack=10,attacks=[f'A{i:02d}' for i in range(9,17)],bootstrap_seed=20260921,bootstrap_rounds=10000,primary_models='six CE/KD models equal weight per family',primary_test='one-sided centered identity bootstrap, Holm over two families',fa_margin=None,confirmation_scores_exist=False)
 save(O/'selfcal_preregistration.json',rec);snap=O/'frozen_code';snap.mkdir()
 for p in paths:
  if p.suffix in ['.py','.md']:(snap/p.name).write_bytes(p.read_bytes())
 print('PREREGISTRATION SEALED',flush=True)
def score():
 import torch
 from aasist_replication_common import setup_gpu,make_model,preprocess
 from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Model
 from transformers.utils.hub import cached_file
 setup_gpu();torch.manual_seed(20260921);reg=json.loads((O/'selfcal_preregistration.json').read_text())
 for p,h in reg['input_sha256'].items():assert sha(R/p)==h,p
 assert sha(O/'windows.npy')==json.loads((O/'audio_audit.json').read_text())['windows_sha256']
 models=reg['models'];rows=read(O/'manifest.csv');aa=read(O/'audio_audit_rows.csv');waves=np.load(O/'windows.npy',mmap_mode='r');sys.path.insert(0,str(R/'external/w2v2lcnn'));import model
 for m in models:assert sha(R/m['path'])==m['sha256']
 front=json.loads((R/'results/preservation_v1/frontend_provenance.json').read_text());repo=front['repo'];rev=front['cached_files'][0]['cache_snapshot']
 for f in front['cached_files']:assert sha(cached_file(repo,f['filename'],revision=rev,local_files_only=True))==f['sha256']
 proc=Wav2Vec2FeatureExtractor.from_pretrained(repo,revision=rev,local_files_only=True);enc=Wav2Vec2Model.from_pretrained(repo,revision=rev,local_files_only=True,use_safetensors=False).eval().cuda();nets=[(torch.load(R/m['path'],map_location='cpu',weights_only=False) if m['family']=='xlsr_lcnn' else make_model(R/m['path'])).eval().cuda() for m in models]
 def predict(x):
  with torch.inference_mode(),contextlib.redirect_stdout(io.StringIO()):
   h=enc(proc(list(x),sampling_rate=16000,return_tensors='pt').input_values.cuda()).last_hidden_state.unsqueeze(1).transpose(2,3).half().float();tx=torch.from_numpy(x.copy()).cuda();zz=[]
   for m,net in zip(models,nets):
    z=net(h if m['family']=='xlsr_lcnn' else tx)[1].cpu().numpy();zz.append(z if m['family']=='xlsr_lcnn' else z[:,::-1])
  z=np.stack(zz,axis=1).astype(float);assert np.isfinite(z).all();return z
 probes=read(R/'results/learning_vs_calibration_v1/historical_probe_manifest.csv');x=np.stack([preprocess(r['path']) for r in probes]);bat=predict(x);single=np.concatenate([predict(xx[None]) for xx in x]);assert abs(bat-single).max()<=.005
 checks=[]
 for j,m in enumerate(models):
  d=dict(index=j,batch_single_error=float(abs(bat[:,j]-single[:,j]).max()))
  if m['method']=='original':
   folder='replication_v1' if m['family']=='xlsr_lcnn' else 'aasist_replication_v1';old={r['utterance_id']:r for r in read(R/'results'/folder/'scores/external_original.csv')};expected=np.array([[float(old[r['utterance_id']][k]) for k in ['bonafide_logit','spoof_logit']] for r in probes]);d['historical_error']=float(abs(bat[:,j]-expected).max());assert d['historical_error']<=.005
  checks.append(d)
 if not (O/'historical_probes.json').exists():save(O/'historical_probes.json',checks)
 for stage in ['enrollment','test']:
  ix=[i for i,r in enumerate(rows) if r['role']==stage];folder=O/('chunks_'+stage);folder.mkdir(exist_ok=True);parts=[];th=sha(O/'thresholds_selfcal_frozen.json') if stage=='test' else None
  if stage=='test' and not (O/'test_started.json').exists():save(O/'test_started.json',dict(started_utc=utc(),thresholds_sha256=th,preregistration_sha256=sha(O/'selfcal_preregistration.json')))
  for begin in range(0,len(ix),160):
   end=min(begin+160,len(ix));p=folder/f'{begin:05d}_{end:05d}.npz';done=p.with_suffix('.json')
   if done.exists():
    d=json.loads(done.read_text());assert d['sha256']==sha(p) and d['thresholds_sha256']==th and d['preregistration_sha256']==sha(O/'selfcal_preregistration.json');parts.append(np.load(p)['logits']);continue
   zs=[]
   for off in range(begin,end,8):
    idx=ix[off:min(off+8,end)];x=np.array(waves[idx])
    for i,xx in zip(idx,x):assert hashlib.sha256(xx.tobytes()).hexdigest()==aa[i]['window_sha256']
    zs.append(predict(x))
   z=np.concatenate(zs);tmp=p.with_suffix('.partial.npz');np.savez(tmp,logits=z);tmp.replace(p);save(done,dict(sha256=sha(p),thresholds_sha256=th,preregistration_sha256=sha(O/'selfcal_preregistration.json'),completed_utc=utc()));parts.append(z);print(stage,end,'/',len(ix),flush=True)
  z=np.concatenate(parts);p=O/f'{stage}_logits.npz';done=O/f'{stage}_complete.json'
  if not done.exists():np.savez(p,logits=z,indices=ix);save(done,dict(sha256=sha(p),completed_utc=utc(),records=len(ix)))
  else:assert json.loads(done.read_text())['sha256']==sha(p)
  if stage=='enrollment':
   margin=z[:,:,0]-z[:,:,1];er=[rows[i] for i in ix];old=json.loads((B/'thresholds_frozen.json').read_text())['thresholds'];tt=[]
   for j,m in enumerate(models):
    pooled=threshold(margin[:,j]);assert old[j]['model']==m['model']
    for s in reg['identities']:
     si=[i for i,r in enumerate(er) if r['speaker_id']==s];assert len(si)==20
     tt.append(dict(index=j,speaker_id=s,old_global_001=old[j]['threshold'],pooled_005=pooled,self_005=threshold(margin[si,j])))
   p=O/'thresholds_selfcal_frozen.json'
   if not p.exists():save(p,dict(frozen_utc=utc(),thresholds=tt,enrollment_sha256=sha(O/'enrollment_logits.npz'),preregistration_sha256=sha(O/'selfcal_preregistration.json')))
   else:assert json.loads(p.read_text())['thresholds']==tt
   print('ALL SELF/POOLED THRESHOLDS SEALED',flush=True)
 save(O/'scoring_complete.json',dict(success=True,completed_utc=utc(),preregistration_sha256=sha(O/'selfcal_preregistration.json'),enrollment_sha256=sha(O/'enrollment_logits.npz'),test_sha256=sha(O/'test_logits.npz'),thresholds_sha256=sha(O/'thresholds_selfcal_frozen.json')))
if __name__=='__main__':
 {'audit':audit,'seal':seal,'score':score}[sys.argv[1]]()
