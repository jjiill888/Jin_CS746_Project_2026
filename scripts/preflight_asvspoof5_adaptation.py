"""Verify frozen source/cache contracts and historical GPU probes, no target eval."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import sys,json,io,contextlib,hashlib
from pathlib import Path
import numpy as np
import torch
from aasist_replication_common import setup_gpu,preprocess,make_model,sha,read
R=Path(__file__).resolve().parents[1];O=R/'results/asvspoof5_adaptation_v1'
def main():
 setup_gpu(); checks={}
 for p,h in json.loads((O/'metadata_validation.json').read_text())['sha256'].items():assert sha(R/p)==h,p
 audio=json.loads((O/'audio_audit.json').read_text());assert audio['passed'];assert sha(O/'windows.npy')==audio['windows_sha256']
 old=read(R/'results/replication_v1/adaptation_training.csv');prev=read(R/'results/preservation_v1/training.csv');rows=read(O/'training.csv');assert len(rows)==len(old)==len(prev)==10320
 source=0; labels={0:0,1:0}
 waveinfo=json.loads((R/'results/aasist_replication_v1/waveform_cache.json').read_text())
 for i,(r,a,b) in enumerate(zip(rows,old,prev)):
  assert a['path']==b['path'] and a['label_0bona_1spoof']==b['label_0bona_1spoof'] and int(a['feature_index'])==i
  assert r['label_0bona_1spoof']==a['label_0bona_1spoof']
  if r['is_source']=='True':
   assert r['path']==a['path'];source+=1;labels[int(r['label_0bona_1spoof'])]+=1
   assert hashlib.sha256(preprocess(r['path']).tobytes()).hexdigest()==waveinfo['model_input_sha256'][i]
  else:assert r['label_0bona_1spoof']=='0' and a['path'].startswith('data/l2arctic')
 assert source==7740 and labels=={0:2580,1:5160};checks['source_slots']=source
 cacheinfo=json.loads((R/'results/preservation_v1/feature_cache.json').read_text())
 assert sha(R/'results/preservation_v1/training.csv')==cacheinfo['training_manifest_sha256']
 assert sha(R/'results/preservation_v1/training_features.npy')==cacheinfo['sha256']
 checks['source_waveforms_and_feature_cache_verified']=True
 print('SOURCE CONTRACTS PASSED',flush=True)
 from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Model
 from transformers.utils.hub import cached_file
 front=json.loads((R/'results/preservation_v1/frontend_provenance.json').read_text());repo=front['repo'];rev=front['cached_files'][0]['cache_snapshot']
 for f in front['cached_files']:assert sha(cached_file(repo,f['filename'],revision=rev,local_files_only=True))==f['sha256']
 proc=Wav2Vec2FeatureExtractor.from_pretrained(repo,revision=rev,local_files_only=True);enc=Wav2Vec2Model.from_pretrained(repo,revision=rev,local_files_only=True,use_safetensors=False).eval().cuda()
 probes=read(R/'results/learning_vs_calibration_v1/historical_probe_manifest.csv');x=np.stack([preprocess(r['path']) for r in probes])
 with torch.inference_mode():
  h=enc(proc(list(x),sampling_rate=16000,return_tensors='pt').input_values.cuda()).last_hidden_state.unsqueeze(1).transpose(2,3).half().float()
  hs=torch.cat([enc(proc([xx],sampling_rate=16000,return_tensors='pt').input_values.cuda()).last_hidden_state.unsqueeze(1).transpose(2,3).half().float() for xx in x])
 sys.path.insert(0,str(R/'external/w2v2lcnn'));import model
 checks['models']=[]
 for family,folder,base in [('xlsr_lcnn','replication_v1','external/w2v2lcnn/models_0.63/try/anti-spoofing_feat_model.pt'),('aasist','aasist_replication_v1','external/aasist/models/weights/AASIST.pth')]:
  complete=json.loads((R/'results'/folder/'training_complete.json').read_text());prep=json.loads((R/'results'/folder/'training_prepared.json').read_text());assert sha(R/base)==prep['base_sha256']
  choices=[('original',dict(path=base,sha256=sha(R/base)))]+[(f'source_only_s{s}',complete['models'][f'source_only_s{s}']) for s in [20260912,20260913,20260914]]
  for name,m in choices:
   assert sha(R/m['path'])==m['sha256']
   if name!='original':assert m['steps']==2584
   net=(torch.load(R/m['path'],map_location='cpu',weights_only=False) if family=='xlsr_lcnn' else make_model(R/m['path'])).eval().cuda()
   tx=h if family=='xlsr_lcnn' else torch.from_numpy(x).cuda();sx=hs if family=='xlsr_lcnn' else tx
   with torch.inference_mode(),contextlib.redirect_stdout(io.StringIO()):
    bat=net(tx)[1].cpu().numpy();single=torch.cat([net(xx[None])[1] for xx in sx]).cpu().numpy()
   if family=='aasist':bat=bat[:,::-1];single=single[:,::-1]
   historic={r['utterance_id']:r for r in read(R/'results'/folder/'scores'/f'external_{name}.csv')}
   expected=np.array([[float(historic[r['utterance_id']][k]) for k in ['bonafide_logit','spoof_logit']] for r in probes])
   a=float(abs(bat-expected).max());b=float(abs(bat-single).max());assert a<=.005 and b<=.005,(family,name,a,b)
   checks['models'].append(dict(family=family,model=name,sha256=m['sha256'],historical_error=a,batch_single_error=b));del net
 checks.update(passed=True,probe_records=len(probes),torch=torch.__version__,gpu=torch.cuda.get_device_name(),no_target_evaluation=True,
  input_sha256={str(p.relative_to(R)):sha(p) for p in [Path(__file__),O/'manifest.csv',O/'training.csv',O/'audio_audit.json',R/'scripts/train_asvspoof5_adaptation.py']})
 with (O/'preflight.json').open('x') as f:json.dump(checks,f,indent=2)
 print('PREFLIGHT PASSED',flush=True)
if __name__=='__main__':main()
