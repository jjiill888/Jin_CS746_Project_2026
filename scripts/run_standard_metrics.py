"""Post-hoc score diagnostics; raw margins are not certified calibrated LLRs."""
import csv,json,hashlib,time,importlib.util
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'results/standard_metrics_v1';B=R/'results/asvspoof5_adaptation_v1';S=R/'results/asvspoof5_selfcal_v1'
def read(p):return list(csv.DictReader(p.open()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,rr):
 with (O/n).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
paths=[Path(__file__),O/'protocol.md',O/'official_calculate_modules.py',B/'training_complete.json',B/'source_logits.npz',B/'source_selection.csv',B/'evaluation_logits.npz',B/'manifest.csv',B/'thresholds_frozen.json',S/'test_logits.npz',S/'manifest.csv',S/'thresholds_selfcal_frozen.json']
with (O/'execution_seal.json').open('x') as f:json.dump(dict(posthoc=True,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),input_sha256={str(p.relative_to(R)):sha(p) for p in paths}),f,indent=2)
spec=importlib.util.spec_from_file_location('official',O/'official_calculate_modules.py');official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
models=json.loads((B/'training_complete.json').read_text())['models'];sets=[]
for folder,stem,name in [(B,'source','LA2019_dev_selection'),(B,'evaluation','ASV5_eval_raw'),(S,'test','ASV5_confirmation_test')]:
 d=np.load(folder/(stem+'_logits.npz'));rr=read(folder/('source_selection.csv' if stem=='source' else 'manifest.csv'));rr=[rr[i] for i in d['indices']];x=d['logits'][:,:,0]-d['logits'][:,:,1];mask=np.array([r.get('codec','-')=='-' for r in rr]);rr=[r for r,k in zip(rr,mask) if k];x=x[mask];sets.append((name,rr,x))
rows=[];deployment=[];dets={};checks=[];old=json.loads((B/'thresholds_frozen.json').read_text())['thresholds'];st={(r['index'],r['speaker_id']):r for r in json.loads((S/'thresholds_selfcal_frozen.json').read_text())['thresholds']}
for name,rr,x in sets:
 isb=np.array([r['label']=='bonafide' for r in rr]);assert isb.any() and (~isb).any()
 for j,m in enumerate(models):
  b=x[isb,j];s=x[~isb,j];vals=np.r_[b,s];labels=np.r_[np.ones(len(b),int),np.zeros(len(s),int)];order=np.argsort(vals,kind='stable');cb=np.cumsum(labels[order]);cs=np.arange(1,len(vals)+1)-cb;fr=np.r_[0,cb/len(b)];fa=np.r_[1,1-cs/len(s)];k=np.argmin(abs(fr-fa));eer=(fr[k]+fa[k])/2;cost=1.9*fr+fa;mindcf=cost.min();tau=-np.log(1.9);act=1.9*np.mean(b<tau)+np.mean(s>=tau);cllr=(np.logaddexp(0,-b).mean()+np.logaddexp(0,s).mean())/(2*np.log(2))
  # Tie groups are indivisible under a strict scalar threshold.
  ends=np.r_[np.flatnonzero(np.diff(vals[order])!=0)+1,len(vals)];ix=np.r_[0,ends];tf=fr[ix];ta=fa[ix];tk=np.argmin(abs(tf-ta));tieeer=(tf[tk]+ta[tk])/2;tiemin=(1.9*tf+ta).min()
  oe,of,oa,ot,_=official.compute_eer(b,s);om,_=official.compute_mindcf(of,oa,ot,.05,1,10);od,_=official.compute_actDCF(b,s,.05,1,10);oc=official.calculate_CLLR(b,s)
  assert np.allclose([eer,mindcf,act,cllr],[oe,om,od,oc],rtol=1e-12,atol=1e-12)
  rows.append(dict(dataset=name,index=j,family=m['family'],model=m['model'],method=m['method'],seed=m['seed'],bona_support=len(b),spoof_support=len(s),EER=float(eer),minDCF=float(mindcf),actDCF_raw_margin_as_LLR=float(act),Cllr_raw_margin_as_LLR=float(cllr),tie_aware_EER=float(tieeer),tie_aware_minDCF=float(tiemin),duplicate_score_count=len(vals)-len(np.unique(vals))))
  dets[name+'__'+str(j)+'__FRR']=tf;dets[name+'__'+str(j)+'__spoof_accept']=ta
  policies={'study1_target_threshold':np.full(len(rr),old[j]['threshold'])} if name!='ASV5_confirmation_test' else {arm:np.array([st[j,r['speaker_id']][arm] for r in rr]) for arm in ['old_global_001','pooled_005','self_005']}
  for arm,t in policies.items():hit=x[:,j]<t;bf=float(hit[isb].mean());sp=float((~hit[~isb]).mean());deployment.append(dict(dataset=name,index=j,family=m['family'],model=m['model'],policy=arm,bona_reject=bf,spoof_accept=sp,DCF_frozen_threshold=1.9*bf+sp))
write('metrics.csv',rows);write('deployment_DCF.csv',deployment);np.savez(O/'det_curves.npz',**dets)
summary=[]
for dataset in [q[0] for q in sets]:
 for fam in ['xlsr_lcnn','aasist']:
  for method in ['original','full_ce','kd10','source_only']:
   rr=[r for r in rows if r['dataset']==dataset and r['family']==fam and r['method']==method]
   summary.append(dict(dataset=dataset,family=fam,method=method,models=len(rr),**{key:float(np.mean([r[key] for r in rr])) for key in ['EER','minDCF','actDCF_raw_margin_as_LLR','Cllr_raw_margin_as_LLR']}))
write('method_summary.csv',summary)
with (O/'validation.json').open('x') as f:json.dump(dict(passed=True,model_dataset_cases=len(rows),metrics_per_case=4,reference='downloaded official ASVspoof5 implementation',official_sha256=sha(O/'official_calculate_modules.py'),scope='independent formulas versus official implementation on all real score sets; not independent neural inference'),f,indent=2)
with (O/'analysis_complete.json').open('x') as f:json.dump(dict(success=True,artifacts={str(p.relative_to(R)):sha(p) for p in O.iterdir() if p.is_file()}),f,indent=2)
print('STANDARD METRICS COMPLETE; OFFICIAL COMPARISON PASSED')
