import csv,json,hashlib,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'results/selfcal_stability_v1';S=R/'results/asvspoof5_selfcal_v1';B=R/'results/asvspoof5_adaptation_v1'
def read(p):return list(csv.DictReader(p.open()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (O/n).open('x') as f:json.dump(x,f,indent=2)
def write(n,rows):
 with (O/n).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
paths=[Path(__file__),R/'clawsbot/stability_analysis_rules_v1.md',O/'execution_addendum.md',S/'manifest.csv',S/'enrollment_logits.npz',S/'test_logits.npz',S/'selfcal_preregistration.json',B/'manifest.csv',B/'calibration_logits.npz',R/'results/selfcal_posthoc_v1/b4_identities.csv']
save('execution_seal.json',dict(posthoc=True,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),input_sha256={str(p.relative_to(R)):sha(p) for p in paths}))
models=json.loads((S/'selfcal_preregistration.json').read_text())['models'];ids=sorted(json.loads((S/'selfcal_preregistration.json').read_text())['identities']);families={fam:[j for j,m in enumerate(models) if m['family']==fam and m['method'] in ['full_ce','kd10']] for fam in ['xlsr_lcnn','aasist']};arms=['self','pooled','znorm_internal','znorm_external']
def load(folder,stage):
 rows=read(folder/'manifest.csv');d=np.load(folder/(stage+'_logits.npz'));return [rows[i] for i in d['indices']],d['logits'][:,:,0]-d['logits'][:,:,1]
er,ev=load(S,'enrollment');tr,tv=load(S,'test');cr,cv=load(B,'calibration');rows=er+tr;values=np.concatenate([ev,tv]);g=[];f=[];gids=[]
for s in ids:
 ii=sorted([i for i,r in enumerate(rows) if r['speaker_id']==s and r['label']=='bonafide'],key=lambda i:rows[i]['record_id']);assert len(ii)==50;g.append(values[ii]);gids.append([rows[i]['record_id'] for i in ii]);ff=[]
 for a in [f'A{k:02d}' for k in range(9,17)]:
  jj=[i for i,r in enumerate(tr) if r['speaker_id']==s and r['attack']==a];assert len(jj)==10;ff.append(tv[jj])
 f.append(ff)
g=np.array(g);f=np.array(f);old=np.stack([cv[[i for i,r in enumerate(cr) if r['speaker_id']==s]] for s in sorted({r['speaker_id'] for r in cr})]);om=old.mean(1);os=old.std(1,ddof=1);of=os<1e-6;os[of]=1;ze=np.sort(((old-om[:,None])/os[:,None]).reshape(-1,20),axis=0)[59]
allrates=[];thresholds=[];perms=[];splits=[];fallback=[]
for d in range(100):
 rng=np.random.default_rng(20260930+d);perm=np.stack([rng.permutation(50) for s in ids]);perms.append(perm);en=g[np.arange(60)[:,None],perm[:,:20]];test=g[np.arange(60)[:,None],perm[:,20:]];mu=en.mean(1);sd=en.std(1,ddof=1);fb=sd<1e-6;sd[fb]=1;zi=np.sort(((en-mu[:,None])/sd[:,None]).reshape(-1,20),axis=0)[59];tt=np.stack([en.min(1),np.broadcast_to(np.sort(en.reshape(-1,20),axis=0)[59],(60,20)),mu+sd*zi,mu+sd*ze]);thresholds.append(tt);rate=[]
 for t in tt:
  fa=(test<t[:,None]).mean(1);tp=(f<t[:,None,None]).mean(2);rate.append(np.concatenate([fa[:,None],tp],axis=1).transpose(2,0,1))
 allrates.append(rate)
 for i,s in enumerate(ids):splits.append(dict(draw=d,seed=20260930+d,speaker_id=s,enrollment_ids='|'.join(gids[i][k] for k in perm[i,:20]),test_ids='|'.join(gids[i][k] for k in perm[i,20:])))
 for j in range(20):fallback.append(dict(draw=d,index=j,enrollment_fallback=int(fb[:,j].sum()),old_calibration_fallback=int(of[:,j].sum())))
a=np.array(allrates);np.savez(O/'arrays.npz',rates=a,thresholds=thresholds,permutations=perms,identities=ids,arms=arms)
write('split_manifest.csv',splits);write('fallback_counts.csv',fallback);mr=[];fr=[];dist=[]
def stats(x):
 sign=np.sign(np.round(x,12));return dict(median=float(np.median(x)),p025=float(np.quantile(x,.025)),p975=float(np.quantile(x,.975)),minimum=float(min(x)),maximum=float(max(x)),positive=int(sum(sign>0)),zero=int(sum(sign==0)),negative=int(sum(sign<0)),positive_fraction=float(np.mean(sign>0)),adjacent_sign_changes=int(sum(sign[1:]!=sign[:-1])))
for fam,ix in families.items():
 for ai,arm in enumerate(arms):
  v=a[:,ai,ix].mean(axis=1)
  for d in range(100):fr.append(dict(draw=d,family=fam,arm=arm,FA=float(v[d,:,0].mean()),TPR=float(v[d,:,1:].mean())))
 for ref in [1,2]:
  v=(a[:,0,ix]-a[:,ref,ix]).mean(axis=1)
  for metric,x in [('FA',v[:,:,0].mean(1)),('TPR',v[:,:,1:].mean((1,2)))]:dist.append(dict(family=fam,contrast='self-'+arms[ref],metric=metric,**stats(x)))
for j,m in enumerate(models):
 for ref in [1,2]:
  v=a[:,0,j]-a[:,ref,j]
  for metric,x in [('FA',v[:,:,0].mean(1)),('TPR',v[:,:,1:].mean((1,2)))]:mr.append(dict(index=j,family=m['family'],model=m['model'],contrast='self-'+arms[ref],metric=metric,**stats(x)))
write('family_draws.csv',fr);write('family_distributions.csv',dist);write('model_distributions.csv',mr)
write('model_draws.csv',[dict(draw=d,index=j,family=m['family'],model=m['model'],arm=arm,FA=float(a[d,ai,j,:,0].mean()),TPR=float(a[d,ai,j,:,1:].mean())) for d in range(100) for ai,arm in enumerate(arms) for j,m in enumerate(models)])
oldb={r['speaker_id']:float(r['delta_TPR']) for r in read(R/'results/selfcal_posthoc_v1/b4_identities.csv')};idrows=[]
for fam,ix in families.items():
 x=(a[:,0,ix,:,1:]-a[:,1,ix,:,1:]).mean((1,3));hits=np.rint(x*480).astype(int)
 for i,s in enumerate(ids):idrows.append(dict(family=fam,speaker_id=s,positive_draws=int(sum(hits[:,i]>0)),negative_draws=int(sum(hits[:,i]<0)),zero_draws=int(sum(hits[:,i]==0)),mean_delta_TPR=float(x[:,i].mean()),original_beneficiary=oldb[s]>0 if fam=='aasist' else ''))
write('identity_stability.csv',idrows)
save('analysis_complete.json',dict(success=True,artifacts={str(p.relative_to(R)):sha(p) for p in O.iterdir() if p.is_file()}));print('STABILITY ANALYSIS COMPLETE')
