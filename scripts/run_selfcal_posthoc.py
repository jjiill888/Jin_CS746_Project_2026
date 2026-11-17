"""Transparent B1-B5 supplement; no inference or main-result mutations."""
import sys,json,csv,hashlib,time
from pathlib import Path
from fractions import Fraction
import numpy as np
from scipy.stats import rankdata,spearmanr
R=Path(__file__).resolve().parents[1];O=R/'results/selfcal_posthoc_v1';S=R/'results/asvspoof5_selfcal_v1';B=R/'results/asvspoof5_adaptation_v1'
def read(p):return list(csv.DictReader(p.open()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,d):
 with (O/n).open('x') as f:json.dump(d,f,indent=2,allow_nan=False)
def write(n,rr):
 with (O/n).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
models=json.loads((S/'selfcal_preregistration.json').read_text())['models'];ids=sorted(json.loads((S/'selection.json').read_text())['identities']);attacks=[f'A{i:02d}' for i in range(9,17)]
paths=[R/'clawsbot/posthoc_analysis_rules_v1.1.md',O/'execution_addendum.md',Path(__file__),S/'manifest.csv',B/'manifest.csv',S/'enrollment_logits.npz',S/'test_logits.npz',B/'calibration_logits.npz',S/'selfcal_preregistration.json',S/'thresholds_selfcal_frozen.json']
save('execution_seal.json',dict(posthoc=True,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),input_sha256={str(p.relative_to(R)):sha(p) for p in paths}))
def load(folder,stem):
 rr=read(folder/'manifest.csv');d=np.load(folder/(stem+'_logits.npz'));return [rr[i] for i in d['indices']],d['logits'][:,:,0]-d['logits'][:,:,1]
er,em=load(S,'enrollment');tr,tm=load(S,'test');cr,cm=load(B,'calibration');cids=sorted({r['speaker_id'] for r in cr})
def grouped(rr,x,ss,attack=None):
 return np.stack([x[sorted([i for i,r in enumerate(rr) if r['speaker_id']==s and (attack is None or r['attack']==attack)],key=lambda i:rr[i]['record_id'])] for s in ss])
e=grouped(er,em,ids);c=grouped(cr,cm,cids);b=grouped(tr,tm,ids,'bonafide');f=np.stack([grouped(tr,tm,ids,a) for a in attacks],axis=1)
assert e.shape==(60,20,20) and c.shape==(40,30,20) and f.shape==(60,8,10,20)
counts=np.random.default_rng(20260922).multinomial(60,np.full(60,1/60),10000)
def stat(x):
 point=float(x.mean());draw=counts@x/60;lo,hi=np.quantile(draw,[.025,.975]);return dict(point=point,low=float(lo),high=float(hi),p=float((1+np.sum(abs(draw-point)>=abs(point)-1e-12))/10001))
def threshold(x,a):
 k=int((len(x)+1)*Fraction(str(a)));return np.full(x.shape[1:],-np.inf) if k==0 else np.sort(x,axis=0)[k-1]
def norm(x):
 mu=x.mean(axis=1);sd=x.std(axis=1,ddof=1);fallback=sd<1e-6;sd=np.where(fallback,1,sd);return mu,sd,fallback
fallbacks=[]
def build(en,old,a,tag):
 mu,sd,fb=norm(en);om,os,ofb=norm(old)
 for j,m in enumerate(models):fallbacks.append(dict(setting=tag,index=j,enrollment_fallback=int(fb[:,j].sum()),old_calibration_fallback=int(ofb[:,j].sum())))
 zi=threshold(((en-mu[:,None])/sd[:,None]).reshape(-1,20),a);ze=threshold(((old-om[:,None])/os[:,None]).reshape(-1,20),a)
 return dict(pooled=np.broadcast_to(threshold(en.reshape(-1,20),a),(60,20)),self=np.stack([threshold(x,a) for x in en]),znorm_internal=mu+sd*zi,znorm_external=mu+sd*ze)
def rates(t):
 fa=(b<t[:,None]).mean(axis=1);tp=(f<t[:,None,None]).mean(axis=2);return np.concatenate([fa[:,None],tp],axis=1).transpose(2,0,1)
def meta(j):return dict(index=j,family=models[j]['family'],model=models[j]['model'],method=models[j]['method'])
families={fam:[j for j,m in enumerate(models) if m['family']==fam and m['method'] in ['full_ce','kd10']] for fam in ['xlsr_lcnn','aasist']}
ts=build(e,c,.05,'B1');arms=list(ts);rr=np.stack([rates(ts[a]) for a in arms]);np.savez(O/'b1_arrays.npz',rates=rr,thresholds=np.stack([ts[a] for a in arms]),enrollment=e,old_calibration=c,test_bona=b,test_fake=f,ids=ids,arms=arms,counts=counts)
# Reconcile independent main artifact (different identity order).
main=np.load(S/'analysis/rates.npy');oldids=json.loads((S/'selfcal_preregistration.json').read_text())['identities'];order=[oldids.index(s) for s in ids]
assert np.array_equal(rr[0],main[:,1,order,:]) and np.array_equal(rr[1],main[:,2,order,:])
summary=[];pairs=[]
for ai,a in enumerate(arms):
 for j in range(20):
  for metric,x in [('FA',rr[ai,j,:,0]),('TPR',rr[ai,j,:,1:].mean(axis=1))]:summary.append(dict(**meta(j),arm=a,metric=metric,**stat(x)))
for fam,ix in families.items():
 for left,right in [('self','pooled'),('self','znorm_internal'),('self','znorm_external'),('znorm_internal','pooled'),('znorm_external','pooled')]:
  delta=(rr[arms.index(left),ix]-rr[arms.index(right),ix]).mean(axis=0)
  for metric,x in [('FA',delta[:,0]),('TPR',delta[:,1:].mean(axis=1))]:pairs.append(dict(family=fam,contrast=left+'-'+right,metric=metric,**stat(x),holm_p=''))
selected=[r for r in pairs if r['metric']=='TPR' and r['contrast'].startswith('self-znorm')];prev=0
for k,row in enumerate(sorted(selected,key=lambda r:r['p'])):prev=max(prev,min(1,(4-k)*row['p']));row['holm_p']=prev
write('b1_models.csv',summary);write('b1_family_contrasts.csv',pairs)
deltas={fam:(rr[1,ix]-rr[0,ix]).mean(axis=0) for fam,ix in families.items()};d=deltas['aasist']-deltas['xlsr_lcnn'];write('b2_family_difference.csv',[dict(metric=metric,**stat(x)) for metric,x in [('FA',d[:,0]),('TPR',d[:,1:].mean(axis=1))]])
write('b2_model_differences.csv',[dict(**meta(j),metric=metric,**stat(x)) for ix in families.values() for j in ix for metric,x in [('FA',rr[1,j,:,0]-rr[0,j,:,0]),('TPR',(rr[1,j,:,1:]-rr[0,j,:,1:]).mean(axis=1))]])
attackrows=[]
for j in families['aasist']:
 for k,a in enumerate(attacks):attackrows.append(dict(**meta(j),attack=a,support=600,pooled_TPR=float(rr[0,j,:,k+1].mean()),self_TPR=float(rr[1,j,:,k+1].mean()),**stat(rr[1,j,:,k+1]-rr[0,j,:,k+1])))
write('b3_per_model_attack.csv',attackrows);write('b3_family_attack.csv',[dict(attack=a,support_per_model=600,models=6,**stat(deltas['aasist'][:,k+1])) for k,a in enumerate(attacks)])
x=np.rint(deltas['aasist'][:,1:].mean(axis=1)*480)/480;assoc=[];idrows=[dict(speaker_id=s,delta_TPR=float(x[i])) for i,s in enumerate(ids)]
for name,v in [('std',e.std(axis=1,ddof=1)),('range',np.ptp(e,axis=1))]:
 rank=np.mean([rankdata(v[:,j])/60 for j in families['aasist']],axis=0)
 for i,r in enumerate(idrows):r[name+'_rank_mean']=float(rank[i])
 rho=spearmanr(rank,x).statistic;assoc.append(dict(scope='family_mean_within_model_rank',model='six_adapted',dispersion=name,spearman=float(rho) if np.isfinite(rho) else None))
 for j in families['aasist']:
  y=(rr[1,j,:,1:]-rr[0,j,:,1:]).mean(axis=1);rho=spearmanr(v[:,j],y).statistic;assoc.append(dict(scope='individual_model',model=models[j]['model'],dispersion=name,spearman=float(rho) if np.isfinite(rho) else None))
write('b4_identities.csv',idrows);write('b4_associations.csv',assoc)
con=dict(benefited=int(sum(x>0)),harmed=int(sum(x<0)),unchanged=int(sum(x==0)),median_delta_TPR=float(np.median(x)),top_shares={})
for side,z in [('positive_gain',np.maximum(x,0)),('absolute_loss',np.maximum(-x,0))]:con['top_shares'][side]={str(k):float(np.sort(z)[-k:].sum()/z.sum()) if z.sum()>0 else None for k in [5,10,20]}
save('b4_concentration.json',con)
rng=np.random.default_rng(20260923);ep=np.stack([rng.permutation(20) for s in ids]);cp=np.stack([rng.permutation(30) for s in cids]);selections=[];curve=[];cache=[]
for a,ns in [(.05,[5,10,15,20]),(.10,[10,15,20])]:
 for n in ns:
  en=e[np.arange(60)[:,None],ep[:,:n]];old=c[np.arange(40)[:,None],cp[:,:n]];tt=build(en,old,a,f'B5_{a}_{n}');rs=np.stack([rates(tt[arm]) for arm in arms]);cache.append(rs)
  for ai,arm in enumerate(arms):
   for j in range(20):
    for metric,z in [('FA',rs[ai,j,:,0]),('TPR',rs[ai,j,:,1:].mean(axis=1))]:curve.append(dict(alpha=a,n=n,arm=arm,scope='model',name=models[j]['model'],family=models[j]['family'],metric=metric,**stat(z)))
   for fam,ix in families.items():
    z=rs[ai,ix].mean(axis=0)
    for metric,y in [('FA',z[:,0]),('TPR',z[:,1:].mean(axis=1))]:curve.append(dict(alpha=a,n=n,arm=arm,scope='family',name='six_adapted',family=fam,metric=metric,**stat(y)))
  if a==.05 and n==20:assert np.array_equal(rs[:3],rr[:3])
  if a==.05 and n<19:assert not rs[1].any()
for role,rows,ss,perm in [('enrollment',er,ids,ep),('old_calibration',cr,cids,cp)]:
 for i,s in enumerate(ss):
  records=sorted(r['record_id'] for r in rows if r['speaker_id']==s)
  for k,v in enumerate(perm[i]):selections.append(dict(role=role,speaker_id=s,rank=k+1,record_id=records[v]))
write('b5_selection_order.csv',selections);write('b5_curves.csv',curve);write('fallback_counts.csv',fallbacks);np.savez(O/'b5_arrays.npz',rates=np.stack(cache),enrollment_permutations=ep,old_permutations=cp)
save('analysis_complete.json',dict(posthoc=True,artifacts={str(p.relative_to(R)):sha(p) for p in O.iterdir() if p.is_file()}));print('B1-B5 ANALYSIS COMPLETE')
