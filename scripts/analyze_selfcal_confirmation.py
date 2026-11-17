"""Frozen family-level confirmation analysis, full secondary model reporting."""
from selfcal_common import *
def main():
 reg=json.loads((O/'selfcal_preregistration.json').read_text());sc=json.loads((O/'scoring_complete.json').read_text());assert sc['success']
 for p,h in reg['input_sha256'].items():assert sha(R/p)==h,p
 assert sc['test_sha256']==sha(O/'test_logits.npz') and sc['thresholds_sha256']==sha(O/'thresholds_selfcal_frozen.json')
 A=O/'analysis';A.mkdir(exist_ok=False);rows=read(O/'manifest.csv');d=np.load(O/'test_logits.npz');rr=[rows[i] for i in d['indices']];v=d['logits'][:,:,0]-d['logits'][:,:,1];models=reg['models'];ids=reg['identities'];arms=reg['arms'];attacks=reg['attacks'];tt={(r['index'],r['speaker_id']):r for r in json.loads((O/'thresholds_selfcal_frozen.json').read_text())['thresholds']}
 rates=np.empty((20,3,60,9));per=[]
 for j,m in enumerate(models):
  for ai,arm in enumerate(arms):
   for si,s in enumerate(ids):
    t=tt[j,s][arm]
    for k,a in enumerate(['bonafide',*attacks]):
     ix=[i for i,r in enumerate(rr) if r['speaker_id']==s and r['attack']==a];assert len(ix)==(30 if k==0 else 10);hit=int((v[ix,j]<t).sum());rates[j,ai,si,k]=hit/len(ix);per.append(dict(index=j,family=m['family'],model=m['model'],arm=arm,speaker_id=s,attack=a,support=len(ix),rejected=hit,rate=hit/len(ix)))
 write(A/'per_identity_attack.csv',per);np.save(A/'rates.npy',rates);counts=np.random.default_rng(20260921).multinomial(60,np.full(60,1/60),size=10000);summary=[];diff=[];coverage=[];oracle=[]
 for j,m in enumerate(models):
  meta=dict(index=j,family=m['family'],model=m['model'],method=m['method'],seed=m['seed'])
  for ai,arm in enumerate(arms):
   fa,_=bootstrap(rates[j,ai,:,0],counts);tp,_=bootstrap(rates[j,ai,:,1:].mean(axis=1),counts);summary.append(dict(**meta,arm=arm,FA=fa['point'],FA_low=fa['low'],FA_high=fa['high'],TPR=tp['point'],TPR_low=tp['low'],TPR_high=tp['high']))
  for ref in [0,1]:
   fa,_=bootstrap(rates[j,2,:,0]-rates[j,ref,:,0],counts);tp,_=bootstrap((rates[j,2,:,1:]-rates[j,ref,:,1:]).mean(axis=1),counts);diff.append(dict(**meta,reference=arms[ref],delta_FA=fa['point'],FA_low=fa['low'],FA_high=fa['high'],delta_TPR=tp['point'],TPR_low=tp['low'],TPR_high=tp['high']))
  x=rates[j,2,:,0];coverage.append(dict(**meta,mean_FA=float(x.mean()),nominal_marginal_bound=1/21,identities_over_005=int((x>.05).sum()),identities=60,minimum=float(x.min()),maximum=float(x.max()),histogram_rejected_counts=json.dumps({str(k):int(np.sum(np.rint(x*30)==k)) for k in range(31)})))
  b=np.array([r['label']=='bonafide' for r in rr]);bs=np.sort(v[b,j]);fs=np.sort(v[~b,j]);budget=float(rates[j,1,:,0].mean());k=int(round(budget*len(bs)));threshold=bs[k] if k<len(bs) else np.inf;ceiling=float(np.mean(v[~b,j]<threshold));pooled=float(rates[j,1,:,1:].mean());selftp=float(rates[j,2,:,1:].mean());den=ceiling-pooled
  oracle.append(dict(**meta,pooled_achieved_FA=budget,pooled_TPR=pooled,self_TPR=selftp,global_oracle_TPR=ceiling,denominator=den,ratio=(selftp-pooled)/den if den>.02 else '',self_FA=float(x.mean()),interpretation='global threshold benchmark; self uses identity, ratio may exceed one'))
 family=[]
 for fam in ['xlsr_lcnn','aasist']:
  idx=[j for j,m in enumerate(models) if m['family']==fam and m['method'] in ['full_ce','kd10']];assert len(idx)==6
  dif=rates[idx,2]-rates[idx,1];xfa=dif[:,:,0].mean(axis=0);xtp=dif[:,:,1:].mean(axis=(0,2));fa,_=bootstrap(xfa,counts);tp,draw=bootstrap(xtp,counts);hits=np.rint(xtp*480).astype(np.int64);p=(1+int(np.sum(counts@hits>=2*hits.sum())))/10001
  family.append(dict(family=fam,adapted_models=6,delta_FA=fa['point'],FA_low=fa['low'],FA_high=fa['high'],delta_TPR=tp['point'],TPR_low=tp['low'],TPR_high=tp['high'],bootstrap_one_sided_p=p,FA_reading='positive_cost' if fa['low']>0 else 'lower_FA' if fa['high']<0 else 'unresolved'))
 adj=holm([r['bootstrap_one_sided_p'] for r in family])
 for r,p in zip(family,adj):r.update(holm_adjusted_p=float(p),primary_TPR_positive=bool(p<=.05 and r['delta_TPR']>0))
 for name,rr in [('summary.csv',summary),('paired_differences.csv',diff),('coverage.csv',coverage),('oracle_descriptive.csv',oracle),('family_primary.csv',family)]:write(A/name,rr)
 save(A/'analysis_complete.json',dict(success=True,completed_utc=utc(),preregistration_sha256=sha(O/'selfcal_preregistration.json'),artifact_sha256={str(p.relative_to(R)):sha(p) for p in A.iterdir() if p.is_file()}));print('ANALYSIS COMPLETE',flush=True)
if __name__=='__main__':main()
