import csv,json,hashlib,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'results/selfcal_stability_v1_1';S=R/'results/asvspoof5_selfcal_v1';P=R/'results/selfcal_stability_v1'
def read(p):return list(csv.DictReader(p.open()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (O/n).open('x') as f:json.dump(x,f,indent=2)
def write(n,rows):
 with (O/n).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
paths=[Path(__file__),R/'clawsbot/stability_analysis_rules_v1.1.md',O/'execution_addendum.md',P/'arrays.npz',P/'validation.json',S/'manifest.csv',S/'enrollment_logits.npz',S/'test_logits.npz',S/'selfcal_preregistration.json']
save('execution_seal.json',dict(posthoc=True,v10_results_already_seen=True,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),input_sha256={str(p.relative_to(R)):sha(p) for p in paths}))
z=np.load(P/'arrays.npz');per=z['permutations'];ids=list(z['identities']);rates=z['rates'];models=json.loads((S/'selfcal_preregistration.json').read_text())['models'];rows=read(S/'manifest.csv');data={}
for stage in ['enrollment','test']:
 d=np.load(S/(stage+'_logits.npz'))
 for i,v in zip(d['indices'],d['logits']):data[rows[i]['record_id']]=(rows[i],v[:,0]-v[:,1])
g=[];f=[]
for s in ids:
 rr=sorted([(r,v) for r,v in data.values() if r['speaker_id']==s and r['label']=='bonafide'],key=lambda x:x[0]['record_id']);g.append([v for r,v in rr]);f.append([v for r,v in data.values() if r['speaker_id']==s and r['label']=='spoof'])
g=np.array(g);f=np.array(f);out=[];hz=[];hb=[];fself=[];bself=[];theta=[]
for d in range(100):
 en=g[np.arange(60)[:,None],per[d,:,:20]];test=g[np.arange(60)[:,None],per[d,:,20:]];mu=en.mean(1);sd=en.std(1,ddof=1);sd[sd<1e-6]=1;ne=(en-mu[:,None])/sd[:,None];nb=(test-mu[:,None])/sd[:,None];nf=(f-mu[:,None])/sd[:,None];selft=en.min(1);ss=(f<selft[:,None]).sum(1);bb=(test<selft[:,None]).sum(1);ff=np.zeros((60,20),int);bc=np.zeros((60,20),int);th=[]
 for j,m in enumerate(models):
  cand=np.unique(ne[:,:,j]);errs=np.searchsorted(np.sort(nb[:,:,j].flatten()),cand,side='left');budget=int(bb[:,j].sum());valid=np.flatnonzero(errs<=budget);t=float(cand[valid[-1]]) if len(valid) else -np.inf;th.append(t);ff[:,j]=(nf[:,:,j]<t).sum(1);bc[:,j]=(nb[:,:,j]<t).sum(1)
  assert bc[:,j].sum()<=budget
  out.append(dict(draw=d,index=j,family=m['family'],model=m['model'],threshold=t,candidates=len(cand),fallback_no_candidate=not len(valid),self_FA=budget/1800,z_FA=float(bc[:,j].sum()/1800),FA_budget_gap=float((budget-bc[:,j].sum())/1800),self_TPR=float(ss[:,j].sum()/4800),z_TPR=float(ff[:,j].sum()/4800),delta_TPR=float((ss[:,j].sum()-ff[:,j].sum())/4800)))
 hz.append(ff);hb.append(bc);fself.append(ss);bself.append(bb);theta.append(th)
write('model_draws.csv',out);np.savez(O/'matched_arrays.npz',z_fake_hits=hz,z_bona_hits=hb,self_fake_hits=fself,self_bona_hits=bself,thresholds=theta)
def desc(x):return dict(median=float(np.median(x)),p025=float(np.quantile(x,.025)),p975=float(np.quantile(x,.975)),minimum=float(min(x)),maximum=float(max(x)),positive=int(sum(x>0)),zero=int(sum(x==0)),negative=int(sum(x<0)))
modeldist=[]
for j,m in enumerate(models):
 rr=[r for r in out if r['index']==j]
 for metric in ['delta_TPR','FA_budget_gap']:modeldist.append(dict(index=j,family=m['family'],model=m['model'],metric=metric,**desc(np.array([r[metric] for r in rr]))))
write('model_distributions.csv',modeldist);family=[]
for fam in ['xlsr_lcnn','aasist']:
 ix=[j for j,m in enumerate(models) if m['family']==fam and m['method'] in ['full_ce','kd10']];diff=(np.array(fself)[:,:,ix]-np.array(hz)[:,:,ix]).sum((1,2));gap=(np.array(bself)[:,:,ix]-np.array(hb)[:,:,ix]).sum((1,2))
 for metric,x in [('delta_TPR',diff/28800),('FA_budget_gap',gap/10800)]:family.append(dict(family=fam,metric=metric,**desc(x)))
write('family_distributions.csv',family)
st=read(P/'identity_stability.csv');pop=[]
for fam in ['xlsr_lcnn','aasist']:
 x=np.array([int(r['positive_draws'])/100 for r in st if r['family']==fam]);pop.append(dict(family=fam,median=float(np.median(x)),q25=float(np.quantile(x,.25)),q75=float(np.quantile(x,.75)),minimum=float(x.min()),maximum=float(x.max()),identities_ge95=int(sum(x>=.95)),identities_eq0=int(sum(x==0))))
write('all_identity_positive_proportions.csv',pop);save('analysis_complete.json',dict(posthoc=True,artifacts={str(p.relative_to(R)):sha(p) for p in O.iterdir() if p.is_file()}));print('MATCHED-FA COMPLETE')
