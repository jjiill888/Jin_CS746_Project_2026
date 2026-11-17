"""Independent direct-count validation; does not import analysis/metric functions."""
import csv,json,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
R=Path(__file__).resolve().parents[1];O=R/'results/asvspoof5_adaptation_v1';A=O/'analysis'
def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8388608),b''):h.update(x)
 return h.hexdigest()
def close(x,y):assert abs(float(x)-float(y))<1e-12,(x,y)
def main():
 done=json.loads((A/'analysis_complete.json').read_text());assert done['success']
 for p,h in done['artifact_sha256'].items():assert sha(R/p)==h,p
 sc=json.loads((O/'scoring_complete.json').read_text())
 for p,h in sc['artifact_sha256'].items():assert sha(R/p)==h,p
 seal=json.loads((O/'thresholds_frozen.json').read_text());gate=json.loads((O/'evaluation_started.json').read_text());assert gate['thresholds_sha256']==sha(O/'thresholds_frozen.json') and gate['started_utc']>=seal['frozen_utc']
 models=json.loads((O/'training_complete.json').read_text())['models'];assert len(models)==20 and sum(not m['reused'] for m in models)==12
 for m in models:assert sha(R/m['path'])==m['sha256']
 rows=read(O/'manifest.csv');cz=np.load(O/'calibration_logits.npz');cr=[rows[i] for i in cz['indices']];cm=cz['logits'][:,:,0]-cz['logits'][:,:,1];ez=np.load(O/'evaluation_logits.npz');er=[rows[i] for i in ez['indices']];em=ez['logits'][:,:,0]-ez['logits'][:,:,1]
 roles={k:{r['speaker_id'] for r in rows if r['role']==k} for k in ['adaptation','calibration','evaluation']};assert [len(roles[k]) for k in roles]==[86,40,60]
 assert not any(roles[a]&roles[b] for a in roles for b in roles if a<b)
 assert len(cr)==1200 and len(er)==45530 and all(r['role']=='calibration' and r['codec']=='-' and r['label']=='bonafide' for r in cr) and all(r['role']=='evaluation' for r in er)
 cg={s:[i for i,r in enumerate(cr) if r['speaker_id']==s] for s in roles['calibration']};assert all(len(g)==30 for g in cg.values())
 for j,m in enumerate(models):
  t=seal['thresholds'][j]['threshold'];values=np.unique(cm[:,j]);feasible=[x for x in values if np.mean([np.mean(cm[ii,j]<x) for ii in cg.values()])<=.01+1e-12];close(t,max(feasible));assert sha(O/'thresholds'/f'{j:02d}.json')==seal['thresholds'][j]['record_sha256']
  record=json.loads((O/'thresholds'/f'{j:02d}.json').read_text());assert record['model_sha256']==m['sha256'];payload=sorted([dict(r,margin=float(v)) for r,v in zip(cr,cm[:,j])],key=lambda r:r['record_id']);assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==record['calibration_sha256']
 for stage in ['evaluation','source']:
  for f in (O/('score_chunks_'+stage)).glob('*.json'):
   d=json.loads(f.read_text());assert d['threshold_seal']==sha(O/'thresholds_frozen.json') and d['completed_utc']>=gate['started_utc'] and sha(f.with_suffix('.npz'))==d['sha256']
 raw=[i for i,r in enumerate(er) if r['codec']=='-'];rr=[er[i] for i in raw];rv=em[raw];ids=done['identities'];attacks=done['attacks'];rates=np.zeros((20,len(ids),17));lookup={}
 for si,s in enumerate(ids):
  for ai,a in enumerate(['bonafide',*attacks]):
   ix=[i for i,r in enumerate(rr) if r['speaker_id']==s and r['attack']==a];assert len(ix)==(30 if ai==0 else 10)
   for j in range(20):
    hit=sum(rv[i,j]<seal['thresholds'][j]['threshold'] for i in ix);rates[j,si,ai]=hit/len(ix);lookup[j,s,a]=(hit,len(ix))
 for r in read(A/'per_identity_attack.csv'):
  hit,n=lookup[int(r['index']),r['speaker_id'],r['attack']];assert int(r['support'])==n and int(r['predicted_spoof'])==hit;close(r['rate'],hit/n)
 stored=np.load(A/'identity_rates.npy');assert np.array_equal(stored,rates)
 # Shared multinomial identity counts; explicit weighted sum independently of producer's matrix product.
 counts=np.random.default_rng(20260919).multinomial(60,np.full(60,1/60),size=10000)
 def check_ci(r,x,delta=False):
  for c,field in enumerate(['FA','TPR']):
   vals=x[:,0] if c==0 else x[:,1:].mean(axis=1);draw=(counts*vals).sum(axis=1)/60
   close(r[('delta_'+field) if delta else ('macro_FA' if c==0 else 'attack_macro_TPR')],vals.mean());close(r[field+'_low'],np.percentile(draw,2.5));close(r[field+'_high'],np.percentile(draw,97.5))
 for r in read(A/'primary.csv'):check_ci(r,rates[int(r['index'])])
 modelidx={(m['family'],m['model']):j for j,m in enumerate(models)}
 for r in read(A/'paired_differences.csv'):check_ci(r,rates[int(r['index'])]-rates[modelidx[r['family'],r['reference']]],True)
 for r in read(A/'per_attack.csv'):close(r['TPR'],rates[int(r['index']),:,1+attacks.index(r['attack'])].mean())
 ranking_cases=0
 for r in read(A/'ranking.csv'):
  j=int(r['index']);s=r['speaker_id'];b=rv[[i for i,x in enumerate(rr) if x['speaker_id']==s and x['label']=='bonafide'],j];f=rv[[i for i,x in enumerate(rr) if x['speaker_id']==s and x['label']=='spoof'],j];n=len(b);m=len(f)
  rank=rankdata(np.r_[b,f],method='average');close(r['auc'],(rank[:n].sum()-n*(n+1)/2)/(n*m))
  thresholds=np.r_[-np.inf,np.nextafter(np.unique(np.r_[b,f]),np.inf)];fa=np.array([(b<t).mean() for t in thresholds]);tp=np.array([(f<t).mean() for t in thresholds])
  for alpha,col in [(0,'oracle_0'),(.01,'oracle_001'),(.05,'oracle_005')]:close(r[col],tp[fa<=alpha+1e-12].max())
  area=0.
  for k in range(n):
   lo=k/n;hi=min((k+1)/n,.05)
   if hi>lo:area+=(hi-lo)*tp[fa<=lo+1e-12].max()
  close(r['step_pauc_005'],area/.05);ranking_cases+=1
 # Independent parent joins and root-weighted class rates for every codec/model/group.
 rawidx={r['record_id']:i for i,r in enumerate(er) if r['codec']=='-'};groups=defaultdict(list)
 for i,r in enumerate(er):
  if r['codec']=='-':continue
  k=rawidx[r['parent_record_id']];p=er[k];assert all(p[f]==r[f] for f in ['speaker_id','label','attack'])
  groups[r['codec'],r['speaker_id'],r['label'],r['attack']].append((i,k,r['parent_record_id']))
 codec_cases=0
 for r in read(A/'codec_identity_attack.csv'):
  j=int(r['index']);t=seal['thresholds'][j]['threshold'];g=groups[r['codec'],r['speaker_id'],r['label'],r['attack']];parents=defaultdict(list)
  for i,k,parent in g:parents[parent].append((int(em[i,j]<t),int(em[k,j]<t)))
  enc=np.mean([sum(q[0] for q in pp)/len(pp) for pp in parents.values()]);base=np.mean([pp[0][1] for pp in parents.values()]);close(r['encoded_rate'],enc);close(r['matched_raw_rate'],base);close(r['delta'],enc-base);assert int(r['parent_support'])==len(parents) and int(r['file_support'])==len(g);codec_cases+=1
 cgrows=read(A/'codec_identity_attack.csv')
 for r in read(A/'codec_paired.csv'):
  g=[x for x in cgrows if x['index']==r['index'] and x['codec']==r['codec']];b=[x for x in g if x['label']=='bonafide'];f=[x for x in g if x['label']=='spoof']
  for field,fb,ff in [('encoded_rate','FA','TPR'),('matched_raw_rate','matched_raw_FA','matched_raw_TPR'),('delta','delta_FA','delta_TPR')]:
   close(r[fb],np.mean([float(x[field]) for x in b]));close(r[ff],np.mean([np.mean([float(x[field]) for x in f if x['attack']==a]) for a in {x['attack'] for x in f}]))
 source=read(O/'source_selection.csv');sz=np.load(O/'source_logits.npz')['logits']
 for r in read(A/'source_retention.csv'):
  j=int(r['index']);zz=sz[:,j];x=zz[:,0]-zz[:,1]
  if r['score']=='bonafide_probability':e=np.exp(zz-zz.max(axis=1,keepdims=True));x=e[:,0]/e.sum(axis=1)
  ix=[i for i,x in enumerate(source) if x['label']==r['label']];hit=x<float(r['threshold']);close(r['micro'],hit[ix].mean());ss={source[i]['speaker_id'] for i in ix};close(r['macro'],np.mean([hit[[i for i in ix if source[i]['speaker_id']==s]].mean() for s in ss]))
 result=dict(passed=True,models=20,calibration_records=1200,evaluation_records=45530,raw_identity_ranking_cases=ranking_cases,codec_identity_attack_cases=codec_cases,threshold_freeze_before_eval=True,independent_direct_count_and_rank_checks=True,validator_sha256=sha(Path(__file__)),analysis_complete_sha256=sha(A/'analysis_complete.json'))
 with (A/'validation.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
