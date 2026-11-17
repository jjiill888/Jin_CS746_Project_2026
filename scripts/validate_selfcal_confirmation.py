"""Separate direct-count reference validation and sealed report creation."""
import csv,json,hashlib,time
from pathlib import Path
from fractions import Fraction
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'results/asvspoof5_selfcal_v1';A=O/'analysis'
def read(p):return list(csv.DictReader(p.open()))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def eq(a,b):assert abs(float(a)-float(b))<1e-12,(a,b)
def main():
 reg=json.loads((O/'selfcal_preregistration.json').read_text());done=json.loads((A/'analysis_complete.json').read_text())
 for p,h in reg['input_sha256'].items():assert sha(R/p)==h,p
 for p,h in done['artifact_sha256'].items():assert sha(R/p)==h,p
 assert done['preregistration_sha256']==sha(O/'selfcal_preregistration.json')
 rows=read(O/'manifest.csv');en=np.load(O/'enrollment_logits.npz');te=np.load(O/'test_logits.npz');er=[rows[i] for i in en['indices']];tr=[rows[i] for i in te['indices']];em=en['logits'][:,:,0]-en['logits'][:,:,1];tm=te['logits'][:,:,0]-te['logits'][:,:,1];seal=json.loads((O/'thresholds_selfcal_frozen.json').read_text());gate=json.loads((O/'test_started.json').read_text());sc=json.loads((O/'scoring_complete.json').read_text())
 assert sc['preregistration_sha256']==sha(O/'selfcal_preregistration.json') and sc['enrollment_sha256']==sha(O/'enrollment_logits.npz') and sc['test_sha256']==sha(O/'test_logits.npz') and sc['thresholds_sha256']==sha(O/'thresholds_selfcal_frozen.json')
 assert reg['frozen_utc']<=seal['frozen_utc']<=gate['started_utc'];assert gate['thresholds_sha256']==sha(O/'thresholds_selfcal_frozen.json') and seal['enrollment_sha256']==sha(O/'enrollment_logits.npz')
 assert len(er)==1200 and len(tr)==6600 and not {r['record_id'] for r in er}&{r['record_id'] for r in tr}
 ids=reg['identities'];assert len(ids)==60
 for s in ids:assert sum(r['speaker_id']==s for r in er)==20 and sum(r['speaker_id']==s and r['label']=='bonafide' for r in tr)==30
 tt={(r['index'],r['speaker_id']):r for r in seal['thresholds']};old=json.loads((R/'results/asvspoof5_adaptation_v1/thresholds_frozen.json').read_text())['thresholds']
 for j,m in enumerate(reg['models']):
  assert sha(R/m['path'])==m['sha256']
  for s in ids:
   t=tt[j,s];eq(t['self_005'],min(em[[i for i,r in enumerate(er) if r['speaker_id']==s],j]));eq(t['pooled_005'],sorted(em[:,j])[59]);eq(t['old_global_001'],old[j]['threshold'])
 for stage in ['enrollment','test']:
  for f in (O/('chunks_'+stage)).glob('*.json'):
   d=json.loads(f.read_text());assert d['preregistration_sha256']==sha(O/'selfcal_preregistration.json') and sha(f.with_suffix('.npz'))==d['sha256'];assert d['completed_utc']>=reg['frozen_utc']
   if stage=='test':assert d['thresholds_sha256']==sha(O/'thresholds_selfcal_frozen.json') and d['completed_utc']>=seal['frozen_utc']
 rates=np.zeros((20,3,60,9));groups={}
 for si,s in enumerate(ids):
  for k,a in enumerate(['bonafide',*reg['attacks']]):groups[s,a]=[i for i,r in enumerate(tr) if r['speaker_id']==s and r['attack']==a]
 per=read(A/'per_identity_attack.csv')
 for r in per:
  j=int(r['index']);s=r['speaker_id'];arm=r['arm'];ix=groups[s,r['attack']];hit=sum(tm[i,j]<tt[j,s][arm] for i in ix);assert int(r['support'])==len(ix) and int(r['rejected'])==hit;eq(r['rate'],hit/len(ix));rates[j,reg['arms'].index(arm),ids.index(s),['bonafide',*reg['attacks']].index(r['attack'])]=hit/len(ix)
 assert np.array_equal(rates,np.load(A/'rates.npy'))
 counts=np.random.default_rng(20260921).multinomial(60,np.full(60,1/60),size=10000)
 def check(r,x,field):
  draw=(counts*x).sum(axis=1)/60;eq(r[field],x.mean());prefix='FA' if 'FA' in field else 'TPR';eq(r[prefix+'_low'],np.percentile(draw,2.5));eq(r[prefix+'_high'],np.percentile(draw,97.5));return draw
 summary=read(A/'summary.csv')
 for r in summary:
  j=int(r['index']);ai=reg['arms'].index(r['arm']);check(r,rates[j,ai,:,0],'FA');check(r,rates[j,ai,:,1:].mean(axis=1),'TPR')
 for r in read(A/'paired_differences.csv'):
  j=int(r['index']);ai=reg['arms'].index(r['reference']);check(r,rates[j,2,:,0]-rates[j,ai,:,0],'delta_FA');check(r,(rates[j,2,:,1:]-rates[j,ai,:,1:]).mean(axis=1),'delta_TPR')
 families=read(A/'family_primary.csv');ps=[]
 for r in families:
  ix=[j for j,m in enumerate(reg['models']) if m['family']==r['family'] and m['method'] in ['full_ce','kd10']];assert len(ix)==6
  xfa=np.array([sum(rates[j,2,s,0]-rates[j,1,s,0] for j in ix)/6 for s in range(60)]);xtp=np.array([sum(float(np.mean(rates[j,2,s,1:]-rates[j,1,s,1:])) for j in ix)/6 for s in range(60)]);check(r,xfa,'delta_FA');draw=check(r,xtp,'delta_TPR');hits=np.rint(xtp*480).astype(np.int64);p=(1+int(np.sum((counts*hits).sum(axis=1)>=2*hits.sum())))/10001;eq(p,r['bootstrap_one_sided_p']);ps.append(p)
 order=np.argsort(ps);adjusted=[0.,0.];adjusted[order[0]]=min(1,2*ps[order[0]]);adjusted[order[1]]=max(adjusted[order[0]],ps[order[1]])
 for i,r in enumerate(families):eq(r['holm_adjusted_p'],adjusted[i]);assert (r['primary_TPR_positive']=='True')==(adjusted[i]<=.05 and float(r['delta_TPR'])>0)
 for r in read(A/'coverage.csv'):
  j=int(r['index']);x=rates[j,2,:,0];eq(r['mean_FA'],x.mean());assert int(r['identities_over_005'])==sum(x>.05);assert json.loads(r['histogram_rejected_counts'])=={str(k):int(sum(np.rint(x*30)==k)) for k in range(31)}
 for r in read(A/'oracle_descriptive.csv'):
  j=int(r['index']);b=np.array([x['label']=='bonafide' for x in tr]);score=tm[:,j];budget=float(r['pooled_achieved_FA']);thresholds=np.r_[-np.inf,np.nextafter(np.unique(score),np.inf)];best=max(float(np.mean(score[~b]<t)) for t in thresholds if np.mean(score[b]<t)<=budget+1e-12);eq(r['global_oracle_TPR'],best)
  den=best-rates[j,1,:,1:].mean();eq(r['denominator'],den)
  if den>.02:eq(r['ratio'],(rates[j,2,:,1:].mean()-rates[j,1,:,1:].mean())/den)
  else:assert r['ratio']==''
 validation=dict(passed=True,models=20,identities=60,enrollment_records=1200,test_records=6600,per_identity_attack_cases=len(per),preregistration_sha256=sha(O/'selfcal_preregistration.json'),analysis_sha256=sha(A/'analysis_complete.json'),completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
 with (A/'validation.json').open('x') as f:json.dump(validation,f,indent=2)
 fmt=lambda x:f'{100*float(x):.3f}'
 text='''# Per-speaker calibration confirmation: complete results

The study uses 60 previously unused development speakers: 20 genuine enrollment and 30 genuine test recordings per speaker, plus ten spoofs per attack for eight attacks. All 20 models and three arms are retained. Rules, manifests, and code were frozen before enrollment scoring; thresholds were sealed before test scoring. The primary contrast is self5%-pooled5%; the transferred original 1% threshold is secondary. The independent recomputation passed without tuning or seed selection.

## Primary family-level comparison

Each family equally averages all six CE/KD models, with speakers as the analysis units. Detection tests use an approximate centered speaker bootstrap and Holm correction over two families, not an exact finite-sample test. False-alarm 95% intervals are reported separately. No non-inferiority margin was preregistered, so a positive detection result does not establish acceptable cost or overall dominance.

| Family | Delta FA, pp [95% CI] | Delta TPR, pp [95% CI] | Holm p | Primary TPR criterion |
|---|---|---|---:|---|
'''
 for r in families:text+=f"| {r['family']} | {fmt(r['delta_FA'])} [{fmt(r['FA_low'])}, {fmt(r['FA_high'])}] | {fmt(r['delta_TPR'])} [{fmt(r['TPR_low'])}, {fmt(r['TPR_high'])}] | {float(r['holm_adjusted_p']):.5f} | {r['primary_TPR_positive']} |\n"
 text+='\n## All models and arms (percent)\n\n| Family | Model | Arm | FA | TPR |\n|---|---|---|---:|---:|\n'
 for r in summary:text+=f"| {r['family']} | {r['model']} | {r['arm']} | {fmt(r['FA'])} | {fmt(r['TPR'])} |\n"
 text+='''
## Interpretation limits

self uses each speaker's minimum margin among 20 genuine enrollment recordings. Under within-speaker exchangeability of enrollment and future genuine speech, with a fixed model, the marginal false-rejection bound is 1/21. It is not conditional on a fixed enrollment set, a guarantee for realized finite tests, or a simultaneous guarantee for all speakers. Observed false alarms above or below nominal do not alone prove or disprove exchangeability. The pooled arm is an empirical grouped-sample baseline without a per-speaker conformal guarantee. No arbitrary +2 pp rule is used to declare acceptable cost or coverage failure.

These outcomes concern only the specified family-level detection hypotheses. They do not establish a causal shape/location mechanism, methodological novelty, or publication readiness. Development attacks differ from the earlier evaluation attacks, and pretraining overlap cannot be excluded for anonymized identities. This cohort is now exposed and cannot be relabeled as independent confirmation for a design changed after seeing the results.

[Per-speaker/attack results](analysis/per_identity_attack.csv), [all arms and intervals](analysis/summary.csv), [paired model differences](analysis/paired_differences.csv), [coverage distribution](analysis/coverage.csv), [descriptive global oracle](analysis/oracle_descriptive.csv), [primary tests](analysis/family_primary.csv), and [independent recomputation](analysis/validation.json). The oracle ratio is reported only when its denominator exceeds 2 pp. It compares against a single global-threshold benchmark while self uses speaker-specific thresholds, so it may exceed one; it is not a recovered fraction of an information-theoretic upper bound.
'''
 with (O/'interpretation.md').open('x') as f:f.write(text)
 with (O/'completion.json').open('x') as f:json.dump(dict(success=True,completed_utc=validation['completed_utc'],artifact_sha256={str(p.relative_to(R)):sha(p) for p in [O/'interpretation.md',O/'scoring_complete.json',A/'analysis_complete.json',A/'validation.json',O/'selfcal_preregistration.json']}),f,indent=2)
 print('SELFCAL COMPLETE; VALIDATION PASSED',flush=True)
if __name__=='__main__':main()
