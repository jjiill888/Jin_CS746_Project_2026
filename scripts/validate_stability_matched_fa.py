import csv,json,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'results/selfcal_stability_v1_1';S=R/'results/asvspoof5_selfcal_v1';P=R/'results/selfcal_stability_v1'
def read(p):return list(csv.DictReader(p.open()))
for file,key in [('execution_seal.json','input_sha256'),('analysis_complete.json','artifacts')]:
 for p,h in json.loads((O/file).read_text())[key].items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h
archive=np.load(O/'matched_arrays.npz');z={k:archive[k] for k in archive.files};manifest=read(S/'manifest.csv');lookup={}
for stage in ['enrollment','test']:
 d=np.load(S/(stage+'_logits.npz'))
 for i,v in zip(d['indices'],d['logits']):lookup[manifest[i]['record_id']]=(manifest[i],v[:,0]-v[:,1])
split=sorted([r for r in read(P/'split_manifest.csv') if r['draw']=='0'],key=lambda r:r['speaker_id']);en=np.array([[lookup[k][1] for k in r['enrollment_ids'].split('|')] for r in split]);te=np.array([[lookup[k][1] for k in r['test_ids'].split('|')] for r in split]);fake=np.array([[v for r,v in lookup.values() if r['speaker_id']==s['speaker_id'] and r['label']=='spoof'] for s in split])
for j in range(20):
 mu=en[:,:,j].mean(1);sd=en[:,:,j].std(1,ddof=1);sd[sd<1e-6]=1;ne=(en[:,:,j]-mu[:,None])/sd[:,None];nb=(te[:,:,j]-mu[:,None])/sd[:,None];nf=(fake[:,:,j]-mu[:,None])/sd[:,None];budget=int(np.sum(te[:,:,j]<en[:,:,j].min(1)[:,None]));cand=np.unique(ne);counts=(nb.flatten()[:,None]<cand).sum(0);valid=cand[counts<=budget];t=valid[-1] if len(valid) else -np.inf;assert np.isclose(t,z['thresholds'][0,j],rtol=1e-12,atol=1e-12);assert np.array_equal((nb<t).sum(1),z['z_bona_hits'][0,:,j]);assert np.array_equal((nf<t).sum(1),z['z_fake_hits'][0,:,j])
models=json.loads((S/'selfcal_preregistration.json').read_text())['models']
for row in read(O/'family_distributions.csv'):
 ix=[j for j,m in enumerate(models) if m['family']==row['family'] and m['method'] in ['full_ce','kd10']];x=[]
 for d in range(100):
  if row['metric']=='delta_TPR':v=sum(int(z['self_fake_hits'][d,:,j].sum())-int(z['z_fake_hits'][d,:,j].sum()) for j in ix)/28800
  else:v=sum(int(z['self_bona_hits'][d,:,j].sum())-int(z['z_bona_hits'][d,:,j].sum()) for j in ix)/10800
  x.append(v)
 for key,val in [('median',np.median(x)),('p025',np.percentile(x,2.5)),('p975',np.percentile(x,97.5)),('minimum',min(x)),('maximum',max(x))]:assert abs(float(row[key])-val)<1e-12
 assert int(row['positive'])==sum(v>0 for v in x) and int(row['zero'])==sum(v==0 for v in x) and int(row['negative'])==sum(v<0 for v in x)
assert np.all(z['z_bona_hits'].sum(1)<=z['self_bona_hits'].sum(1))
(O/'validation.json').write_text(json.dumps(dict(passed=True,draw0_models_independent_candidate_enumeration=20,all_draw_budget_cases=2000,family_distribution_cases=4,shared_producer_code=False),indent=2));print('S6 INDEPENDENT VALIDATION PASSED')
