from pathlib import Path
import numpy as np,csv,json,hashlib
R=Path(__file__).resolve().parents[1];O=R/'results/selfcal_stability_v1';S=R/'results/asvspoof5_selfcal_v1';B=R/'results/asvspoof5_adaptation_v1'
def read(p):return list(csv.DictReader(p.open()))
for name,key in [('execution_seal.json','input_sha256'),('analysis_complete.json','artifacts')]:
 for p,h in json.loads((O/name).read_text())[key].items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h
archive=np.load(O/'arrays.npz');z={k:archive[k] for k in archive.files};ids=list(z['identities']);mods=json.loads((S/'selfcal_preregistration.json').read_text())['models'];lookup={}
for folder,stages in [(S,['enrollment','test']),(B,['calibration'])]:
 rows=read(folder/'manifest.csv')
 for stage in stages:
  data=np.load(folder/(stage+'_logits.npz'))
  for i,x in zip(data['indices'],data['logits']):lookup[rows[i]['record_id']]=(rows[i],x[:,0]-x[:,1])
splits=read(O/'split_manifest.csv');assert len(splits)==6000
for row in splits:
 en=row['enrollment_ids'].split('|');te=row['test_ids'].split('|');assert len(en)==20 and len(te)==30 and len(set(en+te))==50;assert all(lookup[k][0]['speaker_id']==row['speaker_id'] for k in en+te)
for d in range(100):
 rng=np.random.default_rng(20260930+d)
 for i in range(60):assert np.array_equal(rng.permutation(50),z['permutations'][d,i])
sel=[r for r in splits if r['draw']=='0'];sel=sorted(sel,key=lambda r:r['speaker_id']);en=np.array([[lookup[k][1] for k in r['enrollment_ids'].split('|')] for r in sel]);oldids=sorted({r['speaker_id'] for r in read(B/'manifest.csv') if r['role']=='calibration'});old=np.array([[v for r,v in lookup.values() if r['speaker_id']==s and r['role']=='calibration'] for s in oldids])
for j in range(20):
 mu=en[:,:,j].mean(1);sd=en[:,:,j].std(1,ddof=1);sd[sd<1e-6]=1;om=old[:,:,j].mean(1);os=old[:,:,j].std(1,ddof=1);os[os<1e-6]=1
 zi=sorted(((en[:,:,j]-mu[:,None])/sd[:,None]).flatten())[59];ze=sorted(((old[:,:,j]-om[:,None])/os[:,None]).flatten())[59];pool=sorted(en[:,:,j].flatten())[59]
 for i,row in enumerate(sel):
  tt=[min(en[i,:,j]),pool,mu[i]+sd[i]*zi,mu[i]+sd[i]*ze];assert np.allclose(tt,z['thresholds'][0,:,i,j],rtol=1e-12,atol=1e-12)
  test=[lookup[k][1][j] for k in row['test_ids'].split('|')]
  for ai,t in enumerate(tt):
   vals=[sum(v<t for v in test)/30]
   for k in range(9,17):
    fake=[v[j] for r,v in lookup.values() if r['speaker_id']==ids[i] and r['attack']==f'A{k:02d}'];assert len(fake)==10;vals.append(sum(v<t for v in fake)/10)
   assert np.allclose(vals,z['rates'][0,ai,j,i],rtol=0,atol=1e-14)
for row in read(O/'family_distributions.csv'):
 ix=[j for j,m in enumerate(mods) if m['family']==row['family'] and m['method'] in ['full_ce','kd10']];ref=1 if row['contrast']=='self-pooled' else 2;x=[]
 for d in range(100):
  vals=[]
  for j in ix:
   v=z['rates'][d,0,j]-z['rates'][d,ref,j];vals.append(v[:,0].mean() if row['metric']=='FA' else v[:,1:].mean())
  x.append(sum(vals)/6)
 for k,v in [('median',np.median(x)),('p025',np.percentile(x,2.5)),('p975',np.percentile(x,97.5)),('minimum',min(x)),('maximum',max(x))]:assert abs(float(row[k])-v)<1e-12
 assert int(row['positive'])==sum(v>1e-12 for v in x)
(O/'validation.json').write_text(json.dumps(dict(passed=True,split_partitions=6000,draw0_models=20,draw0_arms=4,family_distributions=8,neural_inference=False),indent=2));print('INDEPENDENT VALIDATION PASSED')
