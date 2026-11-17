"""Independent scalar threshold/count validation of saved post-hoc outputs."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'results/selfcal_posthoc_v1'
def read(n):return list(csv.DictReader((O/n).open()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for manifest,key in [('execution_seal.json','input_sha256'),('analysis_complete.json','artifacts')]:
 for p,h in json.loads((O/manifest).read_text())[key].items():assert sha(R/p)==h,p
z=np.load(O/'b1_arrays.npz');v=np.load(O/'b5_arrays.npz');e=z['enrollment'];c=z['old_calibration'];b=z['test_bona'];f=z['test_fake'];counts=z['counts'];arms=list(z['arms']);cases=0

def quant(x,alpha):
 k=((len(x)+1)*(5 if alpha==.05 else 10))//100
 return sorted(x)[k-1] if k else -np.inf

def check(en,old,alpha,expected,thresholds=None):
 global cases
 for j in range(20):
  mu=np.array([sum(row[:,j])/len(row) for row in en]);sd=np.array([np.sqrt(sum((row[:,j]-mu[i])**2)/(len(row)-1)) for i,row in enumerate(en)]);sd[sd<1e-6]=1
  om=np.array([sum(row[:,j])/len(row) for row in old]);os=np.array([np.sqrt(sum((row[:,j]-om[i])**2)/(len(row)-1)) for i,row in enumerate(old)]);os[os<1e-6]=1
  internal=quant([(val-mu[i])/sd[i] for i,row in enumerate(en) for val in row[:,j]],alpha);external=quant([(val-om[i])/os[i] for i,row in enumerate(old) for val in row[:,j]],alpha);pool=quant(en[:,:,j].flatten(),alpha)
  for i in range(60):
   tt=[pool,quant(en[i,:,j],alpha),mu[i]+sd[i]*internal,mu[i]+sd[i]*external]
   for a,t in enumerate(tt):
    if thresholds is not None:assert np.isclose(t,thresholds[a,i,j],rtol=1e-12,atol=1e-12)
    hit=[sum(b[i,:,j]<t)/30]+[sum(f[i,k,:,j]<t)/10 for k in range(8)]
    assert np.array_equal(hit,expected[a,j,i]),(a,j,i);cases+=1
check(e,c,.05,z['rates'],z['thresholds'])
for q,(alpha,n) in enumerate([(.05,5),(.05,10),(.05,15),(.05,20),(.1,10),(.1,15),(.1,20)]):
 en=np.stack([e[i,v['enrollment_permutations'][i,:n]] for i in range(60)]);old=np.stack([c[i,v['old_permutations'][i,:n]] for i in range(40)]);check(en,old,alpha,v['rates'][q])
models=json.loads((R/'results/asvspoof5_selfcal_v1/selfcal_preregistration.json').read_text())['models'];ix={fam:[j for j,m in enumerate(models) if m['family']==fam and m['method'] in ['full_ce','kd10']] for fam in ['aasist','xlsr_lcnn']}
def statcheck(row,x):
 draws=np.sum(counts*x,axis=1)/60;lo,hi=np.percentile(draws,[2.5,97.5]);p=(1+sum(abs(draws-x.mean())>=abs(x.mean())-1e-12))/10001
 for key,val in [('point',x.mean()),('low',lo),('high',hi),('p',p)]:assert abs(float(row[key])-val)<1e-10,(key,row,val)
rates=z['rates']
for row in read('b1_models.csv'):
 v0=rates[arms.index(row['arm']),int(row['index'])];statcheck(row,v0[:,0] if row['metric']=='FA' else v0[:,1:].mean(axis=1))
for row in read('b1_family_contrasts.csv'):
 left,right=row['contrast'].split('-');dif=np.mean(rates[arms.index(left),ix[row['family']]]-rates[arms.index(right),ix[row['family']]],axis=0);statcheck(row,dif[:,0] if row['metric']=='FA' else dif[:,1:].mean(axis=1))
dif={fam:np.mean(rates[1,jj]-rates[0,jj],axis=0) for fam,jj in ix.items()}
for row in read('b2_family_difference.csv'):
 d=dif['aasist']-dif['xlsr_lcnn'];statcheck(row,d[:,0] if row['metric']=='FA' else d[:,1:].mean(axis=1))
for row in read('b3_family_attack.csv'):statcheck(row,dif['aasist'][:,int(row['attack'][1:])-8])
xx=np.rint(dif['aasist'][:,1:].mean(axis=1)*480)/480;conc=json.loads((O/'b4_concentration.json').read_text());assert [conc[k] for k in ['benefited','harmed','unchanged']]==[sum(xx>0),sum(xx<0),sum(xx==0)]
for row in read('b5_curves.csv'):
 q=[(.05,5),(.05,10),(.05,15),(.05,20),(.1,10),(.1,15),(.1,20)].index((float(row['alpha']),int(row['n'])));a=arms.index(row['arm']);data=v['rates'][q,a]
 if row['scope']=='family':data=data[ix[row['family']]].mean(axis=0)
 else:data=data[next(j for j,m in enumerate(models) if m['model']==row['name'] and m['family']==row['family'])]
 statcheck(row,data[:,0] if row['metric']=='FA' else data[:,1:].mean(axis=1))
with (O/'validation.json').open('x') as out:json.dump(dict(passed=True,independent_threshold_count_cases=cases,b1_all_model_intervals=True,b1_family_intervals=True,b2_family_intervals=True,b3_family_intervals=True,b4_counts=True,b5_all_intervals=True,neural_inference_reimplemented=False),out,indent=2)
print('POSTHOC INDEPENDENT VALIDATION PASSED',cases)
