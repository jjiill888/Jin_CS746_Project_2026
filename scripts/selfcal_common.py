import csv,json,hashlib,time
from pathlib import Path
from fractions import Fraction
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'results/asvspoof5_selfcal_v1';B=R/'results/asvspoof5_adaptation_v1'
def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8388608),b''):h.update(x)
 return h.hexdigest()
def save(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2)
def write(p,rr):
 with p.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def threshold(x,alpha='.05'):
 x=np.asarray(x);assert len(x)>0 and np.isfinite(x).all();k=int((len(x)+1)*Fraction(alpha));return -np.inf if k==0 else float(np.sort(x)[k-1])
def bootstrap(x,counts):
 x=np.asarray(x);draw=counts@x/len(x);return dict(point=float(x.mean()),low=float(np.quantile(draw,.025)),high=float(np.quantile(draw,.975))),draw
def holm(p):
 p=np.asarray(p);order=np.argsort(p);out=np.empty(len(p));prev=0.
 for k,i in enumerate(order):prev=max(prev,min(1.,float(p[i])*(len(p)-k)));out[i]=prev
 return out
