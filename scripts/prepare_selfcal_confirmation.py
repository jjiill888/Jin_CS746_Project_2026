"""Score-blind candidate inventory/selection for corrected confirmation design."""
import csv,json,hashlib
from collections import defaultdict
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'results/asvspoof5_selfcal_v1';B=R/'results/asvspoof5_adaptation_v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8388608),b''):h.update(x)
 return h.hexdigest()
def read(p):return list(csv.DictReader(p.open()))
def save(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2)
def write(p,rows):
 with p.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def key(kind,s):return hashlib.sha256(('selfcal-confirm-v1|'+kind+'|'+s).encode()).hexdigest()
def main():
 O.mkdir(exist_ok=False);g=defaultdict(list);old=read(B/'manifest.csv');excluded={r['speaker_id'] for r in old if r['split']=='dev'}
 for line in (R/'results/independent_data_audit_v1/ASVspoof5.dev.track_1.tsv').read_text().splitlines():
  s,rec,gender,codec,cq,seed,tag,attack,label,res=line.split()
  if s not in excluded and codec=='-':g[s].append(dict(speaker_id=s,record_id=rec,label=label,attack=attack))
 attacks=[f'A{i:02d}' for i in range(9,17)];pool=[s for s,rr in g.items() if sum(r['label']=='bonafide' for r in rr)>=50 and all(sum(r['attack']==a for r in rr)>=10 for a in attacks)];assert len(pool)==236
 ids=sorted(pool,key=lambda s:(key('identity',s),s))[:60];selected=[]
 for s in ids:
  bona=sorted([r for r in g[s] if r['label']=='bonafide'],key=lambda r:(key('genuine',r['record_id']),r['record_id']))[:50]
  for i,r in enumerate(bona):selected.append(dict(**r,role='enrollment' if i<20 else 'test',path='data/asvspoof5_selfcal/flac_D/'+r['record_id']+'.flac'))
  for a in attacks:
   for r in sorted([r for r in g[s] if r['attack']==a],key=lambda r:(key('spoof',r['record_id']),r['record_id']))[:10]:selected.append(dict(**r,role='test',path='data/asvspoof5_selfcal/flac_D/'+r['record_id']+'.flac'))
 assert len(selected)==7800 and len({r['record_id'] for r in selected})==7800 and not set(ids)&excluded
 assert not {r['record_id'] for r in selected}&{r['record_id'] for r in old}
 write(O/'manifest.csv',selected);write(O/'identity_pool.csv',[dict(speaker_id=s,selection_hash=key('identity',s),selected=s in ids) for s in sorted(pool)])
 save(O/'selection.json',dict(identity_count=60,pool_size=236,identities=ids,attacks=attacks,records=7800,enrollment=1200,test_genuine=1800,test_spoof=4800,manifest_sha256=sha(O/'manifest.csv'),pool_sha256=sha(O/'identity_pool.csv'),selection_script_sha256=sha(Path(__file__)),score_blind=True))
 print('SELECTED 60 identities 7800 records; NO SCORING')
if __name__=='__main__':main()
