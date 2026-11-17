"""File/PCM/window integrity and role-disjoint duplicate audit before training."""
import csv,json,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
import soundfile as sf
R=Path(__file__).resolve().parents[1];O=R/'results/asvspoof5_adaptation_v1'
def read(p):return list(csv.DictReader(p.open()))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def main():
 done=O/'audio_audit.json'
 if done.exists():
  d=json.loads(done.read_text());assert d['passed'];assert sha(O/'windows.npy')==d['windows_sha256'];return
 complete=json.loads((O/'acquisition_complete.json').read_text());assert complete['success'] and sha(O/'audio_files.csv')==complete['audio_manifest_sha256'];rows=read(O/'manifest.csv');files={r['record_id']:r for r in read(O/'audio_files.csv')};assert len(rows)==len(files)==49310
 waves=np.lib.format.open_memmap(O/'windows.partial.npy',mode='w+',dtype=np.float32,shape=(len(rows),64600));output=[];groups=defaultdict(set);duplicate_records=defaultdict(list)
 for i,r in enumerate(rows):
  p=R/r['path'];assert sha(p)==files[r['record_id']]['sha256'];x,sr=sf.read(p,dtype='float32',always_2d=True);assert sr==16000 and x.shape[1]==1 and len(x)>0 and np.isfinite(x).all(),r['path'];x=x[:,0];w=x[:64600] if len(x)>=64600 else np.resize(x,64600);waves[i]=w;pcm=hashlib.sha256(x.tobytes()).hexdigest();window=hashlib.sha256(w.tobytes()).hexdigest()
  for kind,h in [('pcm',pcm),('window',window)]:groups[kind,h].add(r['role']);duplicate_records[kind,h].append(r['record_id'])
  output.append(dict(index=i,record_id=r['record_id'],role=r['role'],path=r['path'],frames=len(x),file_sha256=files[r['record_id']]['sha256'],pcm_sha256=pcm,window_sha256=window))
  if (i+1)%2000==0:print('AUDIO AUDIT',i+1,len(rows),flush=True)
 waves.flush();del waves
 cross=[dict(kind=k[0],sha256=k[1],roles=sorted(v),record_ids=duplicate_records[k]) for k,v in groups.items() if len(v)>1];(O/'cross_role_duplicates.json').write_text(json.dumps(cross,indent=2));assert not cross,'Cross-role duplicates: audit required, no training'
 with (O/'audio_audit_rows.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
 (O/'windows.partial.npy').rename(O/'windows.npy');done.write_text(json.dumps(dict(passed=True,files=len(rows),cross_role_exact_duplicates=0,within_role_duplicate_groups=sum(len(v)>1 for v in duplicate_records.values()),windows_sha256=sha(O/'windows.npy'),audit_rows_sha256=sha(O/'audio_audit_rows.csv'),near_duplicate_identity_overlap_not_excluded=True),indent=2));print('AUDIO AUDIT COMPLETE',flush=True)
if __name__=='__main__':main()
