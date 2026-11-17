"""D5 consistency check + D2 speaker-pool verification (newdoc/route_c_integrated_p1_v1.md §5).

Independent re-implementation of the Study 1 primary metrics, reconciled against
results/asvspoof5_adaptation_v1/analysis. Public score files are filtered to the
60 already-exposed Study 1 eval speakers while reading; no other score is kept.
Refuses to overwrite its output directory.
"""
import csv,json,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'results/asvspoof5_adaptation_v1';S=R/'data/external_scores/sasb_asvspoof5';O=R/'results/multidetector_d5_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<24),b''):h.update(b)
 return h.hexdigest()
def read(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def auc(b,s):
 ss=np.sort(s);return float(np.mean((np.searchsorted(ss,b,'left')+np.searchsorted(ss,b,'right'))/(2*len(ss))))
def eer(b,s):
 t=np.unique(np.r_[b,s]);fa=np.searchsorted(np.sort(b),t,'left')/len(b);miss=1-np.searchsorted(np.sort(s),t,'left')/len(s);i=np.argmin(abs(fa-miss));return float((fa[i]+miss[i])/2)
def metrics(v,spk,lab,att,ids,attacks,threshold=None):
 b=lab=='bonafide';d=dict(mean_identity_auc=float(np.mean([auc(v[b&(spk==q)],v[~b&(spk==q)]) for q in ids])),pooled_auc=auc(v[b],v[~b]),pooled_eer=eer(v[b],v[~b]),tied_bonafide_fraction=float(1-len(np.unique(v[b]))/b.sum()))
 if threshold is not None:
  d['macro_FA']=float(np.mean([(v[b&(spk==q)]<threshold).mean() for q in ids]))
  d['attack_macro_TPR']=float(np.mean([np.mean([(v[(att==a)&(spk==q)]<threshold).mean() for q in ids]) for a in attacks]))
 return d
def main():
 O.mkdir(exist_ok=False)
 # --- D2 speaker pool (protocol metadata and frozen split only; no scores) ---
 proto=[l.split() for l in (R/'results/independent_data_audit_v1/ASVspoof5.eval.track_1.tsv').read_text().splitlines()]
 pspk={p[0] for p in proto};man=read(A/'manifest.csv');ev=[r for r in man if r['split']=='eval'];s1={r['speaker_id'] for r in ev}
 split=read(R/'results/speakercal_split_v1/split.csv');conf={r['speaker_id'] for r in split if r['role']=='confirm'};dev={r['speaker_id'] for r in split if r['role']=='dev'}
 c40={r['speaker_id'] for r in split if r['role']=='confirm' and int(r['unencoded_bonafide'])>=40};d40={r['speaker_id'] for r in split if r['role']=='dev' and int(r['unencoded_bonafide'])>=40}
 uncoded=[r for r in ev if r['codec']=='-'];ids=sorted({r['speaker_id'] for r in uncoded})
 pool=dict(protocol_records=len(proto),protocol_speakers=len(pspk),protocol_speaker_prefixes=sorted({s[:2] for s in pspk}),local_manifest_eval_records=len(ev),local_manifest_eval_speakers=len(s1),
  protocol_records_of_those_speakers=sum(p[0] in s1 for p in proto),local_eval_records_all_in_protocol=bool({r['record_id'] for r in ev}<={p[1] for p in proto}),
  confirm=len(conf),dev=len(dev),confirm_eligible=len(c40),dev_eligible=len(d40),confirm_and_dev_within_protocol=bool((conf|dev)<=pspk),
  remaining_after_removing_confirm_and_dev=len(pspk-conf-dev),remaining_equals_study1_eval_speakers=bool(pspk-conf-dev==s1),
  used_speakers=ids,used_intersection=dict(confirm_451=len(set(ids)&conf),dev_226=len(set(ids)&dev),confirm_413=len(set(ids)&c40),dev_204=len(set(ids)&d40)),split_sha256=sha(R/'results/speakercal_split_v1/split.csv'))
 assert not set(ids)&(conf|dev) and len(ids)==60
 # --- D5a: recompute the 20 local models and reconcile with the frozen analysis ---
 zz=np.load(A/'evaluation_logits.npz');rows=[man[i] for i in zz['indices']];keep=np.array([r['codec']=='-' for r in rows]);rr=[r for r,k in zip(rows,keep) if k];assert len(rr)==11400
 v=(zz['logits'][:,:,0]-zz['logits'][:,:,1])[keep];bona_logit=zz['logits'][keep][:,:,0]
 spk=np.array([r['speaker_id'] for r in rr]);lab=np.array([r['label'] for r in rr]);att=np.array([r['attack'] for r in rr]);attacks=sorted(set(att[lab=='spoof']));assert len(attacks)==16 and sorted(set(spk))==ids
 primary=read(A/'analysis/primary.csv');rank=read(A/'analysis/ranking_summary.csv');thr=json.loads((A/'thresholds_frozen.json').read_text())['thresholds'];local=[];worst=0.
 for j,(p,q,t) in enumerate(zip(primary,rank,thr)):
  assert p['model']==q['model']==t['model'] and p['family']==t['family']
  m=metrics(v[:,j],spk,lab,att,ids,attacks,t['threshold']);diff={k:abs(m[k]-float(src[c])) for k,src,c in [('macro_FA',p,'macro_FA'),('attack_macro_TPR',p,'attack_macro_TPR'),('mean_identity_auc',q,'auc'),('pooled_auc',q,'pooled_auc')]}
  worst=max(worst,*diff.values());local.append(dict(family=p['family'],model=p['model'],threshold=t['threshold'],**m,max_abs_diff_vs_frozen=max(diff.values())))
 # --- D5b + D2 preview: public scores, only records of the 60 exposed speakers are retained ---
 want={r['record_id']:i for i,r in enumerate(rr)};public=[];aasist_public=None
 for path in sorted(S.glob('*.scores.txt')):
  x=np.full(len(rr),np.nan);n=0
  with path.open() as f:
   for line in f:
    n+=1;k=line[:12]
    if k in want:x[want[k]]=float(line.split()[1])
  assert np.isfinite(x).all(),path
  name=path.name.replace('.scores.txt','');public.append(dict(system=name,file_rows=n,sha256=sha(path),**metrics(x,spk,lab,att,ids,attacks)))
  if name=='AASIST':aasist_public=x
 j=next(i for i,p in enumerate(primary) if p['family']=='aasist' and p['model']=='original');d=abs(aasist_public-bona_logit[:,j])
 same=dict(local_model='aasist/original',public_score_vs_local_bonafide_logit=dict(median_abs_diff=float(np.median(d)),max_abs_diff=float(d.max())),
  public_score_definition='bona fide logit only; the manuscript uses the margin bonafide_logit - spoof_logit',
  local_margin=metrics(v[:,j],spk,lab,att,ids,attacks),local_bonafide_logit=metrics(bona_logit[:,j],spk,lab,att,ids,attacks),public=metrics(aasist_public,spk,lab,att,ids,attacks))
 out=dict(code_sha256=sha(__file__),scope='60 Study 1 eval speakers, unencoded only: 1800 bona fide + 9600 spoof (16 attacks)',pool=pool,
  d5a=dict(models=len(local),max_abs_diff_vs_frozen_analysis=worst,passed=bool(worst<1e-9)),d5b_same_checkpoint=same,
  frozen_threshold_metrics_available_for_public_systems=False,reason='the frozen 1% thresholds come from 40 ASVspoof 5 dev (D_) calibration speakers; public score files contain eval (E_) records only')
 with (O/'summary.json').open('x') as f:json.dump(out,f,indent=2,ensure_ascii=False)
 for name,rws in (('local_models.csv',local),('public_systems.csv',public)):
  with (O/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rws[0]));w.writeheader();w.writerows(rws)
 out['pool'].pop('used_speakers');print(json.dumps(out,indent=1,ensure_ascii=False))
 for r in local+public:print({k:(round(x,4) if isinstance(x,float) else x) for k,x in r.items() if k!='sha256'})
if __name__=='__main__':main()
