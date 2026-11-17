"""All seeds, identity paired uncertainty, root-matched codec and source retention."""
import json,csv
from pathlib import Path
from collections import defaultdict
import numpy as np
from aasist_replication_common import read,sha
from asvspoof5_metrics import identity_rates,ranking,paired_bootstrap,codec_pairs
R=Path(__file__).resolve().parents[1];O=R/'results/asvspoof5_adaptation_v1';A=O/'analysis'
def save(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2)
def write(name,rows):
 with (A/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def prepare():
 A.mkdir(exist_ok=True)
 inputs=[Path(__file__),R/'scripts/asvspoof5_metrics.py',R/'scripts/validate_asvspoof5_adaptation.py',R/'docs/asvspoof5-scoring-v1.md',O/'scoring_prepared.json']
 rec=dict(input_sha256={str(p.relative_to(R)):sha(p) for p in inputs},bootstrap_rounds=10000,bootstrap_seed=20260919,ranking_condition='unencoded_only',source_threshold_policy='original_family_threshold_fixed_for_all_models',codec_weighting='parent then identity then attack',no_evaluation_selection=True)
 if (A/'prepared.json').exists():assert json.loads((A/'prepared.json').read_text())==rec
 else:save(A/'prepared.json',rec)
def main():
 prep=json.loads((A/'prepared.json').read_text())
 for p,h in prep['input_sha256'].items():assert sha(R/p)==h,p
 completed=json.loads((O/'scoring_complete.json').read_text());assert completed['success']
 for p,h in completed['artifact_sha256'].items():assert sha(R/p)==h,p
 models=json.loads((O/'training_complete.json').read_text())['models'];rows=read(O/'manifest.csv');zz=np.load(O/'evaluation_logits.npz');ev=[rows[i] for i in zz['indices']];z=zz['logits'];v=z[:,:,0]-z[:,:,1];thresholds=json.loads((O/'thresholds_frozen.json').read_text())['thresholds']
 raw=np.array([r['codec']=='-' for r in ev]);rr=[r for r in ev if r['codec']=='-'];rv=v[raw];assert len(rr)==11400
 attacks=sorted({r['attack'] for r in rr if r['label']=='spoof'});assert len(attacks)==16
 identities=sorted({r['speaker_id'] for r in rr});assert len(identities)==60
 rates=[];summary=[];perid=[];perattack=[];rankrows=[];ranksummary=[];codecs=[];codecgroups=[]
 def meta(j):
  m=models[j];return dict(index=j,family=m['family'],model=m['model'],method=m['method'],seed=m['seed'])
 for j,m in enumerate(models):
  mm=meta(j);t=thresholds[j]['threshold'];ids,a,n=identity_rates(rr,rv[:,j],t,attacks);assert ids==identities and np.all(n[:,0]==30) and np.all(n[:,1:]==10);rates.append(a)
  ci=paired_bootstrap(a);summary.append(dict(**mm,threshold=t,macro_FA=ci['point'][0],FA_low=ci['low'][0],FA_high=ci['high'][0],attack_macro_TPR=ci['point'][1],TPR_low=ci['low'][1],TPR_high=ci['high'][1],genuine_support=1800,spoof_support=9600,identities=60))
  for k,ident in enumerate(ids):
   for c,attack in enumerate(['bonafide',*attacks]):perid.append(dict(**mm,speaker_id=ident,attack=attack,rate=float(a[k,c]),support=int(n[k,c]),predicted_spoof=int(round(a[k,c]*n[k,c]))))
  for k,attack in enumerate(attacks):perattack.append(dict(**mm,attack=attack,TPR=float(a[:,k+1].mean()),support=600,identities=60))
  local=[]
  for ident in identities:
   bi=np.array([r['speaker_id']==ident and r['label']=='bonafide' for r in rr]);si=np.array([r['speaker_id']==ident and r['label']=='spoof' for r in rr]);d=ranking(rv[bi,j],rv[si,j]);row=dict(**mm,speaker_id=ident,genuine_support=d['genuine_support'],spoof_support=d['spoof_support'],auc=d['auc'],step_pauc_005=d['step_pauc'],oracle_0=d['oracle']['0.0'],oracle_001=d['oracle']['0.01'],oracle_005=d['oracle']['0.05']);assert row['oracle_0']==row['oracle_001'];rankrows.append(row);local.append(row)
  b=np.array([r['label']=='bonafide' for r in rr]);ss=np.sort(rv[~b,j]);bv=rv[b,j];pooled=float(np.mean((np.searchsorted(ss,bv,'left')+np.searchsorted(ss,bv,'right'))/(2*len(ss))))
  ranksummary.append(dict(**mm,**{k:float(np.mean([r[k] for r in local])) for k in ['auc','step_pauc_005','oracle_0','oracle_001','oracle_005']},pooled_auc=pooled))
  for codec in sorted({r['codec'] for r in ev}-{'-'}):
   pairs=codec_pairs(ev,v[:,j],t,codec);parents=defaultdict(list)
   for p in pairs:parents[p['speaker_id'],p['label'],p['attack'],p['parent_record_id']].append(p)
   groups=defaultdict(list)
   for (ident,label,attack,parent),pp in parents.items():groups[ident,label,attack].append((np.mean([p['codec_spoof'] for p in pp]),pp[0]['raw_spoof'],len(pp)))
   cg=[]
   for (ident,label,attack),gg in sorted(groups.items()):
    encoded=float(np.mean([g[0] for g in gg]));base=float(np.mean([g[1] for g in gg]));item=dict(**mm,codec=codec,speaker_id=ident,label=label,attack=attack,encoded_rate=encoded,matched_raw_rate=base,delta=encoded-base,parent_support=len(gg),file_support=sum(g[2] for g in gg));codecgroups.append(item);cg.append(item)
   bona=[r for r in cg if r['label']=='bonafide'];fake=[r for r in cg if r['label']=='spoof'];fa={};tp={}
   for field in ['encoded_rate','matched_raw_rate','delta']:
    fa[field]=float(np.mean([r[field] for r in bona]));tp[field]=float(np.mean([np.mean([r[field] for r in fake if r['attack']==attack]) for attack in sorted({r['attack'] for r in fake})]))
   codecs.append(dict(**mm,codec=codec,FA=fa['encoded_rate'],matched_raw_FA=fa['matched_raw_rate'],delta_FA=fa['delta'],TPR=tp['encoded_rate'],matched_raw_TPR=tp['matched_raw_rate'],delta_TPR=tp['delta'],genuine_identities=len(bona),spoof_identity_attack_groups=len(fake),attacks=len({r['attack'] for r in fake}),genuine_parent_support=sum(r['parent_support'] for r in bona),spoof_parent_support=sum(r['parent_support'] for r in fake)))
  print('ANALYZED',j,m['family'],m['model'],flush=True)
 ratearray=np.stack(rates);np.save(A/'identity_rates.npy',ratearray);diffs=[];lookup={(m['family'],m['model']):j for j,m in enumerate(models)}
 for j,m in enumerate(models):
  if m['method']=='original':continue
  refs=['original']
  if m['method'] in ['full_ce','kd10']:refs.append('source_only_s'+str(m['seed']))
  if m['method']=='kd10':refs.append('full_ce_s'+str(m['seed']))
  for ref in refs:
   k=lookup[m['family'],ref];ci=paired_bootstrap(ratearray[j],ratearray[k]);diffs.append(dict(**meta(j),reference=ref,delta_FA=ci['point'][0],FA_low=ci['low'][0],FA_high=ci['high'][0],delta_TPR=ci['point'][1],TPR_low=ci['low'][1],TPR_high=ci['high'][1]))
 # Source selection: report class micro and identity macro, plus attacks, under both explicit rules.
 source=read(O/'source_selection.csv');sz=np.load(O/'source_logits.npz')['logits'];sourceout=[];sourcegroups=[]
 for j,m in enumerate(models):
  folder='replication_v1' if m['family']=='xlsr_lcnn' else 'aasist_replication_v1';old=next(r for r in json.loads((R/'results'/folder/'thresholds.json').read_text())['models'] if r['model']=='original')['threshold'];margin=sz[:,j,0]-sz[:,j,1]
  for policy,t in [('new_target_threshold',thresholds[j]['threshold']),('original_source_threshold',old)]:
   score=1/(1+np.exp(-margin)) if policy=='original_source_threshold' and m['family']=='xlsr_lcnn' else margin;hit=score<t
   for label in ['bonafide','spoof']:
    ix=np.array([r['label']==label for r in source]);ids=sorted({r['speaker_id'] for r in source if r['label']==label});macro=np.mean([hit[[i for i,r in enumerate(source) if r['label']==label and r['speaker_id']==s]].mean() for s in ids]);sourceout.append(dict(**meta(j),policy=policy,threshold=t,score='bonafide_probability' if policy=='original_source_threshold' and m['family']=='xlsr_lcnn' else 'margin',label=label,micro=float(hit[ix].mean()),macro=float(macro),support=int(ix.sum()),identities=len(ids)))
   for attack in sorted({r['attack_id'] for r in source if r['label']=='spoof'}):
    ix=np.array([r['label']=='spoof' and r['attack_id']==attack for r in source]);sourcegroups.append(dict(**meta(j),policy=policy,attack=attack,TPR=float(hit[ix].mean()),support=int(ix.sum())))
 seedrows=[]
 for family in ['xlsr_lcnn','aasist']:
  for method in ['full_ce','kd10','source_only']:
   rs=[r for r in summary if r['family']==family and r['method']==method];assert len(rs)==3
   for field in ['macro_FA','attack_macro_TPR']:
    values=np.array([r[field] for r in rs]);seedrows.append(dict(family=family,method=method,metric=field,mean=float(values.mean()),sample_sd=float(values.std(ddof=1)),minimum=float(values.min()),maximum=float(values.max()),seeds=3))
 for name,rows in [('primary.csv',summary),('per_identity_attack.csv',perid),('per_attack.csv',perattack),('ranking.csv',rankrows),('ranking_summary.csv',ranksummary),('codec_paired.csv',codecs),('codec_identity_attack.csv',codecgroups),('paired_differences.csv',diffs),('source_retention.csv',sourceout),('source_per_attack.csv',sourcegroups),('seed_summary.csv',seedrows)]:write(name,rows)
 save(A/'analysis_complete.json',dict(success=True,attacks=attacks,identities=identities,scoring_complete_sha256=sha(O/'scoring_complete.json'),artifact_sha256={str(p.relative_to(R)):sha(p) for p in sorted(A.iterdir()) if p.is_file()}))
if __name__=='__main__':
 import sys
 if len(sys.argv)>1 and sys.argv[1]=='prepare':prepare()
 else:main()
