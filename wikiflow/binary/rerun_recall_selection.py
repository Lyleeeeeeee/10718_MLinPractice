"""Refit three frozen Recall@50 configurations and verify shared evaluation support."""
import argparse, csv, gzip, hashlib, importlib.metadata, json, platform, subprocess, sys, time
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
repo=args.repo.resolve(); root=repo/'wikiflow/binary'; out=args.output.resolve()
out.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(root))
from fit_fold import fit_recipe
from wikiflow.core import FORECASTS, aggregate, rank_metric, candidate_allowed
from wikiflow.reproduce import load, validate_panel, write_csv

ids=['binary_signed_linear_0.001','binary_svd_0.01','pairwise_expanded_1_a0.5']
recipes=load(root/'config/recipes.json')
panel=load(repo/'wikiflow/data/panel.json.gz'); validate_panel(panel)
rows=panel['rows']; labels=panel['labels']
assert FORECASTS[0]=='2024-09' and FORECASTS[-1]=='2026-08' and len(FORECASTS)==24
assert max(r['target_month'] for r in rows)<'2026-09'
manifest={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
 'python':platform.python_version(),'platform':platform.platform(),
 'versions':{p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','lightgbm','threadpoolctl']},
 'recipes':{k:recipes[k] for k in ids},'months':FORECASTS,'holdout_accessed':False,
 'selection':'Frozen before this rerun: pooled early Recall@50, then macro binary NDCG@50; no new search.',
 'inputs_sha256':{str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [repo/'wikiflow/data/panel.json.gz',root/'data/text/b2/documents.json.gz',root/'config/recipes.json']}}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))

# Verify the current fitter against previously saved early results before development refits.
checks=[]
for mid,batch in zip(ids,['batch1','batch2','batch4_rule_residual']):
 expected=list(csv.DictReader((root/f'evidence/{batch}/validation_monthly.csv').open()))
 months=['2024-02','2024-05','2024-08'] if batch!='batch4_rule_residual' else [f'2024-{m:02}' for m in range(1,9)]
 for month in months:
  scores=fit_recipe(recipes[mid],month)
  ca=[r for r in rows if r['target_month']==month and candidate_allowed(r)]
  metric=rank_metric(ca,labels,scores,50)
  e=next(r for r in expected if r['method']==mid and r['month']==month and r['K']=='50')
  assert metric['TP']==int(e['TP']), (mid,month,'TP',metric,e)
  assert abs(metric['binary_NDCG']-float(e['binary_NDCG']))<1e-9,(mid,month,'NDCG',metric,e)
  checks.append({'method':mid,'month':month,'TP':metric['TP'],'binary_NDCG':metric['binary_NDCG'],'match':True})
  print('EARLY VERIFIED',mid,month,flush=True)
(out/'early_verification.json').write_text(json.dumps(checks,indent=2))

monthly=[]; predictions={}; start=time.time()
for mid in ids:
 predictions[mid]={}
 for month in FORECASTS:
  scores=fit_recipe(recipes[mid],month)
  predictions[mid].update(scores)
  ca=[r for r in rows if r['target_month']==month and candidate_allowed(r)]
  m=rank_metric(ca,labels,scores,50)
  monthly.append(dict(method=mid,month=month,K=50,**m))
  print('FIT',mid,month,'hits',m['TP'],'events',m['events'],flush=True)
  (out/f'{mid}_predictions.json').write_text(json.dumps(predictions[mid],allow_nan=False))
  write_csv(out/'new_monthly.csv',monthly)

# Re-score original saved predictions under exactly the same evaluator/support.
saved=load(root/'data/development_predictions.json.gz')
expected={r['method']:r for r in csv.DictReader((root/'evidence/expected_summary.csv').open())}
for mid,scores in saved.items():
 records=[]
 for month in FORECASTS:
  ca=[r for r in rows if r['target_month']==month and candidate_allowed(r)]
  m=rank_metric(ca,labels,{r['key']:scores[r['key']] for r in ca},50)
  records.append(dict(method=mid,month=month,K=50,**m))
 a=aggregate(records)
 for k in ['TP','events','selected','micro_Recall','macro_binary_NDCG','macro_Precision']:
  assert abs(a[k]-float(expected[mid][k]))<1e-10,(mid,k)
 monthly.extend(records)
summaries=[]
for mid in ids+list(saved):
 rec=[r for r in monthly if r['method']==mid]
 a=aggregate(rec)
 assert a['events']==190 and a['selected']==1200 and sum(r['N'] for r in rec)==5387
 summaries.append(dict(method=mid,source='new_refit' if mid in ids else 'saved_predictions_replay',**a))
write_csv(out/'all_monthly.csv',monthly)
write_csv(out/'comparison.csv',summaries)

# Paired month bootstrap: re-compute pooled recall for each sampled 24-month period.
rng=np.random.default_rng(718); ix=rng.integers(0,24,size=(10000,24)); comparisons=[]
for mid in ids:
 a=[r for r in monthly if r['method']==mid]; av=np.array([r['TP'] for r in a]); ev=np.array([r['events'] for r in a])
 for base in ['cs_momentum','binary_log13','binary_missing_0.01','rank50_base']:
  b=[r for r in monthly if r['method']==base]; bv=np.array([r['TP'] for r in b])
  assert [r['month'] for r in a]==[r['month'] for r in b]
  dist=(av[ix].sum(axis=1)-bv[ix].sum(axis=1))/ev[ix].sum(axis=1)
  lo,hi=np.quantile(dist,[.025,.975])
  comparisons.append(dict(method=mid,baseline=base,delta_recall=float((av.sum()-bv.sum())/190),ci95_low=float(lo),ci95_high=float(hi),wins=int((av>bv).sum()),losses=int((av<bv).sum()),ties=int((av==bv).sum())))
write_csv(out/'paired_bootstrap.csv',comparisons)
manifest.update(development_fits=72,early_verification_fits=len(checks),elapsed_development_seconds=time.time()-start,completed=True)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(summaries,indent=2),flush=True)
