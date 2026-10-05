"""Fixed 24-fold retraining and independent replay of the reported development metrics."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
import sklearn
import threadpoolctl
from threadpoolctl import threadpool_limits

from .core import (BLOCKED_PAGEIDS,HISTORICAL_EXCLUSIONS,FORECASTS,KS,aggregate,candidate_allowed,days,
                   joint_threshold,make_label,ordered,preprocess,rank_metric,select_training,shift)
from .learners import MODELS,fit_predict

ROOT=Path(__file__).resolve().parents[2]
BINARY=['volume','binary_log13']
RELATIVE=['volume','relative_ridge12']
SCORE_TOLERANCE=1e-10
METRIC_TOLERANCE=5e-12


def load(path):
    path=Path(path)
    if '2026-09' in path.name:raise ValueError('Holdout filename forbidden')
    body=path.read_bytes()
    return json.loads(gzip.decompress(body) if path.suffix=='.gz' else body)


def save(path,obj):
    body=(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    path=Path(path)
    path.write_bytes(gzip.compress(body,mtime=0) if path.suffix=='.gz' else body)


def digest(values):
    return hashlib.sha256(json.dumps(values,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def validate_panel(panel):
    if panel['schema']!='wikiflow.course.panel.v1':raise ValueError('Unsupported panel schema')
    rows=panel['rows'];labels=panel['labels'];keys=[r['key'] for r in rows]
    if len(set(keys))!=len(keys) or set(labels)!=set(keys):raise ValueError('Duplicate or mismatched label keys')
    for r in rows:
        if r['target_month']>='2026-09' or r['target_month']<'2023-10':raise ValueError('Panel crosses historical boundary')
        if r['pageid'] in BLOCKED_PAGEIDS|HISTORICAL_EXCLUSIONS:raise ValueError('Identity quarantine violated')
        if r['key']!=f"{r['target_month']}:{r['pageid']}":raise ValueError('Key does not match identity/date')
        if r['cutoff_month']!=shift(r['target_month'],-1) or r['max_feature_date_declared'][:7]>r['cutoff_month']:
            raise ValueError('Feature cutoff violated')
        if len(r['features'])!=12:raise ValueError('Original 12 features required')
        if (r.get('metadata_quality_complete') is not True or not r['editor_activity_asof']['coverage_complete']
            or r['current_count']<100 or r['current_count']/days(r['cutoff_month'])>r['U_count']/days(r['target_month'])
            or r['current_count']>r['current_precurrent_U']):
            raise ValueError('Expanded TRAIN requires complete quality and both current-cold gates')
        threshold=r['effective_event_U_asof'];lab=labels[r['key']]
        if abs(joint_threshold(r)-threshold)>1e-8:raise ValueError('Joint threshold changed')
        expected=make_label(lab['actual'],threshold)
        for field in expected:
            a,b=lab[field],expected[field]
            if a is None or b is None:
                if a!=b:raise ValueError('Unknown converted to zero')
            elif isinstance(a,(bool,int)):
                if a!=b:raise ValueError('Event/identity label mismatch')
            elif abs(a-b)>1e-12:raise ValueError('Relative target must be once compressed')
    wanted=[r['key'] for r in rows if r['target_month'] in FORECASTS and candidate_allowed(r)]
    if set(wanted)!=set(panel['evaluation_keys']) or len(set(panel['evaluation_keys']))!=len(panel['evaluation_keys']):
        raise ValueError('Candidate eligibility drift')
    rowmap={r['key']:r for r in rows}
    for key in panel['evaluation_keys']:
        if not candidate_allowed(rowmap[key]):raise ValueError('Disallowed evaluation candidate')
    return rowmap


def verify_prepared(canonical,rebuilt):
    validate_panel(rebuilt)
    if [r['key'] for r in canonical['rows']]!=[r['key'] for r in rebuilt['rows']] or canonical['evaluation_keys']!=rebuilt['evaluation_keys']:
        raise ValueError('Prepared keys/order differ from corrected benchmark')
    if canonical['labels']!=rebuilt['labels']:
        raise ValueError('Prepared labels differ from corrected benchmark')
    maxerror=0.
    for old,new in zip(canonical['rows'],rebuilt['rows']):
        for field in ['pageid','article','target_month','cutoff_month','current_count','metadata_quality_complete','editor_activity_asof','evaluation_eligible_asof','max_feature_date_declared']:
            if old[field]!=new[field]:raise ValueError(f'Prepared identity/eligibility changed: {field}')
        pairs=list(zip(old['features'],new['features']))+[(old[f],new[f]) for f in ['U_count','current_precurrent_U','effective_event_U_asof']]
        for a,b in pairs:
            if a is None or b is None:
                if a!=b:raise ValueError('Prepared missingness changed')
            else:maxerror=max(maxerror,abs(a-b))
    if maxerror>1e-8:raise ValueError('Prepared values differ beyond audited floating tolerance')
    return maxerror


def baseline_scores(candidates):
    return {'volume':{r['key']:float(r['current_count']) for r in candidates}}


def evaluate(candidates,labels,scores):
    monthly=[];summaries=[]
    for task,names in [('binary',BINARY),('relative',RELATIVE)]:
        for name in names:
            for k in KS:
                group=[]
                for month in FORECASTS:
                    ca=[r for r in candidates if r['target_month']==month]
                    q=rank_metric(ca,labels,{r['key']:scores[name][r['key']] for r in ca},k)
                    q.update(task=task,method=name,month=month,K=k)
                    group.append(q);monthly.append(q)
                summaries.append(dict(task=task,method=name,K=k,**aggregate(group)))
    return monthly,summaries


def verify_monthly(monthly,expected):
    lookup={(r['task'],r['method'],r['month'],r['K']):r for r in monthly}
    reference_keys={(r['task'],r['method'],r['month'],r['K']) for r in expected}
    if len(lookup)!=len(monthly) or len(reference_keys)!=len(expected) or set(lookup)!=reference_keys:
        raise ValueError('Monthly metric support differs from frozen reference')
    maxerror=0.
    for ref in expected:
        q=lookup[ref['task'],ref['method'],ref['month'],ref['K']]
        for actual,field in [('binary_NDCG' if ref['task']=='binary' else 'relative_NDCG','NDCG'),('AP','AP'),('TP','TP')]:
            a,b=q[actual],ref[field]
            if a is None or b is None:
                if a!=b:raise ValueError(f'Unknown support mismatch: {ref}')
            else:
                error=abs(a-b);maxerror=max(maxerror,error)
                if error>METRIC_TOLERANCE:raise ValueError(f'Metric mismatch {ref}, {actual}: {a}')
    return maxerror


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def run(mode,output,prepared_panel=None):
    started=time.monotonic();output=Path(output)
    if output.exists():raise ValueError('Use a new output directory; prior runs are preserved')
    if ROOT/'data'==output or ROOT/'config'==output:raise ValueError('Output cannot overwrite inputs')
    provenance=load(ROOT/'data/provenance.json')
    for name,sha in provenance['package_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise ValueError(f'Input hash changed: {name}')
    panel=load(ROOT/'data/panel.json.gz');prepared_error=None
    if prepared_panel is not None:
        rebuilt=load(prepared_panel);prepared_error=verify_prepared(panel,rebuilt);panel=rebuilt
    rowmap=validate_panel(panel)
    rows=panel['rows'];labels=panel['labels'];ca=[rowmap[k] for k in panel['evaluation_keys']]
    if len(rows)!=11080 or len(ca)!=5388:raise ValueError('Benchmark support changed')
    expected=load(ROOT/'data/expected_scores.json.gz');folds=load(ROOT/'data/folds.json')
    if set(expected)!=set(BINARY+RELATIVE):raise ValueError('Unexpected course score methods')
    if any(set(ss)!=set(panel['evaluation_keys']) for ss in expected.values()):
        raise ValueError('Score support differs from shared candidates')
    scores=baseline_scores(ca);baseline_error={}
    for name,ss in scores.items():
        baseline_error[name]=max(abs(v-expected[name][key]) for key,v in ss.items())
        if baseline_error[name]>1e-12:raise ValueError(f'Baseline formula mismatch: {name}')
    output.mkdir(parents=True)
    trace=[];prediction_error={};rank_match={};scaler_error=0.
    if mode=='refit':
        with threadpool_limits(limits=2):
            for month in FORECASTS:
                train=select_training(rows,labels,month);candidates=[r for r in ca if r['target_month']==month]
                fold=folds[month];keys=[r['key'] for r in train]
                if len(train)!=fold['TRAIN_N'] or digest(keys)!=fold['TRAIN_keys_sha256'] or len(candidates)!=fold['candidate_N']:
                    raise ValueError('Own mature TRAIN or evaluation keys changed')
                for name in MODELS:
                    x,z,prep=preprocess(train,candidates,append_joint=name=='binary_log13')
                    ref=fold['preprocessing']['Log13' if name=='binary_log13' else 'Ridge12']
                    for field in prep:scaler_error=max(scaler_error,float(np.max(abs(np.asarray(prep[field])-ref[field]))))
                    if scaler_error>1e-10:raise ValueError('TRAIN preprocessing drift')
                    y=np.asarray([labels[r['key']]['event' if name.startswith('binary') else 'gR'] for r in train],float)
                    t=time.monotonic();pred,info=fit_predict(name,x,y,z)
                    scores.setdefault(name,{}).update({r['key']:float(v) for r,v in zip(candidates,pred)})
                    error=max(abs(float(v)-expected[name][r['key']]) for r,v in zip(candidates,pred))
                    prediction_error[name]=max(prediction_error.get(name,0.),error)
                    own={r['key']:scores[name][r['key']] for r in candidates};refss={r['key']:expected[name][r['key']] for r in candidates}
                    same=[r['key'] for r in ordered(candidates,own)]==[r['key'] for r in ordered(candidates,refss)]
                    rank_match[name]=rank_match.get(name,True) and same
                    record=dict(month=month,model=name,TRAIN_N=len(train),TRAIN_positive=int(sum(labels[r['key']]['event'] for r in train)),
                                TRAIN_last_label=max(r['target_month'] for r in train),TRAIN_feature_max=max(r['max_feature_date_declared'] for r in train),
                                train_outside_edit_candidates=sum(not r['evaluation_eligible_asof'] for r in train),candidate_N=len(candidates),
                                TRAIN_keys_sha256=digest(keys),preprocessing_sha256=digest(prep),seconds=time.monotonic()-t,
                                score_max_error=error,exact_rank_match=same,numeric=info)
                    trace.append(record);save(output/'fit_trace.json',trace)
                    print(f'{month} {name}: fit {len(trace)}/48, score error={error:.3g}, ranks={same}',flush=True)
                    if error>SCORE_TOLERANCE or not same:
                        raise ValueError(f'Retraining mismatch: {month} {name}; diagnose before publishing')
        if len(trace)!=48:raise ValueError('48 actual fits required')
    elif mode=='replay':
        scores=expected
    else:raise ValueError('Mode must be refit or replay')
    save(output/'predictions.json.gz',scores)
    save(output/'prediction_seal.json',dict(mode=mode,actual_fits=len(trace),evaluation_started=False,
        predictions_sha256=hashlib.sha256((output/'predictions.json.gz').read_bytes()).hexdigest()))
    monthly,summaries=evaluate(ca,labels,scores)
    metric_error=verify_monthly(monthly,load(ROOT/'data/expected_monthly.json.gz'))
    write_csv(output/'monthly.csv',monthly);write_csv(output/'summary.csv',summaries)
    write_csv(output/'summary_at50.csv',[r for r in summaries if r['K']==50])
    write_csv(output/'monthly_at50.csv',[r for r in monthly if r['K']==50])
    report=dict(status='PASS',mode=mode,actual_fits=len(trace),fixed_model_fits={n:sum(x['model']==n for x in trace) for n in MODELS},
        months=FORECASTS,rows=11080,candidates=5388,seed=718,threads=2,source_raw_downloads=0,new_tuning_fits=0,holdout_read=False,
        prepared_panel_used=prepared_panel is not None,prepared_value_max_error=prepared_error,
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,threadpoolctl=threadpoolctl.__version__,
        prediction_tolerance=SCORE_TOLERANCE,metric_tolerance=METRIC_TOLERANCE,model_prediction_max_error=prediction_error,
        all_monthly_rank_orders_match=rank_match,TRAIN_scaler_max_error=scaler_error,baseline_formula_max_error=baseline_error,
        canonical_baseline_floats_retained=False,monthly_metric_max_error=metric_error,monthly_metric_checks=len(monthly)*3,
        elapsed_seconds=time.monotonic()-started,development_only=True)
    save(output/'verification.json',report)
    print(json.dumps(report,indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['refit','replay'],default='refit')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepared-panel',type=Path,help='Optional newly prepared cached panel; checked against frozen benchmark before fitting')
    args=parser.parse_args();run(args.mode,args.output,args.prepared_panel)

if __name__=='__main__':main()
