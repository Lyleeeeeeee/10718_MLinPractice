"""Pure historical contracts shared by cached preparation, fitting and evaluation."""
import calendar
import math
import statistics

import numpy as np

BLOCKED_PAGEIDS = {79745475}
HISTORICAL_EXCLUSIONS = {1164,72417803}
FORECASTS = [f'{i//12:04d}-{i%12+1:02d}' for i in range(2024*12+8,2026*12+8)]
EARLY = ['2024-02','2024-05','2024-08']
KS = [5,10,20,50,100,200]


def days(month):
    return calendar.monthrange(*map(int,month.split('-')))[1]


def shift(month,offset):
    year,m = map(int,month.split('-'))
    i = year*12+m-1+offset
    return f'{i//12:04d}-{i%12+1:02d}'


def historical_bound(rates,next_days):
    if len(rates)!=6 or any(v is None or not math.isfinite(v) or v<0 for v in rates):
        raise ValueError('Six published daily rates are required')
    median = statistics.median(rates)
    sigma = 1.4826*statistics.median(abs(v-median) for v in rates)
    return max(200,next_days*median+max(100,.5*next_days*median,3*next_days*sigma))


def joint_threshold(row):
    forecast = row['current_count']*days(row['target_month'])/days(row['cutoff_month'])
    return max(row['U_count'],1.25*forecast)


def joint_margin(row):
    forecast = row['current_count']*days(row['target_month'])/days(row['cutoff_month'])
    return math.log1p(forecast)-math.log1p(joint_threshold(row))


def make_label(actual,threshold):
    if isinstance(threshold,bool) or not math.isfinite(threshold) or threshold<200:
        raise ValueError('Joint threshold must be finite and at least 200')
    if actual is None:
        return dict(known=False,actual=None,T=threshold,R=None,gR=None,event=None)
    if isinstance(actual,bool) or not isinstance(actual,int) or actual<0:
        raise ValueError('Published count must be a nonnegative integer or unknown')
    r = max(0.,actual/threshold-1.)
    return dict(known=True,actual=actual,T=threshold,R=r,gR=math.log1p(r),event=int(actual>threshold))


def candidate_allowed(row):
    activity = row['editor_activity_asof']
    return bool(row['pageid'] not in BLOCKED_PAGEIDS|HISTORICAL_EXCLUSIONS and row.get('metadata_quality_complete',True)
                and row['current_count']>=100 and activity['coverage_complete']
                and activity['max_abs_nonminor'] is not None and activity['cumulative_abs_nonminor'] is not None
                and activity['max_abs_nonminor']<500 and activity['cumulative_abs_nonminor']<1000
                and row['current_count']/days(row['cutoff_month'])<=row['U_count']/days(row['target_month'])
                and row['current_count']<=row['current_precurrent_U'])


def select_training(rows,labels,month):
    if month not in FORECASTS+EARLY:
        raise ValueError('Only authorized historical folds; holdout is sealed')
    train = [r for r in rows if r['target_month']<month and labels[r['key']]['known']
             and r['pageid'] not in BLOCKED_PAGEIDS|HISTORICAL_EXCLUSIONS]
    for row in train:
        if row['cutoff_month']!=shift(row['target_month'],-1) or row['max_feature_date_declared'][:7]>row['cutoff_month']:
            raise ValueError('Features later than own target minus one')
    return train


def preprocess(train,candidates,append_joint=False):
    if not train:
        raise ValueError('Nonempty mature TRAIN required')
    def values(rows):
        return [r['features']+([joint_margin(r)] if append_joint else []) for r in rows]
    train_values = values(train)
    n = len(train_values[0]);medians=[];means=[];scales=[]
    for j in range(n):
        observed = [v[j] for v in train_values if v[j] is not None]
        median = statistics.median(observed) if observed else 0.
        filled = [median if v[j] is None else v[j] for v in train_values]
        sd = statistics.pstdev(filled)
        medians.append(median);means.append(statistics.mean(filled));scales.append(sd if sd>=1e-12 else 1.)
    def transform(rows):
        arr = np.asarray([[medians[j] if x is None else x for j,x in enumerate(v)]
                          for v in values(rows)],float).reshape(-1,n)
        out = (arr-np.asarray(means))/np.asarray(scales)
        if not np.isfinite(out).all():
            raise ValueError('Nonfinite feature after TRAIN imputation')
        return out
    return transform(train),transform(candidates),dict(medians=medians,means=means,scales=scales)


def select_lambda(values):
    if set(values)!={.01,.1} or any(len(v)!=3 or any(x is None or not math.isfinite(x) for x in v) for v in values.values()):
        raise ValueError('Exactly two frozen lambdas on three early folds')
    means = {lam:statistics.mean(v) for lam,v in values.items()}
    best = max(means.values())
    return max(lam for lam,v in means.items() if best-v<=1e-12)


def ordered(rows,scores):
    if set(scores)!={r['key'] for r in rows} or not all(math.isfinite(x) for x in scores.values()):
        raise ValueError('Exact candidate keys and finite scores required')
    return sorted(rows,key=lambda r:(-scores[r['key']],-r['current_count'],r['pageid']))


def rank_metric(rows,labels,scores,k):
    if k<=0 or not rows:
        raise ValueError('Positive K and nonempty candidates required')
    ranked = ordered(rows,scores);ak=min(k,len(rows))
    base = dict(actual_K=ak,N=len(rows))
    if any(not labels[r['key']]['known'] for r in rows):
        return dict(base,status='unknown',events=None,TP=None,Precision=None,Recall=None,AP=None,
                    binary_NDCG=None,relative_NDCG=None,selected_gR=None,total_gR=None)
    ls = [labels[r['key']] for r in ranked]
    events = sum(x['event'] for x in ls);tp=sum(x['event'] for x in ls[:ak])
    def ndcg(field):
        vals=[x[field] for x in ls]
        def dcg(v):return math.fsum(x/math.log2(i+2) for i,x in enumerate(v[:ak]))
        ideal=dcg(sorted(vals,reverse=True))
        return dcg(vals)/ideal if ideal else None
    hits=0;terms=[]
    for i,lab in enumerate(ls,1):
        hits+=lab['event']
        if lab['event']:terms.append(hits/i)
    return dict(base,status='defined' if events else 'no_events',events=events,TP=tp,
                Precision=tp/ak,Recall=tp/events if events else None,
                AP=math.fsum(terms)/events if events else None,
                binary_NDCG=ndcg('event'),relative_NDCG=ndcg('gR'),
                selected_gR=math.fsum(x['gR'] for x in ls[:ak]),total_gR=math.fsum(x['gR'] for x in ls))


def aggregate(monthly):
    complete=[r for r in monthly if r['status']!='unknown']
    def avg(field):
        vals=[r[field] for r in monthly if r[field] is not None]
        return statistics.mean(vals) if vals else None
    total=sum(r['total_gR'] for r in complete)
    return dict(months=len(monthly),complete_months=len(complete),event_months=sum(r['binary_NDCG'] is not None for r in monthly),
                events=sum(r['events'] for r in complete),TP=sum(r['TP'] for r in complete),
                selected=sum(r['actual_K'] for r in complete),macro_binary_NDCG=avg('binary_NDCG'),
                macro_relative_NDCG=avg('relative_NDCG'),macro_AP=avg('AP'),macro_Precision=avg('Precision'),
                macro_Recall=avg('Recall'),pooled_gR_capture=sum(r['selected_gR'] for r in complete)/total if total else None,
                micro_Precision=sum(r['TP'] for r in complete)/sum(r['actual_K'] for r in complete) if complete else None,
                micro_Recall=sum(r['TP'] for r in complete)/sum(r['events'] for r in complete) if sum(r['events'] for r in complete) else None)
