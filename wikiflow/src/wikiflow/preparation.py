"""Read-only raw-cache preparation of the fixed identity-corrected historical panel."""
import argparse
import collections
import csv
import datetime as dt
import gzip
import json
import math
import statistics
from pathlib import Path

from .acquisition import BLOCKED, fresh, historical
from .core import days, shift, historical_bound, make_label, candidate_allowed

FEATURES = ['log_current_daily_rate','log_previous_daily_rate','log_mean_last3_daily_rate',
            'log_median_last6_daily_rate','last6_daily_rate_slope','log_seasonal_daily_rate',
            'seasonal_missing','log_current_month_mean_daily_PV','last14_prior14_log_PV_growth',
            'PV_month_missing','log_robust_daily_rate_sigma','log_daily_persistence_margin_U']


def months(start,end):
    historical(start);historical(end)
    if start>end:raise ValueError('Reversed month interval')
    current=start
    while current<=end:
        yield current
        current=shift(current,1)


def metadata_activity(records,coverage,target):
    """Two full calendar months; exact parent net bytes, including reverts."""
    start=shift(target,-2)+'-01T00:00:00Z';end=target+'-01T00:00:00Z';left=start
    for interval in sorted(coverage,key=lambda q:q['start']):
        if interval['end_exclusive']<=left:continue
        if interval['start']>left:break
        left=max(left,interval['end_exclusive'])
        if left>=end:break
    unknown=dict(coverage_complete=False,max_abs_nonminor=None,cumulative_abs_nonminor=None,window_start=start,window_end_exclusive=end)
    if left<end:return unknown
    byid={}
    for r in records:
        if r['timestamp']>=end:continue
        if r['revid'] in byid and byid[r['revid']]!=r:raise ValueError('Conflicting revision ID')
        byid[r['revid']]=r
    ordered=sorted(byid.values(),key=lambda r:(r['timestamp'],r['revid']))
    prior=[r for r in ordered if r['timestamp']<start]
    inside=[r for r in ordered if r['timestamp']>=start]
    if not prior:return unknown
    chain=[prior[-1]]+inside
    if any(not r.get('sha1') or type(r.get('size')) is not int or any(r.get(k) for k in ['sha1hidden','texthidden','commenthidden','suppressed']) for r in chain):return unknown
    if any(child['parentid']!=parent['revid'] for parent,child in zip(chain,chain[1:])):return unknown
    if any(type(r.get('minor')) is not bool for r in inside):return unknown
    deltas=[abs(r['size']-byid[r['parentid']]['size']) for r in inside if not r['minor']]
    return dict(coverage_complete=True,max_abs_nonminor=max(deltas,default=0),cumulative_abs_nonminor=sum(deltas),window_start=start,window_end_exclusive=end)


def article_features(article,cutoff,volumes,pageviews):
    target=shift(cutoff,1);history=[shift(cutoff,k) for k in range(-5,1)]
    rates=[volumes[m][article]/days(m) for m in history]
    median=statistics.median(rates);sigma=1.4826*statistics.median(abs(v-median) for v in rates)
    bound=historical_bound(rates,days(target));season=shift(target,-12)
    missing=article not in volumes.get(season,{})
    seasonal=rates[-1] if missing else volumes[season][article]/days(season)
    daily=pageviews.get(article,{})
    dates=[cutoff+f'-{i:02d}' for i in range(1,days(cutoff)+1)]
    complete=all(d in daily for d in dates)
    trend=None
    if complete:
        end=dt.date.fromisoformat(dates[-1])
        recent=[str(end-dt.timedelta(days=i)) for i in range(14)]
        previous=[str(end-dt.timedelta(days=i)) for i in range(14,28)]
        trend=math.log1p(sum(daily[d] for d in recent)/14)-math.log1p(sum(daily[d] for d in previous)/14)
    forecast=rates[-1]*days(target)
    features=[math.log1p(rates[-1]),math.log1p(rates[-2]),math.log1p(sum(rates[-3:])/3),
              math.log1p(median),sum((i-2.5)*r for i,r in enumerate(rates))/17.5,math.log1p(seasonal),int(missing),
              math.log1p(sum(daily[d] for d in dates)/len(dates)) if complete else None,trend,int(not complete),
              math.log1p(sigma),math.log1p(forecast)-math.log1p(bound)]
    rules=dict(daily_persistence=forecast,PV_trend=forecast*math.exp(trend) if complete else forecast)
    return features,bound,rules


def build_panel(mapping,volumes,pageviews,revisions,coverage,start='2023-10',end='2026-08'):
    if start<'2023-10':raise ValueError('Historical training starts at 2023-10')
    rows=[];labels={};evaluation_keys=[];unknown_quality=0
    for target in months(start,end):
        cut=shift(target,-1);history=[shift(cut,k) for k in range(-6,1)]
        for article,pid in sorted(mapping.items()):
            if pid in BLOCKED:continue
            current=volumes.get(cut,{}).get(article)
            if current is None or current<100 or not all(article in volumes.get(m,{}) for m in history):continue
            features,bound,rules=article_features(article,cut,volumes,pageviews)
            precurrent=historical_bound([volumes[m][article]/days(m) for m in history[:-1]],days(cut))
            if current/days(cut)>bound/days(target) or current>precurrent:continue
            activity=metadata_activity(revisions.get(str(pid),[]),coverage.get(str(pid),[]),target)
            if not activity['coverage_complete']:
                unknown_quality+=1;continue
            row=dict(key=f'{target}:{pid}',pageid=pid,article=article,target_month=target,cutoff_month=cut,
                     current_count=current,U_count=bound,current_precurrent_U=precurrent,features=features,
                     editor_activity_asof=activity,metadata_quality_complete=True,
                     max_feature_date_declared=cut+f'-{days(cut):02d}',rule_forecasts=rules)
            # Preserve the v2 label preparation's multiplication order. The Log13
            # input margin separately uses core.joint_threshold's frozen recipe.
            row['effective_event_U_asof']=max(bound,1.25*current*days(target)/days(cut))
            row['v2_evaluation_eligible_asof']=candidate_allowed(row)
            rows.append(row);labels[row['key']]=make_label(volumes.get(target,{}).get(article),row['effective_event_U_asof'])
            if target>='2024-09' and row['v2_evaluation_eligible_asof']:evaluation_keys.append(row['key'])
    evaluation_keys.sort(key=lambda key:(key[:7],int(key.split(':')[1])))
    return dict(schema='wikiflow.corrected.panel.v1',rows=rows,labels=labels,
                evaluation_keys=evaluation_keys,feature_names=FEATURES,
                preparation=dict(unknown_quality_rows_excluded=unknown_quality,canonical_only=True,
                                 catalogue_count=len(mapping),label_release_delay_verified=False))


def read_mapping(path):
    mapping={}
    with open(path,newline='') as f:
        for r in csv.DictReader(f):
            pid=int(r['pageid']);article=r['article']
            if article in mapping and mapping[article]!=pid:raise ValueError('Conflicting title page ID')
            mapping[article]=pid
    if not mapping:raise ValueError('Empty fixed catalogue')
    return mapping


def read_incoming(directory,start,end,catalogue):
    volumes={}
    for month in months(start,end):
        path=Path(directory)/f'incoming_{month}.csv.gz'
        if not path.exists():continue
        counts=collections.Counter();seen=set()
        with gzip.open(path,'rt',encoding='utf-8',newline='') as f:
            for r in csv.DictReader(f):
                n=int(r['count']);pair=(r['source'],r['destination'])
                if r['month']!=month or r['destination'] not in catalogue or n<10 or pair in seen:raise ValueError('Invalid cached clickstream extract')
                seen.add(pair);counts[r['destination']]+=n
        volumes[month]=dict(counts)
    return volumes


def read_pageviews(path,max_date):
    pageviews=collections.defaultdict(dict)
    with gzip.open(path,'rt',encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            date=r['date']
            if date>max_date:continue
            n=int(r['views']);title=r['destination']
            if n<0 or date in pageviews[title]:raise ValueError('Invalid or duplicate daily cached PV')
            dt.date.fromisoformat(date);pageviews[title][date]=n
    return dict(pageviews)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mapping',required=True);p.add_argument('--incoming',required=True);p.add_argument('--pageviews',required=True)
    p.add_argument('--revisions',required=True);p.add_argument('--coverage',required=True);p.add_argument('--output',required=True)
    p.add_argument('--start',default='2023-10');p.add_argument('--end',default='2026-08')
    args=p.parse_args();historical(args.end);fresh(args.output)
    mapping=read_mapping(args.mapping);volumes=read_incoming(args.incoming,'2022-09',args.end,set(mapping))
    last=shift(args.end,-1);pageviews=read_pageviews(args.pageviews,last+f'-{days(last):02d}')
    with open(args.revisions) as f:revisions=json.load(f)
    with open(args.coverage) as f:coverage=json.load(f)
    panel=build_panel(mapping,volumes,pageviews,revisions,coverage,args.start,args.end)
    with gzip.open(args.output,'xt',encoding='utf-8') as f:json.dump(panel,f,separators=(',',':'),allow_nan=False)
    print(json.dumps(dict(rows=len(panel['rows']),evaluation=len(panel['evaluation_keys']),**panel['preparation'])))


if __name__=='__main__':main()
