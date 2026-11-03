"""Frozen panel I/O and validation, shared by replay and opt-in fitting."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from .core import (BLOCKED_PAGEIDS, HISTORICAL_EXCLUSIONS, FORECASTS, candidate_allowed, days,
                   joint_threshold, make_label, shift)

ROOT = Path(__file__).resolve().parents[2]

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

def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
