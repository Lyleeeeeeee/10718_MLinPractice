"""Official public inputs. Commands create new outputs and never replace caches."""
import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

API = 'https://en.wikipedia.org/w/api.php'
AGENT = 'WikiFlow-CMU-10718/3.0 (https://github.com/Lyleeeeeeee/10718_MLinPractice)'
BLOCKED = {1164, 72417803, 79745475}


def historical(month):
    dt.date.fromisoformat(month+'-01')
    if month >= '2026-09':
        raise ValueError('2026-09 and later are sealed')


def fresh(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Existing output is preserved: {path.name}')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def request_json(params):
    url = API+'?'+urlencode(dict(action='query',format='json',formatversion=2,maxlag=5,**params))
    with urlopen(Request(url,headers={'User-Agent':AGENT}),timeout=60) as f:
        data = json.load(f)
    if 'error' in data or 'warnings' in data:
        raise ValueError('Official API returned an error or warning')
    time.sleep(.4)
    return data


def fetch_topic(get_json=request_json):
    """Freeze current maintenance membership and its redirect mapping, not AI semantics."""
    params = dict(list='categorymembers',cmtitle='Category:WikiProject Artificial Intelligence articles',cmnamespace=1,cmlimit='max')
    titles = set(); seen = set()
    while True:
        data = get_json(params)
        titles.update(x['title'][5:].replace('_',' ') for x in data['query']['categorymembers'] if x['ns']==1 and x['title'].startswith('Talk:'))
        cont = data.get('continue')
        if not cont:break
        token = json.dumps(cont,sort_keys=True)
        if token in seen:raise ValueError('Repeated category continuation')
        seen.add(token);params.update(cont)
    mapping = []
    for start in range(0,len(titles),50):
        batch = sorted(titles)[start:start+50]
        query = get_json(dict(prop='info',titles='|'.join(batch),redirects=1))['query']
        aliases = {r['from']:r['to'] for name in ['normalized','converted','redirects'] for r in query.get(name,[])}
        pages = {p['title']:p for p in query['pages']}
        for title in batch:
            canonical = title;visited=set()
            while canonical in aliases:
                if canonical in visited:raise ValueError('Redirect cycle')
                visited.add(canonical);canonical=aliases[canonical]
            page = pages.get(canonical,{})
            if page.get('ns')==0 and 'missing' not in page:
                mapping.append(dict(original_title=title,article=canonical,pageid=page['pageid']))
    if not mapping:raise ValueError('No valid catalogue articles')
    return mapping


def extract_clickstream(raw_path, output, month, catalogue):
    """Literal canonical destinations only; any internal link source, published >=10."""
    historical(month);output=fresh(output);rows=[];seen=set()
    with gzip.open(raw_path,'rt',encoding='utf-8') as f:
        for line in f:
            fields=line.rstrip('\r\n').split('\t')
            if len(fields)!=4:raise ValueError('Expected four literal TSV fields')
            source,destination,kind,count=fields
            source=source.replace('_',' ');destination=destination.replace('_',' ')
            if kind!='link' or destination not in catalogue:continue
            n=int(count);pair=(source,destination)
            if n<10 or pair in seen or not source:raise ValueError('Invalid published count or duplicate pair')
            seen.add(pair);rows.append(dict(month=month,source=source,destination=destination,count=n))
    if not rows:raise ValueError('Empty extract')
    write_csv(output,rows,['month','source','destination','count'])
    return dict(month=month,retained_rows=len(rows),raw_sha256=checksum(raw_path),extract_sha256=checksum(output),canonical_only=True)


def parse_pageviews(payload,title,start,end):
    historical(start[:7]);historical(end[:7]);rows=[];seen=set()
    for item in payload.get('items',[]):
        if (item.get('project')!='en.wikipedia' or item.get('article','').replace('_',' ')!=title.replace('_',' ')
            or item.get('granularity')!='daily' or item.get('access')!='all-access' or item.get('agent')!='user'):
            raise ValueError('Pageview identity or measurement mismatch')
        date=dt.datetime.strptime(str(item['timestamp']),'%Y%m%d%H').date().isoformat();n=item['views']
        if type(n) is not int or n<0 or not start<=date<=end or date in seen:raise ValueError('Invalid daily observation')
        seen.add(date);rows.append(dict(destination=title,date=date,views=n))
    return sorted(rows,key=lambda r:r['date'])


def parse_revisions(payload,pageid,title):
    pages=payload.get('query',{}).get('pages',[])
    if len(pages)!=1:raise ValueError('Ambiguous revision page')
    page=pages[0]
    if page.get('pageid')!=pageid or page.get('title')!=title or page.get('ns')!=0 or any(k in page for k in ['missing','invalid','redirect']):
        raise ValueError('Revision identity mismatch')
    fields={'revid','parentid','timestamp','size','sha1','sha1hidden','texthidden','commenthidden','suppressed'}
    return [dict({k:v for k,v in r.items() if k in fields},minor=bool(r.get('minor',False))) for r in page.get('revisions',[])]


def fetch_revision_windows(mapping,start,end,get_json=request_json):
    """Acquire complete [start,end) and pre-window parent state; no actors or text."""
    historical(start[:7])
    if not start<end or end>'2026-09-01':raise ValueError('Invalid revision interval or sealed holdout')
    end_stamp=(dt.datetime.fromisoformat(end)-dt.timedelta(seconds=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
    before_stamp=(dt.datetime.fromisoformat(start)-dt.timedelta(seconds=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
    records={};coverage={}
    for title,pid in sorted(mapping.items()):
        if pid in BLOCKED:continue
        params=dict(prop='info|revisions',pageids=pid,rvprop='ids|timestamp|size|flags|sha1',rvdir='older',rvstart=end_stamp,rvend=start+'T00:00:00Z',rvlimit=500)
        rs={};seen=set()
        while True:
            payload=get_json(params)
            for r in parse_revisions(payload,pid,title):
                if not start+'T00:00:00Z'<=r['timestamp']<end+'T00:00:00Z':raise ValueError('Revision outside interval')
                rs[r['revid']]=r
            cont=payload.get('continue')
            if not cont:break
            token=json.dumps(cont,sort_keys=True)
            if set(cont)-{'continue','rvcontinue'} or token in seen:raise ValueError('Invalid revision continuation')
            seen.add(token);params.update(cont)
        baseline=get_json(dict(prop='info|revisions',pageids=pid,rvprop='ids|timestamp|size|flags|sha1',rvdir='older',rvstart=before_stamp,rvlimit=1))
        for r in parse_revisions(baseline,pid,title):
            if r['timestamp']>=start+'T00:00:00Z':raise ValueError('Parent state after window start')
            rs[r['revid']]=r
        records[str(pid)]=sorted(rs.values(),key=lambda r:(r['timestamp'],r['revid']))
        coverage[str(pid)]=[dict(start=start+'T00:00:00Z',end_exclusive=end+'T00:00:00Z')]
    return records,coverage


def write_csv(path,rows,fields):
    path=Path(path);opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'xt',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('topic');a.add_argument('--output',required=True)
    a=sub.add_parser('clickstream');a.add_argument('--month',required=True);a.add_argument('--mapping',required=True);a.add_argument('--raw',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('pageviews');a.add_argument('--mapping',required=True);a.add_argument('--start',default='2022-09-01');a.add_argument('--end',default='2026-07-31');a.add_argument('--output',required=True)
    a=sub.add_parser('revisions');a.add_argument('--mapping',required=True);a.add_argument('--start',default='2023-08-01');a.add_argument('--end-exclusive',default='2026-08-01');a.add_argument('--output',required=True);a.add_argument('--coverage',required=True)
    args=p.parse_args();fresh(args.output)
    if args.command=='topic':
        write_csv(args.output,fetch_topic(),['original_title','article','pageid']);return
    with open(args.mapping) as f:mapping={r['article']:int(r['pageid']) for r in csv.DictReader(f)}
    if args.command=='clickstream':
        historical(args.month);raw=fresh(args.raw)
        url=f'https://dumps.wikimedia.org/other/clickstream/{args.month}/clickstream-enwiki-{args.month}.tsv.gz'
        with urlopen(Request(url,headers={'User-Agent':AGENT}),timeout=120) as response,raw.open('xb') as f:
            for block in iter(lambda:response.read(1048576),b''):f.write(block)
        print(json.dumps(extract_clickstream(raw,args.output,args.month,set(mapping))));return
    if args.command=='revisions':
        fresh(args.coverage);records,coverage=fetch_revision_windows(mapping,args.start,args.end_exclusive)
        Path(args.output).write_text(json.dumps(records));Path(args.coverage).write_text(json.dumps(coverage));return
    historical(args.start[:7]);historical(args.end[:7]);rows=[]
    for title in sorted(mapping):
        if mapping[title] in BLOCKED:continue
        url='https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/user/'+quote(title.replace(' ','_'),safe='')+'/daily/'+args.start.replace('-','')+'00/'+args.end.replace('-','')+'00'
        with urlopen(Request(url,headers={'User-Agent':AGENT}),timeout=60) as f:payload=json.load(f)
        rows.extend(parse_pageviews(payload,title,args.start,args.end));time.sleep(.4)
    write_csv(args.output,rows,['destination','date','views'])


if __name__=='__main__':main()
