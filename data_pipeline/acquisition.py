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
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError
from email.utils import parsedate_to_datetime

API = 'https://en.wikipedia.org/w/api.php'
AGENT = 'WikiFlow-CMU-10718/3.0 (https://github.com/Lyleeeeeeee/10718_MLinPractice)'
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING = ROOT / 'data/title_mapping.csv'
HOSTS = {'en.wikipedia.org', 'wikimedia.org', 'dumps.wikimedia.org'}


def validate_month(month):
    date = dt.date.fromisoformat(month + '-01')
    if date.strftime('%Y-%m') != month:
        raise ValueError('Expected YYYY-MM')
    return month


def validate_interval(start, end, exclusive=False):
    first, last = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    if first > last or (exclusive and first == last):
        raise ValueError('Invalid date interval')
    return first, last


def fresh(path, root=None):
    root = Path(root or ROOT).resolve()
    path = Path(path).absolute()
    resolved = path.resolve()
    if not any(resolved.is_relative_to(root / name) and resolved != root / name
               for name in ('runs', 'local_data')):
        raise ValueError('Outputs must be inside repository runs/ or local_data/')
    if path.exists() or path.is_symlink():
        raise FileExistsError(f'Existing output is preserved: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def validate_url(url):
    parts = urlsplit(url)
    if parts.scheme != 'https' or parts.hostname not in HOSTS or parts.username or parts.password or parts.port not in (None, 443):
        raise ValueError('Only official Wikimedia HTTPS endpoints are permitted')


class OfficialRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def retry_request(operation, wait=time.sleep):
    for attempt in range(3):
        try:
            return operation()
        except (URLError, TimeoutError) as error:
            status = getattr(error, 'code', None)
            if attempt == 2 or (status is not None and status != 429 and status < 500):
                raise
            delay = 30 * 2 ** attempt
            value = (getattr(error, 'headers', None) or {}).get('Retry-After')
            if value:
                try:
                    delay = max(delay, float(value))
                except ValueError:
                    target = parsedate_to_datetime(value)
                    delay = max(delay, (target - dt.datetime.now(dt.timezone.utc)).total_seconds())
            if delay > 300:
                raise  # Stop rather than bypass the server cooldown.
            wait(delay)


class PublicClient:
    """Serial requests; each invocation owns a fresh auditable response cache."""
    def __init__(self, cache, delay=1.0):
        self.cache = fresh(cache)
        self.cache.mkdir()
        self.delay = delay
        self.sequence = 0
        self.opener = build_opener(OfficialRedirects())

    def log(self, record):
        record['utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        with (self.cache / 'requests.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')

    def fetch(self, url, target=None, as_json=True):
        validate_url(url)
        if target is not None:
            target = fresh(target)
        def operation():
            self.sequence += 1
            response_path = self.cache / f'{self.sequence:06d}.response'
            time.sleep(self.delay)
            try:
                with self.opener.open(Request(url, headers={'User-Agent': AGENT}), timeout=120) as response:
                    validate_url(response.geturl())
                    with response_path.open('xb') as stream:
                        for block in iter(lambda: response.read(1048576), b''):
                            stream.write(block)
                    self.log(dict(url=url, status=response.status, response_file=response_path.name,
                                  bytes=response_path.stat().st_size, sha256=checksum(response_path)))
                if as_json:
                    data = json.loads(response_path.read_text(encoding='utf-8'))
                    error = data.get('error', {})
                    if error.get('code') in ('maxlag', 'ratelimited', 'readonly'):
                        raise HTTPError(url, 503, str(error), {'Retry-After': '30'}, None)
                    if error or data.get('warnings'):
                        raise ValueError('Official API returned an error or warning: ' + str(data))
                    return data
                # Exclusive creation preserves a concurrently-created output too.
                with response_path.open('rb') as source, target.open('xb') as destination:
                    for block in iter(lambda: source.read(1048576), b''):
                        destination.write(block)
                return target
            except Exception as error:
                self.log(dict(url=url, error=str(error), status=getattr(error, 'code', None)))
                raise
        return retry_request(operation)

    def query(self, params):
        return self.fetch(API + '?' + urlencode(dict(action='query', format='json', formatversion=2, maxlag=5, **params)))


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def fetch_topic(get_json):
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
    validate_month(month);output=fresh(output);rows=[];seen=set()
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
    validate_interval(start, end);rows=[];seen=set()
    for item in payload.get('items',[]):
        if (item.get('project')!='en.wikipedia' or item.get('article','').replace('_',' ')!=title.replace('_',' ')
            or item.get('granularity')!='daily' or item.get('access')!='all-access' or item.get('agent')!='user'):
            raise ValueError('Pageview identity or measurement mismatch')
        date=dt.datetime.strptime(str(item['timestamp']),'%Y%m%d%H').date().isoformat();n=item['views']
        if type(n) is not int or n<0 or not start<=date<=end or date in seen:raise ValueError('Invalid daily observation')
        seen.add(date);rows.append(dict(destination=title,date=date,views=n))
    return sorted(rows,key=lambda r:r['date'])


class RevisionIdentityError(ValueError):
    """An identified page does not match the frozen catalogue; never remap it."""
    def __init__(self, pageid, title, page):
        self.identity = dict(expected_title=title, returned_title=page.get('title'),
                             expected_pageid=pageid, returned_pageid=page.get('pageid'),
                             returned_namespace=page.get('ns'), status='excluded',
                             reason='Revision identity mismatch',
                             flags=[key for key in ('missing', 'invalid', 'redirect') if key in page])
        super().__init__(f'Revision identity mismatch for {pageid}: expected {title!r}, returned {page.get("title")!r}')


def parse_revisions(payload,pageid,title):
    pages=payload.get('query',{}).get('pages',[])
    if len(pages)!=1:raise ValueError('Ambiguous revision page')
    page=pages[0]
    if page.get('pageid')!=pageid or page.get('title')!=title or page.get('ns')!=0 or any(k in page for k in ['missing','invalid','redirect']):
        raise RevisionIdentityError(pageid, title, page)
    fields={'revid','parentid','timestamp','size','sha1','sha1hidden','texthidden','commenthidden','suppressed'}
    return [dict({k:v for k,v in r.items() if k in fields},minor=bool(r.get('minor',False))) for r in page.get('revisions',[])]


def fetch_revision_windows(mapping,start,end,get_json,identity_policy='error',identities=None):
    """Acquire complete [start,end) and pre-window parent state; no actors or text."""
    validate_interval(start, end, exclusive=True)
    if identity_policy not in ('error', 'exclude'):
        raise ValueError('Unknown identity policy')
    if identity_policy == 'exclude' and identities is None:
        raise ValueError('Exclude policy requires an identity report')
    end_stamp=(dt.datetime.fromisoformat(end)-dt.timedelta(seconds=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
    before_stamp=(dt.datetime.fromisoformat(start)-dt.timedelta(seconds=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
    records={};coverage={}
    for title,pid in sorted(mapping.items()):
        try:
            params=dict(prop='info|revisions',pageids=pid,rvprop='ids|timestamp|size|flags|sha1',rvdir='older',rvstart=end_stamp,rvend=start+'T00:00:00Z',rvlimit=500)
            rs={};seen=set()
            while True:
                payload=get_json(params)
                for r in parse_revisions(payload,pid,title):
                    if not start+'T00:00:00Z'<=r['timestamp']<end+'T00:00:00Z':raise ValueError('Revision outside interval')
                    if r['revid'] in rs:raise ValueError('Duplicate revision across pages')
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
            if identities is not None:
                identities[str(pid)] = dict(expected_title=title, returned_title=title, status='verified')
        except RevisionIdentityError as error:
            if identity_policy == 'error':
                raise
            identities[str(pid)] = error.identity
            # Partial windows are not valid coverage, even if an earlier page matched.
            records.pop(str(pid), None)
            coverage.pop(str(pid), None)
    return records,coverage


def write_csv(path,rows,fields):
    path=Path(path);opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'xt',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def write_json(path, value):
    path = fresh(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'xt', encoding='utf-8') as stream:
        json.dump(value, stream, allow_nan=False)


def load_mapping(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        mapping = {}
        for row in csv.DictReader(stream):
            title, pid = row['article'], int(row['pageid'])
            if title in mapping and mapping[title] != pid:
                raise ValueError('Conflicting catalogue identity')
            mapping[title] = pid
    if not mapping:
        raise ValueError('Empty catalogue')
    return mapping


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    topic = sub.add_parser('topic', help='Fetch LIVE membership, not the frozen research catalogue')
    click = sub.add_parser('clickstream')
    pv = sub.add_parser('pageviews')
    rev = sub.add_parser('revisions')
    for item in (topic, click, pv, rev):
        item.add_argument('--output', required=True)
        item.add_argument('--cache', required=True, help='New directory under runs/ or local_data/')
    for item in (click, pv, rev):
        item.add_argument('--mapping', default=str(DEFAULT_MAPPING))
    click.add_argument('--month', required=True)
    click.add_argument('--raw', required=True, help='New .tsv.gz download path')
    pv.add_argument('--start', required=True)
    pv.add_argument('--end', required=True, help='Inclusive day')
    rev.add_argument('--start', required=True, help='Inclusive day; include both months needed by feature windows')
    rev.add_argument('--end-exclusive', required=True, help='First day outside complete query window')
    rev.add_argument('--coverage', required=True)
    rev.add_argument('--identity-output', help='Identity report, required with exclude policy')
    rev.add_argument('--identity-policy', choices=('error', 'exclude'), default='error', help='Fail on changed identity, or explicitly exclude it without coverage/remapping')
    args = parser.parse_args(argv)
    if args.command == 'revisions' and args.identity_policy == 'exclude' and not args.identity_output:
        parser.error('--identity-policy exclude requires --identity-output')
    outputs = [Path(args.output), Path(args.output + '.metadata.json'), Path(args.cache)]
    if args.command == 'clickstream':
        validate_month(args.month)
        outputs.append(Path(args.raw))
    elif args.command == 'pageviews':
        validate_interval(args.start, args.end)
    elif args.command == 'revisions':
        validate_interval(args.start, args.end_exclusive, exclusive=True)
        outputs.append(Path(args.coverage))
        if args.identity_output:
            outputs.append(Path(args.identity_output))
    resolved = [p.resolve() for p in outputs]
    if len(set(resolved)) != len(resolved) or any(x != y and x.is_relative_to(y) for x in resolved for y in resolved):
        raise ValueError('Output and cache paths must be distinct and non-nested')
    for path in outputs:
        fresh(path)
    mapping = load_mapping(args.mapping) if args.command != 'topic' else None
    client = PublicClient(args.cache)
    metadata = dict(command=args.command, arguments=vars(args), acquired_at=dt.datetime.now(dt.timezone.utc).isoformat())
    if mapping is not None:
        metadata['mapping_sha256'] = checksum(args.mapping)
    if args.command == 'topic':
        write_csv(args.output, fetch_topic(client.query), ['original_title', 'article', 'pageid'])
        metadata['catalogue_kind'] = 'live membership, not identical to frozen research cohort'
    elif args.command == 'clickstream':
        url = f'https://dumps.wikimedia.org/other/clickstream/{args.month}/clickstream-enwiki-{args.month}.tsv.gz'
        client.fetch(url, args.raw, as_json=False)
        metadata.update(extract_clickstream(args.raw, args.output, args.month, set(mapping)))
    elif args.command == 'revisions':
        identities = {}
        records, coverage = fetch_revision_windows(mapping, args.start, args.end_exclusive, client.query, args.identity_policy, identities)
        write_json(args.output, records)
        write_json(args.coverage, coverage)
        if args.identity_output:
            write_json(args.identity_output, identities)
            metadata['identity_sha256'] = checksum(args.identity_output)
        metadata['excluded_identities'] = {pid: item for pid, item in identities.items() if item['status'] == 'excluded'}
        metadata['coverage_sha256'] = checksum(args.coverage)
    else:
        rows = []
        for title in sorted(mapping):
            url = 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/user/' + quote(title.replace(' ', '_'), safe='') + '/daily/' + args.start.replace('-', '') + '00/' + args.end.replace('-', '') + '00'
            observations = parse_pageviews(client.fetch(url), title, args.start, args.end)
            expected = (dt.date.fromisoformat(args.end) - dt.date.fromisoformat(args.start)).days + 1
            if len(observations) != expected:
                raise ValueError(f'Incomplete pageview coverage for {title}: {len(observations)}/{expected}')
            rows.extend(observations)
        write_csv(args.output, rows, ['destination', 'date', 'views'])
    metadata['output_sha256'] = checksum(args.output)
    write_json(args.output + '.metadata.json', metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
