import gzip
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from data_pipeline import acquisition as a


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        root = Path(self.workspace.name)
        (root / 'runs').mkdir()
        patcher = patch.object(a, 'ROOT', root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_october_and_exclusive_window(self):
        self.assertEqual(a.validate_month('2026-10'), '2026-10')
        calls = []
        def query(params):
            calls.append(dict(params))
            stamp = '2026-08-31T23:59:59Z' if params['rvlimit'] == 1 else '2026-10-31T23:59:59Z'
            return {'query': {'pages': [{'pageid': 12, 'title': 'AI', 'ns': 0,
                    'revisions': [{'revid': len(calls), 'parentid': 0, 'timestamp': stamp, 'size': 1}]}]}}
        records, coverage = a.fetch_revision_windows({'AI': 12}, '2026-09-01', '2026-11-01', query)
        self.assertEqual(len(records['12']), 2)
        self.assertEqual(calls[0]['rvstart'], '2026-10-31T23:59:59Z')
        self.assertEqual(calls[0]['rvend'], '2026-09-01T00:00:00Z')
        self.assertEqual(coverage['12'][0]['end_exclusive'], '2026-11-01T00:00:00Z')

    def test_pageview_identity_and_duplicates(self):
        item = dict(project='en.wikipedia', article='Artificial_intelligence', granularity='daily', access='all-access', agent='user', timestamp='2026100100', views=4)
        self.assertEqual(a.parse_pageviews({'items': [item]}, 'Artificial intelligence', '2026-10-01', '2026-10-01')[0]['views'], 4)
        for items in ([item, item], [dict(item, agent='spider')], [dict(item, views=-1)], [dict(item, timestamp='2026100200')]):
            with self.assertRaises(ValueError):
                a.parse_pageviews({'items': items}, 'Artificial intelligence', '2026-10-01', '2026-10-01')

    def test_fresh_paths_and_symlinks(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            path = a.fresh(root / 'runs' / 'new.csv', root=root)
            path.write_text('preserved')
            with self.assertRaises(FileExistsError): a.fresh(path, root=root)
            for path in (root / 'data' / 'x', root / 'runs' / '..' / 'x'):
                with self.assertRaises(ValueError): a.fresh(path, root=root)
            (root / 'local_data').symlink_to(root / 'data', target_is_directory=True)
            with self.assertRaises(ValueError): a.fresh(root / 'local_data' / 'x', root=root)

    def test_clickstream_canonical_links(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT / 'runs') as directory:
            raw = Path(directory) / 'raw.tsv.gz'
            with gzip.open(raw, 'wt') as f: f.write('Some_page\tAI\tlink\t12\nother-search\tAI\texternal\t40\nX\tAlias\tlink\t20\n')
            output = Path(directory) / 'out.csv'
            result = a.extract_clickstream(raw, output, '2026-10', {'AI'})
            self.assertEqual(result['retained_rows'], 1)
            self.assertIn('Some page,AI,12', output.read_text())
            with self.assertRaises(FileExistsError): a.extract_clickstream(raw, output, '2026-10', {'AI'})

    def test_retry_after_and_no_retry_404(self):
        calls = []; waits = []
        def operation():
            calls.append(1)
            if len(calls) < 3: raise HTTPError('https://en.wikipedia.org', 429, 'slow', {'Retry-After': '40'}, None)
            return 7
        self.assertEqual(a.retry_request(operation, wait=waits.append), 7)
        self.assertEqual(waits, [40, 60])
        with self.assertRaises(HTTPError):
            a.retry_request(lambda: (_ for _ in ()).throw(HTTPError('x', 404, 'missing', {}, None)), wait=waits.append)
        self.assertEqual(len(waits), 2)

    def test_mocked_client_cache_and_checksums(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT / 'runs') as directory:
            client = a.PublicClient(Path(directory) / 'cache', delay=0)
            response = io.BytesIO(b'{"items": []}')
            response.status = 200
            response.geturl = lambda: 'https://en.wikipedia.org/w/api.php'
            with patch.object(client.opener, 'open', return_value=response):
                self.assertEqual(client.fetch('https://en.wikipedia.org/w/api.php'), {'items': []})
            self.assertTrue((client.cache / 'requests.jsonl').is_file())
            self.assertIn(a.checksum(client.cache / '000001.response'), (client.cache / 'requests.jsonl').read_text())
            with self.assertRaises(FileExistsError): a.PublicClient(client.cache)

    def test_long_cooldown_stops_and_attempts_bounded(self):
        waits = []
        calls = []
        def fail():
            calls.append(1)
            raise HTTPError('x', 503, 'retry', {}, None)
        with self.assertRaises(HTTPError): a.retry_request(fail, wait=waits.append)
        self.assertEqual(len(calls), 3)
        with self.assertRaises(HTTPError):
            a.retry_request(lambda: (_ for _ in ()).throw(HTTPError('x', 429, 'slow', {'Retry-After': '500'}, None)), wait=waits.append)
        self.assertEqual(waits, [30, 60])

    def test_revision_identity_and_interval(self):
        with self.assertRaises(ValueError):
            a.parse_revisions({'query': {'pages': [{'pageid': 99, 'ns': 0, 'title': 'AI'}]}}, 12, 'AI')
        with self.assertRaises(ValueError):
            a.fetch_revision_windows({'AI': 12}, '2026-11-01', '2026-10-01', lambda _: None)

    def test_exclude_identity_and_continue_without_coverage(self):
        identities = {}
        def query(params):
            pid = params['pageids']
            return {'query': {'pages': [{'pageid': pid, 'title': 'Moved' if pid == 60669244 else 'Z valid', 'ns': 0, 'revisions': []}]}}
        records, coverage = a.fetch_revision_windows({'Old': 60669244, 'Z valid': 12}, '2026-09-01', '2026-11-01', query, identity_policy='exclude', identities=identities)
        self.assertNotIn('60669244', coverage)
        self.assertNotIn('60669244', records)
        self.assertIn('12', coverage)
        self.assertEqual(identities['60669244']['returned_title'], 'Moved')
        self.assertEqual(identities['60669244']['expected_title'], 'Old')
        self.assertEqual(identities['60669244']['status'], 'excluded')
        with self.assertRaises(a.RevisionIdentityError):
            a.fetch_revision_windows({'Old': 60669244}, '2026-09-01', '2026-11-01', query)

    def test_baseline_identity_failure_discards_partial_page(self):
        identities = {}
        def query(params):
            return {'query': {'pages': [{'pageid': 12, 'title': 'Changed' if params['rvlimit'] == 1 else 'AI', 'ns': 0,
                'revisions': [{'revid': 1, 'timestamp': '2026-10-01T00:00:00Z', 'size': 1}]}]}}
        records, coverage = a.fetch_revision_windows({'AI': 12}, '2026-09-01', '2026-11-01', query, identity_policy='exclude', identities=identities)
        self.assertEqual(records, {})
        self.assertEqual(coverage, {})
        self.assertEqual(identities['12']['status'], 'excluded')

    def test_exclude_does_not_swallow_other_errors(self):
        for error in (ValueError('invalid data'), HTTPError('x', 503, 'unavailable', {}, None)):
            def query(params): raise error
            with self.assertRaises(type(error)):
                a.fetch_revision_windows({'AI': 12}, '2026-09-01', '2026-11-01', query, identity_policy='exclude', identities={})

    def test_official_hosts(self):
        for url in ('http://en.wikipedia.org/a', 'https://evil.example/a', 'https://en.wikipedia.org.evil.example/a'):
            with self.assertRaises(ValueError): a.validate_url(url)
        a.validate_url('https://dumps.wikimedia.org/other/clickstream/')

    def test_topic_and_continuation(self):
        def query(params):
            if 'list' in params:
                return {'query': {'categorymembers': [{'ns': 1, 'title': 'Talk:AI'}]}}
            return {'query': {'redirects': [{'from': 'AI', 'to': 'Artificial intelligence'}], 'pages': [{'ns': 0, 'pageid': 12, 'title': 'Artificial intelligence'}]}}
        self.assertEqual(a.fetch_topic(query), [dict(original_title='AI', article='Artificial intelligence', pageid=12)])

if __name__ == '__main__': unittest.main()
