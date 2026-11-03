import gzip
import importlib
import json
from pathlib import Path
import tempfile
import unittest


class Acquisition(unittest.TestCase):
    def api(self):
        self.assertTrue((Path(__file__).parents[1]/'src/wikiflow/acquisition.py').exists(), 'acquisition API missing')
        return importlib.import_module('wikiflow.acquisition')

    def test_literal_link_extract_retains_outside_sources_and_boundary_ten(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as d:
            raw=Path(d)/'raw.tsv.gz';out=Path(d)/'incoming.csv.gz'
            with gzip.open(raw,'wt') as f:
                f.write('Outside_source\tExample_article\tlink\t10\nAlias\tExample_article\texternal\t40\nOutside\tOld_title\tlink\t30\n')
            audit=api.extract_clickstream(raw,out,'2024-08',{'Example article'})
            with gzip.open(out,'rt') as f:body=f.read()
            self.assertIn('Outside source,Example article,10',body)
            self.assertNotIn('Old title',body)
            self.assertEqual(audit['retained_rows'],1)
            self.assertTrue(raw.exists())
            with self.assertRaises(FileExistsError):api.extract_clickstream(raw,out,'2024-08',{'Example article'})

    def test_duplicate_pair_bad_count_and_holdout_rejected(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as d:
            raw=Path(d)/'raw.gz'
            for text in ['s\ta\tlink\t9\n','s\ta\tlink\t10\ns\ta\tlink\t10\n']:
                with gzip.open(raw,'wt') as f:f.write(text)
                with self.assertRaises(ValueError):api.extract_clickstream(raw,Path(d)/'out.gz','2024-08',{'a'})
            with self.assertRaises(ValueError):api.extract_clickstream(raw,Path(d)/'out.gz','2026-09',{'a'})

    def test_daily_explicit_zero_kept_absent_day_not_invented(self):
        api=self.api()
        item=dict(project='en.wikipedia',article='Example_article',granularity='daily',access='all-access',agent='user',timestamp='2024080100',views=0)
        rows=api.parse_pageviews(dict(items=[item]),'Example article','2024-08-01','2024-08-31')
        self.assertEqual(rows,[dict(destination='Example article',date='2024-08-01',views=0)])
        self.assertEqual(api.parse_pageviews(dict(items=[]),'Example article','2024-08-01','2024-08-31'),[])
        with self.assertRaises(ValueError):api.parse_pageviews(dict(items=[item,item]),'Example article','2024-08-01','2024-08-31')

    def test_revision_parser_checks_identity_and_omits_actor_and_comment(self):
        api=self.api()
        payload=dict(query=dict(pages=[dict(pageid=1,ns=0,title='Example',revisions=[dict(revid=2,parentid=1,timestamp='2024-08-01T00:00:00Z',size=100,sha1='a',user='private',comment='private')])]))
        rows=api.parse_revisions(payload,1,'Example')
        self.assertEqual(rows[0]['minor'],False)
        self.assertNotIn('user',json.dumps(rows));self.assertNotIn('comment',json.dumps(rows))
        with self.assertRaises(ValueError):api.parse_revisions(payload,2,'Example')

    def test_topic_redirect_mapping_and_revision_baseline_offline(self):
        api=self.api();calls=[]
        def topic_get(params):
            calls.append(params)
            if params.get('list')=='categorymembers':return {'query':{'categorymembers':[{'ns':1,'title':'Talk:Old title'}]}}
            return {'query':{'redirects':[{'from':'Old title','to':'Example'}],'pages':[{'ns':0,'title':'Example','pageid':1}]}}
        self.assertEqual(api.fetch_topic(topic_get),[dict(original_title='Old title',article='Example',pageid=1)])
        def revision_get(params):
            calls.append(params)
            self.assertNotIn('user',params['rvprop']);self.assertNotIn('comment',params['rvprop'])
            r=dict(revid=1,parentid=0,timestamp='2024-06-30T23:59:59Z',size=100,sha1='a') if params['rvlimit']==1 else dict(revid=2,parentid=1,timestamp='2024-07-01T00:00:00Z',size=100,sha1='b')
            return dict(query=dict(pages=[dict(pageid=1,ns=0,title='Example',revisions=[r])]))
        records,coverage=api.fetch_revision_windows({'Example':1},'2024-07-01','2024-09-01',revision_get)
        self.assertEqual(len(records['1']),2);self.assertEqual(coverage['1'][0]['end_exclusive'],'2024-09-01T00:00:00Z')
        with self.assertRaises(ValueError):api.fetch_revision_windows({'Example':1},'2026-09-01','2026-10-01',revision_get)

if __name__=='__main__':unittest.main()
