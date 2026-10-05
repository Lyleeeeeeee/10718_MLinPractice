import datetime
import importlib
import math
from pathlib import Path
import unittest


def cache():
    from wikiflow.core import days,shift
    vols={shift('2024-08',i):{'Example':10*days(shift('2024-08',i))} for i in range(-6,2)}
    vols['2024-09']['Example']=500
    pv={'Example':{f'2024-08-{i:02d}':5 for i in range(1,32)}}
    records={'1':[dict(revid=1,parentid=0,timestamp='2024-06-30T23:59:59Z',size=100,sha1='a',minor=False)]}
    coverage={'1':[dict(start='2024-07-01T00:00:00Z',end_exclusive='2024-09-01T00:00:00Z')]}
    return {'Example':1},vols,pv,records,coverage


class Preparation(unittest.TestCase):
    def api(self):
        self.assertTrue((Path(__file__).parents[1]/'src/wikiflow/preparation.py').exists(), 'preparation API missing')
        return importlib.import_module('wikiflow.preparation')

    def test_exact_twelve_features_and_relative_label(self):
        api=self.api();mapping,vol,pv,records,cov=cache()
        panel=api.build_panel(mapping,vol,pv,records,cov,'2024-09','2024-09')
        self.assertEqual(panel['schema'],'wikiflow.course.panel.v1')
        r=panel['rows'][0];label=panel['labels'][r['key']]
        self.assertEqual(r['features'],[math.log1p(10)]*4+[0,math.log1p(10),1,math.log1p(5),0,0,0,math.log1p(300)-math.log1p(450)])
        self.assertEqual(r['U_count'],450);self.assertEqual(r['current_precurrent_U'],465)
        self.assertEqual(r['effective_event_U_asof'],450)
        self.assertNotIn('rule_forecasts',r)
        self.assertEqual(panel['evaluation_keys'],[r['key']])
        self.assertEqual(label['event'],1);self.assertAlmostEqual(label['R'],500/450-1)

    def test_unknown_future_kept_candidate_and_missing_pv_not_zero(self):
        api=self.api();mapping,vol,pv,records,cov=cache();del vol['2024-09']['Example'];del pv['Example']['2024-08-10']
        p=api.build_panel(mapping,vol,pv,records,cov,'2024-09','2024-09');r=p['rows'][0]
        self.assertFalse(p['labels'][r['key']]['known']);self.assertEqual(p['evaluation_keys'],[r['key']])
        self.assertEqual(r['features'][7:10],[None,None,1])

    def test_metadata_boundaries_minor_reverts_and_training_without_edit_filter(self):
        api=self.api();mapping,vol,pv,records,cov=cache()
        records['1'] += [dict(revid=2,parentid=1,timestamp='2024-07-01T00:00:00Z',size=600,sha1='b',minor=False),dict(revid=3,parentid=2,timestamp='2024-08-31T23:59:59Z',size=100,sha1='c',minor=False),dict(revid=4,parentid=3,timestamp='2024-09-01T00:00:00Z',size=999999,sha1='d',minor=False)]
        p=api.build_panel(mapping,vol,pv,records,cov,'2024-09','2024-09')
        self.assertEqual(len(p['rows']),1);self.assertEqual(p['evaluation_keys'],[])
        self.assertEqual(p['rows'][0]['editor_activity_asof']['cumulative_abs_nonminor'],1000)
        records['1'][1]['minor']=True;records['1'][2]['size']=1099
        p=api.build_panel(mapping,vol,pv,records,cov,'2024-09','2024-09')
        self.assertEqual(p['rows'][0]['editor_activity_asof']['max_abs_nonminor'],499)
        self.assertEqual(p['evaluation_keys'],['2024-09:1'])

    def test_unknown_quality_parent_gaps_identity_and_holdout_fail_closed(self):
        api=self.api();mapping,vol,pv,records,cov=cache()
        self.assertEqual(api.build_panel(mapping,vol,pv,records,{},'2024-09','2024-09')['rows'],[])
        for blocked in [1164,72417803,79745475]:
            self.assertEqual(api.build_panel({'Example':blocked},vol,pv,records,cov,'2024-09','2024-09')['rows'],[])
        with self.assertRaises(ValueError):api.build_panel(mapping,vol,pv,records,cov,'2026-09','2026-09')
        del vol['2024-02']['Example']
        self.assertEqual(api.build_panel(mapping,vol,pv,records,cov,'2024-09','2024-09')['rows'],[])

    def test_history_keeps_article_order_but_evaluation_uses_pageid_order(self):
        api=self.api();mapping,vol,pv,records,cov=cache();mapping['A second article']=99
        for v in vol.values():v['A second article']=v['Example']
        pv['A second article']=dict(pv['Example']);records['99']=list(records['1']);cov['99']=list(cov['1'])
        p=api.build_panel(mapping,vol,pv,records,cov,'2024-09','2024-09')
        self.assertEqual([r['pageid'] for r in p['rows']],[99,1])
        self.assertEqual(p['evaluation_keys'],['2024-09:1','2024-09:99'])

if __name__=='__main__':unittest.main()
