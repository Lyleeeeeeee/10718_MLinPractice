import copy
import math
import unittest
from wikiflow.reproduce import BINARY, RELATIVE, validate_panel, baseline_scores, verify_prepared, verify_monthly
from wikiflow.core import make_label

class ReproductionGuards(unittest.TestCase):
    def panel(self):
        r=dict(key='2024-09:1',pageid=1,article='Fixture',target_month='2024-09',cutoff_month='2024-08',
               current_count=100,U_count=200,current_precurrent_U=200,effective_event_U_asof=200,
               features=[math.log1p(100/31),math.log1p(100/31)]+[0.]*10,metadata_quality_complete=True,evaluation_eligible_asof=True,
               max_feature_date_declared='2024-08-31',editor_activity_asof=dict(max_abs_nonminor=0,cumulative_abs_nonminor=0,coverage_complete=True))
        return dict(schema='wikiflow.course.panel.v1',rows=[r],labels={r['key']:make_label(300,200)},evaluation_keys=[r['key']])

    def test_bad_identity_future_gain_and_candidate_rejected(self):
        panel=self.panel();validate_panel(panel)
        for change in ['future','gain','candidate','quality']:
            q=copy.deepcopy(panel)
            if change=='future':q['rows'][0]['target_month']='2026-09'
            if change=='gain':q['labels']['2024-09:1']['gR']=2
            if change=='candidate':q['rows'][0]['editor_activity_asof']['max_abs_nonminor']=500
            if change=='quality':
                q['rows'][0]['metadata_quality_complete']=False;q['evaluation_keys']=[]
            with self.assertRaises(ValueError):validate_panel(q)

    def test_prepared_roundoff_does_not_hide_label_or_identity_changes(self):
        original=self.panel();rebuilt=copy.deepcopy(original)
        rebuilt['rows'][0]['features'][4]=1e-13
        verify_prepared(original,rebuilt)
        rebuilt['labels']['2024-09:1']['actual']=400
        with self.assertRaises(ValueError):verify_prepared(original,rebuilt)

    def test_volume_does_not_depend_on_future_labels(self):
        row=self.panel()['rows'][0]
        before=baseline_scores([row])
        changed=dict(row,actual_count=10**12,event=1,next_count=10**15)
        self.assertEqual(before,baseline_scores([changed]))

    def test_course_baseline_and_task_methods_are_closed(self):
        row=self.panel()['rows'][0]
        self.assertEqual(baseline_scores([row]),{'volume':{row['key']:100.}})
        self.assertEqual(BINARY,['volume','binary_log13'])
        self.assertEqual(RELATIVE,['volume','relative_ridge12'])

    def test_monthly_verification_rejects_incomplete_reference(self):
        q=dict(task='binary',method='volume',month='2024-09',K=50,binary_NDCG=.2,AP=.1,TP=1)
        with self.assertRaises(ValueError):verify_monthly([q],[])

if __name__=='__main__':unittest.main()
