import copy
import math
import unittest

from wikiflow.core import (days, shift, historical_bound, joint_threshold, make_label,
                          select_training, candidate_allowed, preprocess, rank_metric, select_lambda, aggregate)


def row(month='2024-09', pid=1):
    return dict(key=f'{month}:{pid}', pageid=pid, target_month=month,
                cutoff_month=shift(month,-1), current_count=100, U_count=200,
                current_precurrent_U=200, effective_event_U_asof=200,
                features=[1.,None], max_feature_date_declared=shift(month,-1)+'-28',
                metadata_quality_complete=True, v2_evaluation_eligible_asof=True,
                editor_activity_asof=dict(max_abs_nonminor=499,cumulative_abs_nonminor=999,coverage_complete=True))


class Contracts(unittest.TestCase):
    def test_calendar_and_strict_label(self):
        self.assertEqual(days('2024-02'),29)
        self.assertEqual(shift('2024-01',-1),'2023-12')
        self.assertEqual(historical_bound([1]*6,30),200)
        r=row();r['current_count']=800;r['U_count']=200
        t=joint_threshold(r)
        self.assertAlmostEqual(t,1.25*800*30/31)
        self.assertEqual(make_label(int(t),t)['event'],0)
        self.assertEqual(make_label(None,t)['known'],False)
        self.assertEqual(make_label(600,200)['gR'],math.log(3))
        with self.assertRaises(ValueError):make_label(-1,200)

    def test_candidate_boundaries_and_identity(self):
        r=row();self.assertTrue(candidate_allowed(r))
        for change in [dict(pageid=79745475),dict(current_count=99),
                       dict(editor_activity_asof=dict(max_abs_nonminor=500,cumulative_abs_nonminor=999,coverage_complete=True)),
                       dict(editor_activity_asof=dict(max_abs_nonminor=0,cumulative_abs_nonminor=1000,coverage_complete=True)),
                       dict(editor_activity_asof=dict(max_abs_nonminor=0,cumulative_abs_nonminor=0,coverage_complete=False))]:
            self.assertFalse(candidate_allowed(dict(r,**change)))

    def test_future_unknown_and_edit_filter_never_enter_train(self):
        a=row('2024-08');a['v2_evaluation_eligible_asof']=False
        b=row();unknown=row('2024-07',2)
        labels={a['key']:dict(known=True),b['key']:dict(known=True),unknown['key']:dict(known=False)}
        self.assertEqual(select_training([a,b,unknown],labels,'2024-09'),[a])
        altered=copy.deepcopy(labels);altered[b['key']]=dict(known=False)
        self.assertEqual(select_training([a,b,unknown],altered,'2024-09'),[a])
        a['max_feature_date_declared']='2024-08-01'
        with self.assertRaises(ValueError):select_training([a],labels,'2024-09')
        with self.assertRaises(ValueError):select_training([b],labels,'2026-09')

    def test_scaler_uses_train_only_population_std(self):
        train=[dict(features=[0.,None]),dict(features=[2.,4.])]
        x,z,p=preprocess(train,[dict(features=[1000.,None])])
        self.assertEqual(p,dict(medians=[1.,4.],means=[1.,4.],scales=[1.,1.]))
        self.assertEqual(x.tolist(),[[-1.,0.],[1.,0.]])
        self.assertEqual(z.tolist(),[[999.,0.]])

    def test_gain_once_and_macro_month_support(self):
        rr=[row(pid=1),row(pid=2),row(pid=3)]
        labels={r['key']:make_label(a,200) for r,a in zip(rr,[600,200,1000])}
        scores={r['key']:float(3-i) for i,r in enumerate(rr)}
        q=rank_metric(rr,labels,scores,2)
        ideal=math.log(5)+math.log(3)/math.log2(3)
        self.assertAlmostEqual(q['relative_NDCG'],math.log(3)/ideal)
        self.assertEqual(q['TP'],1)
        self.assertAlmostEqual(q['AP'],(1+2/3)/2)
        labels[rr[2]['key']]=make_label(None,200)
        self.assertEqual(rank_metric(rr,labels,scores,2)['status'],'unknown')
        with self.assertRaises(ValueError):rank_metric(rr,labels,{rr[0]['key']:1},2)

    def test_macro_micro_and_unknown_month_support(self):
        rows=[dict(status='defined',events=1,TP=1,actual_K=5,binary_NDCG=1.,relative_NDCG=1.,AP=1.,Precision=.2,Recall=1.,selected_gR=1.,total_gR=1.),
              dict(status='defined',events=9,TP=0,actual_K=5,binary_NDCG=0.,relative_NDCG=0.,AP=.1,Precision=0.,Recall=0.,selected_gR=0.,total_gR=9.)]
        q=aggregate(rows)
        self.assertEqual(q['macro_Recall'],.5)
        self.assertEqual(q['micro_Recall'],.1)
        self.assertEqual(q['micro_Precision'],.1)

    def test_tuning_early_fixed_pairs_and_tie_policy(self):
        self.assertEqual(select_lambda({.01:[.2]*3,.1:[.2]*3}),.1)
        with self.assertRaises(ValueError):select_lambda({.01:[.2]*2,.1:[.2]*2})

if __name__=='__main__':unittest.main()
