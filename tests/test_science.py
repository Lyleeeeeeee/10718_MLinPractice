"""Synthetic final-protocol contracts, independent of archived data and code."""
import copy
import math
import unittest
import numpy as np
from features.assembly import days, shift, upper_bound, explicit_margin, numeric_features, momentum_score, make_label
from features.preprocessing import preprocess
from models.logistic import sigmoid, logistic_objective, fit_logistic
from models.pairwise import training_objective, fit_weights, blend_rank_keys
from textual.representations import represent, age_features
from data_analysis.metrics import score_month, summarize, choose, ORDER, VALIDATION, TEST, LEARNED
from data_analysis.pipeline import training_rows, features


def row(key='a', month='2026-05', eligible=True):
    return dict(key=key, article=key, pageid=10+sum(map(ord,key)), target_month=month,
                cutoff_month=shift(month,-1), max_feature_date_declared=shift(month,-1)+'-28',
                evaluation_eligible_asof=eligible,current_count=300,effective_event_U_asof=400.,
                features=[math.log1p(10.), math.log1p(5.)]+list(map(float,range(2,12))))


def age(stamp='2020-01-01T00:00:00Z'):
    return dict(status='earliest_visible_root',earliest_visible_at=stamp,earliest_visible_revision=1,observed_roots=[])


class FeatureContracts(unittest.TestCase):
    def test_calendar_and_robust_bound(self):
        self.assertEqual(days('2024-02'),29)
        self.assertEqual(days('2025-02'),28)
        self.assertEqual(shift('2026-01',-1),'2025-12')
        self.assertEqual(shift('2025-12',2),'2026-02')
        self.assertEqual(upper_bound([0.]*6,30),200.)
        self.assertEqual(upper_bound([20.]*6,30),750.)
        self.assertEqual(upper_bound([20.]*5+[100000.],30),750.)
        self.assertAlmostEqual(upper_bound([1.,2.,3.,4.,5.,6.],30),105+max(100.,.25*105,2*30*1.4826*1.5))

    def test_label_strict_threshold_and_unknown(self):
        self.assertEqual(make_label(400,400.)['event'],0)
        self.assertEqual(make_label(401,400.)['event'],1)
        self.assertEqual(make_label(400,400.5)['event'],0)
        self.assertEqual(make_label(401,400.5)['event'],1)
        self.assertFalse(make_label(None,400.)['known'])
        self.assertIsNone(make_label(None,400.)['event'])
        for actual in (-1, True, 1.5):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                make_label(actual,400.)

    def test_explicit_margin_and_momentum(self):
        r=row(); pace=300*31/30
        self.assertAlmostEqual(explicit_margin(r),math.log1p(pace)-math.log1p(400))
        self.assertAlmostEqual(momentum_score(r),math.log1p(20*31)-math.log1p(400))
        r['effective_event_U_asof']=pace
        self.assertAlmostEqual(explicit_margin(r),0.)

    def test_features_no_extra_momentum(self):
        r=row(); before=copy.deepcopy(r)
        for family in ('logistic','lambdarank','lightgbm','pairwise_rule'):
            expected=list(r['features'])+[explicit_margin(r)] if family=='logistic' else [explicit_margin(r),r['features'][0]-r['features'][1],8.,r['features'][0],7.]
            self.assertEqual(numeric_features(r,family),expected)
            for kind in ('numeric','svd','minilm'):
                result=features([r],family,kind,{'a':{'missing':False}},{str(r['pageid']):age()},{'a':np.arange(8.)})[0]['features']
                self.assertEqual(len(result),len(expected)+(0 if kind=='numeric' else 11))
                self.assertEqual(result[:len(expected)],expected)
                if kind!='numeric': self.assertEqual(result[-9:],[0.]+list(np.arange(8.)))
        self.assertEqual(r,before)

    def test_train_only_scaling_missing_and_zero_variance(self):
        train=[{'features':[1.,None,4.,None]},{'features':[3.,2.,4.,None]},{'features':[None,6.,4.,None]}]
        x,z,stats=preprocess(train,[{'features':[1001.,None,9.,None]}])
        self.assertEqual(stats['medians'],[2.,4.,4.,0.])
        self.assertEqual(stats['means'],[2.,4.,4.,0.])
        self.assertEqual(stats['scales'][2:],[1.,1.])
        np.testing.assert_allclose(x.mean(axis=0),0.,atol=1e-12)
        np.testing.assert_array_equal(x[:,2:],np.zeros((3,2)))
        np.testing.assert_allclose(z[0],[999/math.sqrt(2/3),0.,5.,0.])
        self.assertEqual(stats,preprocess(train,[{'features':[-1e9,1e9,None,1e9]}])[2])
        self.assertEqual(preprocess(train,[])[1].shape,(0,4))
        with self.assertRaises(ValueError): preprocess([],[])
        with self.assertRaises(ValueError): preprocess([{'features':[1.]}],[{'features':[float('inf')]}])

    def test_training_past_known_and_candidate_sort(self):
        rows=[row('z','2026-03'),row('b','2026-04',False),row('a','2026-03'),row('unknown','2026-04'),row('same'),row('future','2026-06')]
        panel=dict(rows=rows,labels={r['key']:{'known':r['key']!='unknown'} for r in rows})
        self.assertEqual([r['key'] for r in training_rows(panel,'2026-05','expanded')],['z','b','a'])
        self.assertEqual([r['key'] for r in training_rows(panel,'2026-05','candidate')],['a','z'])


class RepresentationContracts(unittest.TestCase):
    def synthetic(self):
        train=[{'key':f'r{i}'} for i in range(12)]
        records={f'r{i}':dict(snapshot_id=f's{i:02}',missing=False,text=' '.join(f'token{j}' for j in range(14) if (i+j)%4!=0),timestamp='2020-01-01T00:00:00Z') for i in range(12)}
        records['target']=dict(snapshot_id='target',missing=False,text='heldoutword unseenword',timestamp='2026-04-01T00:00:00Z')
        records['missing']=dict(snapshot_id='missing',missing=True,text='',timestamp='2020-01-01T00:00:00Z')
        rng=np.random.default_rng(718)
        return train,records,{r['snapshot_id']:rng.normal(size=384) for r in records.values()}

    def test_train_only_projection_eight_dimensions(self):
        for kind in ('minilm','svd'):
            with self.subTest(kind=kind):
                train,records,embeddings=self.synthetic()
                vectors,state=represent(train,records,embeddings,kind)
                self.assertEqual(state['fit_snapshot_ids'],[f's{i:02}' for i in range(12)])
                self.assertTrue(all(v.shape==(8,) and np.isfinite(v).all() for v in vectors.values()))
                np.testing.assert_array_equal(vectors['missing'],np.zeros(8))
                records['target']['text']='changedtarget completelynovel'; embeddings['target']=np.full(384,1e6)
                changed,second=represent(train,records,embeddings,kind)
                for r in train: np.testing.assert_allclose(vectors[r['key']],changed[r['key']],atol=1e-12)
                np.testing.assert_allclose(state['model'].components_,second['model'].components_,atol=1e-12)
                if kind=='svd': self.assertNotIn('heldoutword',state['vectorizer'].vocabulary_)
                else: np.testing.assert_allclose(state['model'].mean_,np.mean([embeddings[f's{i:02}'] for i in range(12)],axis=0))

    def test_snapshot_identity_conflict_guard(self):
        for field,value in (('text','conflicting text'),('missing',True)):
            for kind in ('minilm','svd'):
                with self.subTest(field=field,kind=kind):
                    train,records,embeddings=self.synthetic()
                    records['duplicate']=dict(records['r0'])
                    vectors,state=represent(train+[{'key':'duplicate'}],records,embeddings,kind)
                    self.assertEqual(len(state['fit_snapshot_ids']),12)
                    np.testing.assert_array_equal(vectors['r0'],vectors['duplicate'])
                    records['duplicate'][field]=value
                    with self.assertRaises(ValueError): represent(train,records,embeddings,kind)

    def test_insufficient_train(self):
        train,records,embeddings=self.synthetic()
        for kind in ('minilm','svd'):
            with self.assertRaises(ValueError): represent(train[:9],records,embeddings,kind)

    def test_age_strict_boundary(self):
        cutoff='2026-05-01T00:00:00Z'
        self.assertEqual(age_features(age(cutoff),cutoff),[None,1.])
        self.assertEqual(age_features(age('2026-05-02T00:00:00Z'),cutoff),[None,1.])
        self.assertEqual(age_features(age('2026-04-30T00:00:00Z'),cutoff),[math.log1p(1),0.])
        evidence=age('2026-04-29T00:00:00Z'); evidence['observed_roots']=[dict(revid=2,timestamp=cutoff)]
        self.assertEqual(age_features(evidence,cutoff),[math.log1p(2),0.])
        evidence['observed_roots'][0]['timestamp']='2026-04-30T00:00:00Z'
        self.assertEqual(age_features(evidence,cutoff),[None,1.])


class SolverContracts(unittest.TestCase):
    def test_sigmoid_and_logistic_derivatives(self):
        np.testing.assert_allclose(sigmoid(np.array([-1000.,0.,1000.])),[0.,.5,1.])
        x=np.array([[-2.,1.],[-1.,0.],[1.,0.],[2.,1.]])
        y=np.array([0.,0.,1.,1.]); theta=np.array([.2,-.3,.4])
        value,grad,hessian=logistic_objective(x,y,theta,.01)
        for j in range(3):
            step=np.eye(3)[j]*1e-5
            plus=logistic_objective(x,y,theta+step,.01); minus=logistic_objective(x,y,theta-step,.01)
            self.assertAlmostEqual(grad[j],(plus[0]-minus[0])/2e-5,places=8)
            np.testing.assert_allclose(hessian[:,j],(plus[1]-minus[1])/2e-5,atol=1e-8)
        fitted,info=fit_logistic(x,y,.01)
        self.assertLess(logistic_objective(x,y,fitted,.01)[0],value)
        self.assertLessEqual(info['gradient_inf'],1e-9)

    def test_pairwise_month_balancing_and_constraint(self):
        train=[row(str(i),'2026-03' if i<3 else '2026-04') for i in range(5)]
        labels={str(i):{'event':int(i in (0,3))} for i in range(5)}
        x=np.array([[0.,1.],[1.,0.],[2.,0.],[10.,1.],[11.,0.]])
        pairs,y,weights,intercept=training_objective(train,labels,x,'pairwise')
        np.testing.assert_array_equal(pairs,[[-1.,1.],[-2.,1.],[-1.,1.]])
        np.testing.assert_array_equal(y,np.ones(3))
        np.testing.assert_allclose(weights,[.25,.25,.5]); self.assertFalse(intercept)
        theta,info=fit_weights(pairs,y,weights,1.,intercept)
        self.assertGreaterEqual(theta[0],0.); self.assertGreater(theta[1],0.); self.assertTrue(info['converged'])
        with self.assertRaises(ValueError): training_objective(train,{str(i):{'event':0} for i in range(5)},x,'pairwise')

    def test_blend_exact_ties(self):
        np.testing.assert_array_equal(blend_rank_keys([1.,2.,3.],[3.,2.,1.],.5),[4,4,4])
        with self.assertRaises(ValueError): blend_rank_keys([1.],[float('nan')],.5)


class MetricContracts(unittest.TestCase):
    def table(self,month='2026-05',reverse=False):
        rows=[dict(key=str(i),pageid=i,current_count=100.) for i in range(61)]
        labels={str(i):dict(known=i!=60,event=int(i in (0,49,59))) for i in range(61)}
        scores={str(i):(1e6 if i==60 else 0.) for i in range(61)}
        return score_month(month,list(reversed(rows)) if reverse else rows,labels,{name:dict(scores) for name in ORDER if name!='random_expected'})

    def test_unknown_filtered_before_top50_and_deterministic_ties(self):
        table=self.table(); self.assertEqual(table,self.table(reverse=True)); first=table[0]
        self.assertEqual((first['known'],first['unknown'],first['selected'],first['hits_at_50']),(60,1,50,2))
        self.assertAlmostEqual(first['recall_at_50'],2/3)
        self.assertAlmostEqual(first['precision_at_50'],2/50)
        self.assertAlmostEqual(first['dcg_at_50'],1+1/math.log2(51))

    def test_random_expectation(self):
        random=next(r for r in self.table() if r['model_id']=='random_expected')
        self.assertEqual(random['hits_at_50'],2.5)
        self.assertAlmostEqual(random['recall_at_50'],50/60)
        self.assertAlmostEqual(random['precision_at_50'],3/60)
        self.assertAlmostEqual(random['dcg_at_50'],3/60*sum(1/math.log2(i+2) for i in range(50)))

    def test_summary_pools_counts(self):
        summary=summarize(self.table('2026-05')+self.table('2026-06'))[0]
        self.assertEqual((summary['months'],summary['events'],summary['selected'],summary['hits_at_50']),(2,6,100,4))
        self.assertAlmostEqual(summary['pooled_recall_at_50'],2/3)

    def test_selection_only_validation_and_lexical_tie(self):
        self.assertEqual(VALIDATION[0],'2024-09'); self.assertEqual(VALIDATION[-1],'2026-04')
        self.assertEqual(TEST,[f'2026-{m:02}' for m in range(5,11)])
        self.assertFalse(set(VALIDATION)&set(TEST)); self.assertEqual(len(LEARNED),12)
        self.assertEqual(set(ORDER)-set(LEARNED),{'current_volume','momentum','random_expected'})
        validation=self.table('2026-04')
        self.assertEqual(choose(validation)['model_id'],sorted(LEARNED)[0])
        with self.assertRaises(ValueError): choose(validation+self.table('2026-05'))
        with self.assertRaises(ValueError): choose(self.table('2026-05'))


if __name__=='__main__':
    unittest.main()
