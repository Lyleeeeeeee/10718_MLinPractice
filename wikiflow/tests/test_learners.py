import unittest
import numpy as np
from wikiflow.learners import logistic_objective, fit_logistic, model_parameters

class LearnerChecks(unittest.TestCase):
    def test_newton_matches_finite_difference_and_stationary_point(self):
        x=np.array([[-2.],[-1.],[0.],[1.],[2.]])
        y=np.array([0.,0.,0.,1.,1.]);t=np.array([-.2,.7])
        value,g,h=logistic_objective(x,y,t,.01)
        numeric=[]
        for j in range(2):
            delta=np.zeros(2);delta[j]=1e-5
            numeric.append((logistic_objective(x,y,t+delta,.01)[0]-logistic_objective(x,y,t-delta,.01)[0])/(2e-5))
        np.testing.assert_allclose(g,numeric,atol=1e-9,rtol=0)
        theta,info=fit_logistic(x,y,.01)
        self.assertLess(info['gradient_inf'],1e-9)
        self.assertLess(logistic_objective(x,y,theta,.01)[0],value)
        self.assertGreater(theta[1],0)

    def test_configs_depend_only_on_train_n(self):
        self.assertEqual(model_parameters('relative_ridge12',1000)['alpha'],10.)
        for name in ['binary_hgb12','relative_hgb12']:
            p=model_parameters(name,503)
            self.assertEqual(p['min_samples_leaf'],51)
            self.assertEqual(p['random_state'],718)
            self.assertEqual(p['max_iter'],30)
            self.assertFalse(p['early_stopping'])

if __name__=='__main__':unittest.main()
