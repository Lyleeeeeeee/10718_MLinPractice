"""Small synthetic numerical checks; these do not train historical experiments."""
import unittest
import numpy as np
from wikiflow.learners import logistic_objective, fit_logistic
from wikiflow.models import objective, blend_rank_keys, training_rows
from wikiflow.features import text_vectors
from test_core import row


class ModelContracts(unittest.TestCase):
    def test_logistic_gradient_and_newton_stationary_point(self):
        x = np.array([[-2.], [-1.], [0.], [1.], [2.]])
        y = np.array([0., 0., 0., 1., 1.])
        theta = np.array([-.2, .7])
        value, gradient, _ = logistic_objective(x, y, theta, .01)
        numerical = []
        for j in range(2):
            delta = np.zeros(2)
            delta[j] = 1e-5
            numerical.append((logistic_objective(x, y, theta + delta, .01)[0]
                              - logistic_objective(x, y, theta - delta, .01)[0]) / 2e-5)
        np.testing.assert_allclose(gradient, numerical, atol=1e-9, rtol=0)
        result, info = fit_logistic(x, y, .01)
        self.assertLess(info["gradient_inf"], 1e-9)
        self.assertLess(logistic_objective(x, y, result, .01)[0], value)

    def test_pairwise_gradient_matches_finite_difference(self):
        x = np.array([[1., 2.], [-1., .5], [.5, -2.]])
        theta = np.array([.2, -.3])
        y = np.ones(3)
        weights = np.array([.25, .25, .5])
        _, gradient = objective(theta, x, y, weights, 1., False)
        numerical = []
        for j in range(2):
            delta = np.zeros(2)
            delta[j] = 1e-5
            numerical.append((objective(theta + delta, x, y, weights, 1., False)[0]
                              - objective(theta - delta, x, y, weights, 1., False)[0]) / 2e-5)
        np.testing.assert_allclose(gradient, numerical, atol=1e-9, rtol=0)

    def test_rank_blend_preserves_exact_mathematical_ties(self):
        np.testing.assert_array_equal(blend_rank_keys([3, 2, 1], [1, 2, 3], .5), [4, 4, 4])

    def test_training_pool_and_time_boundary_remain_distinct(self):
        earlier = row("2024-08")
        edited = row("2024-08", 2)
        edited["editor_activity_asof"]["max_abs_nonminor"] = 500
        future = row("2024-09", 3)
        unknown = row("2024-07", 4)
        rows = [earlier, edited, future, unknown]
        labels = {r["key"]: {"known": r is not unknown} for r in rows}
        panel = {"rows": rows, "labels": labels}
        self.assertEqual(training_rows(panel, "2024-09", "expanded"), [earlier, edited])
        self.assertEqual(training_rows(panel, "2024-09", "candidate"), [earlier])

    def test_text_projection_excludes_future_vocabulary_and_retains_missing(self):
        documents = {i: {"text": f"common field term{i % 10} group{i // 2} item{i}", "missing": False}
                     for i in range(20)}
        documents[20] = {"text": "futureonlyword", "missing": False}
        documents[21] = {"text": "", "missing": True}
        vectors, info = text_vectors(list(range(20)) + [1, 2, 21], documents)
        self.assertEqual(info["fit_pageids"], list(range(20)))
        self.assertNotIn("futureonlyword", info["vocabulary"])
        np.testing.assert_array_equal(vectors[21], np.zeros(8))
        self.assertEqual(vectors[20].shape, (8,))


if __name__ == "__main__":
    unittest.main()
