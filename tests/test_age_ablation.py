"""Contracts for the supplementary age vs. text comparison."""
from pathlib import Path
import unittest

import numpy as np

from data_analysis.age_ablation import FEATURE_COUNTS, MODEL_IDS, reference_matches
from data_analysis.metrics import LEARNED, score_month, summarize
from data_analysis.pipeline import features
from data_analysis.io import load

ROOT = Path(__file__).resolve().parents[1]


class AgeAblationTests(unittest.TestCase):
    def test_age_arm_has_no_text_inputs(self):
        row = load(ROOT / "data/evaluation_rows.json.gz")["2024-09"][0]
        key = row["key"]
        ages = load(ROOT / "data/text/ages.json.gz")
        # Empty text records and no vectors: the first two arms must not access text.
        numeric = features([row], "pairwise_rule", "numeric", {}, ages, None)[0]["features"]
        age = features([row], "pairwise_rule", "age", {}, ages, None)[0]["features"]
        both = features([row], "pairwise_rule", "svd", {key: {"missing": False}},
                        ages, {key: np.arange(8)})[0]["features"]
        self.assertEqual(tuple(map(len, (numeric, age, both))), FEATURE_COUNTS)
        self.assertEqual(age[:len(numeric)], numeric)
        self.assertEqual(both[:len(age)], age)
        self.assertEqual(both[len(age):], [0.] + list(range(8)))

    def test_same_ranking_and_no_selection_change(self):
        self.assertNotIn(MODEL_IDS[1], LEARNED)
        rows = [dict(key="a", pageid=1, current_count=100), dict(key="b", pageid=2, current_count=110)]
        labels = {"a": {"known": True, "event": True}, "b": {"known": True, "event": False}}
        predictions = {name: {"a": 1., "b": 1.} for name in MODEL_IDS}
        monthly = score_month("2024-09", rows, labels, predictions, order=MODEL_IDS)
        self.assertEqual([r["model_id"] for r in summarize(monthly, order=MODEL_IDS)], list(MODEL_IDS))
        self.assertTrue(all(r["hits_at_50"] == 1 for r in monthly))
        self.assertAlmostEqual(monthly[0]["ndcg_at_50"], 0.6309297535714575)
        self.assertEqual(reference_matches(predictions, predictions), 2)
        changed = dict(predictions, **{MODEL_IDS[2]: {"a": 2., "b": 1.}})
        self.assertEqual(reference_matches(changed, predictions), 1)
        with self.assertRaises(ValueError):
            reference_matches(dict(predictions, **{MODEL_IDS[2]: {"a": 1.}}), predictions)
