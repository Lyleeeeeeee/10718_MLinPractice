"""Contracts for the binary cleanup: fixed results, readable names, safe fitting."""
import csv
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class WorkflowTests(unittest.TestCase):
    def cli(self, *args):
        self.assertTrue((ROOT / "run.py").exists(), "One root-level reproduction entry point is required")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run([sys.executable, "-B", str(ROOT / "run.py"), *args], capture_output=True, text=True, env=env)

    def test_main_table_and_control_replay_frozen_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "replay"
            result = self.cli("replay", "--output", str(output))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("0 fits", result.stdout)
            with (output / "main_results.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([r["model_id"] for r in rows], [
                "volume", "momentum", "traffic_logistic", "traffic_text_logistic",
                "numeric_lambdarank", "tree_classifier", "linear_pairwise_blend", "random_expected"])
            hits = {r["model_id"]: r["hits"] for r in rows}
            self.assertEqual(hits["traffic_text_logistic"], "93")
            self.assertEqual(hits["tree_classifier"], "78")
            self.assertEqual(hits["linear_pairwise_blend"], "80")
            with (output / "text_ablation.csv").open() as handle:
                controls = list(csv.DictReader(handle))
            self.assertEqual([r["model_id"] for r in controls], [
                "traffic_logistic", "text_availability_control", "traffic_text_logistic"])
            self.assertAlmostEqual(float(controls[1]["recall_at_50"]), 88 / 190)
            manifest = json.loads((output / "verification.json").read_text())
            self.assertEqual(manifest["model_fits"], 0)
            self.assertEqual(manifest["known_candidate_rows"], 5387)
            self.assertEqual(manifest["events"], 190)
            self.assertFalse(manifest["holdout_read"])
            self.assertEqual(rows[-1]["reference_type"], "expected_random_order")
            self.assertEqual(rows[-1]["ndcg_at_50"], "")

    def test_legacy_model_id_is_compatible_and_not_reselected(self):
        result = self.cli("models", "--model", "binary_svd_0.01")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Traffic + Text Logistic", result.stdout)
        self.assertIn("traffic_text_logistic", result.stdout)
        unknown = self.cli("models", "--model", "relative_ridge12")
        self.assertNotEqual(unknown.returncode, 0)
        self.assertIn("binary", unknown.stderr.lower())

    def test_future_month_and_unapproved_fit_are_rejected_before_training(self):
        for args in [
            ("fit", "--model", "traffic_text_logistic", "--month", "2026-09", "--allow-fit"),
            ("fit", "--model", "traffic_logistic", "--month", "2024-09")]:
            run = self.cli(*args)
            self.assertNotEqual(run.returncode, 0)
        self.assertIn("allow-fit", self.cli("fit", "--model", "traffic_logistic", "--month", "2024-09").stderr)

    def test_input_bytes_are_preserved(self):
        self.assertTrue((ROOT / "data" / "panel.json.gz").exists(), "Move the original frozen panel without rewriting it")
        checks = {
            "data/panel.json.gz": "8727d27c1b70096971bcfd8bf8f0d9f6270dc9a0e8d9cdcb613cdfa0282e1fcf",
            "data/historical_text.json.gz": "44020dbbf5040389e153379f2113265e2abaad5a3bac35760c2ddfb77b81e9e2"}
        for path, expected in checks.items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected)

    def test_feature_groups_match_original_positions(self):
        self.assertTrue((ROOT / "src" / "wikiflow" / "features.py").exists(), "Readable feature groups must preserve the numerical mapping")
        features = importlib.import_module("wikiflow.features")
        with gzip.open(ROOT / "data" / "panel.json.gz", "rt") as handle:
            row = json.load(handle)["rows"][100]
        numeric = features.traffic_features(row)
        compact = features.compact_features(row)
        self.assertEqual(len(numeric), 13)
        self.assertEqual(numeric[:12], row["features"])
        self.assertEqual(compact, [numeric[-1], numeric[0] - numeric[1], numeric[8], numeric[0], numeric[10] - numeric[3]])

    def test_fixed_selection_evidence_and_input_hashes_verify(self):
        result = self.cli("verify")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS", result.stdout)
        self.assertIn("selection", result.stdout.lower())

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            sentinel = Path(tmp) / "keep.txt"
            sentinel.write_text("existing work")
            result = self.cli("replay", "--output", tmp)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(sentinel.read_text(), "existing work")


if __name__ == "__main__":
    unittest.main()
