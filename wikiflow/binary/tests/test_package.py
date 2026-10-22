"""Zero-fit contracts for the portable report package."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class PackageTests(unittest.TestCase):
    def test_replay_reproduces_saved_metrics_without_training(self):
        script = ROOT / 'reproduce.py'
        self.assertTrue(script.exists(), 'A portable zero-fit reproduction entry point is required')
        with tempfile.TemporaryDirectory() as tmp:
            run = subprocess.run([sys.executable, str(script), '--output', tmp + '/replay'], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn('0 fits', run.stdout)

    def test_future_month_is_rejected(self):
        script = ROOT / 'fit_fold.py'
        self.assertTrue(script.exists(), 'The explicit development-only fold entry point is required')
        run = subprocess.run([sys.executable, str(script), '--month', '2026-09'], capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('development', run.stderr.lower())

if __name__ == '__main__':
    unittest.main()
