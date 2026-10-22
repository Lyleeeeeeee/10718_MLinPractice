"""Original search choices must survive packaging without refitting."""
import subprocess
import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
class SelectionTests(unittest.TestCase):
    def test_all_four_historical_searches_replay(self):
        script = ROOT / 'replay_selection.py'
        self.assertTrue(script.exists(), 'Historical selection replay is required')
        result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('4 selection records', result.stdout)
