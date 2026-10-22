"""Check two report examples against bundled histories and the baseline panel."""
import calendar
import gzip
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def days(month):
    return calendar.monthrange(*map(int, month.split('-')))[1]

def bound(rates, count_days):
    median = statistics.median(rates)
    scale = 1.4826 * statistics.median(abs(x - median) for x in rates)
    return max(200, count_days * median + max(100, .5 * count_days * median, 3 * count_days * scale))

def main():
    evidence = json.loads((ROOT / 'evidence/case_evidence.json').read_text())
    with gzip.open(ROOT.parent / 'data/panel.json.gz', 'rt') as handle:
        panel = json.load(handle)
    rows = {r['key']: r for r in panel['rows']}
    for case in evidence['cases']:
        key = f"{case['target_month']}:{case['pageid']}"
        row, label = rows[key], panel['labels'][key]
        assert row['article'] == case['article'] and key in panel['evaluation_keys']
        assert case['pageid'] != 525160 and case['eligible']
        history = case['history']
        rates = [v['incoming'] / days(v['month']) for v in history[-7:-1]]
        threshold = max(bound(rates, case['target_days']), 1.25 * case['previous_incoming'] * case['target_days'] / case['previous_days'])
        assert math.isclose(threshold, case['T'], rel_tol=1e-12)
        assert math.isclose(threshold, label['T'], rel_tol=1e-12)
        assert label['actual'] == case['target_incoming']
        assert int(label['actual'] > threshold) == case['y']
        assert label['event'] == case['y']
    print('PASS: 2 cases, fixed development panel, 0 fits; no raw data or holdout access.')

if __name__ == '__main__':
    main()
