"""Recompute report results from saved development rankings; never fit a model."""
import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'src'))
sys.path.insert(0, str(ROOT / 'src'))
from wikiflow.core import FORECASTS, aggregate, rank_metric
from wikiflow.reproduce import load, validate_panel, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'generated' / 'replay')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a fresh directory.')
    panel = load(ROOT.parent / 'data/panel.json.gz')
    rowmap = validate_panel(panel)
    candidates = [rowmap[key] for key in panel['evaluation_keys']]
    predictions = load(ROOT / 'data/development_predictions.json.gz')
    expected = {r['method']: r for r in csv.DictReader((ROOT / 'evidence/expected_summary.csv').open())}
    assert set(predictions) == set(expected)
    monthly, summaries = [], []
    for method, scores in predictions.items():
        assert set(scores) == set(panel['evaluation_keys']), method
        records = []
        for month in FORECASTS:
            rows = [r for r in candidates if r['target_month'] == month]
            metric = rank_metric(rows, panel['labels'], {r['key']: scores[r['key']] for r in rows}, 50)
            records.append(dict(method=method, month=month, K=50, **metric))
        summary = dict(method=method, K=50, **aggregate(records))
        for field in ['TP', 'selected', 'events', 'macro_Precision', 'micro_Recall', 'macro_Recall', 'macro_AP', 'macro_binary_NDCG']:
            assert math.isclose(float(summary[field]), float(expected[method][field]), abs_tol=1e-12), (method, field)
        monthly.extend(records)
        summaries.append(summary)
    args.output.mkdir(parents=True)
    write_csv(args.output / 'monthly.csv', monthly)
    write_csv(args.output / 'summary.csv', summaries)
    print(f'PASS: {len(summaries)} methods, {len(monthly)} method-months; 0 fits; no holdout access.')


if __name__ == '__main__':
    main()
