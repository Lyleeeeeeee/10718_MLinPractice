"""Verify historical choices from saved early scores, without new selection or fits."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / 'wikiflow/src'))
sys.path.insert(0, str(ROOT / 'src'))
from wikiflow.reproduce import load
from experiment import choose_config
from rule_residual import select
from at50_ranker_semantics import select_recipe


def main():
    for batch, selector in [('batch1', choose_config), ('batch2', choose_config), ('batch4_rule_residual', select), ('batch8_at50_ranker_semantics', select_recipe)]:
        directory = ROOT / 'evidence' / batch
        frozen = load(directory / 'selection_frozen_before_development.json')
        source = directory / ('selection_scores.json' if batch == 'batch4_rule_residual' else 'validation_scores.json')
        scores = load(source) if source.exists() else frozen['configs']
        expected = frozen['overall'].get('binary', frozen['overall'])
        actual = selector(scores)
        assert actual['id'] == expected['id'], (batch, actual['id'], expected['id'])
        print(batch, actual['id'])
    print('PASS: 4 selection records; 0 fits. Historical choices are unchanged.')


if __name__ == '__main__':
    main()
