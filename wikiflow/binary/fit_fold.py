"""Explicit opt-in refit of one documented development recipe and month."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'src'))
sys.path.insert(0, str(ROOT / 'src'))


def fit_recipe(recipe, month, exclude_pageid=None):
    """Use preserved numerical kernels; this function performs actual fitting."""
    import numpy as np
    import lightgbm as lgb
    from threadpoolctl import threadpool_limits
    from wikiflow.core import FORECASTS, candidate_allowed, joint_margin, preprocess
    from wikiflow.learners import fit_logistic, sigmoid
    from wikiflow.reproduce import load, validate_panel
    from experiment import feature_values, fit_one, momentum_score
    from rule_residual import EARLY8, train_at, prepared, training_objective, fit_weights
    from rule_blend_v2 import blend_rank_keys
    from nonlinear_focus import PARAMETERS, query_groups, add_initializer
    from text_features import representation
    from at50_ranker_semantics import fit_representation, augmented_prepared

    if month not in EARLY8 + FORECASTS:
        raise ValueError('Only documented development months are allowed')
    panel = load(ROOT.parent / 'data/panel.json.gz')
    validate_panel(panel)
    rows = [r for r in panel['rows'] if r['pageid'] != exclude_pageid]
    labels = panel['labels']
    candidates = [r for r in rows if r['target_month'] == month and candidate_allowed(r)]
    batch = recipe['batch']
    pool = recipe.get('pool', 'candidate' if batch in ['b5', 'b8'] else 'expanded')
    train = train_at(rows, labels, month, pool)
    with threadpool_limits(limits=2):
        if batch == 'rule':
            values = [float(r['current_count']) if recipe['id'] == 'volume' else momentum_score(r) for r in candidates]
        elif batch in ['original', 'b1']:
            return fit_one(train, candidates, labels, recipe)[0]
        elif batch == 'b4':
            x, z, _ = prepared(train, candidates)
            xx, y, weights, intercept = training_objective(train, labels, x, recipe['family'])
            theta, _ = fit_weights(xx, y, weights, recipe['lambda'], intercept)
            learned = theta[0] + z @ theta[1:] if intercept else z @ theta
            values = blend_rank_keys([joint_margin(r) for r in candidates], learned, recipe['alpha'])
        elif batch == 'b2':
            text_dir = ROOT / 'data/text' / batch
            docs = {int(k): v for k, v in load(text_dir / 'documents.json.gz').items()}
            kind = recipe['kind']
            vectors = {i: np.zeros(0) for i in docs}
            if batch == 'b2':
                archive = np.load(text_dir / 'embeddings.npz')
                emb = dict(zip(map(int, archive['pageids']), archive['vectors']))
                vectors, _ = representation([r['pageid'] for r in train], docs, emb, kind)
            def features(items):
                return [dict(r, features=feature_values(r, 'original', 'binary') + vectors[r['pageid']].tolist() + ([] if kind == 'numeric' else [float(docs[r['pageid']]['missing'])])) for r in items]
            x, z, _ = preprocess(features(train), features(candidates))
            theta, _ = fit_logistic(x, np.array([labels[r['key']]['event'] for r in train], float), recipe['lambda'])
            values = sigmoid(theta[0] + z @ theta[1:])
        else:
            eligible = [r for r in rows if candidate_allowed(r)]
            train, groups = query_groups(eligible, labels, month)
            x, z, _ = prepared(train, candidates)
            if batch == 'b8' and recipe['kind'] != 'base':
                folder = ROOT / 'data/text/b8'
                docs = {int(k): v for k, v in load(folder / 'documents.json.gz').items()}
                archive = np.load(folder / 'intro.npz')
                emb = dict(zip(map(int, archive['pageids']), archive['vectors']))
                vectors = {i: np.zeros(0) for i in docs}
                if recipe['kind'] != 'missing':
                    vectors, _, _ = fit_representation([r['pageid'] for r in train], docs, emb, recipe['kind'])
                x, z, _, _ = augmented_prepared(train, candidates, docs, vectors, recipe['kind'])
            y = np.array([labels[r['key']]['event'] for r in train], int)
            params = dict(PARAMETERS, monotone_constraints=[1] + [0] * (x.shape[1] - 1))
            if recipe.get('objective') == 'binary':
                counts = {m: sum(r['target_month'] == m for r in train) for m in set(r['target_month'] for r in train)}
                weights = np.array([1 / counts[r['target_month']] for r in train]); weights /= weights.mean()
                model = lgb.LGBMClassifier(objective='binary', **params).fit(x, y, sample_weight=weights)
                values = model.predict_proba(z)[:, 1]
            else:
                initialized = recipe.get('initialized', True)
                model = lgb.LGBMRanker(objective='lambdarank', label_gain=[0, 1], lambdarank_norm=True, lambdarank_truncation_level=recipe.get('truncation', 50), **params)
                model.fit(x, y, group=groups, init_score=x[:, 0] if initialized else None)
                values = add_initializer(model.booster_.predict(z, raw_score=True), z, initialized)
        assert np.isfinite(values).all()
        return {r['key']: float(v) for r, v in zip(candidates, values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--month', required=True)
    parser.add_argument('--recipe', default='binary_log13')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-fit', action='store_true')
    parser.add_argument('--exclude-pageid', type=int, choices=[525160])
    args = parser.parse_args()
    months = [f'2024-{m:02d}' for m in range(1, 13)] + [f'2025-{m:02d}' for m in range(1, 13)] + [f'2026-{m:02d}' for m in range(1, 9)]
    if args.month not in months:
        parser.error('Only January 2024--August 2026 development months are allowed.')
    if not args.allow_fit or args.output is None:
        parser.error('Actual fitting requires --allow-fit and a fresh --output; use reproduce.py for zero-fit replay.')
    if args.output.exists():
        parser.error('Output exists; refusing replacement.')
    recipes = json.loads((ROOT / 'config/recipes.json').read_text())
    if args.recipe not in recipes:
        parser.error('Unknown recipe; see config/recipes.json.')
    scores = fit_recipe(recipes[args.recipe], args.month, args.exclude_pageid)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, allow_nan=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
