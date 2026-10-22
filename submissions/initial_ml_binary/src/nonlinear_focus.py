"""Fixed LightGBM settings and chronological query grouping."""
import numpy as np
from wikiflow.core import FORECASTS
from rule_residual import EARLY8
PARAMETERS = dict(n_estimators=100, num_leaves=7, max_depth=3, learning_rate=0.03, min_child_samples=30, reg_lambda=10, max_bin=63, monotone_constraints=[1, 0, 0, 0, 0], n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True, random_state=718, data_random_seed=718, feature_fraction_seed=718, bagging_seed=718, subsample=1.0, colsample_bytree=1.0)

def query_groups(rows, labels, month):
    if month not in EARLY8 + FORECASTS:
        raise ValueError('Forbidden target/holdout')
    train = sorted([r for r in rows if r['target_month'] < month and labels[r['key']]['known']], key=lambda r: (r['target_month'], r['article'], r['key']))
    months = sorted({r['target_month'] for r in train})
    if not train:
        raise ValueError('Empty TRAIN')
    if len({r['key'] for r in train}) != len(train):
        raise ValueError('Duplicate TRAIN rows')
    return (train, [sum((r['target_month'] == m for r in train)) for m in months])

def add_initializer(tree_scores, z, enabled):
    return np.asarray(tree_scores) + (z[:, 0] if enabled else 0)

def choose(records):
    p = max((r['P5'] for r in records))
    group = [r for r in records if p - r['P5'] <= 1e-12]
    ap = max((r['AP'] for r in group))
    group = [r for r in group if ap - r['AP'] <= 1e-12]
    return min(group, key=lambda r: (r['complexity'], r['id']))
