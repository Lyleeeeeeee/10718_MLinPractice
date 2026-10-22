"""Preserved tabular feature, fitting and selection kernels."""
import math, statistics
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from wikiflow.core import EARLY, FORECASTS, KS, days, joint_margin, preprocess, rank_metric, shift
from wikiflow.learners import fit_logistic, sigmoid
from wikiflow.reproduce import digest

def momentum_score(row):
    current, previous = (math.expm1(row['features'][j]) for j in [0, 1])
    if previous <= 0:
        raise ValueError('Published positive previous rate required')
    return math.log1p(current * current / previous * days(row['target_month'])) - math.log1p(row['effective_event_U_asof'])

def feature_values(row, feature_set, task):
    vals = list(row['features'])
    margin = joint_margin(row)
    if feature_set not in ['original', 'signed', 'dynamics']:
        raise ValueError('Unknown feature set')
    if task == 'binary':
        vals.append(margin)
    if feature_set != 'original':
        vals[4] = math.copysign(math.log1p(abs(vals[4])), vals[4])
    if feature_set == 'dynamics':
        growth = vals[0] - vals[1]
        vals.extend([momentum_score(row), growth, vals[0] - vals[3], vals[0] - vals[5], margin * growth])
    return vals

def chronological_train(rows, labels, month):
    if month not in EARLY + FORECASTS:
        raise ValueError('Only predefined development months; holdout forbidden')
    train = [r for r in rows if r['target_month'] < month and labels[r['key']]['known']]
    for r in train:
        if r['cutoff_month'] != shift(r['target_month'], -1) or r['max_feature_date_declared'][:7] > r['cutoff_month']:
            raise ValueError('Feature cutoff violated')
    return train

def choose_config(records):
    best = max((r['validation_score'] for r in records))
    tied = [r for r in records if best - r['validation_score'] <= 1e-12]
    best_ap = max((r.get('validation_tie_AP', 0) for r in tied))
    tied = [r for r in tied if best_ap - r.get('validation_tie_AP', 0) <= 1e-12]
    return min(tied, key=lambda r: (r['complexity'], -r.get('lambda', r.get('l2', 0)), r['id']))

def fit_one(train, candidates, labels, config):

    def transformed(rows):
        return [dict(r, features=feature_values(r, config['features'], config['task'])) for r in rows]
    x, z, prep = preprocess(transformed(train), transformed(candidates))
    field = 'event' if config['task'] == 'binary' else 'gR'
    y = np.asarray([labels[r['key']][field] for r in train], float)
    if config['family'] == 'linear':
        if config['task'] == 'binary':
            theta, info = fit_logistic(x, y, config['lambda'])
            pred = sigmoid(theta[0] + z @ theta[1:])
        else:
            model = Ridge(alpha=len(x) * config['lambda'], fit_intercept=True, solver='cholesky').fit(x, y)
            pred = model.predict(z)
            info = {'coefficients': model.coef_.tolist(), 'intercept': float(model.intercept_)}
    else:
        cls = HistGradientBoostingClassifier if config['task'] == 'binary' else HistGradientBoostingRegressor
        model = cls(learning_rate=0.05, max_iter=100, max_depth=config['depth'], max_leaf_nodes=2 ** config['depth'], l2_regularization=config['l2'], max_bins=32, min_samples_leaf=max(30, math.ceil(0.05 * len(x))), early_stopping=False, random_state=718).fit(x, y)
        pred = model.predict_proba(z)[:, 1] if config['task'] == 'binary' else model.predict(z)
        info = {'iterations': int(model.n_iter_)}
    if not np.isfinite(pred).all():
        raise ValueError('Nonfinite prediction')
    return ({r['key']: float(v) for r, v in zip(candidates, pred)}, dict(preprocessing_sha256=digest(prep), numeric=info))

def baseline_scores(rows):
    return {'volume': {r['key']: float(r['current_count']) for r in rows}, 'cs_momentum': {r['key']: momentum_score(r) for r in rows}}

def metrics(rows, labels, scores, method, task, months, ks=KS):
    result = []
    for month in months:
        ca = [r for r in rows if r['target_month'] == month]
        ss = {r['key']: scores[r['key']] for r in ca}
        for k in ks:
            result.append(dict(task=task, method=method, month=month, K=k, **rank_metric(ca, labels, ss, k)))
    return result
