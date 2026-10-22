"""Month-balanced pairwise fitting kernels, without historical batch drivers."""
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from wikiflow.core import FORECASTS, candidate_allowed, preprocess, joint_margin, shift
EARLY8 = [f'2024-{m:02d}' for m in range(1, 9)]

def objective(theta, x, y, w, lam, intercept):
    a = np.column_stack([np.ones(len(x)), x]) if intercept else x
    eta = a @ theta
    penalty = np.r_[0, np.repeat(lam, x.shape[1])] if intercept else np.repeat(lam, x.shape[1])
    value = float(np.dot(w, np.logaddexp(0, eta) - y * eta) + 0.5 * np.dot(penalty, theta ** 2))
    grad = a.T @ (w * (expit(eta) - y)) + penalty * theta
    return (value, grad)

def fit_weights(x, y, w, lam, intercept):
    theta = np.zeros(x.shape[1] + int(intercept))
    if intercept:
        prevalence = float(np.dot(w, y))
        theta[0] = math.log(prevalence / (1 - prevalence))
    bounds = [(None, None)] * len(theta)
    bounds[int(intercept)] = (0, None)
    result = minimize(lambda t: objective(t, x, y, w, lam, intercept), theta, jac=True, method='L-BFGS-B', bounds=bounds, options=dict(maxiter=500, gtol=1e-07, ftol=1e-12))
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(str(result.message))
    return (result.x, dict(iterations=int(result.nit), objective=float(result.fun), converged=bool(result.success), theta=result.x.tolist()))

def prepared(train, test):

    def tx(rr):
        return [dict(r, features=[joint_margin(r), r['features'][0] - r['features'][1], r['features'][8], r['features'][0], r['features'][10] - r['features'][3]]) for r in rr]
    return preprocess(tx(train), tx(test))

def training_objective(train, labels, x, family):
    months = sorted(set((r['target_month'] for r in train)))
    y = np.asarray([labels[r['key']]['event'] for r in train], float)
    if family == 'logistic':
        counts = {m: sum((r['target_month'] == m for r in train)) for m in months}
        return (x, y, np.asarray([1 / (len(months) * counts[r['target_month']]) for r in train]), True)
    pairs = []
    weights = []
    rng = np.random.default_rng(718)
    valid = 0
    for m in months:
        pos = [i for i, r in enumerate(train) if r['target_month'] == m and y[i] == 1]
        neg = [i for i, r in enumerate(train) if r['target_month'] == m and y[i] == 0]
        if not pos or not neg:
            continue
        valid += 1
        block = []
        for i in pos:
            chosen = neg if len(neg) <= 200 else rng.choice(neg, size=200, replace=False)
            block.extend((x[i] - x[j] for j in chosen))
        pairs.extend(block)
        weights.extend([1 / len(block)] * len(block))
    if not valid:
        raise ValueError('No positive-negative TRAIN pairs')
    return (np.asarray(pairs), np.ones(len(pairs)), np.asarray(weights) / valid, False)

def train_at(rows, labels, month, pool):
    if month not in EARLY8 + FORECASTS:
        raise ValueError('Forbidden target/holdout')
    train = [r for r in rows if r['target_month'] < month and labels[r['key']]['known'] and (pool == 'expanded' or candidate_allowed(r))]
    if any((r['cutoff_month'] != shift(r['target_month'], -1) or r['max_feature_date_declared'][:7] > r['cutoff_month'] for r in train)):
        raise ValueError('As-of boundary')
    return train

def select(records):
    best = max((r['P5'] for r in records))
    group = [r for r in records if best - r['P5'] <= 1e-12]
    ap = max((r['AP'] for r in group))
    group = [r for r in group if ap - r['AP'] <= 1e-12]
    return min(group, key=lambda r: (r['alpha'], -r['lambda'], r['pool'] != 'expanded', r['family'], r['id']))
