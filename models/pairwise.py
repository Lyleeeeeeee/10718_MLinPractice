"""Fixed within-month pairwise ranking and equal rule blend."""
from fractions import Fraction
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

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

def training_objective(train, labels, x, family):
    months = sorted(set((r['target_month'] for r in train)))
    y = np.asarray([labels[r['key']]['event'] for r in train], float)
    if family != 'pairwise':
        raise ValueError('Only the fixed pairwise objective is supported')
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

def blend_rank_keys(base, learned, alpha):
    """Return integer keys proportional to the intended rational percentile blend.

    Average ranks are integer or half-integer. Doubling them and using integer
    weights preserves mathematical ties exactly. Common positive denominators
    do not affect a within-month ordering. The shared positive scale does not change ranking.
    """
    if alpha != 0.5:
        raise ValueError('Only the fixed 0.5 rule blend is allowed')
    base = np.asarray(base, dtype=float)
    learned = np.asarray(learned, dtype=float)
    if base.ndim != 1 or base.shape != learned.shape or (not len(base)):
        raise ValueError('Nonempty equal-length score vectors are required')
    if not np.isfinite(base).all() or not np.isfinite(learned).all():
        raise ValueError('Scores must be finite')
    weight = Fraction(str(alpha))
    twice_base = np.rint(2 * (rankdata(base, method='average') - 1)).astype(np.int64)
    twice_learned = np.rint(2 * (rankdata(learned, method='average') - 1)).astype(np.int64)
    return (weight.denominator - weight.numerator) * twice_base + weight.numerator * twice_learned
