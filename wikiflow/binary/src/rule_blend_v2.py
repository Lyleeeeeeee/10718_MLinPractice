"""Exact ordering keys for the five frozen rank-blend weights; no fitting."""
from fractions import Fraction
import numpy as np
from scipy.stats import rankdata

def blend_rank_keys(base, learned, alpha):
    """Return integer keys proportional to the intended rational percentile blend.

    Average ranks are integer or half-integer. Doubling them and using integer
    weights preserves mathematical ties exactly. Common positive denominators
    do not affect a within-month ordering. The original frozen runner is retained.
    """
    if alpha not in (0, 0.1, 0.25, 0.5, 1):
        raise ValueError('Only the five frozen blend weights are allowed')
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
