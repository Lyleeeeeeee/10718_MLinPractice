"""Fixed binary fitters for the retained models; no search or automatic fitting."""
from fractions import Fraction
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from .core import FORECASTS, candidate_allowed, joint_margin, preprocess, shift
from .features import compact_features, momentum_score, text_vectors, traffic_features
from .learners import fit_logistic, sigmoid
from .registry import resolve_model

EARLY_MONTHS = [f"2024-{month:02d}" for month in range(1, 9)]
TREE_PARAMETERS = dict(n_estimators=100, num_leaves=7, max_depth=3,
    learning_rate=.03, min_child_samples=30, reg_lambda=10, max_bin=63,
    n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True,
    random_state=718, data_random_seed=718, feature_fraction_seed=718,
    bagging_seed=718, subsample=1., colsample_bytree=1.)

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


def training_rows(panel, month, pool):
    if month not in EARLY_MONTHS + FORECASTS:
        raise ValueError("Only fixed development months are allowed; holdout is sealed")
    rows = [row for row in panel["rows"]
            if row["target_month"] < month and panel["labels"][row["key"]]["known"]
            and (pool == "expanded" or candidate_allowed(row))]
    if any(row["cutoff_month"] != shift(row["target_month"], -1)
           or row["max_feature_date_declared"][:7] > row["cutoff_month"] for row in rows):
        raise ValueError("As-of boundary violated")
    return rows


def fit_model(model_id, month, panel, documents):
    """Refit one existing fixed recipe; invoked only by the explicit CLI fit command."""
    spec = resolve_model(model_id)
    if month not in EARLY_MONTHS + FORECASTS:
        raise ValueError("Only fixed development months are allowed; holdout is sealed")
    candidates = [row for row in panel["rows"]
                  if row["target_month"] == month and candidate_allowed(row)]
    if spec["family"] == "reference":
        raise ValueError("Random is an expected reference, not a fitted model")
    if spec["family"] == "rule":
        if spec["id"] == "momentum":
            scores = {row["key"]: momentum_score(row) for row in candidates}
        else:
            scores = {row["key"]: float(row["current_count"]) for row in candidates}
        return scores, {"model_fits": 0}
    train = training_rows(panel, month, spec["pool"])
    labels = panel["labels"]
    with threadpool_limits(limits=2):
        if spec["family"] == "logistic":
            vectors = None
            if spec["id"] == "traffic_text_logistic":
                vectors, _ = text_vectors([row["pageid"] for row in train], documents)
            def transform(rows):
                output = []
                for row in rows:
                    values = traffic_features(row)
                    if vectors is not None:
                        values += vectors[row["pageid"]].tolist()
                    if spec["id"] in ("traffic_text_logistic", "text_availability_control"):
                        values += [float(documents[row["pageid"]]["missing"])]
                    output.append(dict(row, features=values))
                return output
            x, z, preparation = preprocess(transform(train), transform(candidates))
            y = np.array([labels[row["key"]]["event"] for row in train], float)
            theta, _ = fit_logistic(x, y, spec["penalty"])
            values = sigmoid(theta[0] + z @ theta[1:])
        else:
            if spec["family"] in ("tree_classifier", "lambdarank"):
                train = sorted(train, key=lambda row: (row["target_month"], row["article"], row["key"]))
            def transform(rows):
                return [dict(row, features=compact_features(row)) for row in rows]
            x, z, preparation = preprocess(transform(train), transform(candidates))
            y = np.array([labels[row["key"]]["event"] for row in train], int)
            if spec["family"] == "pairwise":
                xx, targets, weights, intercept = training_objective(train, labels, x, "pairwise")
                theta, _ = fit_weights(xx, targets, weights, spec["penalty"], intercept)
                values = blend_rank_keys([joint_margin(row) for row in candidates], z @ theta, spec["blend_weight"])
            else:
                import lightgbm as lgb
                parameters = dict(TREE_PARAMETERS, monotone_constraints=[1, 0, 0, 0, 0])
                months = sorted({row["target_month"] for row in train})
                counts = {m: sum(row["target_month"] == m for row in train) for m in months}
                if spec["family"] == "tree_classifier":
                    weights = np.array([1 / counts[row["target_month"]] for row in train])
                    weights /= weights.mean()
                    model = lgb.LGBMClassifier(objective="binary", **parameters)
                    model.fit(x, y, sample_weight=weights)
                    values = model.predict_proba(z)[:, 1]
                else:
                    model = lgb.LGBMRanker(objective="lambdarank", label_gain=[0, 1],
                        lambdarank_norm=True, lambdarank_truncation_level=50, **parameters)
                    model.fit(x, y, group=[counts[m] for m in months], init_score=x[:, 0])
                    values = model.booster_.predict(z, raw_score=True) + z[:, 0]
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite predictions")
    return {row["key"]: float(value) for row, value in zip(candidates, values)}, {
        "model_fits": 1, "training_rows": len(train), "preprocessing": preparation}
