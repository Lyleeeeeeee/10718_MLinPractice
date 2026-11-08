"""Fixed binary LightGBM and LambdaRank recipes."""
import numpy as np

TREE_PARAMETERS = dict(n_estimators=100, num_leaves=7, max_depth=3,
    learning_rate=.03, min_child_samples=30, reg_lambda=10, max_bin=63,
    n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True,
    random_state=718, data_random_seed=718, feature_fraction_seed=718,
    bagging_seed=718, subsample=1., colsample_bytree=1.)
def fit_matrix(family, train, candidates, labels, x, z):
    """Fit one approved tree recipe with fixed weights, offset and constraints."""
    y = np.array([labels[r["key"]]["event"] for r in train], int)
    import lightgbm as lgb
    constraints = [1] + [0] * (x.shape[1] - 1)
    parameters = dict(TREE_PARAMETERS, monotone_constraints=constraints)
    months = sorted({r["target_month"] for r in train})
    counts = {month: sum(r["target_month"] == month for r in train) for month in months}
    if family == "lightgbm":
        weights = np.array([1 / counts[r["target_month"]] for r in train])
        weights /= weights.mean()
        model = lgb.LGBMClassifier(objective="binary", **parameters)
        model.fit(x, y, sample_weight=weights)
        values = model.booster_.predict(z)
    elif family == "lambdarank":
        model = lgb.LGBMRanker(objective="lambdarank", label_gain=[0, 1], lambdarank_norm=True,
            lambdarank_truncation_level=50, **parameters)
        model.fit(x, y, group=[counts[m] for m in months], init_score=np.array(x[:, 0], copy=True))
        values = model.booster_.predict(z, raw_score=True) + z[:, 0]
    else:
        raise ValueError("Unknown fixed model family")
    return values, dict(tree_parameters=parameters, monotone_constraints=constraints,
                        feature_importances=model.feature_importances_.tolist())
