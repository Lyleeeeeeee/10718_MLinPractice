"""Three-arm feature ablation for the final Pairwise + Traffic Rule recipe."""
from pathlib import Path

from threadpoolctl import threadpool_limits

from features.preprocessing import preprocess
from textual.representations import represent
from .io import load, save, write_csv
from .metrics import VALIDATION, TEST, score_month, summarize
from .pipeline import ROOT, features, fit_pairwise_rule, fresh_output, inputs, training_rows

KINDS = ("numeric", "age", "svd")
MODEL_IDS = tuple("pairwise_rule__" + kind for kind in KINDS)
FEATURE_COUNTS = (5, 7, 16)
REFERENCE_KINDS = ("numeric", "svd")


def fold_scores(month, panel, rows, records, ages):
    train = training_rows(panel, month, "expanded")
    subset = {r["key"]: records[r["key"]] for r in train + rows}
    vectors, projection = represent(train, subset, None, "svd")
    wanted = sorted({records[r["key"]]["snapshot_id"] for r in train
                     if not records[r["key"]]["missing"]})
    if projection["fit_snapshot_ids"] != wanted:
        raise ValueError("SVD fit must use distinct usable TRAIN snapshots only")
    scores = {}
    pair_counts = set()
    for kind, name, feature_count in zip(KINDS, MODEL_IDS, FEATURE_COUNTS):
        active_vectors = vectors if kind == "svd" else None
        x, z, _ = preprocess(features(train, "pairwise_rule", kind, records, ages, active_vectors),
                             features(rows, "pairwise_rule", kind, records, ages, active_vectors))
        if x.shape[1] != feature_count or z.shape[1] != feature_count:
            raise ValueError(f"Unexpected feature count for {name}")
        values, details = fit_pairwise_rule(train, rows, panel["labels"], x, z)
        pair_counts.add(details["training_pairs"])
        scores[name] = {r["key"]: float(v) for r, v in zip(rows, values)}
    if len(pair_counts) != 1:
        raise ValueError("Training pair counts differ between arms")
    return scores


def reference_matches(scores, reference):
    """Count exact score-map matches without mixing scores from distinct runs."""
    matches = []
    for kind in REFERENCE_KINDS:
        name = "pairwise_rule__" + kind
        if scores[name].keys() != reference[name].keys():
            raise ValueError(f"Candidate support differs from the main benchmark: {name}")
        matches.append(scores[name] == reference[name])
    return sum(matches)


def tables(evaluation, labels, predictions):
    if set(predictions) != set(VALIDATION + TEST):
        raise ValueError("Ablation prediction months differ from the main benchmark")
    result = {}
    for split, months in (("validation", VALIDATION), ("test", TEST)):
        monthly = []
        for month in months:
            monthly.extend(score_month(month, evaluation[month], labels, predictions[month], order=MODEL_IDS))
        result[split + "_monthly"] = monthly
        result[split + "_summary"] = summarize(monthly, order=MODEL_IDS)
    return result


def run(output, root=ROOT):
    root = Path(root)
    panel, evaluation, labels, records, ages, _ = inputs(root)
    reference = load(root / "results/predictions.json.gz")
    output = fresh_output(output, root)
    predictions = {}
    exact_maps = 0
    with threadpool_limits(limits=2):
        for month in VALIDATION + TEST:
            scores = fold_scores(month, panel, evaluation[month], records, ages)
            exact_maps += reference_matches(scores, reference[month])
            predictions[month] = scores
            print("Age ablation", month, flush=True)
    save(output / "predictions.json.gz", predictions)
    for name, rows in tables(evaluation, labels, predictions).items():
        write_csv(output / (name + ".csv"), rows)
    return dict(months=len(predictions), fits=len(predictions) * len(MODEL_IDS),
                reference_score_maps_compared=len(predictions) * len(REFERENCE_KINDS),
                exact_reference_score_maps=exact_maps,
                output=str(output))
