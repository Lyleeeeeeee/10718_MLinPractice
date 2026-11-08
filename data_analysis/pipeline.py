"""Fresh expanding-window fits for the twelve approved fixed configurations."""
import importlib.metadata
from pathlib import Path
import platform
import time

import numpy as np
from threadpoolctl import threadpool_limits

from features.assembly import numeric_features, explicit_margin, momentum_score
from models.logistic import fit_logistic, sigmoid
from models.pairwise import training_objective, fit_weights, blend_rank_keys
from features.preprocessing import preprocess
from models.trees import fit_matrix
from textual.representations import represent, age_features
from .io import load, save, digest, write_csv
from .metrics import VALIDATION, TEST, LEARNED, ORDER, score_month, summarize, choose

FAMILIES = {"logistic": "expanded", "lambdarank": "candidate", "lightgbm": "candidate", "pairwise_rule": "expanded"}
KINDS = ("numeric", "svd", "minilm")
ROOT = Path(__file__).resolve().parents[1]


def inputs(root=ROOT, data_dir=None):
    data_dir = Path(data_dir).resolve() if data_dir is not None else root / "data"
    panel = load(data_dir / "panel.json.gz")
    evaluation = load(data_dir / "evaluation_rows.json.gz")
    labels = load(data_dir / "evaluation_labels.json.gz")
    snapshots = load(data_dir / "text/snapshots.json.gz")
    mapping = load(data_dir / "text/row_snapshots.json.gz")
    records = {key: snapshots[sid] for key, sid in mapping.items()}
    ages = load(data_dir / "text/ages.json.gz")
    with np.load(data_dir / "text/minilm_embeddings.npz", allow_pickle=False) as arrays:
        keys = arrays["snapshot_ids"].tolist()
        values = arrays["chunked"]
        if len(set(keys)) != len(keys) or values.shape != (len(keys), 384) or not np.isfinite(values).all():
            raise ValueError("Invalid frozen MiniLM vectors")
        embeddings = dict(zip(keys, values.copy()))
    if set(embeddings) != set(snapshots):
        raise ValueError("Embedding and text snapshot support differ")
    for row in panel["rows"] + [r for group in evaluation.values() for r in group]:
        record = records[row["key"]]
        if record["timestamp"] >= row["target_month"] + "-01T00:00:00Z":
            raise ValueError("Text crosses its own feature cutoff")
    return panel, evaluation, labels, records, ages, embeddings


def training_rows(panel, month, pool):
    if pool not in ("expanded", "candidate"):
        raise ValueError("Unknown TRAIN pool")
    if month not in VALIDATION + TEST:
        raise ValueError("Target outside the fixed comparison")
    rows = [r for r in panel["rows"] if r["target_month"] < month
            and panel["labels"][r["key"]]["known"]
            and (pool == "expanded" or r["evaluation_eligible_asof"])]
    if pool == "candidate":
        rows.sort(key=lambda r: (r["target_month"], r["article"], r["key"]))
    return rows


def features(rows, family, kind, records, ages, vectors):
    result = []
    for row in rows:
        values = numeric_features(row, family)
        if kind != "numeric":
            values += age_features(ages[str(row["pageid"])], row["target_month"] + "-01T00:00:00Z")
            values += [float(records[row["key"]]["missing"])] + vectors[row["key"]].tolist()
        result.append(dict(row, features=values))
    return result


def fit_fold(month, panel, rows, records, ages, embeddings):
    scores, fits, pools = {}, {}, {}
    for pool in ("expanded", "candidate"):
        train = training_rows(panel, month, pool)
        subset = {r["key"]: records[r["key"]] for r in train + rows}
        pools[pool] = dict(rows=len(train), events=sum(panel["labels"][r["key"]]["event"] for r in train),
            max_train_target=max(r["target_month"] for r in train),
            train_keys_sha256=digest([r["key"] for r in train]))
        for kind in KINDS:
            vectors = None
            if kind != "numeric":
                vectors, state = represent(train, subset, embeddings, kind)
                wanted = sorted({records[r["key"]]["snapshot_id"] for r in train if not records[r["key"]]["missing"]})
                if state["fit_snapshot_ids"] != wanted:
                    raise ValueError("Projection TRAIN membership differs")
            for family, wanted_pool in FAMILIES.items():
                if pool != wanted_pool:
                    continue
                x, z, preparation = preprocess(features(train, family, kind, records, ages, vectors),
                                               features(rows, family, kind, records, ages, vectors))
                if x.shape[1] != (13 if family == "logistic" else 5) + (0 if kind == "numeric" else 11):
                    raise ValueError("Unexpected feature set")
                if family == "logistic":
                    y = np.array([panel["labels"][r["key"]]["event"] for r in train], float)
                    theta, solver = fit_logistic(x, y, .01)
                    values = sigmoid(theta[0] + z @ theta[1:])
                    details = dict(solver=solver, penalty=.01)
                elif family == "pairwise_rule":
                    pairs, targets, weights, intercept = training_objective(train, panel["labels"], x, "pairwise")
                    theta, solver = fit_weights(pairs, targets, weights, 1., intercept)
                    values = blend_rank_keys([explicit_margin(r) for r in rows], z @ theta, .5)
                    details = dict(solver=solver, penalty=1., blend_weight=.5, training_pairs=len(pairs))
                else:
                    values, details = fit_matrix(family, train, rows, panel["labels"], x, z)
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite model scores")
                name = family + "__" + kind
                scores[name] = {r["key"]: float(v) for r, v in zip(rows, values)}
                fits[name] = dict(pool=pool, feature_count=x.shape[1], preprocessing=preparation, **details)
    scores["current_volume"] = {r["key"]: r["current_count"] for r in rows}
    scores["momentum"] = {r["key"]: momentum_score(r) for r in rows}
    return scores, dict(pools=pools, fits=fits)


def fresh_output(output, root=ROOT):
    output = Path(output).resolve()
    runs = (root / "runs").resolve()
    if not output.is_relative_to(runs) or output == runs:
        raise ValueError("Outputs must be a new child directory under runs/")
    output.mkdir(parents=True, exist_ok=False)
    return output


def refit(output, root=ROOT, data_dir=None, compare_reference=None):
    from .verify import verify, audit_data
    audit = verify(root)
    compare_reference = data_dir is None if compare_reference is None else compare_reference
    source_audit = dict(source="bundled", source_rows_checked=audit["source_rows_checked"])
    if data_dir is not None:
        _, _, _, checked_rows = audit_data(root, data_dir=Path(data_dir))
        source_audit = dict(source="prepared external input directory", source_rows_checked=checked_rows, source_contracts_passed=True)
    output = fresh_output(output, root)
    panel, evaluation, labels, records, ages, embeddings = inputs(root, data_dir)
    start = time.monotonic()
    predictions, metadata, validation_monthly, test_monthly = {}, {}, [], []
    with threadpool_limits(limits=2):
        for month in VALIDATION:
            predictions[month], metadata[month] = fit_fold(month, panel, evaluation[month], records, ages, embeddings)
            validation_monthly.extend(score_month(month, evaluation[month], labels, predictions[month]))
            print("Validation", month, flush=True)
        selection = choose(validation_monthly)
        save(output / "selection.json", dict(model_id=selection["model_id"], validation_metrics=selection,
             selection_rule="Validation pooled Recall@50, mean event-month NDCG@50, lexical ID",
             prior_test_exposure=True, independent_holdout=False))
        for month in TEST:
            predictions[month], metadata[month] = fit_fold(month, panel, evaluation[month], records, ages, embeddings)
            test_monthly.extend(score_month(month, evaluation[month], labels, predictions[month]))
            print("Retrospective test", month, flush=True)
    save(output / "predictions.json.gz", predictions)
    save(output / "fit_metadata.json.gz", metadata)
    for name, values in (("validation_monthly", validation_monthly), ("test_monthly", test_monthly),
                         ("validation_summary", summarize(validation_monthly)), ("test_summary", summarize(test_monthly))):
        write_csv(output / (name + ".csv"), values)
    maximum, compared = None, 0
    if compare_reference:
        expected = load(root / "results/predictions.json.gz")
        maximum = 0.
        compared = 0
        for month in VALIDATION + TEST:
            if set(predictions[month]) != set(expected[month]):
                raise ValueError("Refit configuration support differs")
            for model, values in predictions[month].items():
                if set(values) != set(expected[month][model]):
                    raise ValueError("Refit candidate keys differ")
                maximum = max(maximum, max(abs(v - expected[month][model][k]) for k, v in values.items()))
                compared += 1
        if maximum > 1e-8:
            raise ValueError(f"Refit prediction mismatch: {maximum}")
        expected_selection = load(root / "results/selection.json")["model_id"]
        if selection["model_id"] != expected_selection:
            raise ValueError("Validation selection changed")
    proof = dict(supervised_fits=312, representation_fits=104, encoder_fits=0, downloads=0,
        reference_comparison=compare_reference, score_maps_compared=compared, max_prediction_error=maximum,
        selected_model=selection["model_id"], seconds=time.monotonic()-start, reference_bundle_audit=audit, source_input_audit=source_audit,
        python=platform.python_version(), data_source="bundled" if data_dir is None else "prepared external input directory", versions={n: importlib.metadata.version(n) for n in
            ("numpy", "scipy", "scikit-learn", "lightgbm", "threadpoolctl")})
    save(output / "verification.json", proof)
    return proof
