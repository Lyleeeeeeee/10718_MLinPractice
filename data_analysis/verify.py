"""Read-only integrity, independent label arithmetic and result verification."""
import calendar
import csv
import json
import math
from pathlib import Path
import statistics

from .io import load, sha, digest
from .metrics import VALIDATION, TEST, ORDER, eligible, score_month, summarize, choose


def package_files(root):
    paths = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in {".git", "runs", "local_data", "__pycache__"} or part.startswith(".venv") for part in relative.parts):
            continue
        if path.is_file() and relative.as_posix() != "manifest.json":
            paths.append(relative.as_posix())
    return sorted(paths)


def check_manifest(root):
    expected = load(root / "manifest.json")["files_sha256"]
    actual = package_files(root)
    if set(actual) != set(expected):
        raise ValueError(f"Package file list changed: missing={set(expected)-set(actual)}, extra={set(actual)-set(expected)}")
    for name, digest_value in expected.items():
        if sha(root / name) != digest_value:
            raise ValueError("Package file changed: " + name)
    return len(expected)


def compare_rows(actual, path):
    with path.open() as stream:
        expected = list(csv.DictReader(stream))
    if len(actual) != len(expected):
        raise ValueError("Result row count differs: " + str(path))
    for first, second in zip(actual, expected):
        for key, value in first.items():
            saved = second[key]
            if value is None:
                same = saved == ""
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                same = math.isclose(float(saved), value, rel_tol=1e-12, abs_tol=1e-12)
            else:
                same = str(value) == saved
            if not same:
                raise ValueError(f"Result mismatch in {path.name}: {key}: {value} != {saved}")


def audit_data(root, data_dir=None):
    """Recompute loose and original bands without importing feature/label code."""
    data_dir = Path(data_dir) if data_dir is not None else root / "data"
    panel = load(data_dir / "panel.json.gz")
    evaluation = load(data_dir / "evaluation_rows.json.gz")
    evaluation_labels = load(data_dir / "evaluation_labels.json.gz")
    counts = load(data_dir / "source_monthly_counts.json.gz")
    rows = {row["key"]: row for row in panel["rows"]}
    if len(rows) != len(panel["rows"]) or set(rows) != set(panel["labels"]):
        raise ValueError("TRAIN panel keys differ")
    if set(evaluation) != set(VALIDATION + TEST):
        raise ValueError("Evaluation month support differs")
    for month, group in evaluation.items():
        if len({row["key"] for row in group}) != len(group):
            raise ValueError("Duplicate evaluation candidate")
        for row in group:
            if row["key"] in rows and row != rows[row["key"]]:
                raise ValueError("TRAIN/evaluation row disagreement")
            rows[row["key"]] = row
    labels = dict(panel["labels"])
    for key, value in evaluation_labels.items():
        if key in labels and value != labels[key]:
            raise ValueError("TRAIN/evaluation label disagreement")
        labels[key] = value
    if set(rows) != set(labels):
        raise ValueError("Row/label support differs")
    def days(month):
        return calendar.monthrange(*map(int, month.split('-')))[1]
    def shift(month, offset):
        y, m = map(int, month.split('-'))
        i = y * 12 + m - 1 + offset
        return f"{i//12:04d}-{i%12+1:02d}"
    def bound(rates, length, fraction, sigmas):
        med = statistics.median(rates)
        spread = 1.4826 * statistics.median(abs(x-med) for x in rates)
        return max(200., length*med + max(100., fraction*length*med, sigmas*length*spread))
    def close(a, b):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)
    mapping = load(data_dir / "text/row_snapshots.json.gz")
    snapshots = load(data_dir / "text/snapshots.json.gz")
    ages = load(data_dir / "text/ages.json.gz")
    if set(mapping) != set(rows) or set(mapping.values()) != set(snapshots):
        raise ValueError("Text mapping support differs")
    for key, row in rows.items():
        title, target, cutoff = row["article"], row["target_month"], row["cutoff_month"]
        if key != f"{target}:{row['pageid']}" or cutoff != shift(target,-1):
            raise ValueError("Row identity/cutoff differs")
        if row["max_feature_date_declared"][:7] > cutoff:
            raise ValueError("Declared numeric feature date crosses cutoff")
        activity = row["editor_activity_asof"]
        if not activity["coverage_complete"] or activity["window_end_exclusive"] != target+'-01T00:00:00Z':
            raise ValueError("Historical edit coverage differs")
        rates = [counts[shift(cutoff,i)][title]/days(shift(cutoff,i)) for i in range(-5,1)]
        prior_rates = [counts[shift(cutoff,i)][title]/days(shift(cutoff,i)) for i in range(-6,0)]
        original = bound(rates,days(target),.5,3)
        previous = bound(prior_rates,days(cutoff),.5,3)
        loose = bound(rates,days(target),.25,2)
        pace = row["current_count"]*days(target)/days(cutoff)
        threshold = max(loose,1.1*pace)
        if row["current_count"] != counts[cutoff][title] or not close(original,row["U_count"]) or not close(previous,row["current_precurrent_U"]):
            raise ValueError("Original eligibility threshold differs")
        if not close(loose,row["label_U_count"]) or not close(threshold,row["effective_event_U_asof"]):
            raise ValueError("Loose threshold differs")
        if row["current_count"] < 100 or pace > original+1e-9 or row["current_count"] > previous+1e-9 or not row["metadata_quality_complete"]:
            raise ValueError("Expanded cold-row membership differs")
        if row["evaluation_eligible_asof"] != eligible(row):
            raise ValueError("Original candidate eligibility differs")
        if len(row["features"]) != 12 or not close(row["features"][11],math.log1p(pace)-math.log1p(loose)):
            raise ValueError("Loose numeric margin differs")
        if not close(row["features"][0],math.log1p(rates[-1])) or not close(row["features"][1],math.log1p(rates[-2])):
            raise ValueError("Current/previous rate feature differs")
        label = labels[key]
        actual = label["actual"]
        if actual is not None and (type(actual) is not int or actual < 0):
            raise ValueError("Invalid published count")
        if target in counts and actual != counts[target].get(title):
            raise ValueError("Outcome differs from retained count source")
        if not close(label["T"],threshold) or label["known"] != (actual is not None) or label["event"] != (None if actual is None else int(actual>threshold)):
            raise ValueError("Binary label differs")
        rec = snapshots[mapping[key]]
        if rec["pageid"] != row["pageid"] or rec["timestamp"] >= target+'-01T00:00:00Z':
            raise ValueError("Historical text cutoff/identity differs")
        if rec["snapshot_id"] != f"{rec['pageid']}:{rec['revision_id']}":
            raise ValueError("Revision identity differs")
        if str(row["pageid"]) not in ages:
            raise ValueError("Age evidence missing")
    for month, group in evaluation.items():
        wanted = {key for key,row in rows.items() if row["target_month"]==month and eligible(row)}
        if wanted != {r["key"] for r in group}:
            raise ValueError("Candidate month support differs")
    return panel, evaluation, evaluation_labels, len(rows)


def verify(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    files = check_manifest(root)
    panel, evaluation, labels, row_count = audit_data(root)
    predictions = load(root / "results/predictions.json.gz")
    with (root / "results/training_pools.csv").open() as stream:
        pool_rows = list(csv.DictReader(stream))
    metadata = {(r["target_month"], r["pool"]): r for r in pool_rows}
    expected_pools = {(m, p) for m in VALIDATION + TEST for p in ("expanded", "candidate")}
    if len(metadata) != len(pool_rows) or set(metadata) != expected_pools:
        raise ValueError("TRAIN pool table support differs")
    validation, test = [], []
    pools = 0
    for month in VALIDATION + TEST:
        for pool in ("expanded", "candidate"):
            info = metadata[month, pool]
            train = [r for r in panel["rows"] if r["target_month"] < month and panel["labels"][r["key"]]["known"]
                     and (pool == "expanded" or r["evaluation_eligible_asof"])]
            if pool == "candidate":
                train.sort(key=lambda r:(r["target_month"],r["article"],r["key"]))
            if int(info["rows"]) != len(train) or int(info["events"]) != sum(panel["labels"][r["key"]]["event"] for r in train):
                raise ValueError("TRAIN membership counts differ")
            if info["first_train_target"] != min(r["target_month"] for r in train):
                raise ValueError("TRAIN start boundary differs")
            if info["max_train_target"] != max(r["target_month"] for r in train) or digest([r["key"] for r in train]) != info["train_keys_sha256"]:
                raise ValueError("TRAIN membership or time boundary differs")
            pools += 1
        values = score_month(month,evaluation[month],labels,predictions[month])
        (validation if month in VALIDATION else test).extend(values)
    chosen = choose(validation)
    if chosen["model_id"] != load(root / "results/selection.json")["model_id"]:
        raise ValueError("Validation selection differs")
    for name, values in (("validation_monthly", validation), ("test_monthly",test),
                         ("validation_summary",summarize(validation)),("test_summary",summarize(test))):
        compare_rows(values,root / "results" / (name+".csv"))
    return dict(package_files_checked=files, source_rows_checked=row_count, train_pools_checked=pools,
                monthly_metrics_checked=len(validation)+len(test), selected_model=chosen["model_id"],
                new_model_fits=0, retrospective=True, actual_release_availability_certified=False)
