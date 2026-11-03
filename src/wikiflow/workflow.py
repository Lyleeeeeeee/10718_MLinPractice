"""Replay and verify the retained binary study from frozen scores; no model fitting."""
import csv
import hashlib
import math
from pathlib import Path
import numpy as np
from .core import FORECASTS, aggregate, days, historical_bound, rank_metric
from .data import ROOT, load, save, validate_panel, write_csv
from .features import momentum_score
from .registry import MODELS, ablation_models, main_models


def verify_inputs():
    provenance = load(ROOT / "data/provenance.json")
    checks = {name: entry["sha256"] for name, entry in provenance["unchanged_files"].items()}
    checks.update(load(ROOT / "results/provenance.json")["files_sha256"])
    for name, expected in checks.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen input hash changed: {name}")
    return len(checks)


def load_study():
    panel = load(ROOT / "data/panel.json.gz")
    rowmap = validate_panel(panel)
    scores = load(ROOT / "results/predictions.json.gz")
    expected_ids = {model["id"] for model in MODELS if model["family"] != "reference"}
    if set(scores) != expected_ids:
        raise ValueError("Saved model catalogue differs from the retained binary study")
    for name, values in scores.items():
        if set(values) != set(panel["evaluation_keys"]) or not all(math.isfinite(x) for x in values.values()):
            raise ValueError(f"Invalid prediction support: {name}")
    return panel, rowmap, scores


def verify_selections():
    with (ROOT / "results/early_trials.csv").open() as handle:
        trials = list(csv.DictReader(handle))
    frozen = load(ROOT / "results/selection.json")
    for choice in frozen["current_choices"]:
        candidates = {}
        for row in trials:
            if row["search"] == choice["search"]:
                candidates.setdefault(row["legacy_id"], []).append(row)
        def metrics(rows):
            events = sum(int(row["events"]) for row in rows)
            recall = sum(int(row["hits"]) for row in rows) / events
            ndcg = np.mean([float(row["ndcg_at_50"]) for row in rows if row["ndcg_at_50"]])
            return recall, float(ndcg)
        selected = candidates[choice["legacy_id"]]
        if sum(int(row["events"]) for row in selected) != choice["early_events"]:
            raise ValueError("Early event support changed")
        if sum(int(row["hits"]) for row in selected) != choice["early_hits"]:
            raise ValueError("Recorded early hit count changed")
        recall, ndcg = metrics(selected)
        for rows in candidates.values():
            other_recall, other_ndcg = metrics(rows)
            if other_recall > recall + 1e-12 or (
                abs(other_recall - recall) <= 1e-12 and other_ndcg > ndcg + 1e-12):
                raise ValueError(f"Frozen Recall selection does not replay: {choice['model_id']}")
    return len(frozen["current_choices"])


def verify_cases(panel, rowmap):
    for case in load(ROOT / "results/cases.json")["cases"]:
        key = f"{case['target_month']}:{case['pageid']}"
        row, label = rowmap[key], panel["labels"][key]
        rates = [item["incoming"] / days(item["month"]) for item in case["history"][-7:-1]]
        threshold = max(historical_bound(rates, case["target_days"]),
                        1.25 * case["previous_incoming"] * case["target_days"] / case["previous_days"])
        if (key not in panel["evaluation_keys"] or row["article"] != case["article"]
            or not math.isclose(threshold, label["T"], rel_tol=1e-12)
            or label["actual"] != case["target_incoming"] or label["event"] != case["y"]):
            raise ValueError("Historical report case no longer matches the frozen panel")
    return 2


def replay(output=None):
    if output is not None:
        output = Path(output).resolve()
        if output.exists():
            raise ValueError("Output already exists; choose a fresh directory")
        if any(output.is_relative_to(ROOT / name) for name in ("src", "data", "results", "tests", ".git")):
            raise ValueError("Output must not be placed inside source or frozen-input directories")
    hash_checks = verify_inputs()
    panel, rowmap, predictions = load_study()
    candidates = [rowmap[key] for key in panel["evaluation_keys"]]
    monthly, summary = [], []
    with (ROOT / "results/summary.csv").open() as handle:
        expected = {row["model_id"]: row for row in csv.DictReader(handle)}
    for spec in MODELS:
        if spec["family"] == "reference":
            continue
        name = spec["id"]
        scores, records = predictions[name], []
        if spec["family"] == "rule":
            for row in candidates:
                value = float(row["current_count"]) if name == "volume" else momentum_score(row)
                if abs(value - scores[row["key"]]) > 1e-12:
                    raise ValueError(f"Baseline formula changed: {name}")
        for month in FORECASTS:
            rows = [row for row in candidates if row["target_month"] == month]
            metric = rank_metric(rows, panel["labels"], {row["key"]: scores[row["key"]] for row in rows}, 50)
            records.append(metric)
            monthly.append(dict(model_id=name, month=month, candidates=metric["N"],
                events=metric["events"], hits=metric["TP"], selected_slots=metric["actual_K"],
                recall_at_50=metric["Recall"], ndcg_at_50=metric["binary_NDCG"],
                precision_at_50=metric["Precision"]))
        result = aggregate(records)
        if result["events"] != 190 or result["selected"] != 1200 or sum(row["N"] for row in records) != 5387:
            raise ValueError("Shared evaluation support changed")
        row = dict(model_id=name, model=spec["name"], role=spec["role"],
            reference_type="saved_prediction", hits=result["TP"], recall_at_50=result["micro_Recall"],
            ndcg_at_50=result["macro_binary_NDCG"], precision_at_50=result["macro_Precision"])
        for field in ("hits", "recall_at_50", "ndcg_at_50", "precision_at_50"):
            if not math.isclose(float(row[field]), float(expected[name][field]), abs_tol=1e-12):
                raise ValueError(f"Result changed: {name}, {field}")
        summary.append(row)
    # Independent row-by-row check, including the zero-event month and its undefined NDCG.
    with (ROOT / "results/monthly.csv").open() as handle:
        saved_monthly = list(csv.DictReader(handle))
    if len(saved_monthly) != len(monthly):
        raise ValueError("Monthly evidence support changed")
    for actual, saved in zip(monthly, saved_monthly):
        for key, value in actual.items():
            if value is None:
                if saved[key] != "":
                    raise ValueError("Undefined monthly metric was changed")
            elif isinstance(value, (int, float)):
                if not math.isclose(value, float(saved[key]), abs_tol=1e-12):
                    raise ValueError("Monthly metric changed")
            elif value != saved[key]:
                raise ValueError("Monthly identity changed")
    known_by_month = [[row for row in candidates if row["target_month"] == month
                       and panel["labels"][row["key"]]["known"]] for month in FORECASTS]
    expected_hits = math.fsum(
        min(50, len(rows)) * sum(panel["labels"][row["key"]]["event"] for row in rows) / len(rows)
        for rows in known_by_month)
    summary.append(dict(model_id="random_expected", model="Random reference (expected)", role="main",
        reference_type="expected_random_order", hits=expected_hits, recall_at_50=expected_hits / 190,
        ndcg_at_50=None, precision_at_50=expected_hits / 1200))
    index = {row["model_id"]: row for row in summary}
    main = [index[model["id"]] for model in main_models()]
    ablation = [index[model["id"]] for model in ablation_models()]
    verification = dict(status="PASS",model_fits=0, holdout_read=False, months=24,
        known_candidate_rows=5387, unknown_candidate_rows=1, events=190, review_slots=1200,
        scored_models=len(predictions), method_months=len(monthly), input_hashes_verified=hash_checks,
        selection_records_verified=verify_selections(), cases_verified=verify_cases(panel, rowmap))
    if output is not None:
        output.mkdir(parents=True)
        write_csv(output / "main_results.csv", main)
        write_csv(output / "text_ablation.csv", ablation)
        write_csv(output / "monthly.csv", monthly)
        save(output / "verification.json", verification)
    return main, ablation, verification
