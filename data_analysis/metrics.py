"""Binary Top50 metrics, analytic random expectation and fixed splits."""
import calendar
import math

FAMILIES = ["logistic", "lambdarank", "lightgbm", "pairwise_rule"]

LEARNED = [family + "__" + kind for family in FAMILIES for kind in ["numeric", "svd", "minilm"]]

ORDER = LEARNED + ["current_volume", "momentum", "random_expected"]

VALIDATION = [f"{year}-{month:02d}" for year, months in [(2024, range(9, 13)), (2025, range(1, 13)), (2026, range(1, 5))] for month in months]

TEST = [f"2026-{month:02d}" for month in range(5, 11)]

def eligible(row):
    activity = row["editor_activity_asof"]
    days = lambda month: calendar.monthrange(*map(int, month.split("-")))[1]
    return bool(row["pageid"] not in {79745475, 1164, 72417803} and row.get("metadata_quality_complete", True)
                and activity["coverage_complete"] and row["current_count"] >= 100
                and activity["max_abs_nonminor"] is not None and activity["max_abs_nonminor"] < 500
                and activity["cumulative_abs_nonminor"] is not None and activity["cumulative_abs_nonminor"] < 1000
                and row["current_count"] / days(row["cutoff_month"]) <= row["U_count"] / days(row["target_month"])
                and row["current_count"] <= row["current_precurrent_U"])

def score_month(month, rows, labels, predictions):
    assert set(predictions) == set(ORDER) - {"random_expected"}
    keys = {r["key"] for r in rows}
    known = [r for r in rows if labels[r["key"]]["known"]]
    n = len(known)
    events = sum(labels[r["key"]]["event"] for r in known)
    k = min(50, n)
    idcg = math.fsum(1 / math.log2(i + 2) for i in range(min(k, events)))
    result = []
    for name in ORDER:
        if name == "random_expected":
            hits = k * events / n
            dcg = events / n * math.fsum(1 / math.log2(i + 2) for i in range(k))
        else:
            scores = predictions[name]
            assert set(scores) == keys and all(math.isfinite(v) for v in scores.values())
            ranked = sorted(known, key=lambda r: (-scores[r["key"]], -r["current_count"], r["pageid"]))
            top = ranked[:k]
            hits = sum(labels[r["key"]]["event"] for r in top)
            dcg = math.fsum(labels[r["key"]]["event"] / math.log2(i + 2) for i, r in enumerate(top))
        result.append(dict(month=month, model_id=name, candidates=len(rows), known=n, unknown=len(rows)-n,
                           events=events, selected=k, hits_at_50=hits, recall_at_50=hits/events if events else None,
                           precision_at_50=hits/k, ndcg_at_50=dcg/idcg if idcg else None, dcg_at_50=dcg, idcg_at_50=idcg))
    return result

def summarize(monthly):
    result = []
    for name in ORDER:
        rows = [r for r in monthly if r["model_id"] == name]
        events, hits, slots = (math.fsum(r[key] for r in rows) for key in ["events", "hits_at_50", "selected"])
        recall = [r["recall_at_50"] for r in rows if r["recall_at_50"] is not None]
        ndcg = [r["ndcg_at_50"] for r in rows if r["ndcg_at_50"] is not None]
        result.append(dict(model_id=name, months=len(rows), event_months=len(recall),
            candidates=sum(r["candidates"] for r in rows), known=sum(r["known"] for r in rows), unknown=sum(r["unknown"] for r in rows),
            events=int(events), selected=int(slots), hits_at_50=hits,
            pooled_recall_at_50=hits/events, macro_recall_at_50=math.fsum(recall)/len(recall),
            pooled_precision_at_50=hits/slots, macro_precision_at_50=math.fsum(r["precision_at_50"] for r in rows)/len(rows),
            mean_monthly_ndcg_at_50=math.fsum(ndcg)/len(ndcg),
            dcg_sum_over_idcg_sum=math.fsum(r["dcg_at_50"] for r in rows)/math.fsum(r["idcg_at_50"] for r in rows)))
    return result


def choose(validation_monthly):
    """Choose only on validation months, never on retrospective test metrics."""
    if not validation_monthly or any(row["month"] not in VALIDATION for row in validation_monthly):
        raise ValueError("Selection requires validation-only metric rows")
    summary = summarize(validation_monthly)
    return sorted([row for row in summary if row["model_id"] in LEARNED],
                  key=lambda row: (-row["pooled_recall_at_50"], -row["mean_monthly_ndcg_at_50"], row["model_id"]))[0]
