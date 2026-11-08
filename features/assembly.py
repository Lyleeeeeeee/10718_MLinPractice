"""Approved numeric inputs and the separate momentum baseline."""
import calendar
import math
import statistics

def days(month):
    return calendar.monthrange(*map(int, month.split("-")))[1]

def shift(month, offset):
    year, number = map(int, month.split("-"))
    index = year * 12 + number - 1 + offset
    return f"{index // 12}-{index % 12 + 1:02d}"

def upper_bound(rates, target_days):
    median = statistics.median(rates)
    sigma = 1.4826 * statistics.median(abs(value - median) for value in rates)
    return max(200., target_days * median + max(100., .25 * target_days * median, 2 * target_days * sigma))

def explicit_margin(row):
    pace = row["current_count"] * days(row["target_month"]) / days(row["cutoff_month"])
    return math.log1p(pace) - math.log1p(row["effective_event_U_asof"])

def numeric_features(row, family):
    values = row["features"]
    if family == "logistic":
        return list(values) + [explicit_margin(row)]
    if family not in {"lambdarank", "lightgbm", "pairwise_rule"}:
        raise ValueError("Unknown fixed model family")
    return [explicit_margin(row), values[0] - values[1], values[8], values[0], values[10] - values[3]]
TRAFFIC_FEATURE_NAMES = [
    "current_clickstream_rate", "previous_clickstream_rate", "three_month_mean",
    "six_month_median", "six_month_slope", "seasonal_clickstream_rate",
    "seasonal_missing", "pageview_level", "pageview_growth", "pageviews_incomplete",
    "robust_traffic_spread", "historical_threshold_margin", "surge_threshold_margin"]

COMPACT_FEATURE_NAMES = [
    "surge_threshold_margin", "clickstream_growth", "pageview_growth",
    "traffic_level", "relative_traffic_spread"]

def momentum_score(row):
    current, previous = (math.expm1(row["features"][j]) for j in [0, 1])
    if previous <= 0:
        raise ValueError("Published positive previous rate required")
    return (math.log1p(current * current / previous * days(row["target_month"]))
            - math.log1p(row["effective_event_U_asof"]))


def make_label(actual, threshold):
    """Binary label with an explicit unknown sentinel and strict boundary."""
    if isinstance(threshold, bool) or not math.isfinite(threshold) or threshold < 200:
        raise ValueError("Threshold must be finite and at least 200")
    if actual is None:
        return dict(known=False, actual=None, T=threshold, event=None)
    if isinstance(actual, bool) or not isinstance(actual, int) or actual < 0:
        raise ValueError("Actual count must be a nonnegative integer or unknown")
    return dict(known=True, actual=actual, T=threshold, event=int(actual > threshold))
