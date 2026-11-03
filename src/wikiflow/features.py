"""The three retained feature groups; the original numerical mappings are unchanged."""
import math
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from .core import days, joint_margin

TRAFFIC_FEATURE_NAMES = [
    "current_clickstream_rate", "previous_clickstream_rate", "three_month_mean",
    "six_month_median", "six_month_slope", "seasonal_clickstream_rate",
    "seasonal_missing", "pageview_level", "pageview_growth", "pageviews_incomplete",
    "robust_traffic_spread", "historical_threshold_margin", "surge_threshold_margin"]
COMPACT_FEATURE_NAMES = [
    "surge_threshold_margin", "clickstream_growth", "pageview_growth",
    "traffic_level", "relative_traffic_spread"]


def traffic_features(row):
    """Original Log13: 12 frozen panel columns plus the joint-threshold margin."""
    return list(row["features"]) + [joint_margin(row)]


def compact_features(row):
    values = row["features"]
    return [joint_margin(row), values[0] - values[1], values[8],
            values[0], values[10] - values[3]]


def momentum_score(row):
    current, previous = (math.expm1(row["features"][j]) for j in [0, 1])
    if previous <= 0:
        raise ValueError("Published positive previous rate required")
    return (math.log1p(current * current / previous * days(row["target_month"]))
            - math.log1p(row["effective_event_U_asof"]))


def text_vectors(train_pageids, documents):
    """TF-IDF/SVD8 on unique usable TRAIN articles, with zero missing sentinels."""
    train = sorted({pageid for pageid in train_pageids
                    if not documents[pageid]["missing"]})
    pageids = sorted(documents)
    if len(train) < 10:
        raise ValueError("Insufficient distinct historical TRAIN documents")
    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english", min_df=2, max_df=.95,
        max_features=3000)
    x = vectorizer.fit_transform([documents[i]["text"] for i in train])
    all_x = vectorizer.transform([documents[i]["text"] for i in pageids])
    if x.shape[1] < 8:
        raise ValueError("Insufficient TRAIN vocabulary for 8 dimensions")
    projection = TruncatedSVD(n_components=8, n_iter=7, random_state=718)
    projection.fit(x)
    z = projection.transform(all_x)
    vectors = {pageid: np.zeros(8) if documents[pageid]["missing"] else z[j]
               for j, pageid in enumerate(pageids)}
    if not all(np.isfinite(vector).all() for vector in vectors.values()):
        raise ValueError("Nonfinite text representation")
    return vectors, {"fit_pageids": train, "vocabulary": sorted(vectorizer.vocabulary_)}
