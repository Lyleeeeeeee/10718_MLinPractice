"""TRAIN-only SVD/PCA and qualified visible-history age."""
import datetime as dt
import math
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

def represent(train, records, embeddings, kind):
    documents = {}
    for record in records.values():
        previous = documents.setdefault(record["snapshot_id"], record)
        if any(previous[field] != record[field] for field in ("text", "missing")):
            raise ValueError("Conflicting content for one snapshot")
    ids = sorted(documents)
    fit_ids = sorted({records[row["key"]]["snapshot_id"] for row in train
                      if not records[row["key"]]["missing"]})
    if len(fit_ids) < 10:
        raise ValueError("Insufficient usable TRAIN snapshots")
    if kind == "minilm":
        vectorizer = None
        model = PCA(n_components=8, svd_solver="full")
        model.fit(np.array([embeddings[key] for key in fit_ids]))
        projected = model.transform(np.array([embeddings[key] for key in ids]))
    elif kind == "svd":
        vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", min_df=2,
                                    max_df=.95, max_features=3000)
        matrix = vectorizer.fit_transform([documents[key]["text"] for key in fit_ids])
        model = TruncatedSVD(n_components=8, n_iter=7, random_state=718)
        model.fit(matrix)
        projected = model.transform(vectorizer.transform([documents[key]["text"] for key in ids]))
    else:
        raise ValueError("Only fixed SVD or chunked MiniLM/PCA representations")
    lookup = {key: np.zeros(8) if documents[key]["missing"] else projected[i]
              for i, key in enumerate(ids)}
    if not all(np.isfinite(value).all() for value in lookup.values()):
        raise ValueError("Nonfinite historical representation")
    return {key: lookup[record["snapshot_id"]] for key, record in records.items()}, dict(
        kind=kind, vectorizer=vectorizer, model=model, fit_snapshot_ids=fit_ids)
def age_features(evidence, cutoff):
    if evidence["status"] not in ("earliest_visible_root", "multiple_roots_uncertain"):
        return [None, 1.]
    if evidence["status"] == "multiple_roots_uncertain" and not evidence.get("observed_roots"):
        return [None, 1.]
    stamp = evidence["earliest_visible_at"]
    if not stamp or stamp >= cutoff:
        return [None, 1.]
    other_roots = [r for r in evidence.get("observed_roots", []) if r["timestamp"] < cutoff
                   and r["revid"] != evidence["earliest_visible_revision"]]
    if other_roots:
        return [None, 1.]
    delta = dt.datetime.fromisoformat(cutoff.replace("Z", "+00:00")) - dt.datetime.fromisoformat(
        stamp.replace("Z", "+00:00"))
    return [math.log1p(delta.total_seconds() / 86400), 0.]
