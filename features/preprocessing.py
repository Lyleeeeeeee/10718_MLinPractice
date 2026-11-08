"""Median imputation and population scaling fitted only on TRAIN."""
import statistics
import numpy as np

def preprocess(train,candidates):
    if not train:
        raise ValueError('Nonempty mature TRAIN required')
    def values(rows):
        return [r['features'] for r in rows]
    train_values = values(train)
    n = len(train_values[0]);medians=[];means=[];scales=[]
    for j in range(n):
        observed = [v[j] for v in train_values if v[j] is not None]
        median = statistics.median(observed) if observed else 0.
        filled = [median if v[j] is None else v[j] for v in train_values]
        sd = statistics.pstdev(filled)
        medians.append(median);means.append(statistics.mean(filled));scales.append(sd if sd>=1e-12 else 1.)
    def transform(rows):
        arr = np.asarray([[medians[j] if x is None else x for j,x in enumerate(v)]
                          for v in values(rows)],float).reshape(-1,n)
        out = (arr-np.asarray(means))/np.asarray(scales)
        if not np.isfinite(out).all():
            raise ValueError('Nonfinite feature after TRAIN imputation')
        return out
    return transform(train),transform(candidates),dict(medians=medians,means=means,scales=scales)
