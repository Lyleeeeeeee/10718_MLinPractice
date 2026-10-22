"""Static historical documents with strictly TRAIN-fitted representations."""
import copy
import hashlib
import json
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def build_documents(rows, records):
    docs = {}
    for pid in sorted(set((r['pageid'] for r in rows))):
        rec = records.get(pid, {})
        valid = rec.get('status') == 'ok' and bool(rec.get('intro'))
        if valid and (not rec.get('revision_id') or not rec.get('timestamp') or rec['timestamp'] > '2023-09-30T23:59:59Z'):
            raise ValueError('Text provenance missing or after cutoff')
        cats = ' '.join(rec.get('categories', [])) if valid else ''
        text = rec['intro'] + ' Categories: ' + cats if valid else ''
        docs[pid] = dict(text=text, categories=cats, missing=not valid)
    return docs

def permute_documents(docs, seed):
    ids = sorted((pid for pid, v in docs.items() if not v['missing']))
    permutation = np.random.default_rng(seed).permutation(ids)
    out = copy.deepcopy(docs)
    for pid, source in zip(ids, permutation):
        out[pid] = copy.deepcopy(docs[int(source)])
    return out

def representation(train_pageids, docs, embeddings, kind):
    train = sorted(set((pid for pid in train_pageids if not docs[pid]['missing'])))
    allids = sorted(docs)
    if len(train) < 10:
        raise ValueError('Insufficient distinct historical TRAIN documents')
    info = dict(kind=kind, fit_pageids=train, fit_pageids_sha256=hashlib.sha256(json.dumps(train).encode()).hexdigest())
    if kind == 'missing':
        return ({pid: np.zeros(0) for pid in allids}, info)
    if kind == 'embedding':
        model = PCA(n_components=8, svd_solver='full').fit(np.asarray([embeddings[pid] for pid in train]))
        z = model.transform(np.asarray([embeddings[pid] for pid in allids]))
        info['explained_variance_ratio'] = model.explained_variance_ratio_.tolist()
    else:
        field = 'categories' if kind == 'category' else 'text'
        cls = CountVectorizer if kind == 'lda' else TfidfVectorizer
        v = cls(lowercase=True, stop_words='english', min_df=2, max_df=0.95, max_features=3000)
        x = v.fit_transform([docs[pid][field] for pid in train])
        allx = v.transform([docs[pid][field] for pid in allids])
        info['vocabulary'] = sorted(v.vocabulary_)
        if x.shape[1] < 8:
            raise ValueError('Insufficient TRAIN vocabulary for 8 dimensions')
        if kind == 'lda':
            model = LatentDirichletAllocation(n_components=8, max_iter=20, learning_method='batch', random_state=718, n_jobs=1)
        else:
            model = TruncatedSVD(n_components=8, n_iter=7, random_state=718)
        model.fit(x)
        z = model.transform(allx)
    vectors = {pid: np.zeros(8) if docs[pid]['missing'] else z[i] for i, pid in enumerate(allids)}
    if not all((np.isfinite(v).all() for v in vectors.values())):
        raise ValueError('Nonfinite text representation')
    return (vectors, info)
