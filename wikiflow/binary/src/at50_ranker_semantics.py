"""Fixed numeric/semantic Rank50 kernels."""
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wikiflow.core import joint_margin, preprocess
from wikiflow.reproduce import digest
from rule_residual import prepared
KINDS = ['base', 'missing', 'svd', 'lda', 'intro']

def select_recipe(records):
    group = list(records)
    for f in ['P50', 'N50', 'R50']:
        best = max((r[f] for r in group))
        group = [r for r in group if best - r[f] <= 1e-12]
    return min(group, key=lambda r: KINDS.index(r['id'].removeprefix('rank50_')))

def fit_representation(train_pageids, docs, embeddings, kind):
    ids = sorted(docs)
    train = sorted({i for i in train_pageids if not docs[i]['missing']})
    if len(train) < 10:
        raise ValueError('Too few usable TRAIN documents')
    info = dict(kind=kind, fit_pageids=train, fit_pageids_sha256=digest(train))
    if kind == 'intro':
        model = PCA(n_components=8, svd_solver='full').fit(np.array([embeddings[i] for i in train]))
        z = model.transform(np.array([embeddings[i] for i in ids]))
        state = dict(model=model)
        info.update(components=model.components_.tolist(), mean=model.mean_.tolist(), explained_variance_ratio=model.explained_variance_ratio_.tolist())
    else:
        cls = CountVectorizer if kind == 'lda' else TfidfVectorizer
        vectorizer = cls(lowercase=True, stop_words='english', min_df=2, max_df=0.95, max_features=3000)
        x = vectorizer.fit_transform([docs[i]['text'] for i in train])
        allx = vectorizer.transform([docs[i]['text'] for i in ids])
        if x.shape[1] < 8:
            raise ValueError('Too few TRAIN tokens')
        model = LatentDirichletAllocation(n_components=8, max_iter=20, learning_method='batch', random_state=718, n_jobs=1) if kind == 'lda' else TruncatedSVD(n_components=8, n_iter=7, random_state=718)
        model.fit(x)
        z = model.transform(allx)
        state = dict(vectorizer=vectorizer, model=model)
        info.update(vocabulary=sorted(vectorizer.vocabulary_), components=model.components_.tolist())
    vectors = {i: np.zeros(8) if docs[i]['missing'] else z[j] for j, i in enumerate(ids)}
    assert all((np.isfinite(v).all() for v in vectors.values()))
    return (vectors, info, state)

def augmented_prepared(train, test, docs, vectors, kind):

    def tx(rr):
        result = []
        for r in rr:
            f = [joint_margin(r), r['features'][0] - r['features'][1], r['features'][8], r['features'][0], r['features'][10] - r['features'][3]]
            if kind != 'base':
                f += vectors[r['pageid']].tolist() + [float(docs[r['pageid']]['missing'])]
            result.append(dict(r, features=f))
        return result
    x, z, prep = preprocess(tx(train), tx(test))
    bx, bz, bprep = prepared(train, test)
    np.testing.assert_array_equal(x[:, :5], bx)
    np.testing.assert_array_equal(z[:, :5], bz)
    return (x, z, prep, bprep)
