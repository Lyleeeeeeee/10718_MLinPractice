# WikiFlow

WikiFlow produces a monthly shortlist of 50 WikiProject AI articles for human review. This compact tree contains the agreed **binary surge prediction** study: seven ranked methods, a random-order expectation, and a text-availability control.

Start here, then read [the method and features](docs/method.md). [models.json](models.json) is the runnable catalogue; retained historical model IDs remain accepted.

## Run

Use Python 3.13.5 and the pinned dependencies:

    python -m pip install -r requirements.txt
    python run.py replay
    python run.py verify
    python run.py test

Replay checks all 192 retained method-month records with **zero model fits**, no downloads and no holdout access. Export the two tables to a new directory:

    python run.py replay --output runs/replay-01

List names, inputs and historical IDs:

    python run.py models
    python run.py models --model binary_svd_0.01

## Main results

All rows use September 2024–August 2026: 24 months, 5,387 known candidate article-months, 190 surges and 1,200 review slots. Primary **Recall@50 is pooled**. NDCG averages the 23 event months; Precision includes all 24.

| Method | Hits | Recall@50 | NDCG@50 | Precision@50 |
|---|---:|---:|---:|---:|
| Volume | 47 | .2474 | .1235 | .0392 |
| Momentum | 76 | .4000 | .2412 | .0633 |
| Traffic Logistic | 81 | .4263 | .2651 | .0675 |
| **Traffic + Text Logistic** | **93** | **.4895** | .2738 | **.0775** |
| Numeric LambdaRank | 88 | .4632 | **.2965** | .0733 |
| Boosted-tree Classifier (LightGBM) | 78 | .4105 | .2777 | .0650 |
| Linear Pairwise Ranker + Rule Blend | 80 | .4211 | .2940 | .0667 |
| Random reference, expected | 44.08 | .2320 | — | .0367 |

Traffic + Text Logistic is the current Recall-selected development candidate, not an independently validated deployment winner. Its 12-event gain over Traffic Logistic is 14.8% relative Recall, below the original 20% bar; the prior paired month-bootstrap interval includes zero. Numeric LambdaRank has higher NDCG among the numeric main-table models.

### Text availability control

These rows share Logistic regression, the original 13 traffic/pageview inputs and λ=.01. “Flag only” is the only **addition**, not the only predictor.

| Added input | Hits | Recall@50 | NDCG@50 |
|---|---:|---:|---:|
| None | 81 | .4263 | .2651 |
| Text-missing flag | 88 | .4632 | .2741 |
| TF-IDF/SVD8 + the same flag | 93 | .4895 | .2738 |

The five-event content gain beyond the flag is uncertain. Cleanup does not repair text acquisition, add age controls, change historical identities or run TA-feedback experiments.

## Layout

- run.py: one entry point for replay, verification, tests and explicit fitting.
- src/wikiflow/: contracts, features, fixed fitters and source preparation.
- data/: unchanged frozen panel/text, catalogue and attribution.
- results/: retained scores, metrics, selection evidence and provenance.
- tests/: offline contracts and reproduction checks.
- docs/: method, selection and history recovery.

There is no active continuous learner, exploratory batch runner, duplicate report build or archive folder. Auxiliary fields such as gR remain in the unchanged frozen panel and shared validators.

## Explicit refit

Refitting is optional, uses a fixed existing recipe, and is restricted to the documented January 2024–August 2026 development folds:

    python run.py fit --model traffic_logistic --month 2024-09 --allow-fit --output runs/logistic-2024-09.json

The command also accepts retained legacy IDs. Tree fits require the pinned LightGBM and its OpenMP runtime. Random is an expectation and has no fitted model. No new search is implemented.

Cleanup validation covers saved-score replay and unit contracts; it does not claim a full historical refit of all relocated fitters. See [selection provenance](docs/selection.md) and [recovery](docs/recovery.md). Existing raw caches and local uncommitted work outside this isolated checkout were not deleted.

