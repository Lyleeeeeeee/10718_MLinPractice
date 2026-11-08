# Methodology

## Forecast target and unchanged eligibility

The unit is an article–target-month opportunity. The outcome is the published monthly count of incoming Wikipedia internal-link Clickstream traffic, not total pageviews or article quality. Let D be the number of target-month days, c the preceding month's count, Dc its number of days, and m the median of the six preceding monthly daily rates. Robust dispersion is s = 1.4826 × median absolute deviation from m.

```text
U_label = max(200, D*m + max(100, 0.25*D*m, 2*D*s))
T = max(U_label, 1.10*c*D/Dc)
event = actual target count > T
```

Equality is not an event. Missing outcomes remain unknown and are excluded **before ranking**, not converted to negatives. All training labels and threshold-based learning/rule features use this label definition.

Eligibility is separate and retains the original historical cold bands:

```text
U_eligibility = max(200, D*m + max(100, 0.50*D*m, 3*D*s))
```

The preceding daily rate must not exceed the target cold band; the preceding count must also pass the analogous cold band computed from its own preceding six months. Existing catalogue exclusions, complete metadata and revision coverage, count floor of 100, maximum absolute nonminor edit size below 500 and cumulative absolute nonminor edits below 1,000 remain fixed. Loosening the outcome does not expand the candidate pool.

## Rolling design and exposure

| Stage | Target months | Candidate opportunities | Known | Events | Top50 slots |
|---|---|---:|---:|---:|---:|
| Validation | September 2024–April 2026 | 4,282 | 4,281 | 327 | 1,000 |
| Retrospective test | May–October 2026 | 1,639 | 1,639 | 149 | 300 |

Training targets start October 2023. A fold uses only known labels from earlier target months. September 2024 trains through August 2024. May 2026 trains through April; June trains through May, continuing through October. Earlier test labels therefore enter later folds. Hyperparameters and selection remain fixed.

Each row uses pre-target traffic and its own historical text revision. Imputation, scaling, TF-IDF/SVD and MiniLM PCA are fitted only on the fold's training data. Text transforms use distinct usable training snapshots and are shared where recipes have the same training pool.

All six test months were previously exposed. These temporal checks do not establish a new independent holdout or prove real-time deployability: actual historical first-publication timestamps and release latency are uncertified. Repeated articles and months are dependent observations.

## Fixed systems

| Family | Training pool | Numeric dimensions | Semantic dimensions | Objective / rule |
|---|---|---:|---:|---|
| Logistic | Expanded quality-complete cold rows | 13 | 24 | Unweighted binary logistic loss; L2 0.01 |
| LambdaRank | Eligible historical candidate rows | 5 | 16 | Monthly groups; binary relevance; Top50 truncation; threshold-margin offset |
| LightGBM binary | Eligible historical candidate rows | 5 | 16 | Binary loss; equal total weight per month |
| Pairwise + Traffic Rule | Expanded quality-complete cold rows | 5 | 16 | Monthly weighted positive–negative comparisons; L2 1; fixed 0.5 percentile blend with threshold rule |

Each family has numeric, SVD8 + age and chunked MiniLM8 + age configurations. Logistic uses twelve traffic/pageview/missingness inputs plus an explicit threshold margin. The other families use threshold margin, clickstream growth, pageview growth, traffic level and relative spread. Momentum remains a standalone baseline; its score is not an extra learned-model feature.

Semantic arms append eleven inputs: qualified visible-history log-age, its missingness indicator, a text-availability indicator and eight text dimensions. Age and text are added together, so their separate effects cannot be identified. SVD and MiniLM comparisons hold the age/availability additions and projected dimension fixed.

Both tree families use 100 trees, seven leaves, depth three, learning rate 0.03, L2 10 and a nonnegative first-feature constraint. Pairwise has a nonnegative first coefficient. Training pools, numeric features, objectives and weights differ between recipes, so results are not a causal model-only comparison.

## Text and age

The source is a pre-target historical revision: cleaned lead text capped at 4,000 characters plus literal categories, not the full article body. SVD fits TF-IDF and eight-dimensional truncated SVD on training snapshots. Chunked MiniLM uses fixed 384-dimensional document vectors, reduced to eight dimensions by training-only PCA. Missing text has a zero vector and an explicit marker.

MiniLM originally processed nonoverlapping chunks of at most 254 content tokens, with two special tokens. Each chunk vector was normalized, averaged with content-token-count weights, then the document vector was normalized. The quick bundled reproduction starts with these fixed vectors. The full pipeline also supplies explicit pinned-checkpoint download/verification and encoder inference; these are separate opt-in stages, not part of a normal refit.

Age is time since the earliest qualifying visible revision, with uncertainty checks and missingness, not certified original creation. Deleted, hidden, moved or incomplete history can limit interpretation.

## Selection and metrics

Select by pooled validation Recall@50, then mean monthly validation NDCG@50, then lexical model ID. The global selection is **Pairwise + Traffic Rule / SVD8 + age**. Within-family selections are Logistic/SVD, LambdaRank/MiniLM, LightGBM/SVD and Pairwise/SVD. Pairwise wins validation within each representation. Clear public IDs do not alter these fixed recipes or selections.

Rank by score descending, preceding count descending, then page ID ascending. Recall pools hits over events; precision pools hits over recommendation slots. Reported NDCG is the equal-month mean of binary NDCG@50. Macro Recall and sum(DCG)/sum(IDCG), when supplied in CSVs, are different summaries. Random is an analytic uniform Top50 expectation, not a sampled ranking; fractional hits must not be interpreted as counts of actual recommendations.

| Configuration | Validation hits /327 | Validation Recall | Test hits /149 | Test Recall | Test Precision | Test mean NDCG |
|---|---:|---:|---:|---:|---:|---:|
| Logistic / numeric | 140 | 42.81% | 60 | 40.27% | 20.00% | 0.4093 |
| Logistic / SVD8 + age | 152 | 46.48% | 70 | 46.98% | 23.33% | 0.4537 |
| Logistic / MiniLM8 + age | 146 | 44.65% | 72 | 48.32% | 24.00% | 0.4456 |
| LambdaRank / numeric | 142 | 43.43% | 61 | 40.94% | 20.33% | 0.3794 |
| LambdaRank / SVD8 + age | 145 | 44.34% | 61 | 40.94% | 20.33% | 0.3711 |
| LambdaRank / MiniLM8 + age | 146 | 44.65% | 66 | 44.30% | 22.00% | 0.3976 |
| LightGBM / numeric | 130 | 39.76% | 59 | 39.60% | 19.67% | 0.3564 |
| LightGBM / SVD8 + age | 142 | 43.43% | 64 | 42.95% | 21.33% | 0.3822 |
| LightGBM / MiniLM8 + age | 135 | 41.28% | 68 | 45.64% | 22.67% | 0.3881 |
| Pairwise + Traffic Rule / numeric | 149 | 45.57% | 62 | 41.61% | 20.67% | 0.4061 |
| **Pairwise + Traffic Rule / SVD8 + age** | **159** | **48.62%** | **71** | **47.65%** | **23.67%** | **0.4422** |
| Pairwise + Traffic Rule / MiniLM8 + age | 155 | 47.40% | 70 | 46.98% | 23.33% | 0.4378 |
| Current volume | 98 | 29.97% | 38 | 25.50% | 12.67% | 0.2043 |
| Momentum | 138 | 42.20% | 56 | 37.58% | 18.67% | 0.3401 |
| Analytic random | 78.88777824429863 | 24.12% | 27.328 (rounded) | 18.34% | 9.11% | 0.1443 |

Logistic/MiniLM's 72 test hits are the highest observed result, not a replacement for the validation-selected system. No significance claim is made from six inspected months.

| Test month | Known candidates | Events | Selected Pairwise/SVD hits | Momentum hits |
|---|---:|---:|---:|---:|
| May 2026 | 260 | 16 | 11 | 6 |
| June 2026 | 265 | 15 | 6 | 6 |
| July 2026 | 286 | 16 | 8 | 4 |
| August 2026 | 295 | 16 | 9 | 8 |
| September 2026 | 275 | 63 | 29 | 28 |
| October 2026 | 258 | 23 | 8 | 4 |

September contributes 63/149 events (42.3%) and exceeds the 50-slot budget. Even perfect rankings can recover at most 136/149 pooled events. Retain both equal-month and pooled summaries. Selected Pairwise/SVD has macro Recall 0.4930 and DCG-weighted NDCG 0.4683, distinct from pooled Recall 0.4765 and mean NDCG 0.4422.

## Limits

The moderate-growth label is a task definition, not pilot-calibrated proof of volunteer usefulness. Internal-link traffic is an attention proxy. Catalogue coverage, title matching, visible history and retrospective exposure constrain generalization. Historical first-publication times remain uncertified.
