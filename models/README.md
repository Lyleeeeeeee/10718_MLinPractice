# Fixed model recipes

These public IDs name the final fixed systems. Every learned family is evaluated with numeric, SVD8 + age and chunked MiniLM8 + age features.

| ID | Recipe | Training pool | Numeric / semantic dimensions |
|---|---|---|---:|
| `logistic` | L2 logistic regression, penalty 0.01, unweighted rows | Expanded quality-complete cold rows | 13 / 24 |
| `lambdarank` | LambdaRank, monthly groups, binary relevance, Top50 truncation, threshold-margin offset | Eligible historical candidate rows | 5 / 16 |
| `lightgbm` | Binary LightGBM, equal total training weight per month | Eligible historical candidate rows | 5 / 16 |
| `pairwise_rule` | Monthly weighted pairwise logistic comparisons, L2 1, nonnegative first coefficient, 0.5 percentile blend with threshold rule | Expanded quality-complete cold rows | 5 / 16 |

Both tree recipes use 100 trees, seven leaves, depth three, learning rate 0.03, L2 10 and a nonnegative first-feature constraint. Imputation and scaling use training rows only. Different objectives, weights, pools and inputs make these complete-system comparisons, not controlled comparisons of learning objectives alone.

The compact five inputs are threshold margin, clickstream growth, pageview growth, traffic level and relative spread. Logistic uses twelve traffic/pageview/missingness columns plus the explicit threshold margin. The margin is `log1p(preceding count × target days / preceding days) - log1p(T)`, where T is the final loose event threshold. Semantic arms append eleven age/availability/text inputs. **The standalone momentum score is not an added learned-model feature.**

## Baselines

- `current_volume`: rank the preceding full-month internal-link count.
- `momentum`: extrapolate recent daily-rate growth and score the forecast against T.
- `random_expected`: analytic expected uniform Top50 metrics; no model fit or random sampled ranking.

All systems share evaluation candidates, known outcomes and a 50-slot budget. Unknowns are removed before ranking; ties use score, preceding count and page ID deterministically.

## Refit, not model deserialization

Run `python run.py refit --output runs/refit` with a fresh output directory to perform 312 supervised fits and 104 training-only text transforms. Reference predictions are in `results/predictions.json.gz`; split sizes are in `results/training_pools.csv`. The package deliberately contains no serialized fitted model objects, booster dumps or cached feature matrices.

Global validation selection is Pairwise + Traffic Rule / SVD8 + age. Logistic / MiniLM has the highest observed retrospective test hit count but is not selected on that basis. See [methodology](../docs/methodology.md) for the full comparison and exposure limits.

## Numeric feature definitions

Let r0 and r1 be the current and previous full-month incoming-link daily rates; m and s are the median and robust spread of the last six daily rates. Let p = r0 × target-month days.

| Logistic column | Definition |
|---|---|
| Current clickstream rate | log1p(r0) |
| Previous clickstream rate | log1p(r1) |
| Three-month mean | log1p(mean of the last three daily rates) |
| Six-month median | log1p(m) |
| Six-month slope | Least-squares daily-rate slope across six months |
| Seasonal clickstream rate | log1p(rate from the corresponding month one year earlier); current-rate fallback when absent |
| Seasonal missing | Indicator for the fallback above |
| Pageview level | log1p(mean daily pageviews in the preceding month); missing if incomplete |
| Pageview growth | log1p(last 14 days' mean) minus log1p(prior 14 days' mean); missing if incomplete |
| Pageviews incomplete | Indicator for incomplete preceding-month pageviews |
| Robust traffic spread | log1p(s) |
| Historical threshold margin | log1p(p) minus log1p(U_label) |
| Surge threshold margin | log1p(p) minus log1p(T) |

The compact five are surge threshold margin; log1p(r0) minus log1p(r1); the same pageview growth; log1p(r0); and log1p(s) minus log1p(m). They are not the separate momentum forecast score. All columns use TRAIN-only median filling and population standardization before learning.
