# WikiFlow: Initial ML Solution

> **Updated experiment:** [Recall@50 reselection and development refits](binary/evidence/recall_reselection/README.md) select SVD logistic (Recall@50 0.489). The results below retain the earlier NDCG-focused report for historical reference.

WikiFlow ranks eligible articles in the WikiProject Artificial Intelligence portfolio for a monthly human-review budget of 50 articles. This is one project: the shared data, preprocessing, original learners and evaluator remain in `src/wikiflow`; `binary/` contains the additional binary-ranking methods and report evidence.

- [Current report PDF](../WikiFlow_Report.pdf)
- [Binary study: reproduction instructions and dependencies](binary/README.md)
- [Report-to-code and result mapping](binary/REPORT_MAP.md)

## Current binary study

The report compares Volume and Momentum rules, Logistic regression (the original Log13 reference and an updated input configuration), histogram gradient boosting, LightGBM classification, a linear pairwise/rule blend, and binary LambdaRank. Text representations are input ablations of the same fixed LambdaRank learner, not additional model families.

The primary model metric is **binary NDCG@50**, with P@50, pooled Recall@50 and full-list AP as supporting metrics. Results use the same original-main development support: 24 months, 5,387 known article-months and 190 surge events. NDCG averages the 23 event months. Clean-data sensitivity is reported separately. These are development backtests, not independent test results.

## Main model results

All rows use the same 24-month original-main development split, 5,387 known article-months, 190 events and 1,200 review slots. NDCG@50 and AP average the 23 event months; Recall@50 pools hits and events across months. April 2025 was evaluated with zero events: P@50 is zero, while NDCG is undefined. Figure 2 shows that month at zero for plotting only.

| Algorithm | Features / configuration | NDCG@50 | P@50 | Pooled Recall@50 | AP |
|---|---|---:|---:|---:|---:|
| Volume | Current count | 0.1235 | 0.0392 | 0.2474 | 0.0615 |
| Momentum | Growth margin | 0.2412 | 0.0633 | 0.4000 | 0.1122 |
| Logistic | Log13 reference | 0.2651 | 0.0675 | 0.4263 | 0.1284 |
| Logistic | Log13 + text availability | 0.2741 | 0.0733 | 0.4632 | 0.1323 |
| HGB | Numeric + dynamics | 0.2408 | 0.0667 | 0.4211 | 0.1060 |
| LightGBM | Compact numeric | 0.2777 | 0.0650 | 0.4105 | 0.1365 |
| Linear pairwise | Compact numeric + rule blend | 0.2819 | 0.0675 | 0.4263 | 0.1417 |
| LambdaRank | Compact numeric, Rank50 | 0.2965 | 0.0733 | 0.4632 | 0.1458 |

Values match the report and [saved summary](binary/evidence/expected_summary.csv). The two Logistic rows are configurations of one algorithm family.

## Features and tuning

The compact numeric inputs capture traffic scale, historical variability, recent clickstream and pageview growth, and distance to the surge threshold. Log13 retains its original 13 inputs. Text availability and article-text representations are optional inputs, not models.

Search ranges, fixed parameters and actual historical selection metrics are documented separately in the report's tuning table, [recipes](binary/config/recipes.json) and [report map](binary/REPORT_MAP.md). The report's NDCG@50 evaluation does not relabel earlier P@5 or P@50 selection as NDCG tuning.

## Does article text improve ranking?

All rows use the same LambdaRank learner, parameters and training splits; only the added inputs change. Each text-vector variant also includes the same text-availability flag. These are separate comparisons against numeric-only, not sequential additions.

| Added information | NDCG@50 | Δ vs numeric-only |
|---|---:|---:|
| None (numeric only) | 0.2965 | +0.0000 |
| Text availability only (control) | 0.2929 | -0.0036 |
| Word patterns (TF-IDF / SVD) | 0.3086 | +0.0121 |
| Topic mixture (LDA) | 0.2676 | -0.0289 |
| Semantic meaning (MiniLM / PCA) | 0.2882 | -0.0083 |

Word-pattern features improve development NDCG over both numeric-only and the availability-only control; the topic and semantic representations do not. The previously selected numeric configuration remains the main model rather than being replaced by the highest development score.

## Reproduce the current report without training

After installing the dependencies in [binary/README.md](binary/README.md), run from the repository root:

```bash
cd wikiflow/binary
python reproduce.py
python replay_selection.py
python verify_cases.py
python -m unittest discover -s tests -v
```

The default replay checks saved predictions for **12 displayed configurations across 24 months (288 method-months), with zero model fits**. Selection replay checks four recorded selection decisions; it does not run a new search. The current checks do not establish full refit equivalence for every new model. Report regeneration and optional development-only fold fitting are documented in the binary study README.

## Legacy baseline details

[Historical baseline documentation](docs/legacy_baselines.md) preserves the original Volume/Log13/Ridge12 results, acquisition commands and source records. Its 48-fit verification covers only Log13 and Ridge12, not the new model families.
