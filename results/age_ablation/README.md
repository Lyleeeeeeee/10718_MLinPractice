# Age vs. text feature ablation

All three rows use **Pairwise + Traffic Rule** (L2 = 1, rule blend = 0.5), the same historical training and evaluation rows, and the same ranking and metrics. Only the model inputs change. The age block contains qualified log-days since the earliest visible revision and its missingness indicator. The SVD8 text block contains eight TF-IDF/SVD components and the text-availability indicator. See the [main methodology](../../docs/methodology.md) for the expanding-window and cutoff rules.

| Inputs | Validation pooled Recall@50 | May–Oct 2026 pooled Recall@50 |
|---|---:|---:|
| Numeric | 149/327 (45.57%) | 62/149 (41.61%) |
| Numeric + age | 149/327 (45.57%) | 63/149 (42.28%) |
| Numeric + age + SVD8 text block | 159/327 (48.62%) | 72/149 (48.32%) |

Age added no validation hits and one retrospective hit. Adding the SVD8 text block on top of age added ten validation and nine retrospective hits. This measures the incremental value of the **text block**, not text without age or the embeddings separately from text availability.

## Reproduce

From the repository root, with Python 3.13.5 and the pinned requirements:

```sh
python run.py ablate-age --output runs/age-ablation
```

- [Validation summary](validation_summary.csv) · [Monthly validation](validation_monthly.csv)
- [Retrospective test summary](test_summary.csv) · [Monthly test](test_monthly.csv)
- [Prediction scores](predictions.json.gz), keyed by month, configuration, and article

All three arms were freshly fit in the same Linux x86-64 runtime. The two existing arms were checked against the [archived main predictions](../predictions.json.gz): all 52 month/configuration candidate sets matched; 27 score maps matched exactly (26 numeric, one SVD). The SVD arm found 72 retrospective events in this rerun versus 71 in the archived main run. The original benchmark and its validation selection are unchanged; the table above uses only scores from the same rerun. May–October results are retrospective, not an independent holdout.
