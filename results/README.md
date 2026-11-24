# Saved results

## Main benchmark

The validation stage (Sep 2024–Apr 2026) selects a fixed model. May–Oct 2026 is a separate retrospective reporting period. See the [methodology](../docs/methodology.md) for definitions and the split protocol.

| File | Contents |
|---|---|
| [Validation summary](validation_summary.csv), [test summary](test_summary.csv) | Aggregate metrics by configuration, plus baselines |
| [Validation monthly](validation_monthly.csv), [test monthly](test_monthly.csv) | Per-month hits and ranking metrics |
| [Selection](selection.json) | Model selected on validation only |
| [Monthly support](monthly_support.csv), [training pools](training_pools.csv) | Evaluation counts and audited historical training membership |
| [Predictions](predictions.json.gz) | Exact scores keyed by month, model and article |

## Supplementary experiment

[Age vs. text ablation](age_ablation/README.md) holds the selected model family and evaluation protocol fixed. It is not part of the 12-configuration selection.
