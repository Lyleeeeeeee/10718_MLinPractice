# Features

- `numeric.py`: twelve pre-target traffic/pageview features and separate eligibility/event thresholds.
- `assembly.py`: recipe-specific inputs, event-threshold margin, labels and standalone momentum baseline.
- `preprocessing.py`: training-only median imputation and population scaling.

Logistic uses 13 numeric dimensions; LambdaRank, binary LightGBM and Pairwise + Traffic Rule use five. Semantic arms append eleven: qualified log-age, age missingness, text availability and eight SVD/PCA dimensions fitted on training snapshots. See [methodology](../docs/methodology.md) for definitions.
