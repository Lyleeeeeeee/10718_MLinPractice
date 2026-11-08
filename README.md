# WikiFlow

Monthly Wikipedia attention-growth ranking, from public data collection to model evaluation: **four learning recipes × three representations, plus three baselines**.

## Run

Use **Python 3.13.5**, from the repository root:

```sh
python -m pip install -r requirements.txt
python run.py show
python run.py refit --output runs/refit
```

Use a fresh output directory. The bundled data supports offline training without encoder downloads. To collect sources and regenerate features and embeddings, follow [the pipeline commands](docs/pipeline.md).

## Contents

- [Methodology](docs/methodology.md): labels, features, rolling training and results.
- [Pipeline](docs/pipeline.md): acquisition, preparation, encoding and training.
- [Data](data/README.md) and [attribution](data/ATTRIBUTION.md).
- `data_pipeline/`, `features/`, `textual/`, `models/`, `data_analysis/`: implementation.
- `results/`: final predictions and summaries; `tests/`: automated tests.

The validation-selected Pairwise/SVD system finds **71/149 test events (47.65% Recall@50)**, versus momentum **56/149 (37.58%)**. All May–October 2026 outcomes were previously exposed; results are retrospective, not an independent holdout.
