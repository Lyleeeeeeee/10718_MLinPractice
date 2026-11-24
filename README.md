# WikiFlow

WikiFlow ranks Wikipedia articles each month to help editors find pages likely to see a surge in readership. It produces a 50-article shortlist from historical internal-link traffic, pageviews, revision metadata, and article text.

## Run with the bundled data

Use **Python 3.13.5**, from the repository root:

```sh
python -m pip install -r requirements.txt
python run.py show
python run.py verify
```

| To reproduce | Command |
|---|---|
| Main benchmark: 12 configurations and three baselines | `python run.py refit --output runs/refit` |
| Age vs. text: three feature sets, one fixed model | `python run.py ablate-age --output runs/age-ablation` |
| Unit and regression tests | `python run.py test` |

Always use a fresh output directory. The bundled data supports offline fits without downloading an encoder. To rebuild the inputs from source, use the [data and training pipeline](docs/pipeline.md).

## Read the results

- [Main benchmark](docs/methodology.md): the validation-selected Pairwise + Traffic Rule with SVD8 and age finds **71/149 events (47.65% Recall@50)** in May–October 2026, vs. **56/149 (37.58%)** for momentum. These are retrospective results, not an independent holdout.
- [Age vs. text ablation](results/age_ablation/README.md): isolates what age adds to numeric features, then what text adds on top of age.
- [Results index](results/README.md): summaries, per-month scores, predictions, and validation selection.

## Find the code and data

| Stage | Directory |
|---|---|
| Source acquisition, dataset preparation and assembly | [`data_pipeline/`](data_pipeline/) |
| Bundled reproducible inputs and attribution | [`data/`](data/) |
| Numeric inputs, thresholds and preprocessing | [`features/`](features/) |
| Historical text, article age and representations | [`textual/`](textual/) |
| Learners and ranking rules | [`models/`](models/) |
| Rolling evaluation, ablation and verification | [`data_analysis/`](data_analysis/) |
| Automated tests | [`tests/`](tests/) |
