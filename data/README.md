# Data bundle

The compressed JSON inputs contain the normalized final article-month panel and supporting traffic, label, candidate and provenance evidence. Text inputs are separated under `text/`:

| File | Role |
|---|---|
| `panel.json.gz` | Historical training panel, row features and known/unknown labels |
| `evaluation_rows.json.gz` | Fixed monthly candidate rows |
| `evaluation_labels.json.gz` | Outcomes keyed by article-month identity |
| `source_monthly_counts.json.gz` | Published monthly traffic counts supporting independent threshold checks |
| `catalogue.csv` and `title_mapping.csv` | Article catalogue and source-title alignment |
| `provenance.json` | Bundled source and preparation provenance |
| `text/snapshots.json.gz` | Historical revision text and availability metadata |
| `text/row_snapshots.json.gz` | Article-month to historical snapshot mapping |
| `text/ages.json.gz` | Qualified visible-history age evidence |
| `text/encoder.json` | Pinned MiniLM checkpoint and chunk-pooling provenance |
| `text/minilm_embeddings.npz` | Fixed 384-dimensional chunk-pooled document vectors |

Load JSON with Python's `gzip` and `json`; load the NumPy archive with `numpy.load(..., allow_pickle=False)`. Keep identity fields and missingness intact. Unknown traffic outcomes are not zero counts or negative labels.

Candidate eligibility uses the original 50% / three-dispersion cold bands; event labels use the separate 25% / two-dispersion threshold plus 10% preceding-rate growth. Do not recompute candidate membership from the looser label band. See [methodology](../docs/methodology.md) for formulas and gates.

Text is cleaned lead content capped at 4,000 characters plus categories from a pre-target revision. Visible-history age is not certified page creation. Fixed embeddings are inputs, not model-fitting outputs; eight-dimensional projections are fitted afresh within each training fold.

Raw dumps, API caches and encoder weights are not included. The bundled inputs support offline training; see [pipeline commands](../docs/pipeline.md) to build a new data tree. Fresh public sources may differ from this frozen cohort and source vintage.

See [ATTRIBUTION.md](ATTRIBUTION.md) for source attribution and licensing.
