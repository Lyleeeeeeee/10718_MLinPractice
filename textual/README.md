# Historical text pipeline

Run from the repository root. Imports never fetch data. Outputs are restricted to fresh files beneath this repository's `runs/` or `local_data/`; API cache files are immutable and resumable. No raw inputs or bundled reference artifacts are overwritten.

## 1. Select exact historical revisions

Install `requirements-text.txt`. Supply a prepared bundle directory containing `panel.json.gz` and `evaluation_rows.json.gz`, revision metadata keyed by string page ID, and coverage intervals keyed by string page ID. Metadata records contain `revid`, `parentid`, `timestamp` and, where available, `size` and `sha1`. Coverage records contain `start` and `end_exclusive` in canonical UTC seconds.

```
python -m textual.preparation --data runs/prepared/data --metadata local_data/metadata.json.gz --coverage local_data/coverage.json.gz --output runs/text-plan/plan.json
```

Each row must have an `editor_activity_asof` window ending exactly at its target-month boundary. Selection requires complete union coverage, a pre-window baseline, and an unbroken parent chain. The latest revision strictly before the boundary wins (revision ID breaks timestamp ties). Conflicting duplicate rows/revisions and cross-page revision collisions fail closed. Panel targets must precede October 2026. Evaluation inputs contribute rows only; evaluation labels are never read by this pipeline.

## 2. Fetch selected content and earliest-visible evidence

These commands explicitly opt into serial public MediaWiki requests. The seven-second minimum interval, request budget, bounded retries, and Retry-After are respected. Reuse the same cache to resume after an interruption.

```
python -m textual.acquisition acquire --plan runs/text-plan/plan.json --cache local_data/text-cache --allow-network --max-requests 1000
python -m textual.acquisition age --plan runs/text-plan/plan.json --metadata local_data/metadata.json.gz --cache local_data/text-cache --allow-network --max-requests 1000
```

Exact revision IDs, page identity, timestamp, available byte size, and SHA1 are checked. Redirects, hidden/unavailable content, and empty parsed leads remain explicit missing states. HTTP/API failures abort rather than becoming missing text. Creation evidence queries the earliest visible revision and compares it with **all retained metadata**, including later roots. Original creation is never certified; the age transform checks root conflicts at each row's own cutoff.

## 3. Reverify and normalize

```
python -m textual.acquisition build --plan runs/text-plan/plan.json --metadata local_data/metadata.json.gz --cache local_data/text-cache --data runs/prepared/data --output runs/text-build/data/text
```

This reparses cached raw responses, verifies them against the parsed cache, checks prepared-input fingerprints and per-row historical boundaries, and emits `snapshots.json.gz`, `row_snapshots.json.gz`, `ages.json.gz`, and `text_manifest.json`. Each snapshot has `snapshot_id`, `pageid`, `revision_id`, `missing`, `text`, `status`, `timestamp`, and `source_url`. Text is the cleaned lead capped at 4,000 characters plus literal historical categories. Templates are not expanded against current Wikipedia. These three normalized data files are the inputs consumed by the model pipeline.

## 4. Explicit frozen MiniLM encoding

Install `requirements-encoder.txt` separately. Local checkpoints are verified against embedded SHA256 hashes for `sentence-transformers/all-MiniLM-L6-v2`, revision `7dbbc90392e2f80f3d3c277d6e90027e55de9125`.

```
python -m textual.encoding --snapshots runs/text-build/data/text/snapshots.json.gz --model-dir /path/to/pinned/model --output runs/text-encoding --allow-encode
```

Alternatively, explicitly authorize the official pinned checkpoint download by replacing `--model-dir` with `--download-model local_data/minilm-pinned`. The destination must be fresh. No download occurs with `--model-dir`.

Tokenize without truncation and split content into nonoverlapping chunks of at most 254 tokens; add the original two special tokens, never decode/re-tokenize. The original frozen SentenceTransformer pools attended tokens within each chunk. Normalize each chunk vector, take the content-token-count-weighted mean, then normalize the document vector. Missing text receives a zero vector. Outputs are `minilm_embeddings.npz` (`snapshot_ids`, `chunked`, 384 dimensions) and `encoder.json`. The NPZ belongs beside the normalized text files when assembling a complete model input bundle; keep the original generated artifacts and manifests for provenance. Encoding does not fit PCA; `representations.py` fits PCA/SVD on TRAIN snapshots only.

Run synthetic tests with `python -m unittest discover -s tests -p test_text_pipeline.py`. No full-corpus encoding or public requests run in tests.

### Source vintages

For exact source-vintage reproduction, replace the plan command's `--metadata` and `--coverage` with `--metadata-by-target local_data/vintages.json`. This JSON maps each target month to `{ "metadata": "path.json.gz", "coverage": "path.json.gz" }`; paths are relative to the configuration file. Every target needs an entry. Each source is fingerprinted in the plan. In the subsequent `age` and `build` commands, pass this same configuration as `--metadata`. Age evidence retains the union of metadata across target vintages, with earlier target vintages taking precedence for repeated revision IDs. This retains later-root evidence for cutoff-specific age qualification. A fresh retrieval can differ from frozen artifacts if Wikipedia's visible history changes; the raw response hashes document the retrieved evidence rather than promising historical visibility certification.

For frozen age-evidence reproduction where older pages retain their original metadata vintage, explicitly pass `--age-metadata-by-page local_data/age-vintages.json` to both `age` and `build`. This JSON maps every required string page ID to its selected metadata file path, relative to the configuration. All retained revisions from that page's declared vintage are used. Missing page mappings fail closed. The build manifest fingerprints the configuration and each source. This optional declared mapping replaces union-based age metadata; it does not silently discard later roots or infer an evidence vintage. Without it, the union behavior above applies and may differ from older frozen age artifacts.
