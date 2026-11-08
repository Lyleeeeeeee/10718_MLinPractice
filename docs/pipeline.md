# Source-to-results commands

Run from the repository root with **Python 3.13.5**. `requirements.txt` supports data preparation and bundled-data training; `requirements-text.txt` is a compatible alias. `requirements-encoder.txt` also adds pinned encoder dependencies.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-encoder.txt
```

Raw dumps, API caches and model weights are external inputs, not bundled. Full English Clickstream downloads need substantial disk space; API collection is serial and can take hours. Fresh APIs can change titles, catalogue membership and visible history, so newly collected data may differ from the frozen research bundle.

All generated outputs must be under `runs/` or `local_data/`. Use a new prefix instead of `runs/from-zero` for another run. Acquisition output, raw-file and cache paths must be distinct and fresh.

## 1. Collect traffic and metadata

Use the frozen `data/title_mapping.csv` cohort. Optional `python run.py acquire topic --output runs/live-topic/mapping.csv --cache runs/live-topic/cache` fetches today's catalogue instead; pass that mapping explicitly to all subsequent steps if using it.

Clickstream requires September 2022–October 2026, including the seasonal lookback:

```sh
set -e
python - <<'PYMONTHS' | while IFS= read -r month; do
for index in range(2022 * 12 + 8, 2026 * 12 + 10):
    print(f"{index // 12:04d}-{index % 12 + 1:02d}")
PYMONTHS
  python run.py acquire clickstream --month "$month" \
    --mapping data/title_mapping.csv \
    --raw "runs/from-zero/raw/clickstream-${month}.tsv.gz" \
    --output "runs/from-zero/incoming/incoming_${month}.csv.gz" \
    --cache "runs/from-zero/caches/clickstream-${month}"
done
```

Pageviews need September 2023–September 2026. Revision metadata needs August 2023–September 2026 plus the preceding baseline revision, fetched automatically.

```sh
python run.py acquire pageviews \
  --mapping data/title_mapping.csv --start 2023-09-01 --end 2026-09-30 \
  --output runs/from-zero/pageviews.csv.gz --cache runs/from-zero/caches/pageviews

python run.py acquire revisions \
  --mapping data/title_mapping.csv \
  --start 2023-08-01 --end-exclusive 2026-10-01 \
  --identity-policy error --identity-output runs/from-zero/identity.json \
  --output runs/from-zero/revisions.json.gz \
  --coverage runs/from-zero/coverage.json.gz \
  --cache runs/from-zero/caches/revisions

python run.py prepare \
  --mapping data/title_mapping.csv --incoming runs/from-zero/incoming \
  --pageviews runs/from-zero/pageviews.csv.gz \
  --metadata runs/from-zero/revisions.json.gz --coverage runs/from-zero/coverage.json.gz \
  --identity runs/from-zero/identity.json \
  --start 2023-10 --end 2026-10 --output runs/from-zero/numeric
```

Revision identity defaults to `error`. To exclude changed identities explicitly, use `--identity-policy exclude` with the required `--identity-output`, then pass that report to `prepare --identity`. No silent title repair is performed.

For retained per-target metadata, `prepare --metadata-by-target` accepts a JSON object mapping `YYYY-MM` to `{"metadata":"path","coverage":"path"}`; relative paths resolve from the JSON file. The base metadata/coverage arguments remain required.

## 2. Build historical text

```sh
python run.py text-plan --data runs/from-zero/numeric \
  --metadata runs/from-zero/revisions.json.gz --coverage runs/from-zero/coverage.json.gz \
  --output runs/from-zero/text-plan.json

python run.py text acquire --plan runs/from-zero/text-plan.json \
  --cache runs/from-zero/text-cache --allow-network --delay 7 --max-requests 1000

python run.py text age --plan runs/from-zero/text-plan.json \
  --cache runs/from-zero/text-cache --metadata runs/from-zero/revisions.json.gz \
  --allow-network --delay 7 --max-requests 1000

python run.py text build --plan runs/from-zero/text-plan.json \
  --cache runs/from-zero/text-cache --data runs/from-zero/numeric \
  --metadata runs/from-zero/revisions.json.gz --output runs/from-zero/text-normalized
```

For per-target metadata, replace `text-plan`'s metadata/coverage pair with `--metadata-by-target`. Text network calls require `--allow-network`, delay ≥7 seconds and a request budget. After interruption or budget exhaustion, rerun the same `text acquire` or `text age` command with `--allow-network` and its existing cache. Build only after both finish. Other acquisition commands require fresh paths after failure.

## 3. Encode

```sh
python run.py encode --snapshots runs/from-zero/text-normalized/snapshots.json.gz \
  --download-model runs/from-zero/minilm-model --allow-encode \
  --output runs/from-zero/encoding
```

The checkpoint is `sentence-transformers/all-MiniLM-L6-v2`, revision `7dbbc90392e2f80f3d3c277d6e90027e55de9125`; files are hash-checked. To use an existing matching checkpoint, replace `--download-model` with `--model-dir`. Encoding requires `--allow-encode` and a fresh output directory. Training later fits the eight-dimensional projections within each fold.

## 4. Assemble and train

```sh
python run.py assemble --numeric runs/from-zero/numeric \
  --text runs/from-zero/text-normalized --encoding runs/from-zero/encoding \
  --output runs/from-zero/data

python run.py verify-data --data runs/from-zero/data
python run.py refit --data runs/from-zero/data --output runs/from-zero/refit
```

`assemble` combines numeric, normalized-text and encoding outputs into a new data tree. A custom-data refit does not compare with the frozen scores unless `--compare-reference` is supplied. For bundled-data training, omit `--data`. Run `python run.py test` for automated tests.
