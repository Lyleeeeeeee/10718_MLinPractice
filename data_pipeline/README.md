# Data pipeline

- `acquisition.py`: official Wikimedia Clickstream, pageviews, revision metadata and optional live catalogue collection.
- `preparation.py`: numeric features, metadata quality gates, candidate eligibility and labels.
- `assemble.py`: combine numeric data, normalized text and vectors into training inputs.

See [pipeline commands](../docs/pipeline.md) for required dates, identity handling, input formats and execution.
