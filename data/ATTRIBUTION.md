# Attribution

Wikipedia text is by Wikipedia contributors, licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Every retained text snapshot includes its exact revision URL, revision ID and timestamp in `text/snapshots.json.gz`. Introductions were normalized and limited to 4,000 characters; categories were appended. These are transformed excerpts, not full articles.

Traffic statistics come from [Wikimedia Clickstream](https://dumps.wikimedia.org/other/clickstream/) and the [Wikimedia Pageviews API](https://wikimedia.org/api/rest_v1/metrics/pageviews/). Published clickstream pairs are thresholded; an absent aggregate outcome is unknown, not zero. Catalogue membership and title identity were reconstructed retrospectively.

The frozen [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) encoder is published under Apache-2.0. Only its derived vectors are included; no model weights or encoder runtime are distributed. Exact checkpoint and chunking provenance are in `text/encoder.json`.
