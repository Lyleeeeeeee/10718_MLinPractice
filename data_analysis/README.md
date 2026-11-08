# Evaluation and verification

- `pipeline.py`: prior-known-target training pools, feature assembly and fixed expanding-window fits.
- `metrics.py`: common monthly ranking, binary Recall/Precision/NDCG, analytic random reference and validation-only selection.
- `verify.py`: complete package inventory, independent threshold/label arithmetic, original candidate gates, temporal contracts, training memberships and saved-metric replay.
- `io.py`: portable JSON/CSV helpers.
- `manifest.py`: explicitly refresh the inventory after reviewed edits with `python -m data_analysis.manifest --write`.

Use the root `run.py` entry point. Generated results belong in a fresh child of ignored `runs/`; final reference results are read-only inputs to reproduction. Rebuilding the manifest acknowledges intentional package edits; it does not validate their scientific correctness.

Verification checks all retained labels and candidate gates, but does not reconstruct upstream pageview extraction, edit histories, title resolution or encoder inference. To rebuild them, use the [full pipeline](../docs/pipeline.md).
