# Original Log13 regularization provenance

The report keeps Log13 at lambda .01. The original `wikiflow/src/wikiflow/learners.py:fit_logistic` defaults to .01; `wikiflow/config/benchmark.json` records the fixed 13-feature model. The earlier joint-margin extension inherited its solver and regularization; it was not a fresh two-value lambda search.

The three-fold numbers .21070004540606072 and .19360397215773992 in the baseline `config/early_selection.json` come from continuous Ridge. The generating source is:

`wikiflow_from_scratch/sessions/2026-10-04_warming_magnitude_ranking/relative_warming_v2_user_confirmed_2026-10-04/authorized57_training/run_authorized57.py`, function `early_stage`, lines 137-155.

Its loop fits `('Ridge', .01)`, `('Ridge', .1)` and a fixed HGB regressor. Only Ridge appends `relative_graded_NDCG` to the lambda comparison. Its `fit_one` uses response field `gR` and `Ridge(**params)`. `early_lambda_selection_FROZEN_BEFORE_OUTER.json` preserves the same six values and model IDs beginning `relative_gR_early_..._Ridge_lambda...`. Thus the model-unlabeled packaged JSON cannot substantiate a binary Log13 search. No original file was changed during this correction.

The initial report draft incorrectly attributed these values to Log13. The revised report removes that claim and preserves the actual fixed Log13 recipe. Later B1/B2 P@5 searches and B7/B8 P@50 searches remain distinct and unchanged.
