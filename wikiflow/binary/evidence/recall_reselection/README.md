# Recall@50 reselection: development refits

Completed 72 fits: three frozen recipes over September 2024–August 2026. No sealed holdout data were accessed. The original repository and historical results were not modified.

## Results
All rows use 24 months, 5,387 known candidate article-months, 190 events and 1,200 review slots. Recall is pooled; NDCG averages the 23 event months.

| Method | Hits | Recall@50 | Precision@50 | NDCG@50 |
|---|---:|---:|---:|---:|
| New signed-slope logistic, lambda=.001 | 79 | .415789 | .065833 | .251821 |
| New TF-IDF/SVD logistic, lambda=.01 | 93 | .489474 | .077500 | .273847 |
| New pairwise expanded, lambda=1, alpha=.5 | 80 | .421053 | .066667 | .293981 |
| Original Momentum | 76 | .400000 | .063333 | .241243 |
| Original Log13 | 81 | .426316 | .067500 | .265096 |
| Original Log13 + text-availability flag | 88 | .463158 | .073333 | .274073 |
| Original numeric LambdaRank | 88 | .463158 | .073333 | .296519 |

The SVD logistic has the highest pooled Recall among the 15 configurations evaluated/replayed here. Relative to Log13 it gains 12 events (14.81% relative Recall); relative to numeric LambdaRank it gains five events (5.68%). Its NDCG is lower than numeric LambdaRank. This is a new development candidate, not evidence of independent-test superiority.

Paired month bootstrap (10,000 samples, seed 718, 95% percentile intervals; pooled Recall recalculated for each sample):
- SVD logistic minus Momentum: +.089474, interval [.011493, .172222].
- SVD logistic minus Log13: +.063158, interval [-.013954, .136792].
- SVD logistic minus numeric LambdaRank: +.026316, interval [-.049757, .105263].

The proposed 20% relative improvement bar over Log13 is not reached (.489474 < 1.2*.426316 = .511579). Intervals describe development-period variation; repeated development inspection and temporal dependence limit inferential claims.

## Verification and provenance
- 14 early-fold refits reproduced saved TP exactly and binary NDCG within 1e-9, across all recorded early months for each new recipe.
- All 12 packaged original configurations were re-scored from saved predictions; key aggregate metrics match saved values within 1e-10.
- Every new configuration has exactly the same evaluation support as the baselines.
- Repository implementation and frozen settings were used without changes. No enlarged hyperparameter search was performed. The early metric change was decided after prior development inspection, so this does not constitute an untouched validation experiment.
- Dependencies match the versions pinned by the repository. Python is 3.12.14 rather than the README's recommended 3.13.5; early-score equivalence checks passed. See manifest.json for actual runtime details.

## Files
- comparison.csv: full aggregate comparison (3 new refits plus 12 original replays).
- all_monthly.csv: 360 method-month records.
- new_monthly.csv: 72 new method-month records.
- *_predictions.json: all new prediction scores, including prediction-time candidates with unknown outcomes (excluded only during scoring).
- paired_bootstrap.csv: paired differences versus four comparators.
- early_verification.json: early-fold reproduction checks.
- manifest.json: exact recipes, commit, input hashes, environment, and fit counts.
- ../../rerun_recall_selection.py: reproducer using repository's fit_recipe and evaluator; requires a fresh --output path; --repo defaults to this repository.

From the repository root, using an environment with `wikiflow/binary/requirements.txt` installed:

```sh
python wikiflow/binary/rerun_recall_selection.py --output /path/to/fresh/results
```

The three recipe IDs are frozen in the script; this command verifies early results and refits them, rather than running a new hyperparameter search. The saved manifest describes the original run, before this integration commit.

On macOS, if LightGBM cannot locate libomp, set DYLD_LIBRARY_PATH to the installed scikit-learn .dylibs folder. No LightGBM fitting is performed for these three recipes; its import is required by the repository's shared fitting entry point.
