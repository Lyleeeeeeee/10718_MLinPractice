# WikiFlow: binary Initial ML Solution

This directory is an additive report entry point in the existing repository for the eight-page Team INFO report. It includes only the reported baseline/model families, their stated input ablations and bounded early search configurations. Original research history is preserved outside this directory.

## Quick reproduction (zero model fits)

Use Python 3.13.5 and an isolated environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python reproduce.py
python replay_selection.py
python -m unittest discover -s tests -v
python report/build_assets.py
cd report
tectonic main.tex
```

The default replay reads only bundled development data and saved predictions. It checks all 12 displayed configurations across 24 months against the saved results. Expected support is 5,387 known candidate article-months, 190 events, and 1,200 top-50 slots. Numeric Rank50 has NDCG@50 0.29651889338132986, P@50 0.07333333333333333, pooled Recall@50 0.4631578947368421 and 88 hits. The selection replay verifies four relevant historical search decisions without rerunning optimizers or choosing a new winner.

Outputs go to a fresh `generated/replay` directory; reruns require a fresh `--output`. All paths are relative to the scripts and the existing repository; keep this directory at submissions/initial_ml_binary. Figure generation uses the CSVs inside `report/evidence`, without requiring the original research workspace. Tectonic may need standard TeX packages on its first run.

## Layout

- `../../wikiflow/`: existing core, original learner, preprocessing, evaluator and frozen development panel, reused without changes.
- `src/`: additional numerical kernels for tabular Logistic/HGB, month-balanced pairwise loss, exact rational rule blending, LightGBM grouping/initialization, and TRAIN-only text transforms. The original function ASTs are preserved; long historical drivers, downloads and unrelated trials are omitted.
- `config/recipes.json`: 100 binary recipe entries for the displayed families: 2 rules, 1 original fixed Log13, 16 numeric Logistic/HGB, 15 Logistic text controls, 60 pairwise/blend combinations, 1 fixed LightGBM classifier and 5 LambdaRank feature inputs. Recipe counts are not fit counts; B4 weights share fitted learners and some later models were historically reused.
- `data/`: saved scores for displayed methods, and compact frozen text/embedding inputs needed by the reported variants.
- `evidence/`: all reported search configurations' early scores/monthly metrics, original selection records, source hashes and expected report metrics. Unselected configurations for the compared families remain available in the reproduction evidence.
- `report/`: editable LaTeX, figures, portable chart script and report CSVs.

## Optional refitting, separate from the default replay

No refitting was performed while assembling this package. An opt-in entry point exposes one documented recipe and one development fold:

```sh
python fit_fold.py --recipe binary_log13 --month 2024-09 --allow-fit --output generated/log13-2024-09.json
python fit_fold.py --recipe rank50_base --month 2024-09 --allow-fit --output generated/rank50-2024-09.json
```

Repeat a recipe over the recorded early months for its historical search, or over September 2024--August 2026 for its fixed development evaluation. Use the exact IDs and settings in `config/recipes.json`, and selection functions/records in `src` and `evidence`; do not select a new recipe using the later development months. Logistic/HGB selection uses February/May/August 2024; pairwise/blend and LambdaRank input selection use January--August 2024. The optional `--exclude-pageid 525160` excludes the known Brainiac identity from both TRAIN and evaluation for sensitivity. Every output must be new. September 2026 and later are rejected before data loading. This entry point has boundary/import tests; a new complete refit equivalence run was deliberately not performed.

The original Log13 uses fixed inherited lambda .01. The old two-lambda `early_selection.json` values are from the continuous Ridge search, as confirmed by their generating code; they are not a separate Log13 tuning result. See `evidence/provenance_correction.md`.

## Inputs, dependencies and limits

The baseline panel is reused in place; compact text-derived inputs and report predictions are added here. They contain public article statistics/text, not editor identities or private communication. `../../wikiflow/data/public_download_provenance.json` and `../../wikiflow/data/title_mapping.csv` document the upstream sources/catalogue. Source data are Wikimedia Clickstream, pageviews and MediaWiki revision content. The frozen historical text cutoff is 2023-09-30. MiniLM is `sentence-transformers/all-MiniLM-L6-v2` at revision `7dbbc90392e2f80f3d3c277d6e90027e55de9125`; the bundled embeddings avoid weight downloads or inference during reproduction. Dimensions are fixed at 8; seed is 718 and numerical fitting uses two threads.

Raw dump/API caches and encoder weights are not included. Fresh source acquisition is not byte-equivalent to the fixed historical snapshot and is not an automatic reproduction step. The original baseline acquisition code remains in `../../wikiflow` in this repository. No holdout data, preparation or evaluation entry point is included. The data are retrospective development evidence, with catalogue/title/source-release limitations stated in the report. Exact floating-point behavior can depend on platform/BLAS even with pinned versions.

## Data examples

`evidence/case_evidence.json` records two purposively selected product/model surge examples and compact source histories. `verify_cases.py` verifies the thresholds, identities and labels against the existing baseline panel without raw-data access. Original raw-extract verification was completed separately; the source hashes are retained. These examples are eligible in both original-main and Brainiac-excluded scope. They do not change the primary original-main denominator.

## Numeric input mapping

`src/rule_residual.py:prepared` and `src/at50_ranker_semantics.py:augmented_prepared` use exactly five numeric columns: (1) `joint_margin(row) = log1p(x_M) - log1p(T_M)`; (2) baseline features[0] minus features[1], the difference of log daily incoming rates; (3) features[8], the difference of log mean daily pageviews in successive 14-day windows; (4) features[0], log current daily incoming; (5) features[10] minus features[3], log robust daily-rate scale minus log six-month median. They represent threshold distance, two growth signals, traffic scale and variability. The baseline field list and calculations remain in `../../wikiflow/src/wikiflow/preparation.py:FEATURES` and `article_features`; Log13 appends the joint margin to the original twelve fields. Text missingness is `docs[pageid]['missing']` (1 = unavailable), independent of outcome-label missingness.
