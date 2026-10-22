# Report-to-code map

All paths are relative to this directory. Every displayed row uses the unchanged baseline `../src/wikiflow/core.py:rank_metric` and the identical candidate support. All optional fold fitting is wired by `fit_fold.py`; the default `reproduce.py` only re-evaluates saved rankings.

| Report row / recipe ID | Implementation | Configuration | Evidence |
|---|---|---|---|
| `volume` | `src/experiment.py:baseline_scores / momentum_score` | `config/recipes.json` key `volume` | `data/development_predictions.json.gz`; `evidence/expected_monthly.csv` |
| `cs_momentum` | `src/experiment.py:baseline_scores / momentum_score` | `config/recipes.json` key `cs_momentum` | `data/development_predictions.json.gz`; `evidence/expected_monthly.csv` |
| `binary_log13` | `../src/wikiflow/learners.py:fit_logistic` | `config/recipes.json` key `binary_log13` | `data/development_predictions.json.gz`; `evidence/expected_monthly.csv` |
| `binary_missing_0.01` | `src/text_features.py:representation + baseline fit_logistic` | `config/recipes.json` key `binary_missing_0.01` | `data/development_predictions.json.gz`; `evidence/expected_monthly.csv` |
| `binary_dynamics_hgb_d2_l21` | `src/experiment.py:fit_one / feature_values` | `config/recipes.json` key `binary_dynamics_hgb_d2_l21` | `data/development_predictions.json.gz`; `evidence/expected_monthly.csv` |
| `binary` | `src/nonlinear_focus.py + fit_fold.py` | `config/recipes.json` key `binary` | `data/development_predictions.json.gz`; `evidence/expected_monthly.csv` |
| `pairwise_expanded_0.01_a0.25` | `src/rule_residual.py:fit_weights + src/rule_blend_v2.py:blend_rank_keys` | `config/recipes.json` key `pairwise_expanded_0.01_a0.25` | `data/development_predictions.json.gz`; `evidence/expected_monthly.csv` |
| `rank50_base` | `src/at50_ranker_semantics.py + src/nonlinear_focus.py + fit_fold.py` | `config/recipes.json` key `rank50_base` | `data/development_predictions.json.gz`; `evidence/expected_monthly.csv` |
| `rank50_missing` | `src/at50_ranker_semantics.py + src/nonlinear_focus.py + fit_fold.py` | `config/recipes.json` key `rank50_missing` | `data/development_predictions.json.gz`; `evidence/expected_monthly.csv` |
| `rank50_svd` | `src/at50_ranker_semantics.py + src/nonlinear_focus.py + fit_fold.py` | `config/recipes.json` key `rank50_svd` | `data/development_predictions.json.gz`; `evidence/expected_monthly.csv` |
| `rank50_lda` | `src/at50_ranker_semantics.py + src/nonlinear_focus.py + fit_fold.py` | `config/recipes.json` key `rank50_lda` | `data/development_predictions.json.gz`; `evidence/expected_monthly.csv` |
| `rank50_intro` | `src/at50_ranker_semantics.py + src/nonlinear_focus.py + fit_fold.py` | `config/recipes.json` key `rank50_intro` | `data/development_predictions.json.gz`; `evidence/expected_monthly.csv` |

## Tuning and negative results

| Report section | Full stated search and evidence |
|---|---|
| B1 Logistic/HGB | 16 binary recipes in config; `evidence/batch1/validation_scores.json`, monthly scores and original selection; `src/experiment.py:choose_config`. |
| B2 Logistic text | 15 recipes; `evidence/batch2/`; original categorical/missing/embedding controls included, same selection helper. |
| B4 pairwise/rule blend | 60 scored combinations of 12 fitted recipes and five weights; `evidence/batch4_rule_residual/`; `src/rule_residual.py:select`. Exact rational blend ordering uses rule_blend_v2. |
| B8 Rank50 inputs | Five fixed inputs; `evidence/batch8_at50_ranker_semantics/`; `src/at50_ranker_semantics.py:select_recipe`. Numeric remains selected. |

## Figures and separate sensitivity

- Figure 1: `report/build_assets.py`, volume rows of `report/evidence/monthly_selected.csv`; sum known candidates/events only.
- Figure 2: same script, month-paired rank50_base minus cs_momentum NDCG@50. April 2025 has predictions and 50 review slots. Figure 2 displays its difference as zero only for plotting; saved metrics and the event-month aggregate are unchanged.
- Tables 3/4: values in `report/main.tex`, independently checked against `evidence/expected_summary.csv` and recomputed from saved rankings by reproduce.py.
- Clean sensitivity paragraph: `report/evidence/clean_selected.json`; it is not merged into original-main. The optional fit_fold.py exclusion removes pageid 525160 from both sides. No new clean refit was run.
- Label, candidate filter, missingness and metrics: existing `../src/wikiflow/core.py`.
- Baseline panel: existing `../data/panel.json.gz`, byte-identical to the research input.
- Historical Log13 provenance correction: `evidence/provenance_correction.md`.

The report uses descriptive model/input names rather than internal batch IDs. Batch IDs below identify reproduction sources only; selection metrics remain those actually used, including P@5. The report introduction omits historical workflow narration.

Models are explained separately from feature definitions and tuning. Table 3 has eight configurations from seven algorithms/rules; Table 4 varies only the fixed LambdaRank inputs. Numeric Logistic and Logistic text controls remain as bounded tuning evidence, not additional main-result rows. Matched semantic Logistic and rank8 trials are outside this report entry point and remain in the original research files.

Data examples: `evidence/case_evidence.json` and `verify_cases.py`; existing baseline panel and title mapping are reused.
