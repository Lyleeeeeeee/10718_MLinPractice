# Selection and provenance

The current table has seven scored methods plus random expectation; the availability-only addition is an ablation control.

results/selection.json freezes current choices. results/early_trials.csv consolidates 613 original binary K=50 trial-month rows. Their historical IDs are evidence, not extra active models. The verify command recomputes pooled early Recall and mean event-month NDCG and verifies the recorded chosen recipes and counts:

| Recipe | Early folds | Hits/events |
|---|---|---:|
| Traffic + Text Logistic | February/May/August 2024 | 6/9 |
| Linear Pairwise Ranker + Rule Blend | January–August 2024 | 19/27 |
| Numeric LambdaRank | January–August 2024 | 16/27 |

Numeric LambdaRank ties the flag input on early Recall/NDCG and retains the simpler input. Traffic Logistic is the inherited Log13 λ=.01 reference, not the signed-slope search winner. The tree classifier's parameters were fixed; availability is a mandatory control.

Initial Logistic/pairwise searches used P@5, and LambdaRank used P@50. Recall reselection occurred after prior development inspection. Source commit 3fc6e22 records 72 development refits for three changed recipes and 14 early verification refits. The confirmed SVD/pairwise predictions are retained from that run. Cleanup does not repeat it or relabel it as untouched validation.

Friendly IDs map to original IDs, paths, commit and hashes in models.json and results/provenance.json. Prediction values, labels, row support and numerical feature mappings are preserved. results/monthly.csv contains 192 rows: eight scored methods/control ×24 months. Replay recomputes every monthly and aggregate metric.

Random is analytical: expected hits in each known-label monthly pool = min(50,N)*E/N. Sum and divide by 190 for pooled Recall or 1,200 for Precision. It is not a seeded realized ranking; no expected NDCG is claimed.

No new training, search, holdout evaluation or TA repair occurred. Numerical kernels are preserved; a complete optimizer/refit equivalence run for all relocated fitters was not performed.
