# WikiFlow monthly ranking design

## Purpose and course scope

The study supplies a monthly shortlist for human article review: currently cold articles with little recent substantial edit activity that may gain internal navigation next month. The measured outcome is incoming published Wikipedia link transitions, not reader demand, content quality or a causal need for edits. The WikiProject AI maintenance catalogue has 1,187 canonical articles; it includes people, works and related subjects.

The course package retains one naive comparator, current incoming volume, plus binary Logistic13 and relative continuous Ridge12. It is a subset of the [previous complete benchmark](https://github.com/Lyleeeeeeee/10718_MLinPractice/tree/ccf65a63287330870b43b7cb563d4cd20348d3a8). Earlier comparisons and development are part of the study's history. The subset does not establish these ML methods as globally best.

## Data and time contracts

For target month M, current month c=M-1 is complete. Monthly incoming count I is the sum of published internal `link` edges to a canonical destination, from any internal source, with published count>=10. Redirect aliases define the catalogue but are not separately accumulated as destination traffic. Absent article/months remain unknown. Explicit API daily PV zero is observed zero; an absent PV date is missing.

| Item | Fixed boundary |
|---|---|
| CS input months | 2022-09 through 2026-08 |
| Expanded training target months | 2023-10 through 2026-08 |
| Development evaluation target months | 2024-09 through 2026-08 (24 months) |
| Feature cutoff for own target M | End of M-1 |
| Final PV cutoff | 2026-07-31 |
| Final revision endpoint | 2026-08-01 exclusive |
| Sealed holdout | 2026-09, absent and rejected |

TRAIN at M uses known labels whose target<M. Labels are assumed mature immediately after their month; original monthly release latency is unverified. Current 2026 catalogue/title mappings and revised dump vintage are retrospective. Therefore this backtest is not a certified month-start deployment simulation.

Identity policy **v1** blocks Daniel homonym pageid 79745475; no history is transferred from director 62422990. Existing exclusions 1164/72417803 remain and Naevis 78522383 is review-only. The retained Brainiac 525160 title-slot mismatch affects TRAIN and evaluation and is explicitly described in the [audit](audit.md); no newer identity correction is applied.

## Thresholds and responses

D_m is the calendar month length and v_m=I_m/D_m. From six monthly daily rates c-5 through c, m=median(v) and s=1.4826*median(abs(v-m)):

```text
U_M = max(200, D_M*m + max(100, 0.5*D_M*m, 3*D_M*s))
x_M = I_c * D_M / D_c
T_M = max(U_M, 1.25*x_M)
event_M = 1[actual_M > T_M]
R_M = max(0, actual_M/T_M - 1)
gR_M = log1p(R_M)
```

Bounds are fixed heuristics, not probabilities. U/x/T/actual have monthly transition units; R and gR are dimensionless. Binary uses strict greater-than and combines historical abnormality with more than 25% daily-rate growth. Relative models regress gR and relative evaluation uses gR as gain exactly once. Labels use the original multiplication order `1.25*I_c*D_M/D_c`; Log13's margin computes x first. Those recipes can differ at floating-point roundoff, which is audited without relabeling.

## Expanded pool and candidate eligibility

The expanded pool requires valid identity, seven published monthly inputs c-6 through c, current count>=100, complete metadata/editorial coverage, `I_c/D_c <= U_M/D_M`, and `I_c <= U_c` where U_c is computed using c-6 through c-1. It contains 11,080 rows. TRAIN does not impose the edit-amount filter.

The evaluation pool additionally requires each nonminor edit's absolute parent-net-byte change<500 and their cumulative absolute change<1000 in the two complete UTC months `[start of M-2, start of M)`. Equal thresholds exclude; reverts contribute; minor revisions do not. Missing parent state, incomplete coverage or hidden records imply unknown quality. Net bytes cannot detect equal-size rewrites. All methods use the same 5,388 candidates and candidate order. Unknown future labels do not affect candidate selection.

## Features and TRAIN preprocessing

Natural `log1p` is used. Means over monthly daily rates weight months equally. The seasonal month is target M-12, not current c-12.

| Column | Definition and units before transformation |
|---:|---|
| 0 | log1p current-month CS daily rate |
| 1 | log1p previous-month CS daily rate |
| 2 | log1p mean of last three monthly CS daily rates |
| 3 | log1p median of last six monthly CS daily rates |
| 4 | Last-six daily-rate OLS slope: sum((i-2.5)*v_i)/17.5, rate per month |
| 5 | log1p target M-12 CS daily rate; current-rate fallback if absent |
| 6 | Seasonal observation missing flag |
| 7 | log1p mean daily PV across the complete current calendar month |
| 8 | log1p trailing-14-day mean PV minus log1p preceding-14-day mean PV, ending at c's last day |
| 9 | Current PV month incomplete flag |
| 10 | log1p robust daily sigma s |
| 11 | log1p(x_M)-log1p(U_M) |
| 12, Log13 only | log1p(x_M)-log1p(T_M) |

Incomplete PV months leave columns 7/8 unknown and column 9 set. Each mature TRAIN computes observed medians, then means and population standard deviations after imputation. An entirely missing column uses computational median zero while its missing flag remains; standard deviation<1e-12 becomes one. Candidate values use these TRAIN statistics only. Unknown outcomes are excluded from TRAIN, never imputed as negatives.

## Fixed objectives and selection provenance

Log13 minimizes mean binary cross-entropy plus `0.5*.01*sum(beta^2)`. The intercept is unpenalized. Initialization is zero slopes and TRAIN-prevalence logit; damped Newton uses gradient infinity norm<=1e-9, maximum 200 steps, at most 60 Armijo line-search steps with constant 1e-4. There is no class/month weighting, warm start or retry.

Ridge12 minimizes mean squared error on gR plus `.01*sum(beta^2)` with a fitted intercept. The sklearn parameter is `alpha=TRAIN_N*.01`, solver cholesky. Models fit from scratch for every expanding fold, with two computational threads and recorded seed 718. Their numerical recipes are unchanged from the prior complete commit.

Inherited binary numerical controls used F1 TRAIN 2023-10 through 2023-12 (649 rows/40 positives), validation 2024-01 through 2024-04; F2 TRAIN through 2024-04 (1,590/71), validation 2024-05 through 2024-08. Log12 lambda .01 was the earlier winner (mean binary NDCG50 .155725). Log13's later joint-T feature and interpretation used already observed development diagnostics. Thus the overall procedure was not blind to the development period.

Relative Ridge lambda .01 versus .1 used only 2024-02/05/08, with equal-month graded NDCG50 means .210700045406 and .193603972158. Lambda .01 was selected; ties within 1e-12 favor stronger .1. The [selection record](../config/early_selection.json), pure selection function and tests remain. Subset validation does not rerun parameter search or change thresholds, feature definitions or model selection.

## Ranking and evaluation

Volume scores are float(I_c). Log13 scores are fitted event probabilities. Ridge12 scores are fitted gR values without clipping. Every ranking orders score descending, current count descending, then pageid ascending. Complete monthly ranking matches are required during refit, beyond Top50 agreement.

| Quantity | Definition |
|---|---|
| DCG at K | sum(gain/log2(rank+1)) for first min(K,N) entries |
| Binary gain | event |
| Relative gain | gR directly; no second log and no exponential gain |
| NDCG | DCG divided by ideal DCG of that same monthly candidate pool |
| AP | Rank-tiebroken binary average precision across the full ranking |
| TP/precision/recall | Event hits, hits/actual K, hits/month events |
| Macro | Equal-weight mean over months where that metric is defined |
| Micro | Pooled hits divided by pooled slots/events |
| Graded capture | Summed selected gR divided by total gR over complete months |

K values are 5,10,20,50,100,200. Any unknown candidate outcome makes all metrics for that whole month NA; candidates are not dropped or backfilled. A complete zero-event month contributes TP/precision zero, while NDCG/AP/recall are NA. April 2026 is unknown; April 2025 is event-free. Overall support is 23 complete months, 22 event months, 182 events and 1,150 Top50 slots.

## Reproduction and remaining work

The [README](../README.md) supplies dependency, offline refit, replay, public download and cached preparation commands. Acquisition preserves existing outputs and requests no editor names, comments or revision text. Raw data and sessions are not committed. Default refit uses a small frozen derived snapshot and retained scores, with input/scaler/key checks and predictions sealed before evaluation. It performs 48 fits; replay performs zero. A new raw-cache preparation must match keys/labels/eligibility exactly and features within audited numerical tolerance.

The [integrity report](../results/subset_integrity.json) verifies that subsetting removed extra score/rule fields and renamed a presentation schema without changing v1 data or labels. The [verification](../results/verification.json) checks 48 fits, exact monthly model ranks and 1,728 metric fields. Acquisition/preparation fixtures and numerical/metric tests are offline.

Future historical identity repair, release-vintage certification, fresh acquisition, independent holdout evaluation, production latency validation and uncertainty estimation remain outside this package. Brainiac repair would require valid former-title exposure data and every affected TRAIN fold to be addressed; removing an evaluation row alone would not certify clean models.
