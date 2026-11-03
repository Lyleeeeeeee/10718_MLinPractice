# Binary task and features

The unit is an article-month in a fixed 2026 WikiProject AI catalogue of 1,187 canonical articles. The expanded panel has 580 articles and 11,080 rows, target months 2023-10..2026-08. At target M, features end at M−1 and TRAIN uses known earlier target labels.

Candidates need seven published monthly counts, current count>=100, both current-cold gates, complete metadata coverage and little substantial editing in the prior two complete calendar months. Each nonminor absolute parent-net-byte delta must be <500; their sum must be <1,000. Exact predicates remain in core.py and preparation.py.

Development has 5,388 candidates, of which one unknown outcome is excluded during scoring: 5,387 known rows and 190 surges. Missing text does not remove a candidate. April 2025 has zero events: Precision=0, Recall/NDCG undefined.

## Label

For c=M−1, incoming link transitions I and month length D, compute over c−5..c:

    m = median(I_j / D_j)
    s = 1.4826 * median(abs(I_j / D_j - m))
    U_M = max(200, D_M*m + max(100, .5*D_M*m, 3*D_M*s))
    x_M = I_c * D_M / D_c
    T_M = max(U_M, 1.25*x_M)
    surge = 1[actual_M > T_M]

Absent published outcomes remain unknown. The code retains the frozen label's multiplication order; cleanup changes no thresholds or labels.

## Feature groups

| Group | Meaning |
|---|---|
| Traffic history + pageviews, 13 | Current/previous log daily Clickstream rate; three-month mean; six-month median/slope; seasonal rate/missing flag; pageview level/14-day growth/incomplete flag; robust spread; historical- and joint-threshold log margins. |
| Compact traffic, 5 | Joint-threshold margin, Clickstream growth, pageview growth, current log traffic level, log robust spread minus log six-month median. |
| Text addition, 9 | TF-IDF/SVD8 of cached lead+literal categories, plus text-missing flag. |

The slope remains the original raw slope. features.py names each numerical column and preserves the original index mapping. TRAIN fits imputation/scaling. Text fits distinct usable TRAIN articles only: English stopwords, unigrams, min_df=2, max_df=.95, max_features=3,000; SVD8, 7 power iterations, seed 718. Missing text gets a zero semantic sentinel plus flag before scaling.

## Recipes and historical IDs

| Current ID | Historical ID | TRAIN | Fixed configuration |
|---|---|---|---|
| volume | volume | none | Current incoming count |
| momentum | cs_momentum | none | Recent daily-rate extrapolation relative to T |
| traffic_logistic | binary_log13 | expanded | Logistic λ=.01 |
| traffic_text_logistic | binary_svd_0.01 | expanded | Log13+SVD8+flag; λ=.01 |
| numeric_lambdarank | rank50_base | candidate | Numeric Rank50 with rule initialization |
| tree_classifier | binary | candidate | Month-balanced LightGBM classifier |
| linear_pairwise_blend | pairwise_expanded_1_a0.5 | expanded | λ=1, α=.5 |
| text_availability_control | binary_missing_0.01 | expanded | Log13+flag; λ=.01 |
| random_expected | none | none | Uniform random-order expectation |

Logistic retains the damped Newton solver. Both tree recipes have 100 trees, 7 leaves, depth 3, learning rate .03, minimum leaf 30, 63 bins, L2=10, seed 718 and a monotone-positive first margin. LambdaRank groups by month, uses binary gains/truncation 50 and adds the rule initializer once at prediction. The classifier uses log loss and equal total month weights without that initializer; this is not a pure loss-only comparison.

Pairwise uses same-month positive/negative pairs, up to 200 negatives per positive and equal total weight per event-bearing TRAIN month. The first coefficient is nonnegative. L-BFGS-B uses max_iter=500, gtol=1e-7, ftol=1e-12. The α=.5 blend combines within-month learned and threshold-rule ranks, preserving original exact tie keys.

Ties use score, current count descending, then page ID. Different input sets and TRAIN pools compare complete recipes, not isolated learner effects.

## Sources and unchanged limits

Clickstream is monthly; pageviews and revision histories are separate APIs. Canonical titles connect Clickstream/pageviews to the current title→pageid mapping; revision/text uses pageid/revid. Source acquisition/preparation code and download hashes are retained.

Text is a partial cached snapshot <=2023-09-30, not necessarily the latest online pre-cutoff revision. Its missing flag is not certified article age. Original-main identity policy, including the known Brainiac title-slot limitation, is unchanged. Historical title identities, 2026 catalogue membership and first-release availability remain retrospective limitations. Development has been examined repeatedly; September 2026 remains sealed. No TA remedy is implemented by this cleanup.
