# WikiFlow: identity-corrected minimal reproducibility package

Rank articles in the WikiProject Artificial Intelligence maintenance portfolio each month to identify those that are currently cold, have had few substantial recent edits, and may experience increased internal navigation next month. The fixed portfolio contains 1,187 canonical articles, including people, works and other maintenance-scope subjects. It is not a pure AI technology collection. Edit history is a candidate-eligibility proxy, not a label for content quality or maintenance need.

This package covers only the identity-corrected binary benchmark and latest **relative continuous** benchmark reported on 2026-10-04. It excludes the old absolute continuous version, diagnostic fusion, new channel/source experiments, full sessions, model pickles and large raw datasets.

The [design document](docs/design.md) explains the monthly human-review use case, data and time contracts, all 13 features, model selection, shared-pool evaluation and unfinished work for course teammates. The [audit record](docs/audit.md) preserves the review evidence.

## Quick reproduction: actual retraining

Run from the repository root. Recommended Python version: **3.13.5**. The verified environment used NumPy 2.3.3, SciPy 1.16.2, scikit-learn 1.7.2 and threadpoolctl 3.6.0. Other supported Python versions may work, but use the recorded environment for exact floating-point reproduction.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r wikiflow/requirements.txt
export PYTHONPATH=wikiflow/src
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
python -m unittest discover -s wikiflow/tests -v
python -m wikiflow.reproduce --mode refit --output wikiflow/runs/my-refit
```

For each month from 2024-09 through 2026-08, `refit` trains Log13, binary HGB12, relative Ridge12 and relative HGB12: **24x4=96 actual fits**, with fixed parameters and no new search. After validating inputs and each fold's TRAIN keys, it recomputes imputation and scaling from that TRAIN alone. It checks every model score, each complete monthly ranking and metrics at all six K values against the historical benchmark. Any mismatch beyond tolerance stops the run. Use a new output directory for every run.

```bash
python -m wikiflow.reproduce --mode replay --output wikiflow/runs/my-replay
```

`replay` only recomputes metrics from stored predictions and performs 0 fits. Outputs include `fit_trace.json`, sealed predictions, monthly/aggregate CSVs and `verification.json`. The committed results document actual refitting before publication; replay is not retraining. Packaging verification performed 96 fits from the frozen snapshot and 96 through the complete existing-cache -> preparation -> refit path: 192 fixed fits, with no tuning. Both paths matched all monthly rankings and reported metrics, taking about 10.9 seconds each.

## Reproduced results at K=50

The candidate pool contains **5,388 article-months**. Of 24 historical months, 23 have complete labels and 22 contain events; there are 182 natural events and 1,150 Top50 slots. April 2026 has unknown labels, so all metrics for that month are NA. April 2025 has no events: TP/precision contribute, while NDCG/AP/recall are NA. July 2026 becomes fully evaluable after quarantining the identity-unknown candidate.

| Binary ranking method | macro NDCG50 | macro AP (full ranking) | cumulative TP50 |
|---|---:|---:|---:|
|volume|0.123384165982|0.062143794623|45|
|strong CS|0.262764324285|0.133360599203|70|
|Log13|0.261323643627|0.127705211410|75|
|HGB12|0.254520450786|0.114735066542|77|

| Relative ranking method | macro graded NDCG50 | cumulative TP50 |
|---|---:|---:|
|volume|0.092810605545|45|
|Ridge12|0.138377139655|63|
|HGB12|0.127657749963|57|
|PV trend/T|0.184767834660|77|

The [complete main-method table](results/summary_at50.csv) retains seven binary and eight relative methods. The [monthly table](results/monthly_at50.csv) is checked against the original results. HGB12 means **12 feature columns**; the historical directory names HGB26/authorized26 refer to **26 identity-correction fits**, not 26 features.

There is no evidence of stable ML superiority over strong CS/PV rules. A mean improvement over volume must not be described as stable superiority. All historical months were repeatedly studied and are development backtests, not independent tests. The original research intervals are descriptive and do not establish independent validation. This minimal package reproduces point predictions and metrics; it does not rerun bootstrap estimation or parameter search.

## Data, time and identity

Public sources are [Wikimedia monthly Clickstream](https://dumps.wikimedia.org/other/clickstream/readme.html), [article daily pageviews](https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/reference/page-views.html), the [MediaWiki revisions API](https://www.mediawiki.org/wiki/API:Revisions) and the current WikiProject talk category.

Clickstream sums published counts from all internal Wikipedia `link` sources to canonical destinations. It does not measure unique readers, search exposure or all pageviews. Preparation preserves literal four-column TSV data, the published count=10 boundary and any internal source; sources need not belong to the AI portfolio. Unpublished low-count edges are suppressed. A missing article/month is unknown, not zero. Original acquisition used redirect/normalized mappings to define the canonical portfolio, but **did not separately aggregate redirect-alias destination counts into the canonical article**. This package preserves that counting definition.

Historical inputs span 2022-09 through 2026-08, model target months 2023-10 through 2026-08, and development evaluation targets 2024-09 through 2026-08. CS/PV features for target M end at M-1; the final PV cutoff is 2026-07-31. Revision metadata end at 2026-08-01 exclusive. Labels use the published incoming article count for the target month. The 2026-09 holdout is absent from the derived snapshot, and entry points reject that month and later.

History uses the current 2026 portfolio, current title mapping and revised data vintage, **assuming M-1 CS/labels are mature and available at the start of M**. Actual historical release latency has not been reconstructed. This is not a demonstrated deployable month-start production system.

The authoritative correction is `identity_correction_v1_Daniel_homonym_quarantine`; its [public policy](config/identity_policy.json) preserves the source manifest SHA256. The current Daniel Kokotajlo homonym slot, pageid **79745475**, has uncertain historical identity and is blocked in TRAIN, candidates and labels. The history of director pageid 62422990 is not transferred to it, and no zero labels are fabricated. Existing exclusions 1164/72417803 remain. Naevis 78522383 retains its review-only warning and is not automatically removed. Current mapping does not certify every article's historical identity, particularly for the M-12 seasonal feature.

## Formulas and candidates

Let M be the target month, c=M-1 the current complete month, D the calendar month length, I the published monthly internal incoming count, and v=I/D its daily rate.

For the six complete monthly daily rates ending at c, let m=median(v) and s=1.4826*median(|v-m|):

```text
U_M = max(200, D_M*m + max(100, 0.5*D_M*m, 3*D_M*s))
x_M = I_c * D_M / D_c
T_M = max(U_M, 1.25*x_M)
binary event = 1[actual_M > T_M]                 # strict >
R_M = max(0, actual_M / T_M - 1)
gR_M = log1p(R_M)                               # once
```

U's floor and padding are fixed policies, not a calibrated three-sigma probability interval. Binary requires actual>U **and** daily-rate month-over-month growth strictly greater than 25%; it does not predict raw future volume alone. The old absolute `max(0,actual-T)` is not this continuous target.

All methods share one candidate pool. Candidates require valid identity, current count>=100, published incoming observations for the current and six previous months, and complete quality/edit coverage. The current rate must be <=U_M/D_M, and the current count must be <=U_c frozen from the six preceding months, excluding the current month. During the two complete months `[start of M-2, start of M)`, every nonminor edit must have **absolute net byte change**<500 and their cumulative absolute change must be <1000. Net change is the revision size minus its true parent's size. Reverts count and minor edits are ignored. This net-byte proxy cannot detect large rewrites whose additions and deletions cancel.

**Training uses the quality-complete, current-cold expanded pool without the edit candidate filter.** Each fold uses only known-label rows with target<M; future M+ labels and unknown historical labels cannot enter TRAIN. The derived expanded pool has 11,080 rows.

## Features, preprocessing and frozen models

|index|original 12 columns|
|---:|---|
|0/1|log1p current/previous-month CS daily rate|
|2/3|log1p mean of last 3 / median of last 6 monthly CS daily rates|
|4|OLS slope of last 6 monthly daily rates, denominator 17.5|
|5/6|log1p target M-12 daily rate; current-rate fallback plus missing flag if absent|
|7/8/9|log1p current complete-month mean daily PV; difference between log1p recent-14/prior-14 mean PV; PV missing flag|
|10/11|log1p robust daily sigma; log1p(x_M)-log1p(U_M)|

Log13 appends `log1p(x_M)-log1p(T_M)` as column 13. An incomplete PV month leaves its associated values as None. Each mature TRAIN supplies median imputation, means and **population** standard deviations; std<1e-12 becomes 1. Missing flags and the computational median=0 fallback for an entirely unobserved column remain explicit. Unknown outcomes are not negative examples. Trees use the same TRAIN-standardized inputs to preserve the historical recipe.

Log13 minimizes row-mean BCE + 0.5*.01*||beta||^2 with an unpenalized intercept and no class/month weighting. It starts with zero slopes and a TRAIN-prevalence logit intercept, uses fixed damped Newton optimization, gradient<=1e-9, at most 200 iterations/60 line-search steps, and no retries. Ridge minimizes mean squared error +.01*||beta||^2 on gR, so sklearn `alpha=N*.01`, using cholesky with an intercept.

Both HGB models use 30 iterations, depth2, leaves4, bins32, lr=.03, L2=1, minleaf=max(50,ceil(.1N)) and seed718. There is no early stopping, validation_fraction, warm start or class weighting. Binary loss is log_loss; relative loss is squared_error. Each monthly expanding fold fits from scratch.

Numerical parameter selection must be distinguished from target/model development. Binary numerical recipes inherit early historical controls: F1 trains on 2023-10 through 2023-12 and evaluates 2024-01 through 2024-04; F2 trains through 2024-04 and evaluates 2024-05 through 2024-08. Log12 .01 was the winner; HGB depth2 was retained as a fixed comparator. Later joint-T features and rule/main-model interpretation used already-seen development diagnostics, so the entire selection process cannot be described as blind to development results. Identity correction keeps parameters fixed with no new search. Relative Ridge lambda selection uses only **2024-02/05/08**, comparing .01/.1 by equal-month graded NDCG50; .01 is selected, with ties<=1e-12 favoring stronger regularization .1. HGB is fixed, with no tree grid search. The [frozen early selection](config/early_selection.json), selection function and tests are included; the search was not rerun. Historical validation/dev split names do not restore independence.

## Rules and metrics

All methods score the same candidates, with score descending, current count descending and pageid ascending as tie order. Volume=current count; strong CS=`log1p(x_M)-log1p(T_M)`; momentum continues the current/previous daily-rate ratio, with an additional fixed ratio clip[.5,2] version in the relative benchmark. PV forecast=`x_M*exp(PV14logtrend)`; incomplete PV falls back to x_M. PV scores use log margins against U or T.

Floating-point ties are traceable: each run independently reconstructs rule formulas and retains the original canonical rule scores only after checking errors<=1e-12, matching the corrected evaluators. This handles rule arithmetic/ties and does not substitute for ML retraining. All four ML models were refitted, with identical complete rankings each month.

DCG=sum(gain/log2(rank+1)). Binary gain=event; relative gain=gR directly: **no second log and no 2^gain-1**. NDCG divides by that month's ideal-ranking DCG. AP is rank-tiebroken average precision over the complete candidate ranking, not truncated AP@50. Macro averages valid months equally; TP/slots/events are accumulated, while micro precision/recall and pooled gR capture use corresponding totals. Do not interchange macro and micro. Unknown candidates are not deleted, filled or replaced; any unknown future label makes the entire month NA. For a zero-event month, precision/TP=0 and NDCG/AP/recall are NA.

## Public acquisition and cached preparation

Default retraining is **offline**, using `data/panel.json.gz`, an approximately 2MB derived snapshot, and corresponding checksums. `data/provenance.json` stores portable relative source paths and SHA256 hashes for original artifacts and package inputs; `public_download_provenance.json` records public dump URLs/checksums. Derived data contain public article names/pageids and statistics, but no editor identities, revision text, comments, credentials or machine paths.

The following commands acquire into a fresh `wikiflow/local_data/`, preserve raw/cache files and refuse existing outputs. A fresh portfolio or API history may differ from the frozen vintage and does not automatically reproduce the original experiment exactly. Use the supplied fixed `data/title_mapping.csv` to reproduce the original scope.

```bash
mkdir -p wikiflow/local_data
python -m wikiflow.acquisition topic --output wikiflow/local_data/new_mapping.csv
python -m wikiflow.acquisition clickstream --month 2024-08 \
  --mapping wikiflow/data/title_mapping.csv \
  --raw wikiflow/local_data/clickstream-enwiki-2024-08.tsv.gz \
  --output wikiflow/local_data/incoming_2024-08.csv.gz
python -m wikiflow.acquisition pageviews --mapping wikiflow/data/title_mapping.csv \
  --start 2022-09-01 --end 2026-07-31 --output wikiflow/local_data/daily.csv.gz
python -m wikiflow.acquisition revisions --mapping wikiflow/data/title_mapping.csv \
  --start 2023-08-01 --end-exclusive 2026-08-01 \
  --output wikiflow/local_data/revisions.json --coverage wikiflow/local_data/coverage.json
```

Complete raw reconstruction requires monthly `incoming_YYYY-MM.csv.gz` files for **2022-09 through 2026-08**, which is expensive and outside the quickstart. No new download was performed during verification. Prepare from existing caches with:

```bash
python -m wikiflow.preparation --mapping wikiflow/data/title_mapping.csv \
  --incoming wikiflow/local_data --pageviews wikiflow/local_data/daily.csv.gz \
  --revisions wikiflow/local_data/revisions.json --coverage wikiflow/local_data/coverage.json \
  --output wikiflow/local_data/rebuilt_panel.json.gz
```

The prepared panel can feed the same fixed fitting/verification entry point. Its keys, labels, eligibility and numeric tolerances must match the frozen benchmark; otherwise stop and diagnose first:

```bash
python -m wikiflow.reproduce --mode refit \
  --prepared-panel wikiflow/local_data/rebuilt_panel.json.gz \
  --output wikiflow/runs/my-cached-refit
```

Acquisition retains revision ID/parent/timestamp/size/minor/SHA and coverage intervals and checks parent continuity. It does not request usernames, comments or text. Missing coverage, parents or hidden records imply unknown quality, not zero edits. PV parsing distinguishes explicit API zero from missing dates. Acquisition and preparation reject the holdout. Fresh downloads do not certify historical homonym title-slot identity; retain the quarantine policy and review separately.

The [cached reconstruction check](results/cached_preparation_verification.json) and public synthetic fixtures verify acquisition parsing and preparation boundaries. Historical feature generation used both NumPy and standard-library summation paths, so raw-cache reconstruction allows audited floating-point tolerances. The frozen derived snapshot retains original floats for exact prediction reproduction. Full fresh network acquisition, historical release-latency certification, portfolio-wide historical identity certification and sealed September testing remain unperformed.
