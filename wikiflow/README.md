# WikiFlow: reproducible course baselines

Each month, rank articles that are currently cold and have had few substantial recent edits, for human review of possible increased internal navigation next month. The fixed WikiProject Artificial Intelligence maintenance portfolio contains 1,187 canonical articles, including people and works. Membership is a maintenance scope, not a pure AI technology collection; edit activity is a candidate proxy, not a content-quality label.

This is a course baseline subset of the [complete benchmark](https://github.com/Lyleeeeeeee/10718_MLinPractice/tree/ccf65a63287330870b43b7cb563d4cd20348d3a8); other studied rules and nonlinear comparators remain in that commit and the repository history.

The current methods are **volume** (non-ML current incoming clickstream count descending), **binary Log13** and **relative continuous Ridge12**. Volume is evaluated identically for both tasks. The [design](docs/design.md) explains the data, features, training and evaluation contracts; the [audit](docs/audit.md) records verification and limitations. No method is claimed to be globally optimal or consistently superior to previously studied strong rules.

## Offline reproduction

Run from the repository root. Verified environment: Python 3.13.5, NumPy 2.3.3, SciPy 1.16.2, scikit-learn 1.7.2 and threadpoolctl 3.6.0. Matching these versions is recommended for exact numerical reproduction.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r wikiflow/requirements.txt
export PYTHONPATH=wikiflow/src
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
python -m unittest discover -s wikiflow/tests -v
python -m wikiflow.reproduce --mode refit --output wikiflow/runs/my-refit
```

`refit` performs **24 x 2 = 48 actual fixed-parameter fits** on the supplied cached derived panel. Every month refits Log13 and Ridge12 from scratch, with its own mature TRAIN and TRAIN-only median imputation/scaler. It verifies all model scores, complete monthly rankings and metrics for K=5,10,20,50,100,200 against the retained historical predictions. No network, new tuning or holdout data are needed. Input hashes and TRAIN-key hashes are checked; a mismatch stops the run. Outputs are preserved, so use a fresh output directory.

```bash
python -m wikiflow.reproduce --mode replay --output wikiflow/runs/my-replay
```

`replay` recalculates metrics from stored predictions and performs **0 fits**. Both modes produce sealed predictions, monthly and aggregate CSVs, and `verification.json`; refit also produces `fit_trace.json`. The [48-fit trace](results/fit_trace.json) preserves the previous fixed refit; the current [verification](results/verification.json) records a zero-fit evaluation replay. [Fit accounting](results/fit_accounting.json) distinguishes the two. The [subset integrity report](results/subset_integrity.json) checks the projection from the previous commit. Parameters in `config/benchmark.json` and `config/early_selection.json` document fixed recipes, rather than exposing a new search or runtime tuning interface.

## Retained results at K=50

The raw candidate pool contains **5,388 article-months**; evaluation uses **5,387** across 24 development months (2024-09 through 2026-08), with 23 event months, 190 events and 1,200 Top50 slots. Rows with missing ground-truth labels are excluded from evaluation. April 2025 has no events: TP/precision contribute zero while NDCG/AP/recall are NA.

| Task | Method | Macro task NDCG50 | Macro full-list AP | Cumulative TP50 |
|---|---|---:|---:|---:|
| Binary | Volume | 0.123528442060 | 0.061531406346 | 47 |
| Binary | Log13 | 0.265096468808 | 0.128429231632 | 81 |
| Relative | Volume | 0.089714087859 | 0.061531406346 | 47 |
| Relative | Ridge12 | 0.144105150570 | 0.099854317205 | 69 |

The [summary](results/summary_at50.csv) and [monthly table](results/monthly_at50.csv) contain only these four task/method pairs. Relative NDCG uses `gR` directly. AP always uses binary events across the full ranking, even when scores come from a relative model. These repeatedly studied historical months are **development backtests**, not independent tests. The sealed 2026-09 holdout is absent and remains unopened. Subsetting does not undo earlier model/feature development or establish stable ML superiority.

## Public data and identity v1

Inputs are [Wikimedia monthly Clickstream](https://dumps.wikimedia.org/other/clickstream/readme.html), [daily pageviews](https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/reference/page-views.html), [MediaWiki revisions](https://www.mediawiki.org/wiki/API:Revisions) and current WikiProject talk-category membership. Monthly incoming CS sums published internal `link` edges to literal canonical destinations, from any internal source. It counts transitions, not unique users. Published edges have count>=10; suppressed or missing article/month observations are **unknown**, not zero. Redirects define the canonical catalogue but alias destination counts were not separately aggregated.

CS inputs span 2022-09 through 2026-08. Model target months start at 2023-10; features for target M end at M-1. Final PV date is 2026-07-31; revisions end before 2026-08-01. The derived panel contains 11,080 expanded rows. Current 2026 catalogue/title mappings and revised dump vintage are retrospective; immediate prior-month CS/label maturity is assumed. Historical publication latency and all historical identities have not been certified.

The [identity policy](config/identity_policy.json) retains **v1**: pageid 79745475 (Daniel Kokotajlo homonym) is blocked in TRAIN and evaluation, director 62422990 history is not transferred, and existing exclusions 1164/72417803 remain. Naevis 78522383 remains review-only.

**Known Brainiac limitation, retained without correction:** pageid 525160 identifies the character, but before the August 2026 title swap the cached literal title `Brainiac` identifies disambiguation page 384942. CS/PV under that title were joined to character revision metadata. The mismatch affects 24 expanded rows, five evaluation rows and every rolling TRAIN fold; correct character exposure/labels remain unknown. July dump canonicalization vintage is also uncertified. This package preserves the original v1 rows and labels for reproducibility; it does not apply a newer identity correction, quarantine extra rows, flip labels or infer zero traffic. See the [audit](docs/audit.md) for the evidence boundary.

## Labels, candidates and training

Let target M have D_M days, current complete month c=M-1 have D_c days, I_c be published incoming CS, and v=I/D. For six monthly daily rates ending at c, m is their median and s=1.4826*median(abs(v-m)).

```text
U_M = max(200, D_M*m + max(100, 0.5*D_M*m, 3*D_M*s))
x_M = I_c * D_M / D_c
T_M = max(U_M, 1.25*x_M)
event = 1[actual_M > T_M]          # strict >
R_M = max(0, actual_M/T_M - 1)
gR_M = log1p(R_M)                 # natural log, exactly once
```

U is a fixed heuristic bound, not a calibrated confidence interval. Binary events exceed both historical U and 25% daily-rate growth. The relative response is dimensionless excess over T; the older absolute excess is outside this package.

Evaluation requires valid identity, current count>=100, seven published months through c, complete metadata/edit coverage, current daily rate<=U_M/D_M and current count<=the precurrent U_c built from six months excluding c. Across `[start M-2, start M)`, each nonminor edit must have absolute parent-net-byte change<500 and their summed absolute change must be<1000. Equality excludes; reverts count; minor edits are ignored. Missing parent/coverage or hidden records imply unknown quality. Net bytes cannot detect rewrites whose additions and deletions cancel.

TRAIN expands over quality-complete, current-cold rows **without the edit-amount candidate filter**. Each fold uses only known outcomes with target<M; every row uses features no later than its own target-1. Scalers use only that fold's TRAIN. [Design](docs/design.md) lists all 12 original features, Log13's additional joint-T margin, fixed objectives and inherited early selection.

Score ties use current count descending then pageid ascending. DCG is sum(gain/log2(rank+1)), with gain=event for binary and gain=gR for relative; no second log or exponential gain. AP uses the complete ordered ranking. Macro metrics average valid months equally; micro precision/recall and pooled graded capture use summed totals.

## Acquisition and preparation commands

Quick reproduction is offline with `data/panel.json.gz` and checksum-checked references. Public-derived data contain article names/pageids/statistics, without editor identities, revision text, comments, credentials or machine paths. [Provenance](data/provenance.json) records the previous complete commit, source hashes and package hashes; [public download provenance](data/public_download_provenance.json) preserves dump URLs/checksums.

Optional fresh acquisition can be costly and is **not** part of the verified quickstart. Commands write fresh local outputs and refuse replacement of existing files. A current catalogue or changed API/dump vintage may not reproduce the historical experiment; use the supplied fixed mapping for its scope.

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

Complete raw reconstruction needs incoming extracts for 2022-09 through 2026-08. No such download was run for this subset. With already acquired caches:

```bash
python -m wikiflow.preparation --mapping wikiflow/data/title_mapping.csv \
  --incoming wikiflow/local_data --pageviews wikiflow/local_data/daily.csv.gz \
  --revisions wikiflow/local_data/revisions.json --coverage wikiflow/local_data/coverage.json \
  --output wikiflow/local_data/rebuilt_panel.json.gz
python -m wikiflow.reproduce --mode refit \
  --prepared-panel wikiflow/local_data/rebuilt_panel.json.gz \
  --output wikiflow/runs/my-cached-refit
```

Preparation verifies parent chains, full windows, explicit PV zeros versus missing days, canonical destination scope and holdout boundaries. The optional prepared panel must match the frozen keys, labels and eligibility exactly, with numerical tolerance for historical summation roundoff. Its `wikiflow.course.panel.v1` schema is a packaging name, not a newer identity version. Fresh downloads, complete raw-cache reconstruction for this subset, release-latency certification, portfolio-wide identity repair, bootstrap intervals and independent holdout performance are not claimed.
