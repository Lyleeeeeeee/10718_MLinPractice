# Audit record: 2026-10-05

This audit packages only the reported identity-corrected binary / relative continuous benchmarks and preserves their frozen mathematical definitions. Source experiment directories and shared caches were read-only; work used an isolated clone. The repository's original main contained only a README, and no experiment ledgers were rewritten.

## Reviewed contracts

- `identity_policy.py` / manifest v1: block 79745475, do not join the homonymous director's history, and retain Naevis as review-only. The public identity-policy copy includes only necessary rules and the original SHA.
- `preflight.py` / `fit26.py`: verify Log13's joint-T column 13, mean-BCE L2=.01, Newton initialization/stopping, fixed HGB12 parameters and own-TRAIN minimum leaf size.
- `training_contract.py` / `run_authorized26.py`: verify relative response `gR=log1p(max(0,actual/T-1))`, Ridge alpha=N*lambda, mature-TRAIN-only models and scalers, and the absence of an edit-amount training filter.
- `pilot_math.py` / `relative_contract.py` / both corrected evaluators: whole-month NA for unknown labels, zero-event support, ties, macro/micro aggregation, full-list AP and one transformation of graded gain.
- `rolling_training_support_zero_fit/verify_support_no_fit.py` and v2 `prepare_v2_no_fit.py`: calendar-normalized daily rates, seven complete published months, both current-cold gates, and maximum/cumulative parent-net-byte thresholds over two complete months.
- Acquisition/daily parser: literal TSV, all internal sources, count10 boundary, canonical-only destinations, explicit PV0 versus missing dates, and coverage/parent/hidden-record guards. New acquisition scripts preserve raw inputs and do not use the originals' automatic-deletion option.

The original programs depend on deeply nested sessions and hardcoded machine paths and update shared cumulative ledgers, so they are unsuitable as a safe direct reproduction entry point. This package extracts pure computation/model code, replaces paths with portable directories and stops on failed validation. Identity, mathematics, candidates and model recipes remain consistent. Source hashes and required derived-artifact hashes are in `data/provenance.json`.

## Exact time and selection boundaries

For target M, TRAIN includes only known labels with target<=M-1. Each row's features end at its own target-1, and prediction features end at M-1. Current portfolio/title information and revised vintage are applied retrospectively. Original release latency was not verified; immediate maturity is an operational assumption.

Inherited early binary capacity controls:

| fold | TRAIN | INNER evaluation | TRAIN N/P |
|---|---|---|---:|
|F1|2023-10 through 2023-12|2024-01 through 2024-04|649/40|
|F2|2023-10 through 2024-04|2024-05 through 2024-08|1590/71|

Four fixed candidates were Log12 lambda=.1/.01 and HGB stump/depth2. Equal-month binary NDCG50 across eight INNER months selected Log12 .01 (.155725). Depth2 HGB (.110317) was retained as a fixed comparator, **not the early winner**. Later v2 targets, joint-T Log13 features, rules and main-model choices used historical diagnostics. End-to-end selection therefore cannot be claimed to have used only unseen early history. Numerical parameters remain unchanged; identity correction and packaging introduced no new tuning.

Relative lambda=.01/.1 selection used only the three early historical boundaries 2024-02/05/08, with equal-month graded NDCG50 selecting .01. HGB was fixed. Backtest means do not establish independently validated superiority for the existing strong rules either.

## Reproduction and limitations

Actual retraining covers 24 months x 4 families. Verification checks every score, 96 complete monthly rankings, all K=5/10/20/50/100/200, 2,160 method/month/K records and 6,480 NDCG/AP/TP fields. A separate preparation pass from 48 months of raw caches wrote new temporary outputs with 11,080 expanded rows / 5,388 candidates. Identity, labels, eligibility and thresholds matched exactly; feature floating-point error was <=4.55e-13. Evidence in `results/` records actual fit counts, dependencies, seed, threads, elapsed time and tolerances.

No substantive bug affecting reported metrics was found, and no reported numbers were replaced. Candidate serialization in the new package was corrected to history=target/article and evaluation=target/pageid, matching authoritative views. This does not change the original experiments.

Packaging did not reacquire the full dataset, run new tuning, read September 2026 or rerun bootstrap estimation. Offline fixtures verify parsing of network responses; they do not demonstrate successful fresh API acquisition. Missing historical identity certificates, changing API vintage, release latency, net-byte edit proxies, the non-pure-AI maintenance scope, suppressed clickstream counts and repeated development selection all limit generalization. Independent test performance and actual deployment benefits remain unverified.
