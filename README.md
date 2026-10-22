# WikiFlow course baselines

A reproducible monthly Wikipedia ranking study for 10-718. The current package contains one non-ML baseline (current incoming clickstream volume), binary Logistic13 and relative continuous Ridge12.

This is a course baseline subset of the [complete benchmark](https://github.com/Lyleeeeeeee/10718_MLinPractice/tree/ccf65a63287330870b43b7cb563d4cd20348d3a8); other studied rules and nonlinear comparators remain in that commit and the repository history.

Start with the [README and reproduction commands](wikiflow/README.md), [design](wikiflow/docs/design.md) and [audit](wikiflow/docs/audit.md). All historical results are development backtests. The retained identity-v1 snapshot has a documented Brainiac subject mismatch; these results are reproducible inherited comparisons, not a claim of fully verified historical identities or globally optimal models.

## Initial ML Solution — binary ranking (22 October 2026)

The expanded binary-ranking report and its supporting code are available in the incremental submission directory:

- [Report PDF](submissions/initial_ml_binary/report/main.pdf)
- [Code and reproduction instructions](submissions/initial_ml_binary/README.md)
- [Report-to-code mapping](submissions/initial_ml_binary/REPORT_MAP.md)

The submission reuses the existing WikiFlow baseline modules. The original baseline documentation is retained.
