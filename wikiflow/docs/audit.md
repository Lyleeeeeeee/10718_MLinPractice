# Course-subset audit

The current package is a projection of the [complete benchmark commit ccf65a6](https://github.com/Lyleeeeeeee/10718_MLinPractice/tree/ccf65a63287330870b43b7cb563d4cd20348d3a8). It retains volume, binary Log13 and relative Ridge12, with acquisition, preparation, identity v1, unchanged features/labels/candidates, fixed parameter provenance and evaluation. Other implementations, configs, predictions and results are removed from the current package and remain recoverable in ordinary Git history. Source sessions, caches, experiments and training ledgers are read-only.

## Mathematical review

The prior review checked original identity policy, frozen binary Newton objective/13th feature, relative response/Ridge alpha, own mature TRAIN, TRAIN-only imputation/scaling and expanding folds. Calendar-normalized CS, month lengths, seasonal target M-12, two-month UTC editor windows, parent-net-byte thresholds, strict labels, unknown outcomes, tie order, shared candidates, zero-event support, full-list AP, macro/micro and direct gR gain are preserved in the [design](design.md).

The course-subset extraction preserved the mathematical recipes. The panel schema becomes `wikiflow.course.panel.v1` and the legacy eligibility field is renamed `evaluation_eligible_asof`; these are presentation changes, not identity changes. The unused rule forecast fields are removed. Stored scores and reference metric rows were originally selected without recomputation from the previous commit. All retained row values, labels, raw candidate keys/order, feature floats and prediction values are exactly preserved, as checked in [subset integrity](../results/subset_integrity.json). April evaluation was subsequently replayed without fitting; other monthly values are unchanged. The shared 12-column preprocessing record is named Ridge12 instead of its historical comparator label, with the numeric arrays unchanged.

The current evaluator also requires identical complete monthly reference support, preventing an incomplete expected table from silently passing. Scope tests require precisely two ML learners and a volume-only non-ML scorer. Fixed fitting validates score keys, own TRAIN hashes, original preprocessing and complete monthly model ranks. The source objectives and thresholds are neither tuned nor repaired in this step.

## Retained identity limitation

Identity v1 corrects the Daniel Kokotajlo homonym error by blocking 79745475, retaining the existing exclusions and never transferring director history or fabricating unknown labels. It does not certify every other historical title slot.

A later read-only investigation confirmed that stable character pageid **525160** used title `Brainiac (character)` before the August 2026 swap, while literal title `Brainiac` belonged to disambiguation pageid **384942**. The disambiguation move occurred at 2026-08-01 00:54:28 UTC and the character move at 00:56:49 UTC. Cached CS/PV under literal `Brainiac` were joined to character revision metadata. Official [move logs](https://en.wikipedia.org/w/index.php?title=Special:Log/move&page=Brainiac&year=2026&month=8&day=31) and the [historical disambiguation revision](https://en.wikipedia.org/w/index.php?oldid=929228048) support the title distinction. Pageview requests are title-specific, with [redirect attribution rules](https://wikitech.wikimedia.org/wiki/Analytics/Data_Lake/Traffic/Pageviews/Redirects#Other_titles_covered_by_a_redirect_page) that do not repair this subject mismatch.

The existing investigation finds 24 affected expanded rows, five evaluation rows and dependencies in all 24 TRAIN folds. Former-title character exposure was outside retained extracts; correct character outcomes are unknown, not zero. July dump title/canonicalization vintage is separately unverified. No corrected label, label flip or clean identity certificate is inferred here. This package **retains v1 unchanged**, records the limitation, and does not adopt the newer identity proposal. All reported scores therefore describe this imperfect inherited development snapshot. A future repair would affect TRAIN as well as evaluation and must be versioned separately.

The small investigation report was read solely to document this existing limitation. No new features, negative-control diagnostics, source-channel experiments, canceled correction outputs or holdout outcomes were imported.

## Verification evidence

The [trace](../results/fit_trace.json) preserves the previous **48 actual fixed fits**, 24 Log13 and 24 Ridge12, including original prediction, monthly rank and TRAIN-scaler checks. The current [verification report](../results/verification.json) records **zero new fits** and replays all 576 task/method/month/K records (1,728 NDCG/AP/TP fields). Original scores and raw labels remain byte-identical; only April monthly metrics and resulting aggregates change. The report retains the excluded raw key for traceability. Score tolerance is 1e-10; metric tolerance is 5e-12. Volume is independently reconstructed from current counts and never replaced by another rule.

Twenty-six offline unit tests cover acquisition parsing, published edge boundary, explicit PV zero versus missing, revision identity/privacy, calendar/strict labels, identity/candidates, unknowns, TRAIN exclusions/scalers, one gain transform, macro/micro, early-lambda ties, Newton derivatives, preparation windows and the current subset guards. They do not perform network acquisition or large training.

Committed [summary](../results/summary_at50.csv) and [monthly results](../results/monthly_at50.csv) contain only four task/method pairs. The [fit accounting](../results/fit_accounting.json) applies to this subset validation; historical full-benchmark validation is available in the previous commit. Model/parameter search was not rerun.

## Evidence limits

Historical months were repeatedly studied and remain development backtests. Numerical lambda provenance is early historical, but later Log13 feature development and method interpretation used development diagnostics. No independent-test, stable strong-rule superiority or causal claim follows from retained averages.

Only cached-derived offline refitting and public synthetic preparation/acquisition fixtures were run for this subset. Full fresh acquisition and a new complete raw-cache reconstruction were not run. The previous complete commit preserves its broader cached-path evidence. Historical release latency, current-versus-historical catalogue/identity validity, July dump canonicalization and Brainiac character traffic remain unresolved. The sealed September 2026 holdout was not opened.

The package includes public-derived statistics and small provenance records, without original raw dumps, experiment folders, actor information, revision content, comments, machine paths, model pickles or credentials. Publishing uses an isolated checkout, an explicit file allowlist and a new ordinary descendant commit; previous history and source workspaces are preserved.
