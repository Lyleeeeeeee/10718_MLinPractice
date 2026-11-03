# Recover the previous tree

Source commit: **3fc6e22eb189a9a9769147d0fcf894e859ca904e**, the pre-cleanup recall50-reselection branch.

All 101 tracked worktree files matched their Git blobs. There were no tracked LFS pointers/submodules. The independent ../backups/wikiflow-before-cleanup-3fc6e22.bundle passed bundle verification; Git fsck also passed. Old files are removed from the active tree, not moved to an internal archive.

Create a separate original worktree:

    git worktree add --detach ../wikiflow-before-cleanup 3fc6e22eb189a9a9769147d0fcf894e859ca904e

Extract one old artifact without overwriting current files:

    git show 3fc6e22eb189a9a9769147d0fcf894e859ca904e:wikiflow/binary/evidence/recall_reselection/comparison.csv > /tmp/wikiflow-old-comparison.csv

Clone the independent backup without network:

    git clone ../backups/wikiflow-before-cleanup-3fc6e22.bundle ../wikiflow-from-backup

| Prior content | Path in source commit |
|---|---|
| Binary package/old commands | wikiflow/binary/ |
| Recall reselection refits | wikiflow/binary/evidence/recall_reselection/ |
| Full trial recipes, including HGB/signed slope | wikiflow/binary/config/recipes.json |
| LDA/MiniLM variants and saved scores | wikiflow/binary/data/ and wikiflow/binary/evidence/ |
| Continuous Ridge baseline | wikiflow/src/wikiflow/learners.py, wikiflow/data/expected_scores.json.gz |
| Earlier PDF/LaTeX | WikiFlow_Report.pdf, wikiflow/binary/report/ |

The old root PDF is the older eight-page report, not the 13-page WikiFlow_Highlighted_v2.pdf provided in Library. It is not presented as the current report here. Report bodies were not edited.

## Non-Git work

This does not assert all caches, weights, local edits or research results were committed. Original research and submission checkouts remain unchanged. Status snapshots, staged/unstaged binary patches and the research untracked repair_semantics_audit.json were copied to independent ../backups/original_work/.

Ignored caches outside this isolated clone were not deleted. Untracked/ignored files encountered inside its removed old tree are independently preserved before removal; the external cleanup record lists them. They are not assumed recoverable from Git.

## Commands and compatibility

- Nested binary/reproduce.py → python run.py replay.
- Selection replay → python run.py verify for current Recall choices. Original P@5 replay remains in history.
- fit_fold.py --recipe ... → python run.py fit --model ... --month ... --allow-fit --output ....
- Retained historical model IDs resolve to the same recipes through models.json. Removed continuous/exploratory IDs are rejected instead of mapped to different models.

The reviewed cleanup tree was published to the original repository main branch on 2026-11-03 as commit 2d22541f5f525ec369732e82bd36d0a9018318d1, using a normal fast-forward update. The pre-cleanup commit remains an ancestor. No reset or history rewrite was used. Local backup paths above refer to the original cleanup workspace; they are not files hosted in this repository.
