# Notes on the first version

These notes describe the prototype reviewed on October 10, 2026, before restoring
the original data archive and running the current experiment. They explain what
changed; they are not a list of unresolved issues in the submission.

The first version had a modular Mask R-CNN pipeline, four preprocessing
checkpoints, and a plausible prediction overlay. Its main evidence was training
loss and counts on an unlabeled crop. That was enough to show a model could run,
but not enough to judge the proposed counting workflow.

| Issue in the first version | Change in the current experiment |
| --- | --- |
| Missing original `img1.png` | Restored the source archive and recorded file hashes |
| Regions inferred from labels | Selected inner frame corners from the photographs |
| No separate evaluation photograph | Trained two independent directional folds |
| Overlapping patch edges left uncovered | Added boundary-anchored tiles |
| Closing joined nearby outlines | Compared recovery with and without closing; kept closing off |
| Thousands of full-resolution masks | Stored each mask in its tight bounding box |
| Different training and inference scales | Used tiled inference at the training scale |
| Unlabeled totals treated as the main result | Reported mask matching, count error, baselines, and failure figures |
| No correction workflow | Added review tools, persistence, and traceable downloads |

The earlier raw, Gaussian, CLAHE, and combined runs had similar final training
losses. Those losses did not establish a best preprocessing method. The current
raw-input experiment is a focused new run rather than a continuation of those
checkpoints.

The original long review is retained locally in `archive/initial_review_full.md`
and in Git history. The original notebook and checkpoints are also preserved
locally. The current results, instructions, and limitations are in the README,
notebook, and `protocol.md`.
