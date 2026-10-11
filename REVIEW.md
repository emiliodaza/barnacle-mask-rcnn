# Final submission readiness

**Ready to share as an honestly evaluated DALI prototype.** This is a proposal and
human-review system, not a validated automatic counter. The original assessment
is preserved in [docs/initial_review.md](docs/initial_review.md).

## Addressed

- Original training photograph restored; supplied file hashes verified.
- Annotation-independent frame polygons, explicit centroid counting rule, and region-constrained inference.
- Compact instance masks, lazy patch targets, full edge coverage, retained negative patches, and consistent train/inference scale.
- Separate image-held-out training runs, fixed thresholds, one-to-one mask matching, count errors, baselines, sensitivity results, and failure figures.
- Independent final model for the official unlabeled demonstration; no claimed unseen accuracy.
- Functional browser review: add/remove, undo, threshold exploration, tile navigation, frame selection, persistence, and audit/point export.
- Readable modules, pinned dependencies, source/model hashes, reproduction instructions, and a clean executed notebook.

## Verification

- 12 Python regression checks pass, including tiled coordinate mapping and duplicate removal without label access.
- Browser counting and validation checks pass; JavaScript syntax checks pass.
- Ruff lint and formatting checks pass; installed Python dependencies have no reported conflicts.
- Both five-epoch folds and the separate final model were trained and evaluated locally.
- Repeated final-model inference exactly reproduced all 973 cached instance records at score 0.25, including the 820 proposals above score 0.5. Original image hash verified.
- The notebook executes all eight code cells, embeds five figures, and has no error outputs or stderr streams.
- Manual browser checks verified adding/removing proposals, undo, reload persistence, tile inspection, example selection, and JSON audit export.

## Remaining scientific limitations

Two photographs cannot establish generalization. Labels and manual frame corners
have uncertainty; threshold/mask matching, incomplete labels, and tile seams
limit interpretation. No scientist usability/time study has been performed.
Exports remain provisional. These limits are explicit in the README and notebook,
consistent with the challenge's request for an honest small experiment.

Before the actual application, explain the decisions and limitations in your own
words, and supply the separately required outside code sample. Admission cannot
be guaranteed by this artifact.
