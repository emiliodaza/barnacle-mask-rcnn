# Development notes

The first version tried four preprocessing options with Mask R-CNN. It had saved
checkpoints and prediction figures, but training loss was the main comparison.
The original review, kept in `initial_review.md`, identified missing source data,
annotation-derived crops, uncovered patch edges, and the lack of held-out evaluation.

Restoring the supplied archive made it possible to fix the data pipeline and run
new experiments. The current version selects frame corners from RGB photographs,
keeps neighboring annotation outlines separate, stores compact masks, retains
empty tiles and visible fragments, and covers every image edge. Two independent
training runs evaluate across the original photographs. A third model supplies
the unlabeled demonstration.

The browser tool grew from that result: proposals need correction, so the
interface keeps model output separate from a scientist's edits. Downloads include
the selected frame, score cutoff, source/model identity, and edit history. The
visual design was simplified into a compact count-review workspace.

The original notebook remains locally in `archive/`, and its four checkpoints
remain in `outputs/`; neither is included in Git. Their training losses are not
presented as held-out performance. The raw-input protocol is the focused experiment
reported here, not a winner selected from those earlier loss values.

Codex assisted with implementation, cleanup, verification, and documentation.
Training, inference, and evaluation were executed locally. The application owner
will review and explain the implementation before submitting.

The final cleanup adds function contracts where coordinate conventions or scientific
assumptions are easy to miss, removes unused annotation-cropping helpers, gives the
browser state clearer names, and provides a reading guide. It does not change the
reported model settings or reinterpret the measurements.
