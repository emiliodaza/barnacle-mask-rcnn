# Development record

The initial review is preserved in `docs/initial_review.md`. It described the
project before the source archive was restored, so its missing-file and readiness
statements are historical rather than the final state.

The revision restored `img1.png` from the supplied archive and checked existing
files against the originals. It replaced mask-derived regions with selected frame
polygons; changed default annotation extraction to avoid closing neighboring
outlines; added lazy compact masks and complete boundary patch coverage; retained
negative patches and visible object fragments; implemented matching, tiled
inference, duplicate suppression, provenance, baselines, and new independent
training runs; and added the review interface and executed notebook.

The original four pretrained prototype checkpoints remain locally in `outputs/`,
and the original notebook is preserved locally in `archive/`. Those artifacts are
excluded from the submitted Git history. New held-out metrics are not attributed
to the historical models. The new raw-input protocol was chosen as a focused
experiment; the old preprocessing-loss comparison does not select a winner.

Codex assisted with the revision, including implementation and documentation.
Results are actual executions, not fabricated outcomes or personal learning
claims. The application owner should be able to explain all decisions and
limitations before submitting. The challenge's separate outside code sample is
still an application requirement.

Verification includes Python regression checks, browser-counting tests, actual
training and evaluation of both folds, final-model inference, notebook execution,
and manual browser checks of count edits, undo, reload persistence, tile navigation,
and audit export. See the final readiness note for the exact completed checks.
