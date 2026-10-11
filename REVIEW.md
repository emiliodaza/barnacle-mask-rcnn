# Submission note

The repository is ready to present as a DALI challenge prototype. It includes a
working demonstration, a reproducible experiment, failure analysis, and an
executed notebook. The model is imperfect, which is consistent with the brief;
the results explain where it fails and what to investigate next.

## What to share

Submit the public repository:
[emiliodaza/barnacle-mask-rcnn](https://github.com/emiliodaza/barnacle-mask-rcnn).
The README is the entry point. The notebook is the experiment narrative, and the
browser examples can be tried without installing machine-learning dependencies.
Checkpoints are available in the linked release.

## Checks

- Python regression tests cover tile edges, duplicate suppression, coordinate mapping, label separation, matching, and empty targets.
- Browser logic tests cover counts, thresholds, frame boundaries, and imported data.
- Both held-out folds and the separate demonstration model were trained locally; the results and source hashes are saved.
- Independent inference reproduced the saved demonstration proposals.
- The notebook executes without error or warning output.
- Browser checks cover corrections, undo, persistence, tile navigation, sample selection, and downloads.

The fixed protocol is in `docs/protocol.md`. The earlier assessment in
`docs/initial_review.md` records the project before these repairs; its missing-file
and publication findings are historical.

## Before sending the application

Read the notebook and [code walkthrough](docs/code_walkthrough.md), then explain
the choices and limitations in your own words. In particular, understand why the
two photographs are split before tiling, why the references are imperfect, and
why a plausible total is not enough evidence of good predictions.

The remaining application item is DALI's **separate outside code sample**.
This repository does not replace it. Any personal learning reflection in the
application should describe your own understanding and experience.

Scientific validation remains future work: more independent photographs,
confirmed counting conventions, a label audit, and a comparison of manual versus
assisted review time and final accuracy. Those limits are described in the
submission rather than hidden behind the demo.
