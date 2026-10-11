# Barnacle — counting proposals a scientist can check

A DALI 26F Data Challenge prototype for counting barnacles inside a field frame.
It combines **Mask R-CNN proposals, image-level evaluation, and a human correction
interface**. The aim is to give a scientist a useful starting point and a record
of their corrections. It does **not** claim reliable automatic counting or a
measured reduction in counting time.

![Scientist review interface](results/figures/review_interface.jpg)

## Try it in two minutes

The committed browser examples contain **real cached model outputs**, including
the official unlabeled photograph. Reviewing them requires only Python, with no
ML packages, account, or downloaded checkpoint:

```bash
git clone https://github.com/emiliodaza/barnacle-mask-rcnn.git
cd barnacle-mask-rcnn
python3 -m http.server 8000 --bind 127.0.0.1
```

Open **http://127.0.0.1:8000/demo/**. Choose a photograph, inspect teal proposals
within the gold frame, remove extras, mark missed barnacles, and export the
review. Changes persist locally in your browser. Read the
[review workflow](docs/review_workflow.md) for custom photos, shortcuts, and exports.
Python inference and browser review are separate steps; the browser does not run
the model. The default photo's **820 proposals are not a verified count**.

## Problem and approach

Scientists currently count hundreds or thousands of barnacles by hand. A useful
system should take a photograph and a scientist-selected inner frame region,
propose individual barnacles, expose uncertain decisions for review, and export
the corrected count. Selecting four corners manually is a deliberate prototype
choice: it avoids using annotation masks that will not exist in the field.

I started with instance segmentation because a mask makes a proposal inspectable
and supports individual corrections. A pretrained backbone offers a practical
starting point with only two annotated photographs. The critical subtask is
**instance proposals inside a selected region**, not unattended final counting.
The input is an RGB photo and four corners in original pixels; the output is
compact instance masks, centers, scores, and an editable proposal count.

The counting convention is **instance-mask centroid inside or on the polygon**.
This is an explicit assumption to confirm with scientists. Pixels outside the
region are excluded during training and inference. Full images are processed in
512-pixel overlapping tiles with complete edge coverage and mask-based duplicate
suppression. Compact masks avoid storing thousands of full-resolution arrays.
The corrected count remains provisional until the scientist has inspected it.

## What the experiment found

Each experiment trains on **one original photograph** and evaluates the other.
Splitting before making patches prevents overlapping patches leaking between
training and evaluation. The original checkpoints trained on both photos were
not reused for these folds. Each new run starts from COCO weights with seed 42,
five epochs, a frozen backbone, and raw RGB inputs.

The primary score threshold is **0.5**, with mask and matching IoU thresholds of
0.5. Matching is one-to-one. References are recovered from the supplied blue
contours, which are described as incomplete; these metrics measure annotation
agreement, not verified biological truth.

| Train → evaluation | Reference count | Model count | Absolute error | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Image 1 → image 2 | 389 | 756 | 367 | 31.1% | 60.4% | 0.410 |
| Image 2 → image 1 | 1,702 | 1,027 | 675 | 84.6% | 51.1% | 0.637 |

The naive bright-component baseline reaches F1 **0.003** on image 2 and **0.000**
on image 1. Its count errors are 155 and 720 respectively. Notice that it has
**lower count error than the model on image 2 while barely matching any instances**:
misses and extras can cancel. A training-density baseline also fails across the
two scenes (count errors 1,805 and 1,400). These are deliberately simple reference
methods, not a claim that classical computer vision cannot work.

![Held-out sparse photograph and model errors](results/figures/heldout_2.png)

Teal marks matches, orange marks extra proposals, and blue marks missed
annotations. The dense-image model overcounts the darker photograph; the
sparse-image model misses many instances in the dense photograph. Shape errors
at the IoU cutoff and incomplete labels also contribute to disagreements. Higher
thresholds trade recall for precision but do not repair the scene shift.

[Full metrics](results/metrics.csv), [run provenance](results/report.json), and
[declared protocol](docs/protocol.md) include the score sensitivity analysis,
training losses, source hashes, runtime, and all baseline results. The separate
model trained on both photographs produces 820 proposals on `unseen_img1.png` at
score 0.5. **That photograph has no reference annotations, so its accuracy is
unknown.** These results support further work on assisted review; they do not
support deploying automatic counts.

## Reproduce the model experiment

Python **3.11–3.14** is the intended range. The recorded run used **Python 3.14.6,
macOS ARM64, CPU, and two PyTorch threads**; package versions are pinned. Other
hardware or versions can produce different floating-point results.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.experiment --epochs 5
```

This retrains both folds and the separate final model, evaluates them, and
regenerates metrics, figures, and cached demo proposals. The recorded protocol
took **about eight minutes** on this machine; runtime elsewhere will vary.
Initial training downloads TorchVision's COCO weights. Later checkpoint inference
constructs the architecture without a pretrained download. The dataset images
are included unchanged, with SHA-256 hashes in `data/manifest.json`; credit for
photographs and blue annotations belongs to the supplied DALI challenge dataset.

You can reuse checkpoints whose fold provenance matches the protocol:

```bash
python -m src.experiment --epochs 5 --reuse
```

For inference alone, obtain `final_raw.pth` from the
[model release](https://github.com/emiliodaza/barnacle-mask-rcnn/releases/tag/v1.0.0),
place it in `outputs/`, then run:

```bash
python -m src.predict --image data/unseen_img1.png \
  --checkpoint outputs/final_raw.pth --score-threshold 0.25 \
  --output results/my-proposals.json
```

Load the corresponding original photograph and proposal JSON in the browser
review tool. For a new photograph, first select and export its frame there, then
pass `--region path/to/frame-region.json` to Python inference. The model assumes
comparable pixel scale; changing a region in the browser does not recompute
proposals beyond the original inference region.

[The executed notebook](barnacle_notebook.ipynb) gives a linear explanation of
the data, prototype, metrics, failure figures, and conclusions. Its ordinary
cells inspect committed results; expensive training is an explicit separate
command. To open it interactively:

```bash
python -m jupyterlab barnacle_notebook.ipynb
```

Optional browser logic checks require Node.js: `node tests/test_review.cjs`.
The browser app itself requires no Node installation or build step.

## Learning and next decisions

The main lesson is that the data pipeline changes the experiment as much as the
model. The old 3×3 contour-closing step joined neighboring labels: removing it
changed recovered image-1 contours from 1,500 to 1,718 before region filtering.
This is not proof that every recovered contour is correct, so the annotation
conversion is shown explicitly in audit figures. Compact masks reduce that
image's mask storage from a hypothetical 13.7 GB to about 0.9 MB.

The second lesson is that decreasing training loss and plausible-looking
predictions cannot substitute for held-out evaluation. The third is that good
count agreement can conceal wrong individual decisions. Together these findings
changed the emphasis from an automatic counter to proposals a scientist can
inspect and correct.

Next, obtain independently annotated photographs spanning density, lighting,
and camera scale; confirm live/empty-shell and boundary conventions; audit label
recovery; and reserve a validation set before tuning thresholds. Compare
manual-only and assisted review on verified regions, measuring both final error
and task time. The review interface makes that next experiment possible, but no
usability study has yet been conducted. More model complexity is not the first
priority.

## Files and provenance

| Path | Role |
| --- | --- |
| `barnacle_notebook.ipynb` | Executed, readable experiment narrative |
| `src/` | Compact masks, regions, dataset, model, training, inference, baselines, evaluation, figures |
| `config/regions.json` | Annotation-independent manual inner-frame coordinates |
| `data/` | Unmodified supplied images and hash manifest |
| `results/` | Actual measured metrics, training provenance, comparisons, and audit figures |
| `demo/` | Dependency-free review interface with real cached proposals |
| `tests/` | Counting, matching, seam, input, and browser-logic regression checks |
| `docs/` | Protocol, review instructions, original review, and development record |
| `scripts/build_notebook.py` | Rebuild the notebook narrative from the committed outputs |
| `outputs/` | Locally generated checkpoints; excluded from Git and distributed in the release |

This extends the earlier four-preprocessing Mask R-CNN prototype. Its historical
training losses were not treated as evidence of held-out accuracy. The revision
used Codex assistance for refactoring, checks, the interface, and documentation;
the results above come from actual local runs. Technical lessons are tied to
these experiments rather than invented personal experience. `unseen_img2.png`
also exists in the supplied archive, but its origin relative to the other
photographs is unspecified; it is excluded from quantitative evaluation.

References: [TorchVision fine-tuning tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html),
[Mask R-CNN model contract](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html),
and [SciPy assignment implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html).
The implementation adapts the standard transfer-learning head replacement;
the surrounding counting, sparse-mask, evaluation, and review pipeline is in this
repository. DALI also requires an **outside code sample separate from this challenge**.
