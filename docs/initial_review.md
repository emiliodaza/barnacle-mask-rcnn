# DALI Data Challenge submission review

Reviewed against the supplied 26F challenge on October 10, 2026.

**Decision: not ready to publish as the finished challenge submission.** There is a credible Mask R-CNN prototype here, including trained checkpoints and a prediction figure. The main missing pieces are a reproducible run, task-correct counting, and an honest evaluation. The brief explicitly welcomes imperfect prototypes and values clear reasoning over sophisticated tools. No submission can guarantee admission.

This review adds feedback only; it does not alter the implementation, retrain models, or upload anything to GitHub.

## What is already good

- Responsibilities are separated into annotation processing, dataset construction, preprocessing, training, prediction, and visualization modules. This follows the brief's suggestion to keep notebook code short.
- Blue contour annotations are converted into filled instance masks, and the dataset produces the image/target structure expected by Mask R-CNN.
- Transfer learning, a frozen backbone, and patch training are plausible choices for the small dataset. Increasing the detection limit acknowledges dense scenes.
- Four preprocessing experiments were executed historically. The notebook has concise saved outputs and an interpretable prediction overlay rather than hundreds of warnings.
- All four checkpoint files load as tensor state dictionaries, contain the expected two-class classification head, and have finite floating-point tensors. This establishes that usable artifacts exist, not that they are accurate.

## Fit to the challenge

| Requested evidence | Current state | Needed before submission |
| --- | --- | --- |
| Proposed scientist-facing system | Implied by the code | Explain photo → select frame interior → model proposals → human review → final count |
| Clear subtask, inputs, outputs | Mask R-CNN pipeline exists | Define whether the prototype handles crops or full field images; state the boundary-counting rule |
| Prototype | Present | Repair reproducibility and make inference consistent with training |
| Performance analysis | Training losses and unlabeled counts only | Evaluate against labels and discuss missed/extra detections |
| Conclusions and learning | README headings are empty; notebook mostly labels operations | Explain what worked, what failed, and what you personally learned |
| Run instructions | Only `pip install -r requirements.txt` | Document data placement, environment, model retrieval/training, inference and evaluation commands |
| Public GitHub repository | This directory is not a Git repository | Package it and publish once the blocking issues are addressed |

## Findings in priority order

### 1. Reproduction currently fails because a training input is missing

`src/train.py:32–45` and notebook cell 3 require `data/img1.png`, but that file is absent. The actual dataset construction fails in `cv2.cvtColor` after OpenCV cannot read the file. `mask1.png` exists, but an annotation image cannot substitute for its corresponding photograph. The cropped `masked_img1.png` also cannot be substituted: the brief warns that its coordinates differ and it contains annotation overlays.

Restore the original `img1.png` from the challenge archive. Validate file existence and image/mask dimensions before preprocessing, with useful errors. Specify running commands from the repository root; for example, the training module's intended invocation is `python -m src.train`, not a bare source file invocation. Capture a tested dependency environment rather than leaving every dependency unversioned. The currently installed environment is not evidence of what was used for the historical training run.

### 2. There is no evidence of counting accuracy or generalization

`src/train.py` trains on both labeled images and never evaluates a held-out image. The notebook reports decreasing training loss and counts on `unseen_img2.png`, which has no accompanying labels here. A larger predicted count can mean better recall or more false positives; it does not establish a better preprocessing method.

After restoring the photograph, use image-level evaluation: train on image 1 and evaluate on image 2, then reverse if time allows. Split before making patches. Randomly splitting overlapping patches would allow nearly identical regions into training and evaluation. Existing checkpoints trained on both images cannot establish held-out performance on either of them; an evaluation split needs new training. If retraining is impractical, report existing-model results on training images explicitly as training-set diagnostics and manually audit a small separate region, without claiming generalization.

Report true count, predicted count, signed and absolute count error, and relative error where true count is nonzero. Add precision and recall from one-to-one instance matching with a declared IoU threshold; unmatched predictions are extras and unmatched annotations are misses. Count error alone can hide equal numbers of misses and false positives. Keep the score threshold fixed for the reported evaluation; if you tune it on that image, identify the result as validation rather than an untouched test. With two images, all conclusions remain exploratory.

Because the provided annotations are described as almost complete, report agreement with the available annotations rather than claiming exact biological ground truth. Inspect a few apparent false positives manually. Do not report accuracy for an unlabeled image unless you create and describe an independent manual reference.

### 3. Prediction does not enforce the frame interior

`src/predict.py:42–77` feeds the complete image into the model, counts every retained prediction, and divides by the complete image area. There is no frame polygon or crop argument. `get_annotation_bbox` bounds annotation pixels; it does not detect the green frame and depends on labels that scientists do not have during inference.

For this prototype, manual selection of the four inner frame corners is a reasonable assumption. Mask predictions to that polygon and declare a counting rule, such as counting an instance when its centroid lies inside. Treat that rule as an explicit assumption rather than a scientific standard. Show the selected region in the figure. Name any density output as counts per image pixel; physical density requires a known area/calibration. A polygon's bounding rectangle can include pixels outside the inner quadrilateral.

### 4. Patch generation silently omits boundary regions

`src/dataset.py:163–172` only visits regularly spaced windows that fit. On a 1000×1000 image with size 512 and stride 384, starts are 0 and 384, so coverage ends at pixel 895. Bottom and right strips remain uncovered. A synthetic image with four small objects near its corners retained only the top-left object in the generated samples.

Include a final window anchored at `dimension - patch_size` on each axis, removing duplicate starts, or pad with a declared policy. Preserve useful background-only patches rather than always dropping them. Partial visible objects excluded by the 40% rule also need an intentional treatment, since they otherwise appear as unlabeled foreground during training.

Training uses 512-pixel crops, but inference uses full photographs. Decide whether to support tiled inference with coordinate mapping and duplicate removal, or explicitly limit the prototype to user-selected crops of comparable scale. Merely setting the final detection cap to 1000 does not solve this scale mismatch.

### 5. Annotation conversion needs a visual audit

`src/masks.py:16–24` closes the blue outlines before finding external contours. A synthetic test with two separate rectangular outlines and a one-pixel gap returned one instance: closing can merge neighbors. In the supplied masks, the current algorithm recovers 1500 contours from mask 1 and 390 from mask 2. These are algorithm-derived counts, not verified true barnacle counts.

Overlay recovered instances on each original photograph, inspect dense and sparse regions, and compare extraction with and without closing. Choose any repair rule from that inspection. Do not infer that the different counts themselves prove a bug. Consider filtering anomalous or merged contours and state the label-recovery limitations. Full-resolution arrays for every instance also consume substantial memory: 1500 masks at 3264×2448 require about 12 GB before other arrays. Crop earlier or rasterize per patch if memory becomes an obstacle.

### 6. The submission does not yet explain the experiment

README.md contains empty headings, and notebook prose describes operations without hypotheses, interpretation, or conclusions. The saved notebook records:

| Preprocessing | Final training loss | Unlabeled `unseen_img2.png` count |
| --- | ---: | ---: |
| Raw | 1.0912 | 114 |
| Gaussian | 1.1384 | 119 |
| CLAHE | 1.0873 | 121 |
| Gaussian + CLAHE | 1.1057 | 123 |

These numbers come from existing notebook outputs, not a repeated four-model experiment. There is no seed control or recorded original environment, so small differences cannot confidently be attributed to preprocessing. `unseen_img2.png` is a 500×500 image not listed in the pasted challenge's inventory; document its origin and whether it is independent of training images. Make the official `unseen_img1.png` part of the demonstration once frame handling is defined.

Write the README around the actual prototype: why instance segmentation could help a scientist review count proposals; why two annotated photographs limit claims; what metric answers the counting question; examples of success and failure; and the next experiment justified by the evidence. Add learning reflections in your own words. Cite borrowed code or tutorials where applicable. Explain all files and provide a quick demo path that does not require rerunning four training jobs. Keep expensive training opt-in in the notebook.

### 7. Publication requires packaging

There is no `.git` directory, remote, or `.gitignore`. Each checkpoint is about 168 MiB, exceeding GitHub's 100 MiB ordinary Git file limit. Add ignores for `.venv`, caches, `.DS_Store`, and checkpoints before staging. Provide one documented checkpoint through a GitHub Release or use Git LFS with clear retrieval instructions. Preserve small result figures, metrics, and the cleaned notebook in ordinary Git. Explain how to obtain the challenge data and avoid presenting supplied data as your own work.

GitHub's file guidance: [About large files on GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

## Smaller issues

- `src/visualize.py:49` uses the set `{10,10}` as the default `figsize`, which raises a TypeError when `show_image` is called without overriding it. Use a tuple. `show_prediction` currently passes a tuple, so its saved figure does not demonstrate this default-path bug.
- The overlay adds colors to pixels rather than alpha blending foreground and background. Bright regions can saturate, and random colors make repeated figures inconsistent. A stable, less cluttered overlay would make error inspection easier.
- Model construction downloads pretrained weights even when inference subsequently replaces the entire state with a checkpoint. Support construction without pretrained downloads for the checkpoint-only inference path.
- Save experiment configuration, seed, split membership, threshold, training history, and package versions beside the model. The current bare state dictionary lacks that provenance.

The official [TorchVision fine-tuning tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html) provides a useful reference for the target format and a separate evaluation workflow.

## Recommended scope for the next iteration

1. Restore the missing source photo, validate inputs, and repair patch coverage. Audit recovered annotations before training again.
2. Define the prototype as counting within a manually selected frame region or explicitly supplied crop. Make the same region convention available at inference without labels.
3. Run one raw-input experiment with a clearly separated labeled evaluation image. Save a compact metrics table and representative missed/extra-instance figures. Add other preprocessing comparisons only if time remains.
4. Complete the README and notebook explanation, including limitations and personal learning. Package a reproducible demo and publish a public repository.
5. If time remains, add a small review interface where the scientist can remove a wrong proposal, add a missed barnacle, and export the corrected count. Measure correction effort or time on a small task before claiming the tool is faster than manual counting.

The brief suggests roughly ten hours overall. Prioritize a narrow, well-explained experiment over a larger feature list. A human review interface is an optional extension, not a prerequisite for an honest modeling submission. Remember that DALI separately requires an outside code sample; this challenge does not replace it.

## Verification performed and limits

- Read every source module, notebook cell and saved textual result, README, dependencies, and data/artifact inventory.
- Inspected the saved raw-model prediction figure; it shows plausible proposals, but visual plausibility does not establish precision or recall.
- Reproduced the missing-input error, uncovered-edge patch behavior, neighboring-contour merge, and default visualization failure with targeted local checks.
- Recovered contour counts from both supplied annotation masks and inspected the shape and finiteness of all four saved state dictionaries.
- Reran raw-checkpoint inference on `unseen_img2.png`: 114 detections at the default 0.5 score threshold, matching the saved notebook count. The run took approximately 1.55 seconds on CPU with two PyTorch threads and cached pretrained weights; this is a local smoke check, not a benchmark or accuracy measurement.
- Full training and a labeled accuracy evaluation were not performed; the missing training photograph prevents reproducing the original experiment.

Publication should follow these repairs and an honest documented evaluation, even if model accuracy is modest. The challenge explicitly asks for evidence, conclusions, and learning, rather than a solved barnacle-counting system.
