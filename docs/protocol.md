# Declared evaluation protocol

This file describes the choices used for the committed experiment. It documents
implementation choices; it is not a timestamped preregistration. No threshold or
model was selected by optimizing the two held-out results.

- Data: original `img1.png` / `mask1.png` and `img2.png` / `mask2.png`; input hashes in `data/manifest.json`.
- Regions: four inner frame corners manually selected from the RGB photographs, without reading annotations. Stored in original coordinates in `config/regions.json`. Approximate corner selection is a source of boundary uncertainty.
- Count rule: filled-mask centroid inside or on the region polygon. Apply the same rule to recovered reference instances and predictions. It is a stated prototype convention, not a confirmed scientist protocol.
- Label recovery: HSV blue range H=90–140, S=50–255, V=30–255; external contours of area at least 20 pixels; no closing. Fill each contour. Labels can be open, joined, incomplete, or biologically ambiguous. The recovered references are not independently validated ground truth.
- Split: train on image 1, evaluate image 2; separately train on image 2, evaluate image 1. Make all patches after the split. Each fold starts from COCO weights, not the other fold's weights. Only the training image's labels enter training.
- Model: TorchVision Mask R-CNN ResNet-50 FPN, frozen backbone, new two-class heads. Image size 512. Up to 400 detections per tile. Default anchors. Raw RGB scaled to [0,1]. Outside-region pixels are blacked out in both training and inference.
- Training: five epochs, one patch per batch, AdamW with learning rate 0.0001 and weight decay 0.0001, seed 42, deterministic PyTorch algorithms, two CPU threads. No augmentation. All nontrivial visible fragments receive masks; patches containing no reference instances are retained.
- Tiling: 512 pixels, stride 384, additional boundary-anchored windows so no edge strip is lost. Predictions are mapped to original coordinates. Keep mask-centroid-inside instances, reject masks with fewer than four pixels, and suppress duplicate masks at IoU > 0.3 in descending model-score order. This suppression is approximate and can still leave split duplicates or suppress overlapping neighbors.
- Evaluation: score >= 0.5, mask pixel value >= 0.5, one-to-one matching at mask IoU >= 0.5. Assignment maximizes the number of valid matches first and IoU second. Report counts, signed/absolute/relative error, precision, recall, F1, and mean IoU of matched instances. Sensitivity at scores 0.25 and 0.75 is reported as exploration, not used to select the primary threshold.
- Baseline A: training reference density times evaluation-region pixel area, rounded to an integer. This illustrates why transferring one image's density is unreliable. Perspective and pixel scale also confound this baseline. No localization metrics apply.
- Baseline B: Gaussian blur 3×3, grayscale brightness >=160, external components with contour area 40–3000 pixels, filtered to the region. Fixed parameters; no labels at inference. It is intentionally naive and not presented as the best classical method.
- Final demonstration: train a separate model on both images under the same settings, then infer on official `unseen_img1.png`. This image has no labels, so no accuracy is reported. `unseen_img2.png` is in the supplied archive but not the pasted inventory; its provenance is unspecified and it is excluded from quantitative evaluation.

The two directional folds are descriptive results for two photographs, not a
statistically reliable estimate for other sites, lighting, seasons, or camera
setups. Improved automatic counting would require additional independently
annotated photographs and a separately reserved validation/test protocol.
