# Reading the code

Start with the notebook for the experiment, then follow one photograph through
these files. Most of the code exists to keep labels, tiles, and model output in
the same coordinate system.

## 1. Decide what counts

Read `config/regions.json` and `src/regions.py` first. The region is selected from
the photograph, not from its annotations. A barnacle counts if its mask centroid
is inside or on that polygon. `contains()` implements that rule; `keep_inside()`
applies it to both recovered labels and predictions.

Two conventions recur throughout the code:

- Points are `(x, y)`, but NumPy images are indexed as `[y, x]`.
- Boxes are `(left, top, right, bottom)`, with exclusive right/bottom ends.
  A box `(10, 20, 14, 25)` selects `image[20:25, 10:14]`: five rows, four columns.

`window_starts()` adds a last tile at the edge when a regular stride falls short.
For a 1000-pixel axis, 512-pixel tiles, and stride 384, it returns `[0, 384, 488]`.
Without the last start, pixels 896–999 would never be inspected.

## 2. Recover labels without merging neighbors

`src/masks.py` finds blue outlines in the supplied annotation images and fills
each one. Its output is a list of `Instance` objects, defined in
`src/instances.py`. An instance holds a small binary mask, its position in the
original photograph, and a score. Reference masks use score 1.0; this is a default
value, not a measured confidence in the annotation.

A 4×5 mask takes 20 bytes even if its photograph is thousands of pixels wide.
`compact_mask()` keeps that local mask and adds any tile offset to the box.
`centroid` restores original coordinates when calculating the mean foreground
position. `mask_iou()` compares only the overlap of two boxes; it does not expand
them to full image-sized arrays.

The annotation audit compares extraction with and without closing. Opening
`results/figures/annotation_audit_1.png` is part of understanding this step:
annotation conversion can change the reference count before the model sees it.

## 3. Build one training tile

`BarnacleDataset` in `src/dataset.py` receives only the training image IDs chosen
by the caller. It loads each photo and compact references once, then stores tile
boxes. `__getitem__()` constructs the requested tile:

1. Copy its RGB pixels and black out the region exterior.
2. Copy visible reference fragments into tile-sized masks and clip them to the region.
3. Derive detection boxes and class labels; preserve a valid empty target if no instances remain.
4. Return a float32 image shaped `(3, height, width)` in `[0, 1]` and a target dictionary.

Targets contain `boxes`, `labels`, `masks`, `image_id`, `area`, and `iscrowd`.
TorchVision reserves class 0 for background, so barnacles have class 1.
`collate_fn()` keeps differently sized target lists from being stacked together.

Split photographs before building this dataset. Neighboring tiles overlap;
a random patch split would put nearly the same rock surface in training and testing.

## 4. Train new heads

`src/model.py` builds the TorchVision model. The COCO backbone supplies visual
features; the box and mask heads are replaced for background and barnacle.
Freezing the backbone reduces the number of parameters trained from two photographs.
It does not remove the risk of overfitting.

`train_one_experiment()` in `src/train.py` seeds the generators, shuffles tiles,
sums TorchVision's component losses, and steps AdamW. The checkpoint includes
weights and configuration; the adjacent JSON is a readable copy of that configuration.
Training loss tells us whether optimization is progressing, not whether counts
are right on a new photograph.

For checkpoint loading, `pretrained=False` avoids downloading weights that would
immediately be replaced. Frozen batch normalization is recreated with the COCO
settings so the saved weights run through the same architecture.

## 5. Translate predictions back to the photograph

Read `predict_instances()` in `src/predict.py`. It takes RGB pixels, a polygon,
and a loaded model. There is no annotation argument.

Each tile prediction contains scores, labels, and soft masks. The code keeps the
barnacle class above the score cutoff, thresholds mask pixels, and adds the tile
and region offsets to each compact box. It then checks centroids and suppresses
likely duplicates in descending score order. The overlap threshold is approximate:
it can leave split objects or remove overlapping neighbors.

`export_prediction()` writes the browser JSON. The binary masks become simplified
polygons for display; evaluation still uses the binary masks. Photo dimensions,
original-file hash, training-image names, and checkpoint hash travel with the proposals.

## 6. Match once, then count errors

`evaluate_instances()` in `src/evaluate.py` builds a mask-IoU matrix and uses
SciPy's assignment solver. Each predicted and reference instance can match once.
The reward prioritizes the number of matches above the threshold, then IoU.
Maximizing IoU alone can sacrifice a valid pair for one stronger overlap.

An unmatched prediction is a false positive; an unmatched reference is a false
negative. Precision asks how many proposals matched. Recall asks how many
references were found. Count error asks a different question: extras and misses
can cancel and produce the right total for the wrong reasons.

`src/experiment.py` ties these steps together for train 1 → test 2, then train 2 →
test 1. It compares baselines and saves figures. The final model is trained
separately on both photographs for the unlabeled demonstration.

## 7. Follow a browser edit

`demo/core.js` contains the counting rules without DOM or canvas access.
`demo/app.js` owns the interface, drawing, file loading, and persistence.

The proposal document stays unchanged. `reviewState` stores removed IDs, manual
points, the score cutoff, frame corners, and events. `editReview()` snapshots that
state for undo, applies the edit, records it, saves it, and refreshes the count.
Zoom and pan are display settings; `imagePointFromEvent()` converts clicks back
to original pixels before recording them.

Saved edits are keyed by a fingerprint of the photo, proposals, frame, and model.
That prevents an old correction from silently attaching to different predictions.
Undo restores the earlier count state while retaining the edit history.
Exports remain provisional because navigating through tiles does not verify that
a scientist actually checked every barnacle.

## Questions to work through next

Use these to check your understanding rather than memorize the file names:

- Why can't `masked_img1.png` replace the original photograph during training?
- What would a random split of overlapping patches accidentally measure?
- Where do local tile coordinates become original photograph coordinates?
- Why can a lower count error come with worse instance predictions?
- What changes in the count when the score cutoff rises? What edits remain saved?
- What experiment would establish that the interface saves a scientist time?

The regression tests contain small examples of edge coverage, duplicate masks,
neighboring outlines, empty targets, and misleading count agreement. Read each
alongside the function it protects.
