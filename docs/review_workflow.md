# Reviewing a count

Start the local server from the repository root:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Open [the demo](http://127.0.0.1:8000/demo/). The three examples use saved model
outputs. The browser does not run inference or upload your photographs.

## Work through a photograph

1. Choose an example and check that the gold polygon follows the inner frame.
2. Use **Next** to inspect smaller tiles. Scroll to zoom; drag in **Pan image** mode.
3. In **Remove extra** mode, click an incorrect teal proposal. To restore it, enable **Show removed proposals** and click its dashed outline.
4. In **Mark missed** mode, click the center of a missed barnacle. Click its marker again to remove it. These additions are points, not segmentation masks.
5. Use **Undo** for the last edit and **Fit frame** for the overview. Changing the score cutoff shows or hides proposals; it does not change the reported evaluation.
6. Use **Download review** at the top to save JSON with the count and edit history. The sidebar also downloads accepted points as CSV and the selected frame as JSON.

Shortcuts: **D** removes extras, **A** marks misses, **P** pans, **F** selects a
frame, and **Ctrl/Cmd + Z** undoes an edit. Shortcuts are ignored while a form
input is focused.

Counts include accepted model centers and manual points inside or on the selected
polygon. Raising the cutoff or moving the frame can hide previous corrections;
those edits remain saved. Local storage is scoped to the photo and proposal data,
so another model output starts a separate review. Undo history lasts for the
current page session. Download important work before clearing browser storage.

Exports use original-photo coordinates. Review JSON includes the model and photo
hashes, frame, cutoff, removed IDs, manual points, and timestamped events.
`review_complete` stays false: visiting every tile does not establish that the
count was checked. Event timestamps are an edit record, not a measured time saving.

## Use a new photograph

1. Expand **Use your own photograph** and load the original image.
2. In **Set frame** mode, select its four inner corners in perimeter order. Download the frame JSON from the sidebar.
3. Run Python inference with that photo and frame:

   ```bash
   python -m src.predict --image path/to/photo.png \
     --region path/to/frame-region.json \
     --checkpoint outputs/final_raw.pth \
     --score-threshold 0.25 --output results/my-proposals.json
   ```

4. Keep the same photo loaded and load the proposal JSON. Start reviewing at cutoff 0.5; lowering it to 0.25 shows more proposals.

Photo dimensions must match the proposal coordinates. The original-file hash is
checked when it is available. Changing the frame does not rerun inference;
expanding beyond the original region requires exporting the new frame and running
Python again. The model assumes a pixel scale similar to the training photographs.

## Test whether this helps

A follow-up study should compare manual-only and proposal-assisted counting on
independently annotated regions. Randomize the workflow order and measure both
verified final error and task time. Model precision and recall do not tell us
whether this interface makes a scientist's work faster.
