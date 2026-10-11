"""Build a short notebook from the actual committed experiment, without retraining."""

from pathlib import Path
import argparse
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build():
    md = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    cells = [
        md("""# Barnacle: counting proposals a scientist can check

The DALI problem is to count barnacles inside the inner green frame. This notebook
explains a small prototype, its evidence, and why human review remains necessary.
It reads **actual committed results**, so running all cells does not retrain models.

**System:** photograph → human-selected frame corners → tiled model proposals →
scientist correction → exported provisional count and audit. The critical subtask
is proposing individual instances, not reliably replacing the scientist."""),
        md("""## 1. Hypothesis and alternatives

Instance masks could make dense scenes easier to inspect and correct. With only
two annotated photos, transfer learning is a practical first experiment, but
Mask R-CNN's complexity is only justified if the proposals are useful. Compare it
with a fixed bright-component baseline and a count-only training-density baseline.
A manual frame selection is enough for this prototype; automated frame detection
is a separate subtask. A mask-derived crop would be unavailable on field photos.

**Metrics:** count error and one-to-one mask precision/recall at IoU 0.5. Count
error alone can hide equal numbers of extra and missed instances. The score
threshold is fixed at 0.5; 0.25/0.75 are sensitivity checks, not test-set tuning."""),
        code("""%matplotlib inline
from pathlib import Path
import json
import matplotlib.pyplot as plt
from IPython.display import display, Image, Markdown
from src.dataset import BarnacleDataset
from src.preprocess import preprocess_raw

root = Path.cwd()
if not (root / "results/report.json").exists():
    raise RuntimeError("Start Jupyter from the repository root.")
report = json.loads((root / "results/report.json").read_text())
audit = json.loads((root / "results/annotation_audit.json").read_text())
print("Recorded environment:", report["environment"])
print("Original image-level split: train 1 → evaluate 2; train 2 → evaluate 1")"""),
        md("""## 2. Explore the data and audit the labels

The original photographs and masks share their coordinate system. The supplied
`masked_img*.png` files are cropped overlays, so they are not used as training
inputs. Blue outlines are filled separately, without morphology that joins nearby
instances. A filled-mask centroid must lie inside or on the manually selected
frame. These are recovered annotations, not independently validated truth.

The original closing step recovered 1,500 image-1 contours; the revised extraction
recovers 1,718, of which 1,702 meet the region rule. We should inspect the conversion
before interpreting model accuracy."""),
        code("""for row in audit:
    print(f'{row["image"]}: {row["reference_in_region"]} references in frame; '
          f'{row["excluded_outside_region"]} excluded; '
          f'{row["compact_mask_bytes"] / 1e6:.2f} MB compact masks '
          f'vs {row["full_mask_bytes_if_materialized"] / 1e9:.2f} GB full masks')
display(Image(filename="results/figures/annotation_audit_1.png", width=1000))"""),
        md("""## 3. Inspect the prototype's real inputs

Patches are made **after** separating photographs. Boundary-anchored windows cover
all edge strips; negative patches and visible fragments are retained. Dataset
masks are generated lazily from compact instances. Train and inference use the
same 512-pixel scale and mask pixels outside the selected region.

The frozen COCO backbone supplies general visual features; new heads learn the
barnacle class. Each directional fold starts independently from those weights."""),
        code("""dataset = BarnacleDataset(["data/img1.png"], ["data/mask1.png"],
                           preprocess_fn=preprocess_raw, patch_size=512, patch_stride=384)
image, target = dataset[0]
print("Training image 1 only:", len(dataset), "patches")
print("Input:", tuple(image.shape), image.dtype, "range:", float(image.min()), float(image.max()))
print("Target masks:", tuple(target["masks"].shape), target["masks"].dtype)
print("Target boxes:", tuple(target["boxes"].shape), "label:", target["labels"].unique().tolist())
plt.figure(figsize=(5, 5))
plt.imshow(image.permute(1, 2, 0))
plt.title("One training patch, before model predictions")
plt.axis("off")
plt.show()"""),
        md("""## 4. Run and evaluate

The recorded run used seed 42, five epochs, AdamW, raw RGB, a frozen backbone, and
two CPU threads. Source hashes, component losses, folds, and package versions are
saved beside the measurements. Reproduce all model outputs explicitly with:

```bash
python -m src.experiment --epochs 5
```

The separate final model uses both labeled images for an **unlabeled demonstration
only**. Its outputs are never used to claim held-out accuracy on either training
photo. No random split of overlapping patches is used."""),
        code("""fig, ax = plt.subplots(figsize=(7, 3))
for fold in report["folds"]:
    history = fold["training"]["history"]
    ax.plot([h["epoch"] for h in history], [h["total_loss"] for h in history],
            marker="o", label=f'Train image {fold["train_image"]}')
ax.set(xlabel="Epoch", ylabel="Mean training loss", title="Optimization improves; this does not measure accuracy")
ax.legend()
fig.tight_layout()
plt.show()

rows = [m for m in report["metrics"] if m["score_threshold"] == 0.5 and m["method"] == "mask_rcnn"]
lines = ["| Train → evaluation | Ref / predicted | Precision | Recall | F1 |", "| --- | ---: | ---: | ---: | ---: |"]
for m in rows:
    lines.append(f'| {m["train_image"]} → {m["test_image"]} | {m["reference_count"]} / {m["predicted_count"]} | '
                 f'{m["precision"]:.1%} | {m["recall"]:.1%} | {m["f1"]:.3f} |')
display(Markdown("\\n".join(lines)))"""),
        md("""## 5. What the baseline reveals

On image 2, the bright-component baseline has a lower count error than Mask R-CNN
(155 vs 367), yet its F1 is only 0.003. It is barely finding the annotated instances.
This is a concrete example of count errors canceling. The training-density baseline
also fails across the scenes. These intentionally simple methods do not establish
that a better classical approach would fail."""),
        code("""lines = ["| Method | Evaluation image | Absolute count error | F1 |", "| --- | ---: | ---: | ---: |"]
for m in report["metrics"]:
    if m["method"] == "mask_rcnn" and m["score_threshold"] != 0.5:
        continue
    f1 = f'{m["f1"]:.3f}' if m["f1"] is not None else "N/A (no localization)"
    lines.append(f'| {m["method"]} | {m["test_image"]} | {m["absolute_count_error"]} | {f1} |')
display(Markdown("\\n".join(lines)))"""),
        md("""## 6. Inspect successes and failures

Teal: matched masks. Orange: unmatched proposals. Blue: missed annotations.

The model trained on the dense photograph overcounts the darker one; the reverse
fold has better precision but misses many dense instances. Incomplete labels,
mask-shape disagreement at the IoU cutoff, and tile seams also limit interpretation.
The results motivate assisted review rather than unattended counting."""),
        code("""display(Image(filename="results/figures/heldout_2.png", width=1000))
display(Image(filename="results/figures/heldout_1.png", width=1000))"""),
        md("""## 7. Explore score sensitivity without selecting a winner

Raising the minimum score can improve precision while lowering recall. With two
photographs there is no separate validation set here. The values below are
exploratory; the primary reported results remain fixed at score 0.5. Scores have
not been calibrated as probabilities."""),
        code("""lines = ["| Evaluation image | Score | Predicted count | Precision | Recall |", "| --- | ---: | ---: | ---: | ---: |"]
for m in report["metrics"]:
    if m["method"] == "mask_rcnn":
        lines.append(f'| {m["test_image"]} | {m["score_threshold"]:.2f} | {m["predicted_count"]} | '
                     f'{m["precision"]:.1%} | {m["recall"]:.1%} |')
display(Markdown("\\n".join(lines)))"""),
        md("""## 8. Human feedback and the official unseen photograph

The final model was trained on both labeled images, separately from the folds.
It produces 820 proposals at score 0.5 in the official unlabeled photograph. We
cannot compute its accuracy without an independent reference.

The browser tool allows tile inspection, changing the score threshold, removing
extras, marking misses, undoing edits, and exporting original-coordinate points
and an audit record. It caches real proposals rather than substituting mock data.
The count is provisional; human review completion and time savings are not verified.

```bash
python -m http.server 8000 --bind 127.0.0.1
# Visit http://127.0.0.1:8000/demo/
```

Detailed instructions: [review workflow](docs/review_workflow.md)."""),
        code("""unseen = json.loads(Path("results/unseen_demo.json").read_text())
print("Official unseen photo:", unseen["image"])
print("Proposals at score 0.5:", unseen["proposals_at_0_5"])
print("Accuracy:", unseen["accuracy"], "(no reference labels)")
print("Tiled inference seconds:", round(unseen["inference"]["seconds"], 2))
display(Image(filename="results/figures/review_interface.jpg", width=1000))"""),
        md("""## 9. Conclusions, learning, and the next experiment

**Worth pursuing:** inspectable proposals and a correction workflow. **Not yet
supported:** reliable automatic counts or a claim that counting is faster.

What the experiment teaches:

- Annotation repair can silently change the count before learning even begins.
- Image-level splitting matters more than a large number of overlapping patches.
- Compact masks make dense scenes practical without a huge full-mask allocation.
- Training loss, count agreement, and instance correctness answer different questions.
- A useful system must handle uncertainty and scientific review explicitly.

Next, confirm live/empty-shell and boundary conventions with scientists, audit
annotations, gather independent photographs across lighting/density/scale, and
reserve a validation set before tuning. Compare manual-only and assisted review
on verified regions, measuring both final error and task time. More architecture
complexity is not the first decision to make.

**Limits:** two directional results cannot establish generalization. Manual corner
selection is approximate. Pixel density is not physical density. Duplicate
suppression may leave split objects or suppress neighbors. Additional study is
needed before operational use.

The revision used Codex assistance; measurements are from actual local runs.
References and file-level reproduction instructions are in [README.md](README.md)."""),
    ]
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.14.6"},
        },
    )
    nbf.write(notebook, ROOT / "barnacle_notebook.ipynb")
    return notebook


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    notebook = build()
    if args.execute:
        from nbclient import NotebookClient

        NotebookClient(
            notebook,
            timeout=120,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
        nbf.write(notebook, ROOT / "barnacle_notebook.ipynb")
        print(
            "Executed notebook successfully; all ordinary cells read existing results."
        )
