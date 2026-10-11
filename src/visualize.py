"""Deterministic figures with true alpha blending and explicit frame boundaries."""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def overlay_instances(image, instances, color=(61, 210, 172), alpha=0.3):
    """Blend compact masks onto a copy of an RGB photo and draw their outlines."""
    out = image.copy()
    for p in instances:
        x, y, r, b = p.box
        patch = out[y:b, x:r]
        select = p.mask > 0
        patch[select] = np.rint(
            (1 - alpha) * patch[select] + alpha * np.asarray(color)
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            p.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        shifted = [c + np.array([[[x, y]]], np.int32) for c in contours]
        cv2.drawContours(out, shifted, -1, color, 1)
    return out


def show_image(image, title=None, figsize=(10, 10)):
    """Display an RGB array without axes and return the Matplotlib figure."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    ax.set_title(title or "")
    ax.axis("off")
    plt.show()
    return fig


def show_prediction(result, title="Model proposals; human review required"):
    """Show the selected frame and all proposals returned by inference."""
    out = overlay_instances(result["image"], result["instances"])
    cv2.polylines(
        out, [np.asarray(result["polygon"], np.int32)], True, (255, 199, 92), 5
    )
    return show_image(out, f"{title} | {result['count']} proposals", (12, 9))


def save_comparison(image, reference, predictions, matches, polygon, path, title):
    """Save reference and error views cropped to the frame bounding rectangle.

    Teal marks matches, orange extra proposals, and blue missed references.
    """
    matched_pred = {i for i, _, _ in matches}
    matched_ref = {j for _, j, _ in matches}
    base = image.copy()
    annotated = overlay_instances(base, reference)
    errors = overlay_instances(
        base,
        [p for i, p in enumerate(predictions) if i in matched_pred],
        (61, 210, 172),
    )
    errors = overlay_instances(
        errors,
        [p for i, p in enumerate(predictions) if i not in matched_pred],
        (255, 125, 92),
        0.4,
    )
    errors = overlay_instances(
        errors,
        [p for j, p in enumerate(reference) if j not in matched_ref],
        (123, 157, 255),
        0.4,
    )
    for im in [annotated, errors]:
        cv2.polylines(im, [np.asarray(polygon, np.int32)], True, (255, 205, 92), 5)
    x, y = np.asarray(polygon).min(axis=0).astype(int)
    r, b = np.asarray(polygon).max(axis=0).astype(int) + 1
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, im, label in zip(
        axes,
        [annotated, errors],
        [
            "Recovered annotations",
            "Matches: teal · extra proposals: orange · misses: blue",
        ],
    ):
        ax.imshow(im[y:b, x:r])
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
