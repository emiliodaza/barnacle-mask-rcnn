"""One-to-one mask matching and count metrics; never optimize thresholds on test images."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from src.instances import mask_iou


def evaluate_instances(predictions, reference, iou_threshold=0.5):
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0,1].")
    overlaps = np.zeros((len(predictions), len(reference)), dtype=np.float32)
    if len(reference):
        boxes = np.asarray([p.box for p in reference])
        for i, p in enumerate(predictions):
            x, y, r, b = p.box
            possible = np.where(
                (boxes[:, 0] < r)
                & (boxes[:, 2] > x)
                & (boxes[:, 1] < b)
                & (boxes[:, 3] > y)
            )[0]
            for j in possible:
                overlaps[i, j] = mask_iou(p, reference[j])
    # Cardinality dominates IoU tie-breaking, including adversarial crossing matches.
    valid = overlaps >= iou_threshold
    reward = valid.astype(float) * (
        max(len(predictions), len(reference)) + 1 + overlaps
    )
    rows, cols = linear_sum_assignment(reward, maximize=True)
    matches = [
        (int(i), int(j), float(overlaps[i, j]))
        for i, j in zip(rows, cols)
        if valid[i, j]
    ]
    tp = len(matches)
    fp = len(predictions) - tp
    fn = len(reference) - tp
    precision = tp / len(predictions) if predictions else 0.0
    recall = tp / len(reference) if reference else 0.0
    delta = len(predictions) - len(reference)
    metrics = {
        "reference_count": len(reference),
        "predicted_count": len(predictions),
        "signed_count_error": delta,
        "absolute_count_error": abs(delta),
        "relative_count_error": abs(delta) / len(reference) if reference else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "mean_matched_mask_iou": float(np.mean([v for _, _, v in matches]))
        if matches
        else None,
        "iou_threshold": iou_threshold,
    }
    return metrics, matches
