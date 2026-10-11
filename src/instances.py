"""Compact instance masks in original image coordinates (exclusive box ends)."""

from dataclasses import dataclass
from functools import cached_property
import cv2
import numpy as np


@dataclass
class Instance:
    """One mask stored in its tight bounding box, rather than a full photo-sized array.

    The box is (left, top, right, bottom) in original pixels; right and bottom
    are exclusive, just like NumPy slices. Treat the mask as immutable because
    its area is cached. Masks contain uint8 values of zero or one.
    """

    box: tuple[int, int, int, int]
    mask: np.ndarray
    score: float = 1.0

    @cached_property
    def area(self):
        """Number of foreground pixels, not the area of the bounding box."""
        return int(np.count_nonzero(self.mask))

    @property
    def centroid(self):
        """Mean foreground position as (x, y) in the original photograph."""
        ys, xs = np.nonzero(self.mask)
        return (float(xs.mean() + self.box[0]), float(ys.mean() + self.box[1]))

    def full_mask(self, height, width):
        """Expand a compact mask for small diagnostics; avoid this for dense photos."""
        out = np.zeros((height, width), dtype=np.uint8)
        x1, y1, x2, y2 = self.box
        out[y1:y2, x1:x2] = self.mask
        return out


def compact_mask(mask, offset=(0, 0), score=1.0):
    """Crop a binary mask to its foreground and add the tile offset to its box.

    Return None for an empty mask. The offset is (x, y), not row/column order.
    """
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    dx, dy = offset
    return Instance(
        (x1 + dx, y1 + dy, x2 + dx, y2 + dy),
        (mask[y1:y2, x1:x2] > 0).astype(np.uint8),
        float(score),
    )


def instance_from_contour(contour, offset=(0, 0), score=1.0):
    """Fill an OpenCV contour in a local box and retain its original position."""
    x, y, w, h = cv2.boundingRect(contour)
    local = contour - np.array([[[x, y]]], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [local], -1, 1, cv2.FILLED)
    dx, dy = offset
    return Instance((x + dx, y + dy, x + w + dx, y + h + dy), mask, score)


def intersection(a, b):
    """Count shared foreground pixels using only the overlap of the two boxes."""
    ax, ay, ar, ab = a.box
    bx, by, br, bb = b.box
    x, y, r, bottom = max(ax, bx), max(ay, by), min(ar, br), min(ab, bb)
    if r <= x or bottom <= y:
        return 0
    return int(
        np.count_nonzero(
            a.mask[y - ay : bottom - ay, x - ax : r - ax]
            & b.mask[y - by : bottom - by, x - bx : r - bx]
        )
    )


def mask_iou(a, b):
    """Intersection divided by union; disjoint masks have an IoU of zero."""
    inter = intersection(a, b)
    if not inter:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union else 0.0


def deduplicate(instances, threshold=0.3):
    """Keep the higher-scoring mask when two tiled proposals overlap too much.

    This is approximate: split objects can survive, and neighboring barnacles
    can be suppressed. The threshold is mask IoU, not bounding-box IoU.
    """
    kept = []
    for candidate in sorted(instances, key=lambda p: p.score, reverse=True):
        if all(mask_iou(candidate, previous) <= threshold for previous in kept):
            kept.append(candidate)
    return kept


def instance_record(instance, index):
    """Convert an instance to the browser schema, with original-pixel polygons."""
    contours, _ = cv2.findContours(
        instance.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        polygon = cv2.approxPolyDP(contour, 1.0, True).reshape(-1, 2)
        polygon = polygon + np.array(instance.box[:2])
        if len(polygon) >= 3:
            polygons.append(polygon.tolist())
    return {
        "id": str(index),
        "box": list(instance.box),
        "center": list(instance.centroid),
        "score": instance.score,
        "polygons": polygons,
    }
