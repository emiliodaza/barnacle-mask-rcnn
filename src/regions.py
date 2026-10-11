"""Explicit, annotation-independent frame regions. Coordinates are original pixels."""

import json
from pathlib import Path
import cv2
import numpy as np


def validate_polygon(polygon, shape):
    """Check that the region is convex, ordered around its edge, and in bounds.

    shape is (height, width, ...); polygon points are [x, y]. Return float32
    points for OpenCV. Python accepts any convex polygon; the UI uses four corners.
    """
    points = np.asarray(polygon, dtype=np.float32)
    height, width = shape[:2]
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) < 3
        or not np.isfinite(points).all()
    ):
        raise ValueError("Region must contain at least three finite [x, y] points.")
    if (
        (points[:, 0] < 0).any()
        or (points[:, 0] >= width).any()
        or (points[:, 1] < 0).any()
        or (points[:, 1] >= height).any()
    ):
        raise ValueError("Region points must lie within the original image.")
    if not cv2.isContourConvex(points) or abs(cv2.contourArea(points)) < 1:
        raise ValueError("Select a convex frame polygon in perimeter order.")
    return points


def load_region(image_path, region_file="config/regions.json"):
    """Read either the sample region map or one frame exported by the browser."""
    document = json.loads(Path(region_file).read_text())
    if "polygon" in document:
        return document["polygon"]
    name = Path(image_path).name
    if name not in document["images"]:
        raise ValueError(
            f"No region for {name}; select and export a frame in the review tool."
        )
    return document["images"][name]["polygon"]


def contains(polygon, point):
    """Include points on the boundary as well as points inside the polygon."""
    return (
        cv2.pointPolygonTest(
            np.asarray(polygon, np.float32), tuple(map(float, point)), False
        )
        >= 0
    )


def polygon_mask(polygon, shape):
    """Rasterize the region as a height-by-width uint8 mask of zeros and ones."""
    points = validate_polygon(polygon, shape)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    return mask


def region_bbox(polygon):
    """Return an enclosing (left, top, right, bottom) box with exclusive ends."""
    p = np.asarray(polygon)
    return (
        int(np.floor(p[:, 0].min())),
        int(np.floor(p[:, 1].min())),
        int(np.ceil(p[:, 0].max())) + 1,
        int(np.ceil(p[:, 1].max())) + 1,
    )


def keep_inside(instances, polygon):
    """Apply the same centroid counting rule to references and model proposals."""
    return [instance for instance in instances if contains(polygon, instance.centroid)]


def window_starts(length, size, stride):
    """Cover an axis, adding an edge-anchored tile when the stride leaves a gap.

    For length 1000, size 512, and stride 384, starts are 0, 384, and 488.
    A smaller image gets a single tile; it is not padded here.
    """
    if size <= 0 or stride <= 0 or stride > size:
        raise ValueError("Require 0 < stride <= tile size.")
    if length <= 0:
        raise ValueError("Image dimensions must be positive.")
    if length <= size:
        return [0]
    return sorted(set(range(0, length - size + 1, stride)) | {length - size})


def tile_boxes(shape, size=512, stride=384):
    """Return row-major tile boxes, including the bottom and right edges."""
    height, width = shape[:2]
    return [
        (x, y, min(x + size, width), min(y + size, height))
        for y in window_starts(height, size, stride)
        for x in window_starts(width, size, stride)
    ]
