"""Deliberately simple fixed bright-component baseline, with no label access."""

import cv2
from src.instances import instance_from_contour
from src.regions import polygon_mask, keep_inside


def bright_components(image, polygon, threshold=160, min_area=40, max_area=3000):
    """Treat bright connected regions as instances, without learning or label access.

    The fixed brightness and area cutoffs make this a simple comparison, not a
    tuned classical vision method.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = ((gray >= threshold) & (polygon_mask(polygon, image.shape) > 0)).astype(
        "uint8"
    ) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return keep_inside(
        [
            instance_from_contour(c)
            for c in contours
            if min_area <= cv2.contourArea(c) <= max_area
        ],
        polygon,
    )
