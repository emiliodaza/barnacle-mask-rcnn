"""Recover blue outlines without morphology that can join adjacent instances."""

import cv2
import numpy as np
from src.instances import instance_from_contour


def extract_blue_contours(mask_rgb, min_area=20, closing_kernel=0):
    """Find the supplied blue outlines and sort them top-to-bottom, then left-to-right.

    Closing is off by default because it can join neighboring barnacles.
    closing_kernel is retained for the annotation audit of the earlier approach.
    """
    binary = cv2.inRange(
        cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2HSV),
        np.array([90, 50, 30]),
        np.array([140, 255, 255]),
    )
    if closing_kernel:
        if closing_kernel < 1 or closing_kernel % 2 != 1:
            raise ValueError("Closing kernel must be zero or a positive odd integer.")
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, np.ones((closing_kernel, closing_kernel), np.uint8)
        )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(
        [c for c in contours if cv2.contourArea(c) >= min_area],
        key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]),
    )


def annotation_instances(mask_rgb, min_area=20, closing_kernel=0):
    """Fill each recovered outline into a compact reference mask.

    These references inherit the omissions and ambiguities of the supplied labels.
    """
    return [
        instance_from_contour(c)
        for c in extract_blue_contours(mask_rgb, min_area, closing_kernel)
    ]
