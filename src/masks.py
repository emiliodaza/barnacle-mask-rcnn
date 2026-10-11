"""Recover blue outlines without morphology that can join adjacent instances."""

import cv2
import numpy as np
from src.instances import instance_from_contour


def extract_blue_contours(mask_rgb, min_area=20, closing_kernel=0):
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
    return [
        instance_from_contour(c)
        for c in extract_blue_contours(mask_rgb, min_area, closing_kernel)
    ]


def contours_to_instance_masks(contours, height, width, min_filled_area=20):
    # Compatibility helper. The main pipeline uses compact masks instead.
    instances = [instance_from_contour(c) for c in contours]
    return [p.full_mask(height, width) for p in instances if p.area >= min_filled_area]


def annotation_to_instance_masks(mask_rgb, min_area=20):
    return contours_to_instance_masks(
        extract_blue_contours(mask_rgb, min_area), *mask_rgb.shape[:2]
    )


def get_annotation_bbox(mask_rgb, padding=20):
    contours = extract_blue_contours(mask_rgb)
    if not contours:
        return None
    points = np.concatenate(contours).reshape(-1, 2)
    return (
        max(int(points[:, 0].min()) - padding, 0),
        max(int(points[:, 1].min()) - padding, 0),
        min(int(points[:, 0].max()) + padding, mask_rgb.shape[1] - 1),
        min(int(points[:, 1].max()) + padding, mask_rgb.shape[0] - 1),
    )


def crop_to_bbox(image, bbox):
    x1, y1, x2, y2 = bbox
    return image[y1 : y2 + 1, x1 : x2 + 1]


def shift_masks_to_crop(instance_masks, bbox):
    return [
        cropped
        for mask in instance_masks
        if (cropped := crop_to_bbox(mask, bbox)).any()
    ]
