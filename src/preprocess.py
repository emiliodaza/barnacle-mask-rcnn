"""RGB preprocessing options from the initial experiment.

The reported held-out runs use raw RGB. These alternatives remain available for
inspection; training loss alone did not establish which preprocessing works best.
"""

import cv2
import numpy as np


def preprocess_raw(rgb_image):
    """Leave RGB pixels unchanged; this is the input used in the reported runs."""
    return rgb_image


def preprocess_gaussian(rgb_image, kernel_size=3):
    """Apply mild smoothing; a larger kernel can erase small barnacle boundaries."""
    return cv2.GaussianBlur(rgb_image, (kernel_size, kernel_size), 0)


def preprocess_clahe(rgb_image):
    """Enhance local brightness contrast in LAB while leaving color channels alone."""
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))

    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def preprocess_gaussian_then_clahe(rgb_image):
    """Smooth first, then apply local contrast enhancement."""
    blurred = preprocess_gaussian(rgb_image, kernel_size=3)
    enhanced = preprocess_clahe(blurred)

    return enhanced


def preprocess_sobel_visual(rgb_image):
    """Show edge strength for inspection, not as a pretrained model input."""

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitude = normalize_to_uint8(magnitude)

    return magnitude


def normalize_to_uint8(array):
    """Rescale to [0, 255]; a constant array becomes all zeros."""
    array = array.astype(np.float32)
    min_value = array.min()
    max_value = array.max()

    if max_value - min_value < 1e-8:
        return np.zeros_like(array, dtype=np.uint8)

    normalized = (array - min_value) / (max_value - min_value)
    normalized = normalized * 255

    return normalized.astype(np.uint8)
