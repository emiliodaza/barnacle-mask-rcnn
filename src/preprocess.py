import cv2
import numpy as np


def preprocess_raw(rgb_image):
    # baseline preprocessing: using the way RGB image
    return rgb_image


def preprocess_gaussian(rgb_image, kernel_size=3):
    # mild smoothing b/c strong blur may affect barnacle boundaries
    # to prevent the impact of salt or pepper noise in prediction
    return cv2.GaussianBlur(rgb_image, (kernel_size, kernel_size), 0)


def preprocess_clahe(rgb_image):
    # contraste enhancement: some barnacles blend into dark rock while other are path/white so contrasting is useful to find distinct regions
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))

    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def preprocess_gaussian_then_clahe(rgb_image):
    # combined preprocessing: reducing small noise, then enhancing local contrast
    blurred = preprocess_gaussian(rgb_image, kernel_size=3)
    enhanced = preprocess_clahe(blurred)

    return enhanced


def preprocess_sobel_visual(rgb_image):
    # sobel visualization mainly for analysis
    # hypothesis: not as the main Mask r-CNN input b/c pretrained Mask R-CNN expects natural RGB images

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitude = normalize_to_uint8(magnitude)

    return magnitude


def normalize_to_uint8(array):
    array = array.astype(np.float32)
    min_value = array.min()
    max_value = array.max()

    if max_value - min_value < 1e-8:
        return np.zeros_like(array, dtype=np.uint8)

    normalized = (array - min_value) / (max_value - min_value)
    normalized = normalized * 255

    return normalized.astype(np.uint8)
